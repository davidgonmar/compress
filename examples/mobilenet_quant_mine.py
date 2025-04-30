import argparse
import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from compress.quantization import (
    IntAffineQuantizationSpec,
    prepare_for_qat,
    merge_qat_model,
    IntAffineQuantizationMode,
    get_quant_dict,
    merge_dicts,
    LSQConv2d,
    LSQLinear,
)
from compress.quantization.qat import ActivationCatcher
from compress.knowledge_distillation import knowledge_distillation_loss


def flat_feat(t: torch.Tensor) -> torch.Tensor:
    return t.view(t.size(0), -1)


class MineCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)


def dv_loss(
    t_joint: torch.Tensor, t_marg: torch.Tensor, ma_et: torch.Tensor, ema: float = 0.01
):
    et = torch.exp(t_marg)
    ma_et = ma_et + ema * (et.mean() - ma_et).detach()
    loss = -(t_joint.mean() - (et.mean().log() * (et.mean().detach() / ma_et)))
    return loss, ma_et.detach()


def quantile(tensor, q, dim=None, keepdim=False):
    assert 0 <= q <= 1, "Quantile must be in [0, 1]"
    if dim is None:
        tensor = tensor.flatten()
        dim = 0
    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value
    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


torch.quantile = quantile

parser = argparse.ArgumentParser("PyTorch CIFAR10 QAT with MINE and MobileNetV2")
parser.add_argument("--leave_edge_layers_8_bits", action="store_true")
parser.add_argument("--load_from", type=str)
parser.add_argument("--bits_schedule", type=str, default="8*8")
parser.add_argument("--epochs_schedule", type=str, default="10")
parser.add_argument("--clip_percentile", type=float, default=0.99)
parser.add_argument("--lambda_mi", type=float, default=0.05)
parser.add_argument("--lambda_kd", type=float, default=0.1)
args = parser.parse_args()

weight_sched_str, act_sched_str = args.bits_schedule.split("*")
sched1_w = list(map(int, weight_sched_str.split(",")))
sched1_a = list(map(int, act_sched_str.split(",")))
sched2 = list(map(int, args.epochs_schedule.split(",")))
sched = list(zip(sched1_w, sched1_a, sched2))
print("Bits schedule (weight, act, epoch):", sched)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True, transform=data_transform),
    batch_size=16,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=data_transform),
    batch_size=512,
    shuffle=False,
)

model_fp = mobilenet_v2(num_classes=10)
model_fp.features[0][0] = nn.Conv2d(
    3, 32, kernel_size=3, stride=1, padding=1, bias=False
)
model_fp.classifier[1] = nn.Linear(model_fp.last_channel, 10)

state = torch.hub.load_state_dict_from_url(
    "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", progress=True
)
del state["classifier.1.weight"]
del state["classifier.1.bias"]
model_fp.load_state_dict(state, strict=False)

if args.load_from:
    checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_fp.load_state_dict(checkpoint["model"], strict=False)
    elif isinstance(checkpoint, dict):
        model_fp.load_state_dict(checkpoint, strict=False)
    elif isinstance(checkpoint, nn.Module):
        model_fp.load_state_dict(checkpoint.state_dict(), strict=True)

teacher = copy.deepcopy(model_fp).eval().to(device)
args.nbits_w, args.nbits_a = sched[0][:2]


def get_specs(model):
    specs = {
        "linear": IntAffineQuantizationSpec(
            nbits=args.nbits_a,
            signed=False,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
        "conv2d": IntAffineQuantizationSpec(
            nbits=args.nbits_a,
            signed=False,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
    }
    weight_specs = {
        "linear": IntAffineQuantizationSpec(
            nbits=args.nbits_w,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
        "conv2d": IntAffineQuantizationSpec(
            nbits=args.nbits_w,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
    }
    quant_dict = merge_dicts(
        get_quant_dict(model, "linear", specs["linear"], weight_specs["linear"]),
        get_quant_dict(model, "conv2d", specs["conv2d"], weight_specs["conv2d"]),
    )

    edge_bits = 8 if args.leave_edge_layers_8_bits else args.nbits_a
    for k in ("classifier.1", "features.0.0"):
        quant_dict[k]["input"] = IntAffineQuantizationSpec(
            nbits=edge_bits,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        )
        quant_dict[k]["weight"] = IntAffineQuantizationSpec(
            nbits=edge_bits,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        )
    return quant_dict


student = model_fp.to(device)
one_batch = next(iter(train_loader))[0][:4]
student = prepare_for_qat(
    student, specs=get_specs(student), use_PACT=True, use_lsq=True, data_batch=one_batch
).to(device)

print(student)
print(teacher)
mine_layers = [
    "features.0.0",
    # "features.1.conv.0.0",
    # "features.1.conv.1",
    # "features.2.conv.0.0",
    "features.2.conv.1.0",
    # "features.2.conv.2",
    # "features.3.conv.0.0",
    # "features.3.conv.1.0",
    "features.3.conv.2",
    # "features.4.conv.0.0",
    # "features.4.conv.1.0",
    # "features.4.conv.2",
    "features.5.conv.0.0",
    # "features.5.conv.1.0",
    # "features.5.conv.2",
    # "features.6.conv.0.0",
    # "features.6.conv.1.0",
    "features.6.conv.2",
    # "features.7.conv.0.0",
    # "features.7.conv.1.0",
    # "features.7.conv.2",
    "features.8.conv.0.0",
    # "features.8.conv.1.0",
    # "features.8.conv.2",
    # "features.9.conv.0.0",
    # "features.9.conv.1.0",
    "features.9.conv.2",
    # "features.10.conv.0.0",
    # "features.10.conv.1.0",
    # "features.10.conv.2",
    # "features.11.conv.0.0",
    # "features.11.conv.1.0",
    "features.11.conv.2",
    # "features.12.conv.0.0",
    # "features.12.conv.1.0",
    # "features.12.conv.2",
    # "features.13.conv.0.0",
    # "features.13.conv.1.0",
    # "features.13.conv.2",
    # "features.14.conv.0.0",
    # "features.14.conv.1.0",
    "features.14.conv.2",
    # "features.15.conv.0.0",
    # "features.15.conv.1.0",
    # "features.15.conv.2",
    # "features.16.conv.0.0",
    # "features.16.conv.1.0",
    # "features.16.conv.2",
    # "features.17.conv.0.0",
    # "features.17.conv.1.0",
    # "features.17.conv.2",
    # "features.18.0",
    "classifier.1",
]
catcher_T = ActivationCatcher(layer_types=(nn.Conv2d, nn.Linear), include_inputs=False)
catcher_S = ActivationCatcher(layer_types=(LSQConv2d, LSQLinear), include_inputs=False)
catcher_T.initialize(teacher)
catcher_S.initialize(student)

critics: Dict[str, MineCritic] = {}
ma_et: Dict[str, torch.Tensor] = {}
critic_params: List[nn.Parameter] = []

criterion = nn.CrossEntropyLoss()
optimizer_S = optim.AdamW(student.parameters(), lr=1e-4)
optimizer_critic = None
scheduler_S = StepLR(optimizer_S, step_size=8, gamma=0.1)

epochs = 1000
print("Starting training with MINE regularizer …")

for epoch in range(epochs):
    if sched and epoch == sched[0][2]:
        print(f"[epoch {epoch}] changing bit-width to w={sched[0][0]}, a={sched[0][1]}")
        student = merge_qat_model(student, inplace=False)
        args.nbits_w, args.nbits_a = sched[0][:2]
        sched.pop(0)
        student = prepare_for_qat(
            student,
            specs=get_specs(student),
            use_PACT=True,
            use_lsq=True,
            data_batch=one_batch,
        ).to(device)
        catcher_S.remove_hooks(id(student))
        catcher_S.initialize(student)
        optimizer_S = optim.AdamW(student.parameters(), lr=1e-4)
        scheduler_S = StepLR(optimizer_S, step_size=8, gamma=0.1)

    student.train()
    teacher.eval()
    running_cls, running_mi, running_kd = 0.0, 0.0, 0.0

    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.no_grad():
            _ = student(imgs)
        _ = teacher(imgs)
        act_S = catcher_S.get_last_activations(student, clear=True)
        act_T = catcher_T.get_last_activations(teacher, clear=True)

        if not critics:
            for name in mine_layers:
                d = flat_feat(act_T[name]["output"]).size(1)
                critics[name] = MineCritic(d * 2).to(device)
                ma_et[name] = torch.tensor(1.0, device=device)
                critic_params += list(critics[name].parameters())
            optimizer_critic = optim.Adam(critic_params, lr=5e-4)

        loss_critic_total = torch.tensor(0.0, device=device)
        for name in mine_layers:
            t_feat = flat_feat(act_T[name]["output"])
            s_feat = flat_feat(act_S[name]["output"])
            joint = critics[name](torch.cat([t_feat, s_feat], dim=1))
            s_perm = s_feat.roll(1, 0)
            marg = critics[name](torch.cat([t_feat, s_perm], dim=1))
            l, ma_et[name] = dv_loss(joint, marg, ma_et[name])
            loss_critic_total += l

        optimizer_critic.zero_grad()
        loss_critic_total.backward()
        optimizer_critic.step()

        logits_S = student(imgs)
        with torch.no_grad():
            logits_T = teacher(imgs)
            act_T = catcher_T.get_last_activations(teacher, clear=True)
        act_S = catcher_S.get_last_activations(student, clear=True)

        mi_reg_sum = torch.tensor(0.0, device=device)
        for name in mine_layers:
            t_feat = flat_feat(act_T[name]["output"])
            s_feat = flat_feat(act_S[name]["output"])
            joint = critics[name](torch.cat([t_feat, s_feat], dim=1))
            s_perm = s_feat.roll(1, 0)
            marg = critics[name](torch.cat([t_feat, s_perm], dim=1))
            I_hat = joint.mean() - marg.exp().mean().log()
            mi_reg_sum -= I_hat

        cls_loss = criterion(logits_S, lbls)
        kd_loss = knowledge_distillation_loss(logits_S, logits_T)

        total_loss = (
            0.6 * cls_loss + args.lambda_kd * kd_loss + args.lambda_mi * mi_reg_sum
        )

        optimizer_S.zero_grad()
        total_loss.backward()
        optimizer_S.step()

        running_cls += cls_loss.item() * imgs.size(0)
        running_mi += mi_reg_sum.item() * imgs.size(0)
        running_kd += kd_loss.item() * imgs.size(0)

    scheduler_S.step()
    n = len(train_loader.dataset)
    print(
        f"Epoch {epoch:03d}  |  CE {running_cls/n:.4f}  KD {running_kd/n:.4f}  MI {running_mi/n:.4f}"
    )

    student.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = student(imgs)
            pred = out.argmax(1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
    acc = 100.0 * correct / total
    print(f"          → val-acc {acc:.2f}%")

    if epoch % 20 == 0:
        torch.save(
            {"model": student.state_dict()}, f"qat_mine_mobilenetv2_e{epoch}.pth"
        )

print("Training complete.")
