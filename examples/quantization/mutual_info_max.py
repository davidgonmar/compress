import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from compress.quantization import prepare_for_qat
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.quantization.recipes import get_recipe_quant
from compress.layer_fusion import resnet20_fuse_pairs


def attach_feature_hooks(model, layer_names, store):
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(
                module.register_forward_hook(
                    lambda _, __, out, key=name: store.__setitem__(key, out)
                )
            )
    return hooks


class TwoConvAdapter(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_out, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class ComplexAdapter(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_out, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


parser = argparse.ArgumentParser(
    description="CIFAR-10 QAT with feature-KD (2-conv adapters)"
)
parser.add_argument(
    "--nbits", default=2, type=int, help="bit-width for student activations & weights"
)
parser.add_argument(
    "--leave_last_layer_8_bits",
    type=lambda x: str(x).lower() == "true",
    default=True,
    help="leave edge layers in 8-bit precision",
)
parser.add_argument(
    "--alpha", default=0.5, type=float, help="weight for CE vs feature-KD loss (0-1)"
)
parser.add_argument("--epochs", default=90, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--val_batch_size", default=512, type=int)
parser.add_argument("--model_name", type=str, default="resnet20")
parser.add_argument(
    "--pretrained_path",
    type=str,
    default="resnet20.pth",
    help="path to pretrained model",
)
parser.add_argument(
    "--matcher_steps",
    default=1,
    type=int,
    help="number of matcher updates per student update",
)

parser.add_argument(
    "--complex_adapter",
    action="store_true",
    help="use complex adapter instead of 2-conv adapter",
)

args = parser.parse_args()
assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0,1]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_tf = [transforms.ToTensor(), transforms.Normalize((cifar10_mean), (cifar10_std))]
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "./data",
        True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                *common_tf,
            ],
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "./data", False, download=True, transform=transforms.Compose(common_tf)
    ),
    batch_size=args.val_batch_size,
    shuffle=False,
)

print("Loading teacher model…")
teacher = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    model_args={"num_classes": 10},
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print("Preparing student for QAT…")
stu_raw = load_vision_model(
    "resnet20",
    pretrained_path="resnet20.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    model_args={"num_classes": 10},
)
quant_specs = get_recipe_quant(args.model_name)(
    bits_activation=2,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=0.995,
    symmetric=True,
)
student = prepare_for_qat(
    stu_raw,
    specs=quant_specs,
    use_lsq=True,
    data_batch=next(iter(train_loader))[0][:100].to(device),
    method_args={"online": False},
    fuse_bn_keys=resnet20_fuse_pairs,
).to(device)

writer = SummaryWriter()

print(student)
layer_names = (
    ["layer1", "layer2", "layer3", "layer4"]
    if args.model_name == "resnet18"
    else ["layer1", "layer2", "layer3"]
)
teacher_feats, student_feats = {}, {}
teacher_hooks = attach_feature_hooks(teacher, layer_names, teacher_feats)
student_hooks = attach_feature_hooks(student, layer_names, student_feats)

dummy = next(iter(train_loader))[0][:100].to(device)
with torch.no_grad():
    teacher(dummy)
    student(dummy)

estimators = {}
for name in layer_names:
    c_in = student_feats[name].shape[1]
    c_out = teacher_feats[name].shape[1]
    estimators[name] = (
        TwoConvAdapter(c_in, c_out).to(device)
        if not args.complex_adapter
        else ComplexAdapter(c_in, c_out).to(device)
    )

print(
    {
        n: f"{m.net[0].in_channels}->{m.net[-1].out_channels}"
        for n, m in estimators.items()
    }
)

teacher_feats.clear()
student_feats.clear()

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

optimizer_student = torch.optim.SGD(
    student.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
)

optimizer_matchers = torch.optim.SGD(
    [p for m in estimators.values() for p in m.parameters()],
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
)

scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_student, T_max=args.epochs
)

scheduler_matchers = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_matchers, T_max=args.epochs
)

print("Starting training…")
for epoch in range(1, args.epochs + 1):
    student.train()
    for est in estimators.values():
        est.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    ):
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        teacher_feats.clear()
        student_feats.clear()
        with torch.no_grad():
            teacher(images)
        logits_s = student(images)

        ce_loss = criterion_ce(logits_s, labels)
        kd_losses = [
            criterion_mse(
                estimators[n](student_feats[n]).reshape(-1),
                teacher_feats[n].detach().reshape(-1),
            )
            for n in layer_names
        ]
        kd_loss = torch.stack(kd_losses).mean()
        loss = args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss

        optimizer_student.zero_grad(set_to_none=True)
        loss.backward(retain_graph=args.matcher_steps > 1)

        # log per-layer weight gradient norms
        for name, param in student.named_parameters():
            if "weight" in name and param.grad is not None:
                writer.add_scalar(
                    f"grad_student/{name.replace('.', '/')}",
                    param.grad.norm().item(),
                    global_step,
                )

        writer.add_scalar("train/ce_loss", ce_loss.item(), global_step)
        writer.add_scalar("train/kd_loss", kd_loss.item(), global_step)
        writer.add_scalar("train/loss", loss.item(), global_step)
        optimizer_student.step()

        for i in range(args.matcher_steps):
            optimizer_matchers.zero_grad(set_to_none=True)
            kd_losses = [
                criterion_mse(
                    estimators[n](student_feats[n].detach()).reshape(-1),
                    teacher_feats[n].detach().reshape(-1),
                )
                for n in layer_names
            ]
            kd_loss = torch.stack(kd_losses).mean()
            kd_loss.backward()
            if i == 0:
                grad_norm_matcher = torch.sqrt(
                    sum(
                        p.grad.norm() ** 2
                        for m in estimators.values()
                        for p in m.parameters()
                        if p.grad is not None
                    )
                )
                writer.add_scalar("grad/matcher", grad_norm_matcher, global_step)
            optimizer_matchers.step()

        running_loss += loss.item() * images.size(0)

    scheduler_student.step()
    scheduler_matchers.step()
    avg_loss = running_loss / len(train_loader.dataset)

    student.eval()
    for est in estimators.values():
        est.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            preds = student(images).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100.0 * correct / total

    print(
        f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val Acc {acc:.2f}% | KD {kd_loss.item():.4f}"
    )
    writer.add_scalar("train/avg_loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", acc, epoch)

for h in teacher_hooks + student_hooks:
    h.remove()

writer.close()
