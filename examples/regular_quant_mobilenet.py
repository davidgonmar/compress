import argparse

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
)


# Patch in a quantile function for PyTorch < 1.7
def quantile(tensor, q, dim=None, keepdim=False):
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"
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

parser = argparse.ArgumentParser("PyTorch CIFAR10 QAT with MobileNetV2")
parser.add_argument("--load_from", type=str, required=False)
parser.add_argument("--bits_schedule", type=str, default="8*8")
parser.add_argument("--epochs_schedule", type=str, default="10")
parser.add_argument("--clip_percentile", type=float, default=0.99)
parser.add_argument(
    "--leave_edge_layers_8_bits",
    action="store_true",
    help="keep the first and last layers at 8 bits",
)
args = parser.parse_args()

# args.leave_edge_layers_8_bits = True

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
    batch_size=256,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=data_transform),
    batch_size=512,
    shuffle=False,
)

# Load pretrained MobileNetV2
model_fp = mobilenet_v2(pretrained=True)

# Modify for CIFAR-10
model_fp.classifier[1] = nn.Linear(model_fp.last_channel, 10)

# Optional: reduce first conv stride for small images
model_fp.features[0][0].stride = (1, 1)

if args.load_from:
    checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_fp.load_state_dict(checkpoint["model"], strict=False)
    elif isinstance(checkpoint, dict):
        model_fp.load_state_dict(checkpoint, strict=False)
    elif isinstance(checkpoint, nn.Module):
        model_fp.load_state_dict(checkpoint.state_dict(), strict=True)

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

    qlin = get_quant_dict(model, "linear", specs["linear"], weight_specs["linear"])
    qconv = get_quant_dict(model, "conv2d", specs["conv2d"], weight_specs["conv2d"])
    quant_dict = merge_dicts(qlin, qconv)

    if args.leave_edge_layers_8_bits:
        final_bits = 8
    else:
        final_bits = args.nbits_a

    # Adjust the first and last layers
    if hasattr(model, "features") and hasattr(model, "classifier"):
        first_layer = "features.0.0"
        last_layer = "classifier.1"
        quant_dict[first_layer] = {
            "input": IntAffineQuantizationSpec(
                nbits=final_bits,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=args.clip_percentile,
            ),
            "weight": IntAffineQuantizationSpec(
                nbits=final_bits,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=args.clip_percentile,
            ),
        }
        quant_dict[last_layer] = {
            "input": IntAffineQuantizationSpec(
                nbits=final_bits,
                signed=False,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=args.clip_percentile,
            ),
            "weight": IntAffineQuantizationSpec(
                nbits=final_bits,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=args.clip_percentile,
            ),
        }

    return quant_dict


student = model_fp.to(device)
USE_LSQ = True
one_batch = next(iter(train_loader))[0][:4]

student = prepare_for_qat(
    student,
    specs=get_specs(student),
    use_PACT=True,
    use_lsq=USE_LSQ,
    data_batch=one_batch,
)
student.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_S = optim.AdamW(student.parameters(), lr=1e-4)
scheduler_S = StepLR(optimizer_S, step_size=8, gamma=0.1)

epochs = 1000
print("Starting training with CE only …")

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
            use_lsq=USE_LSQ,
            data_batch=one_batch,
        ).to(device)
        optimizer_S = optim.AdamW(student.parameters(), lr=1e-4)
        scheduler_S = StepLR(optimizer_S, step_size=8, gamma=0.1)

    student.train()
    running_cls = 0.0

    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
        imgs, lbls = imgs.to(device), lbls.to(device)

        logits = student(imgs)
        cls_loss = criterion(logits, lbls)
        total_loss = cls_loss

        optimizer_S.zero_grad()
        total_loss.backward()
        optimizer_S.step()

        running_cls += cls_loss.item() * imgs.size(0)

    scheduler_S.step()
    n = len(train_loader.dataset)
    print(f"Epoch {epoch:03d}  |  CE {running_cls/n:.4f}")

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
        torch.save({"model": student.state_dict()}, f"qat_ce_mobilenetv2_e{epoch}.pth")

print("Training complete.")
