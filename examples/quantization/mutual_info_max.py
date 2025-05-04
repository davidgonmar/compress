import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.quantization import prepare_for_qat
from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.quantization.recipes import get_resnet18_recipe_quant


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


parser = argparse.ArgumentParser(
    description="CIFAR‑10 QAT with feature‑KD (2‑conv adapters)"
)
parser.add_argument(
    "--nbits", default=2, type=int, help="bit‑width for student activations & weights"
)
parser.add_argument(
    "--leave_last_layer_8_bits",
    type=lambda x: str(x).lower() == "true",
    default=True,
    help="leave edge layers in 8‑bit precision",
)
parser.add_argument(
    "--alpha", default=0.5, type=float, help="weight for CE vs feature‑KD loss (0‑1)"
)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--val_batch_size", default=512, type=int)
args = parser.parse_args()
assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0,1]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_tf = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "./data", True, download=True, transform=transforms.Compose(common_tf)
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
    "resnet18",
    pretrained_path="resnet18.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    model_args={"num_classes": 10},
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print("Preparing student for QAT…")
stu_raw = load_vision_model(
    "resnet18",
    pretrained_path="resnet18.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    model_args={"num_classes": 10},
)
quant_specs = get_resnet18_recipe_quant(
    bits_activation=args.nbits,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=0.99,
    symmetric=True,
)
student = prepare_for_qat(
    stu_raw,
    specs=quant_specs,
    use_lsq=True,
    use_PACT=True,
    data_batch=next(iter(train_loader))[0][:4].to(device),
).to(device)

layer_names = ["layer1", "layer2", "layer3", "layer4"]
teacher_feats, student_feats = {}, {}
teacher_hooks = attach_feature_hooks(teacher, layer_names, teacher_feats)
student_hooks = attach_feature_hooks(student, layer_names, student_feats)

dummy = next(iter(train_loader))[0][:4].to(device)
with torch.no_grad():
    teacher(dummy)
    student(dummy)

estimators = {}
for name in layer_names:
    c_in = student_feats[name].shape[1]
    c_out = teacher_feats[name].shape[1]
    estimators[name] = TwoConvAdapter(c_in, c_out).to(device)

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

optimizer = torch.optim.AdamW(
    list(student.parameters())
    + [p for m in estimators.values() for p in m.parameters()],
    lr=1e-4,
)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

print("Starting training…")
for epoch in range(1, args.epochs + 1):
    student.train()
    for est in estimators.values():
        est.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
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
            criterion_mse(estimators[n](student_feats[n]), teacher_feats[n].detach())
            for n in layer_names
        ]
        kd_loss = torch.stack(kd_losses).mean()
        loss = args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()
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
        f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val Acc {acc:.2f}% | KD {kd_loss.item():.4f}"
    )

for h in teacher_hooks + student_hooks:
    h.remove()
