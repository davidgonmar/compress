import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

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


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ComplexAdapter(nn.Module):
    """
    Adapter with multi-scale dilated convs + SE global context.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.net = nn.Sequential(
            # local 3×3 conv
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # wider context via dilation=2
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # even wider via dilation=4
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # squeeze-and-excitation for global context
            SEBlock(hidden),
            # project down to teacher’s channel size
            nn.Conv2d(hidden, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
parser.add_argument("--epochs", default=180, type=int)
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
    "--student_each",
    default=1,
    type=int,
    help="number of consecutive batches to update the student network",
)
parser.add_argument(
    "--matchers_each",
    default=3,
    type=int,
    help="number of consecutive batches to update the matcher adapters",
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
    bits_activation=args.nbits,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=0.995,
    symmetric=True,
)
student = prepare_for_qat(
    stu_raw,
    specs=quant_specs,
    use_lsq=True,
    data_batch=next(iter(train_loader))[0][:1024].to(device),
    method_args={"online": False},
    fuse_bn_keys=resnet20_fuse_pairs,
).to(device)

print(student)
layer_names = (
    ["layer1", "layer2", "layer3", "layer4"]
    if args.model_name == "resnet18"
    else ["layer1", "layer2", "layer3"]
)
teacher_feats, student_feats = {}, {}
teacher_hooks = attach_feature_hooks(teacher, layer_names, teacher_feats)
student_hooks = attach_feature_hooks(student, layer_names, student_feats)

# gather layer output sizes
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
cm = nn.MSELoss()

optimizer_student = torch.optim.SGD(
    student.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
)

optimizer_matchers = torch.optim.SGD(
    [p for m in estimators.values() for p in m.parameters()],
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
)

scheduler_student = torch.optim.lr_scheduler.StepLR(
    optimizer_student, step_size=25, gamma=0.1
)

scheduler_matchers = torch.optim.lr_scheduler.StepLR(
    optimizer_matchers, step_size=1000, gamma=0.1
)


def criterion_mse(a, b):
    return cm(a, b)
    a_hat = (a - a.mean((2, 3), keepdim=True)) / (a.std((2, 3), keepdim=True) + 1e-5)
    b_hat = (b - b.mean((2, 3), keepdim=True)) / (b.std((2, 3), keepdim=True) + 1e-5)
    return cm(a_hat, b_hat)


print(teacher)

print("Starting training…")
for epoch in range(1, args.epochs + 1):
    student.train()
    for est in estimators.values():
        est.train()
    running_loss = 0.0

    cycle_len = args.student_each + args.matchers_each
    for batch_idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    ):
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
                estimators[n](student_feats[n]),
                teacher_feats[n].detach(),
            )
            for n in layer_names
        ]
        kd_loss = torch.stack(kd_losses).mean()
        # Decide which set of parameters to update this batch
        if batch_idx % cycle_len:
            # Student update phase
            loss = args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss
            optimizer_student.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_student.step()
            running_loss += loss.item() * images.size(0)
        else:
            # Matcher update phase (student frozen)
            optimizer_matchers.zero_grad(set_to_none=True)
            kd_losses_detached = [
                criterion_mse(
                    estimators[n](student_feats[n].detach()),
                    teacher_feats[n].detach(),
                )
                for n in layer_names
            ]
            kd_loss_detached = torch.stack(kd_losses_detached).mean()
            kd_loss_detached.backward()
            optimizer_matchers.step()

    scheduler_student.step()
    scheduler_matchers.step()
    avg_loss = running_loss / len(train_loader.dataset)

    # Validation (student network only)
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
        f"Epoch {epoch:3d} | Avg Loss {avg_loss:.4f} | Val Acc {acc:.2f}% | KD {kd_loss.item():.4f}"
    )

# Clean up hooks
for h in teacher_hooks + student_hooks:
    h.remove()
