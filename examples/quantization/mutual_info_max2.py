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
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            SEBlock(hidden),
            nn.Conv2d(hidden, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 QAT with feature-KD (stage-wise training)"
    )
    parser.add_argument(
        "--nbits",
        default=2,
        type=int,
        help="bit-width for student activations & weights",
    )
    parser.add_argument(
        "--leave_last_layer_8_bits",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="leave edge layers in 8-bit precision",
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="weight for CE vs feature-KD loss (0-1)",
    )
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
        "--complex_adapter",
        action="store_true",
        help="use complex adapter instead of 2-conv adapter",
    )
    parser.add_argument(
        "--stage_epochs",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="list of epochs for each stage (one per layer)",
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=1000,
        help="epochs for full training after all stages",
    )

    args = parser.parse_args()
    assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0,1]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_tf = [
        transforms.ToTensor(),
        transforms.Normalize((cifar10_mean), (cifar10_std)),
    ]

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
                ]
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

    # Load teacher
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

    # Load and prepare student
    stu_raw = load_vision_model(
        "resnet20",
        pretrained_path=args.pretrained_path,
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

    # Hooks and adapters
    layer_names = ["layer1", "layer2", "layer3"]
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
    teacher_feats.clear()
    student_feats.clear()

    # Loss and optimizers
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    opt_s = torch.optim.SGD(
        student.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
    )
    opt_m = torch.optim.SGD(
        [p for m in estimators.values() for p in m.parameters()],
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )
    sch_s = torch.optim.lr_scheduler.StepLR(opt_s, step_size=80, gamma=0.1)
    sch_m = torch.optim.lr_scheduler.StepLR(opt_m, step_size=80, gamma=0.1)

    def freeze_student(layers_to_unfreeze):
        for n, p in student.named_parameters():
            p.requires_grad = (
                any(n.startswith(l) for l in layers_to_unfreeze)
                or n.startswith("conv1")
                or n.startswith("fc")
            )

    # Stage-wise training
    n_layers = len(layer_names)
    assert (
        len(args.stage_epochs) == n_layers
    ), "--stage_epochs length must match number of stages"

    for idx, epochs in enumerate(args.stage_epochs):
        active = layer_names[: idx + 1]
        print(
            f"\nStage {idx+1}/{n_layers}: training up to {active[-1]} for {epochs} epochs"
        )
        freeze_student(active)
        for nm, m in estimators.items():
            for p in m.parameters():
                p.requires_grad = nm in active

        for ep in range(1, epochs + 1):
            student.train()
            [m.train() for m in estimators.values()]
            total_loss = 0
            for bi, (imgs, lbls) in enumerate(
                tqdm(train_loader, desc=f"Stage {idx+1} Ep {ep}/{epochs}")
            ):
                imgs, lbls = imgs.to(device), lbls.to(device)
                teacher_feats.clear()
                student_feats.clear()
                with torch.no_grad():
                    teacher(imgs)
                out_s = student(imgs)
                loss_ce = ce(out_s, lbls)
                kd_losses = [
                    mse(estimators[n](student_feats[n]), teacher_feats[n].detach())
                    for n in active
                ]
                loss_kd = torch.stack(kd_losses).mean()
                # update student and matchers alternately
                if bi % 2:
                    loss = args.alpha * loss_ce + (1 - args.alpha) * loss_kd
                    opt_s.zero_grad()
                    loss.backward()
                    opt_s.step()
                    total_loss += loss.item() * imgs.size(0)
                else:
                    opt_m.zero_grad()
                    det_losses = [
                        mse(
                            estimators[n](student_feats[n].detach()),
                            teacher_feats[n].detach(),
                        )
                        for n in active
                    ]
                    lm = torch.stack(det_losses).mean()
                    lm.backward()
                    opt_m.step()

            sch_s.step()
            sch_m.step()
            avg_loss = total_loss / len(train_loader.dataset)
            # validation
            student.eval()
            [m.eval() for m in estimators.values()]
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = student(imgs).argmax(1)
                    total += lbls.size(0)
                    correct += (preds == lbls).sum().item()
            print(
                f"Stage {idx+1} Ep {ep} | Loss {avg_loss:.4f} | Acc {100*correct/total:.2f}%"
            )

    # Final full training
    if args.final_epochs > 0:
        print(
            f"\nFinal training for {args.final_epochs} epochs with all layers and adapters active"
        )
        # unfreeze all
        freeze_student(layer_names)
        for m in estimators.values():
            for p in m.parameters():
                p.requires_grad = True
        for ep in range(1, args.final_epochs + 1):
            student.train()
            [m.train() for m in estimators.values()]
            total_loss = 0
            for bi, (imgs, lbls) in enumerate(
                tqdm(train_loader, desc=f"Final Ep {ep}/{args.final_epochs}")
            ):
                imgs, lbls = imgs.to(device), lbls.to(device)
                teacher_feats.clear()
                student_feats.clear()
                with torch.no_grad():
                    teacher(imgs)
                out_s = student(imgs)
                loss_ce = ce(out_s, lbls)
                kd_losses = [
                    mse(estimators[n](student_feats[n]), teacher_feats[n].detach())
                    for n in layer_names
                ]
                loss_kd = torch.stack(kd_losses).mean()
                if bi % 2:
                    loss = (1 - args.alpha) * loss_kd
                    opt_s.zero_grad()
                    loss.backward()
                    opt_s.step()
                    total_loss += loss.item() * imgs.size(0)
                else:
                    opt_m.zero_grad()
                    det_losses = [
                        mse(
                            estimators[n](student_feats[n].detach()),
                            teacher_feats[n].detach(),
                        )
                        for n in layer_names
                    ]
                    lm = torch.stack(det_losses).mean()
                    lm.backward()
                    opt_m.step()
            sch_s.step()
            sch_m.step()
            avg_loss = total_loss / len(train_loader.dataset)
            student.eval()
            [m.eval() for m in estimators.values()]
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = student(imgs).argmax(1)
                    total += lbls.size(0)
                    correct += (preds == lbls).sum().item()
            print(f"Final Ep {ep} | Loss {avg_loss:.4f} | Acc {100*correct/total:.2f}%")

    # cleanup
    for h in teacher_hooks + student_hooks:
        h.remove()


if __name__ == "__main__":
    main()
