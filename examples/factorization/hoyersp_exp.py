import argparse
import math
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.factorization.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
)
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.factorization.utils import matrix_approx_rank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune ResNet on CIFAR-10 with constant regularization and dynamic adjustment"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size for training and validation",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="weight decay for optimizer"
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=60,
        help="step size (in epochs) for StepLR scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="multiplicative factor of learning rate decay",
    )

    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.003,
        help="constant regularization weight",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=0.90,
        help="accuracy threshold to trigger regularization adjustment",
    )
    parser.add_argument(
        "--reg_divisor",
        type=float,
        default=10000.0,
        help="divisor to reduce regularization weight by when below threshold",
    )

    parser.add_argument(
        "--model_name", type=str, default="resnet20", help="vision model architecture"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="resnet20.pth",
        help="path to pretrained weights",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="cifar10_resnet20_hoyer_finetuned.pth",
        help="path to save the best model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model(
        args.model_name,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model_name),
        modifier_after_load=None,
        model_args={"num_classes": 10},
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    params_and_reshapers = extract_weights_and_reshapers(
        model,
        cls_list=(torch.nn.Conv2d,),
        keywords={"weight", "kernel"},
    )
    regularizer = SingularValuesRegularizer(
        metric="squared_hoyer_sparsity",
        params_and_reshapers=params_and_reshapers,
        weights=1.0,
        normalize=False,
    )

    base_reg_weight = args.reg_weight
    reg_weight = base_reg_weight
    reg_reduced = False

    for epoch in range(args.epochs):
        model.train()
        train_loss, reg_loss = 0.0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)
            reg = regularizer()
            total_loss = loss + reg_weight * reg

            total_loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            reg_loss += reg.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        reg_loss /= len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, Reg Loss: {reg_loss:.4f}"
        )

        scheduler.step()
        print(
            f"Learning Rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.5f}"
        )
        print(f"reg_weight after epoch {epoch+1}: {reg_weight:.6f}")

        print("Approximate ranks per layer:")

        def _mul(shape):
            res = 1
            for d in shape:
                res *= d
            return res

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                approx = matrix_approx_rank(module.weight)
                total = _mul(module.weight.shape)
                print(
                    f"{name}: rank {approx} / {module.weight.shape[0]}, total elems {total}"
                )

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                val_loss += criterion(y_hat, y).item() * x.size(0)
                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}")

        if acc < args.acc_threshold and not reg_reduced:
            reg_weight /= args.reg_divisor
            reg_reduced = True
            print(
                f"Accuracy below {args.acc_threshold:.2f}, "
                f"reducing reg_weight by factor {args.reg_divisor}: now {reg_weight:.6f}"
            )
        elif acc >= args.acc_threshold and reg_reduced:
            reg_weight = base_reg_weight
            reg_reduced = False
            print(
                f"Accuracy recovered above {args.acc_threshold:.2f}, "
                f"restoring reg_weight to {reg_weight:.6f}"
            )

        torch.save(model, args.save_path)


if __name__ == "__main__":
    main()
