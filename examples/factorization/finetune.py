import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune ResNet on CIFAR-10 with configurable hyperparameters"
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

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
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
        default=0.1,
        help="multiplicative factor of learning rate decay",
    )

    parser.add_argument(
        "--model_name", type=str, default="resnet20", help="vision model architecture"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="resnet20_lr_0.9997.pth",
        help="path to pretrained weights",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="any.pth",
        help="path to save the best model",
    )

    return parser.parse_args()


if __name__ == "__main__":
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
        accept_model_directly=True,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        scheduler.step()
        print(
            f"Learning Rate after epoch {epoch}: {optimizer.param_groups[0]['lr']:.5f}"
        )

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss += criterion(outputs, y).item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best accuracy: {best_acc:.4f}. Model saved to {args.save_path}")
