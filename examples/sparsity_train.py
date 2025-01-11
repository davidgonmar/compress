import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.regularizers import (
    SparsityRegularizer,
    extract_weights_and_pruning_granularities,
)
from examples.utils.models import MLPClassifier, ConvClassifier
import argparse
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--sparsity_metric", type=str, default="noop")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--regularizer_weight", type=float, default=1.0)
parser.add_argument("--model_type", type=str, default="simple")
parser.add_argument("--dataset", type=str, default="mnist")
args = parser.parse_args()


def get_mnist():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    val_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    return train_loader, val_loader, {"input_size": (1, 28, 28), "num_classes": 10}


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    val_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    return train_loader, val_loader, {"input_size": (3, 32, 32), "num_classes": 10}


train_loader, val_loader, model_params = (
    get_mnist()
    if args.dataset == "mnist"
    else get_cifar10()
    if args.dataset == "cifar10"
    else (None, None, None)
)

assert train_loader is not None, "Invalid dataset"

model = (
    MLPClassifier(**model_params)
    if args.model_type == "simple"
    else (
        ConvClassifier(**model_params)
        if args.model_type == "conv"
        else (
            resnet18(num_classes=model_params["num_classes"])
            if args.model_type == "resnet18"
            else None
        )
    )
)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)
sched = StepLR(optimizer, step_size=10, gamma=0.2)

regularizer_kwargs = {
    "entropy": {},
    "hoyer_sparsity": {"normalize": True},
    "scad": {"lambda_val": 0.1, "a_val": 3.7},
    "noop": {},
}

regularizer = SparsityRegularizer(
    metric=args.sparsity_metric,
    params_and_pruning_granularities=extract_weights_and_pruning_granularities(
        model,
        cls_list=(
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.LazyLinear,
            torch.nn.LazyConv2d,
        ),
        keywords={"weight", "kernel"},
    ),
    weights=args.regularizer_weight,
    **regularizer_kwargs[args.sparsity_metric],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        reg = regularizer()
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        reg_loss += reg.item() * x.size(0)

    train_loss /= len(train_loader.dataset)
    reg_loss /= len(train_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Regularization Loss: {reg_loss:.4f}"
    )

    if epoch % 5 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item() * x.size(0)

                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        torch.save(model, args.save_path)


print("Finished training. Saving model...")
torch.save(model, args.save_path)
