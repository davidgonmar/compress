import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.prune import to_pruned, PruningPolicy
from compress.pruning_strats import (
    linear_granularity_from_str,
    conv2d_granularity_from_str,
)
import copy
import argparse
from torch import nn


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--print_model", action="store_true")
parser.add_argument("--dataset", type=str, default="mnist")
args = parser.parse_args()


def maybe_print_model(model):
    if args.print_model:
        print(model)


args.linear_pruning = "2:4"
args.conv2d_pruning = "2:4"


def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss += criterion(y_hat, y).item() * x.size(0)
            correct += (y_hat.argmax(dim=-1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


model = torch.load(args.save_path)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_dataset = (
    datasets.MNIST(root="data", train=False, transform=transform, download=True)
    if args.dataset == "mnist"
    else datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
loss = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {loss[0]}, Test Accuracy: {loss[1]}")


ratios = [1.0]
for ratio in ratios:
    model_lr = to_pruned(
        model,
        policy=PruningPolicy(
            cfg={
                nn.Linear: linear_granularity_from_str(args.linear_pruning),
                nn.Conv2d: conv2d_granularity_from_str(args.conv2d_pruning),
                nn.LazyLinear: linear_granularity_from_str(args.linear_pruning),
                nn.LazyConv2d: conv2d_granularity_from_str(args.conv2d_pruning),
            }
        ),
        ratio_to_keep=ratio,
        ratio_to_keep_in_group=0.5,  # 2:4 keeps 50% of the weights in each group
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
    )
    test_loss, test_acc = evaluate(model_lr, test_loader, criterion, device)
    print(f"Ratio: {ratio}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    maybe_print_model(model_lr)
