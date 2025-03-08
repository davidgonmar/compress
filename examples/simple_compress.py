import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorize import to_low_rank_global, to_low_rank, to_low_rank_global2
import copy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--print_model", action="store_true")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--keep_last_layer", action="store_true")
parser.add_argument("--do_global", action="store_true")
parser.add_argument("--do_global2", action="store_true")
args = parser.parse_args()


def maybe_print_model(model):
    if args.print_model:
        print(model)


def evaluate(model, loader, criterion, device):
    import time

    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        start = time.time()
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss += criterion(y_hat, y).item() * x.size(0)
            correct += (y_hat.argmax(dim=-1) == y).sum().item()
        torch.cuda.synchronize()
        elapsed = time.time() - start
    return loss / len(loader.dataset), correct / len(loader.dataset), elapsed


model = torch.load(args.save_path, weights_only=False)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_dataset = (
    datasets.MNIST(root="data", train=False, transform=transform, download=True)
    if args.dataset == "mnist"
    else datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

train_set = (
    datasets.MNIST(root="data", train=True, transform=transform, download=True)
    if args.dataset == "mnist"
    else datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
)

subset_train_set = torch.utils.data.Subset(
    train_set, torch.randint(0, len(train_set), (10000,))
)
# only get a subset of train_loader
train_loader = DataLoader(subset_train_set, batch_size=100, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
loss0, loss1, elapsed = evaluate(model, test_loader, criterion, device)
n_params = sum(p.numel() for p in model.parameters())
print(
    f"Test Loss: {loss0}, Test Accuracy: {loss1}, Number of Parameters: {n_params}, Elapsed Time: {elapsed}"
)

energies_to_remove = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
energies_to_keep = [1 - energy for energy in energies_to_remove]


def should_do(module, name):
    return isinstance(module, torch.nn.Conv2d) or (
        isinstance(module, torch.nn.Linear) and (not args.keep_last_layer)
    )


import functools

fn = None
if args.do_global:
    fn = to_low_rank_global
elif args.do_global2:
    fn = functools.partial(to_low_rank_global2, dataloader=train_loader)
else:
    fn = to_low_rank

ratios = [0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for ratio in ratios:
    model_lr = fn(
        model,
        ratio_to_keep=ratio,
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
        should_do=should_do,
    )
    n_params = sum(p.numel() for p in model_lr.parameters())
    test_loss, test_acc, elapsed = evaluate(model_lr, test_loader, criterion, device)
    print(
        f"Ratio: {ratio:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Ratio of parameters: {n_params / sum(p.numel() for p in model.parameters()):.4f}, Elapsed Time: {elapsed:.4f}, Global: {args.do_global}"
    )
    maybe_print_model(model_lr)
