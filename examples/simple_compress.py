import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorize import to_low_rank
from examples.utils.models import MLPClassifier, ConvClassifier
import copy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--print_model", action="store_true")
args = parser.parse_args()


def maybe_print_model(model):
    if args.print_model:
        print(model)


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
    if model.input_size == (1, 28, 28)
    else datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

"""entropies = {
    name: singular_values_entropy(module.weight).item()
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}

# print(f"Per-layer entropies: {entropies}")

singular_vals = {
    name: torch.linalg.svd(module.weight, full_matrices=False).S.tolist()
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}

print("Per-layer singular values:")
for name, values in singular_vals.items():
    print(f"Layer {name}: {values}")"""

criterion = torch.nn.CrossEntropyLoss()


energies_to_remove = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
energies_to_keep = [1 - energy for energy in energies_to_remove]
for energy_keep, energy_remove in zip(energies_to_keep, energies_to_remove):
    model_lr = to_low_rank(
        model,
        energy_to_keep=energy_keep,
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
    )
    test_loss, test_acc = evaluate(model_lr, test_loader, criterion, device)
    print(
        f"Energy kept: {energy_keep}, Energy removed: {energy_remove}, Test Loss: {test_loss}, Test Accuracy: {test_acc}"
    )
    maybe_print_model(model_lr)

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
model_cls = MLPClassifier if isinstance(model, MLPClassifier) else ConvClassifier
for ratio in ratios:
    model_lr = to_low_rank(
        model,
        ratio_to_keep=ratio,
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
    )
    test_loss, test_acc = evaluate(model_lr, test_loader, criterion, device)
    print(f"Ratio: {ratio}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    maybe_print_model(model_lr)
