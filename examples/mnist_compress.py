import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from examples.utils.models import SimpleMNISTModel
from compress.factorize import to_low_rank
from compress.regularizers import singular_values_entropy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
args = parser.parse_args()


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


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_dataset = datasets.MNIST(
    root="data", train=False, transform=transform, download=True
)

test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

model = SimpleMNISTModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(
    torch.load(args.save_path, map_location=device, weights_only=True)
)

criterion = torch.nn.CrossEntropyLoss()

ratios = [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

entropies = {
    name: singular_values_entropy(module.weight).item()
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}

print(f"Per-layer entropies: {entropies}")

singular_vals = {
    name: torch.linalg.svd(module.weight, full_matrices=False).S.tolist()
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}

print("Per-layer singular values:")
for name, values in singular_vals.items():
    print(f"Layer {name}: {values}")

for ratio in ratios:
    model_lr = to_low_rank(
        model,
        ratio_to_keep=ratio,
        inplace=False,
        model_initializer=lambda: SimpleMNISTModel().to(device),
    )
    test_loss, test_acc = evaluate(model_lr, test_loader, criterion, device)
    weights = [
        module.weight
        for module in model_lr.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    print(f"Ratio: {ratio}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
