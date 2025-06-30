import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from compress.quantization import prepare_for_qat, get_fuse_bn_keys
import argparse
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.quantization.recipes import get_recipe_quant
import json

torch.manual_seed(0)
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument(
    "--method", default="qat", type=str, help="method to use"
)  # qat, lsq
parser.add_argument(
    "--nbits_activations", default=2, type=int, help="number of bits for quantization"
)
parser.add_argument(
    "--nbits_weights",
    default=2,
    type=int,
    help="number of bits for quantization (weights)",
)
parser.add_argument(
    "--leave_last_layer_8_bits",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
)
parser.add_argument(
    "--model_name",
    default="resnet20",
    type=str,
    help="model name to use",
)

parser.add_argument(
    "--pretrained_path",
    default="resnet20.pth",
    type=str,
    help="path to pretrained model",
)

parser.add_argument(
    "--batch_size", default=256, type=int, help="batch size for training"
)

parser.add_argument(
    "--epochs", default=90, type=int, help="number of epochs for training"
)

parser.add_argument("--lr", default=0.01, type=float, help="learning rate for training")

parser.add_argument(
    "--momentum", default=0.9, type=float, help="momentum for SGD optimizer"
)

parser.add_argument(
    "--weight_decay", default=5e-4, type=float, help="weight decay for SGD optimizer"
)

args = parser.parse_args()

results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=data_transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    ),
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

specs = get_recipe_quant(args.model_name)(
    bits_activation=args.nbits_activations,
    bits_weight=args.nbits_weights,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=99,
    symmetric=True,
)

model = prepare_for_qat(
    model,
    specs=specs,
    use_lsq=args.method == "lsq",
    data_batch=next(iter(train_loader))[0][:100].to(device),
    fuse_bn_keys=get_fuse_bn_keys(args.model_name),
)

print(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(args.epochs):
    model.train()
    train_loss_acc = 0.0
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

        train_loss_acc += train_loss.item() * images.size(0)

    scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss_acc / len(train_loader.dataset):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")
    results.append(
        {
            "epoch": epoch + 1,
            "loss": train_loss_acc / len(train_loader.dataset),
            "accuracy": accuracy,
        }
    )

filename = f"qat_results_w{args.nbits_weights}a{args.nbits_activations}.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=4)
