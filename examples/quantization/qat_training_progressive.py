import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from compress.quantization import prepare_for_qat, get_fuse_bn_keys, requantize_lsq
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
parser.add_argument("--method", default="qat", type=str)
parser.add_argument("--bits_list", nargs="+", type=int, default=[8, 4, 2])
parser.add_argument("--epoch_milestones", nargs="+", type=int, default=[25, 55])
parser.add_argument(
    "--leave_last_layer_8_bits", type=lambda x: str(x).lower() == "true", default=True
)
parser.add_argument("--model_name", default="resnet20", type=str)
parser.add_argument("--pretrained_path", default="resnet20.pth", type=str)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--epochs", default=90, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
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
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
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


bits_schedule = list(zip(args.epoch_milestones, args.bits_list[1:]))
bits_schedule.sort()
schedule_index = 0

current_bits = args.bits_list[0]
specs = get_recipe_quant(args.model_name)(
    bits_activation=current_bits,
    bits_weight=current_bits,
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
    if (
        schedule_index < len(bits_schedule)
        and epoch == bits_schedule[schedule_index][0]
    ):
        current_bits = bits_schedule[schedule_index][1]
        specs = get_recipe_quant(args.model_name)(
            bits_activation=current_bits,
            bits_weight=current_bits,
            leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
            clip_percentile=99,
            symmetric=True,
        )
        requantize_lsq(model, specs=specs)
        print(model)
        schedule_index += 1

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
            "bits": current_bits,
        }
    )

filename = "qat_results_schedule.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=4)
