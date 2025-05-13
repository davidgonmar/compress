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
from compress.quantization.recipes import (
    get_resnet20_recipe_quant,
)


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument(
    "--method", default="qat", type=str, help="method to use"
)  # qat, lsq
parser.add_argument(
    "--nbits", default=2, type=int, help="number of bits for quantization"
)
parser.add_argument(
    "--leave_last_layer_8_bits",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
)
args = parser.parse_args()

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
    "resnet20",
    pretrained_path="resnet20.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

specs = get_resnet20_recipe_quant(
    bits_activation=4,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=0.99,
    symmetric=True,
)

model = prepare_for_qat(
    model,
    specs=specs,
    use_lsq=True,
    use_PACT=True,
    data_batch=next(iter(train_loader))[0][:100].to(device),
    fuse_bn_keys=get_fuse_bn_keys("resnet20"),
    online=False,
)  # W8A8
print(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(100):
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
