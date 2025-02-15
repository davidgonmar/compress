import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from tqdm import tqdm
from compress.quantization import (
    IntQuantizationSpec,
    prepare_for_qat,
    prepare_for_qat_lsq,
    to_quantized_online,
    merge_qat_model,
    merge_qat_lsq_into_offline_quantized_model,
)
import torchvision
import argparse


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument(
    "--method", default="qat", type=str, help="method to use"
)  # qat, lsq
parser.add_argument(
    "--nbits", default=4, type=int, help="number of bits for quantization"
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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
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
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
model = resnet18(num_classes=10).to(device)
# load weights from torch's pretrained model
torch_weights = torchvision.models.resnet18(pretrained=True).state_dict()
del torch_weights["fc.weight"]
del torch_weights["fc.bias"]
# do not load weights for the final layer (classification layer)
model.load_state_dict(torch_weights, strict=False)
specs = {
    "linear": IntQuantizationSpec(nbits=args.nbits, signed=True),
    "conv2d": IntQuantizationSpec(nbits=args.nbits, signed=True),
}
if args.method == "qat":
    model = prepare_for_qat(model, input_specs=specs, weight_specs=specs)  # W8A8
elif args.method == "lsq":
    model = prepare_for_qat_lsq(
        model,
        input_specs=specs,
        weight_specs=specs,
        data_batch=list(
            next(
                iter(
                    torch.utils.data.DataLoader(
                        train_dataset, batch_size=512, shuffle=True
                    )
                )
            )
        )[0].to(device),
    )  # W8A8

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

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

    if args.method == "qat":
        model_requantized = to_quantized_online(
            merge_qat_model(model, inplace=False), input_specs=specs, weight_specs=specs
        )  # W8A8

        model_requantized.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model_requantized(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Requantized Accuracy: {accuracy:.2f}%")
        torch.save(merge_qat_model(model, inplace=False), "merged_qat_resnet18.pth")
        torch.save(model, "qat_resnet18.pth")

    elif args.method == "lsq":
        model_requantized = merge_qat_lsq_into_offline_quantized_model(
            model, inplace=False
        )

        model_requantized.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model_requantized(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f"Epoch {epoch + 1}, Requantized into offline Accuracy: {accuracy:.2f}%")

        torch.save(model_requantized, "merged_lsq_into_offline_resnet18.pth")
        torch.save(model, "lsq_resnet18.pth")
