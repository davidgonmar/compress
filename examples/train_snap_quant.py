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
    to_quantized_online,
    merge_qat_model,
    SnapRegularizer,
)
import torchvision
import argparse


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument(
    "--leave_edge_layers_8_bits",
    action="store_true",
)

parser.add_argument("--snap_loss_activations", action="store_true")
parser.add_argument("--snap_loss_params", action="store_true")
parser.add_argument("--load_from", type=str, default=None)

# usage example --bits_schedule=8,4,2 --epochs_schedule=10,20,30
parser.add_argument(
    "--bits_schedule",
    type=str,
    default="8",
    help="comma separated list of bitwidths",
)

parser.add_argument(
    "--epochs_schedule",
    type=str,
    default="10",
    help="comma separated list of epochs",
)

args = parser.parse_args()

sched1 = list(map(int, args.bits_schedule.split(",")))
sched2 = list(map(int, args.epochs_schedule.split(",")))
sched = list(zip(sched1, sched2))
print("Bits schedule:", sched, sched1, sched2)

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

if args.load_from:
    loaded = torch.load(args.load_from, weights_only=False)
    if isinstance(loaded, dict):
        model.load_state_dict(loaded["model"])
    else:
        model = loaded

args.nbits = sched[0][0]

specs = {
    "linear": IntQuantizationSpec(nbits=args.nbits, signed=True),
    "conv2d": IntQuantizationSpec(nbits=args.nbits, signed=True),
}


if args.leave_edge_layers_8_bits:
    # last layer key is "fc" for resnet18
    specs["fc"] = IntQuantizationSpec(nbits=8, signed=True)
    # first layer key is "conv1" for resnet18
    specs["conv1"] = IntQuantizationSpec(nbits=8, signed=True)

model = prepare_for_qat(model, input_specs=specs, weight_specs=specs)  # W8A8


reg = SnapRegularizer(
    model,
    do_activations=args.snap_loss_activations,
    do_params=args.snap_loss_params,
)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

print("Starting training. Bits schedule:", sched)
del sched[0]

for epoch in range(100):
    # update nbits
    if sched and epoch == sched[0][1]:
        print(f"Changing nbits to {sched[0][0]}")
        model = merge_qat_model(model, inplace=False)
        args.nbits = sched[0][0]
        del sched[0]
        specs = {
            "linear": IntQuantizationSpec(nbits=args.nbits, signed=True),
            "conv2d": IntQuantizationSpec(nbits=args.nbits, signed=True),
        }
        if args.leave_edge_layers_8_bits:
            specs["fc"] = IntQuantizationSpec(nbits=8, signed=True)
            specs["conv1"] = IntQuantizationSpec(nbits=8, signed=True)

        model = prepare_for_qat(model, input_specs=specs, weight_specs=specs)  # W8A8
        reg = SnapRegularizer(
            model,
            do_activations=args.snap_loss_activations,
            do_params=args.snap_loss_params,
        )

    model.train()
    train_loss_acc = 0.0
    snap_loss_params_acc = 0.0
    snap_loss_acts_acc = 0.0

    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        snap_loss_dict = reg.snap_loss()
        snap_loss_acts = (
            snap_loss_dict["activations"] if args.snap_loss_activations else 0
        )
        snap_loss_params = snap_loss_dict["params"] if args.snap_loss_params else 0
        (train_loss + snap_loss_acts * 0.5 + snap_loss_params * 0.5).backward()
        optimizer.step()
        train_loss_acc += train_loss.item() * images.size(0)
        snap_loss_params_acc += snap_loss_params * images.size(0)
        snap_loss_acts_acc += snap_loss_acts * images.size(0)

    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss_acc / len(train_loader.dataset):.4f}, Snap Loss Params: {snap_loss_params_acc / len(train_loader.dataset):.4f}, Snap Loss Acts: {snap_loss_acts_acc / len(train_loader.dataset):.4f}"
    )

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
    torch.save(merge_qat_model(model, inplace=False), "merged_qat_snap_resnet18.pth")
