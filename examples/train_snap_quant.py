import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from tqdm import tqdm
from compress.quantization import (
    IntAffineQuantizationSpec,
    prepare_for_qat,
    to_quantized_online,
    merge_qat_model,
    get_regularizer_for_pact,
    SnapRegularizer,
    IntAffineQuantizationMode,
)
import torchvision
import argparse


def get_specs():
    specs = {
        "linear": IntAffineQuantizationSpec(
            nbits=args.nbits_a,
            signed=False,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
        "conv2d": IntAffineQuantizationSpec(
            nbits=args.nbits_a,
            signed=False,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
    }
    weight_specs = {
        "linear": IntAffineQuantizationSpec(
            nbits=args.nbits_w,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
        "conv2d": IntAffineQuantizationSpec(
            nbits=args.nbits_w,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=args.clip_percentile,
        ),
    }

    if args.leave_edge_layers_8_bits:
        nbits = 8
    else:
        nbits = args.nbits_a

    specs["fc"] = IntAffineQuantizationSpec(
        nbits=nbits,
        signed=False,
        quant_mode=IntAffineQuantizationMode.SYMMETRIC,
        percentile=args.clip_percentile,
    )
    specs["conv1"] = IntAffineQuantizationSpec(
        nbits=nbits,
        signed=True,
        quant_mode=IntAffineQuantizationMode.SYMMETRIC,
        percentile=args.clip_percentile,
    )  # signed=True since the input is signed
    weight_specs["fc"] = IntAffineQuantizationSpec(
        nbits=nbits,
        signed=True,
        quant_mode=IntAffineQuantizationMode.SYMMETRIC,
        percentile=args.clip_percentile,
    )
    weight_specs["conv1"] = IntAffineQuantizationSpec(
        nbits=nbits,
        signed=True,
        quant_mode=IntAffineQuantizationMode.SYMMETRIC,
        percentile=args.clip_percentile,
    )

    return specs, weight_specs


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument(
    "--leave_edge_layers_8_bits",
    action="store_true",
)

parser.add_argument("--snap_loss_activations", action="store_true")
parser.add_argument("--snap_loss_params", action="store_true")
parser.add_argument("--load_from", type=str, default=None)

parser.add_argument(
    "--bits_schedule",
    type=str,
    default="8*8",
    help="format: weights_bits,weights_bits*acts_bits,acts_bits",
)

parser.add_argument(
    "--epochs_schedule",
    type=str,
    default="10",
    help="comma separated list of epochs",
)

parser.add_argument(
    "--clip_percentile",
    type=float,
    default=0.995,  # cuts low 0.5% and high 0.5% of the values
    help="percentile for clipping",
)

args = parser.parse_args()

weight_sched_str, act_sched_str = args.bits_schedule.split("*")
sched1_w = list(map(int, weight_sched_str.split(",")))
sched1_a = list(map(int, act_sched_str.split(",")))
sched2 = list(map(int, args.epochs_schedule.split(",")))
sched = list(zip(sched1_w, sched1_a, sched2))
print("Bits schedule (weight, act, epoch):", sched)

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
torch_weights = torchvision.models.resnet18(pretrained=True).state_dict()
del torch_weights["fc.weight"]
del torch_weights["fc.bias"]
model.load_state_dict(torch_weights, strict=False)

if args.load_from:
    loaded = torch.load(args.load_from, weights_only=False)
    if isinstance(loaded, dict):
        model.load_state_dict(loaded["model"])
    else:
        model = loaded

args.nbits_w = sched[0][0]
args.nbits_a = sched[0][1]

specs, weight_specs = get_specs()

model = prepare_for_qat(
    model, input_specs=specs, use_PACT=True, weight_specs=weight_specs
)
model.to(device)

pact_reg = get_regularizer_for_pact(model)
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

print(model)

print(merge_qat_model(model, inplace=False))
for epoch in range(1000):
    if sched and epoch == sched[0][2]:
        print(f"Changing nbits: weights={sched[0][0]}, activations={sched[0][1]}")
        model = merge_qat_model(model, inplace=False)

        args.nbits_w = sched[0][0]
        args.nbits_a = sched[0][1]
        del sched[0]

        specs, weight_specs = get_specs()

        model = prepare_for_qat(model, input_specs=specs, weight_specs=weight_specs)
        reg = SnapRegularizer(
            model,
            do_activations=args.snap_loss_activations,
            do_params=args.snap_loss_params,
        )

        model.to(device)
        pact_reg = get_regularizer_for_pact(model)

        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    model.train()
    train_loss_acc = 0.0
    snap_loss_params_acc = 0.0
    snap_loss_acts_acc = 0.0
    pact_reg_loss = 0.0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.register_forward_hook(
                lambda module, input, output, name=name: print(
                    f"Layer: {name}, Input: {input[0].min().item(), input[0].max().item(), input[0].mean().item()}, Output: {output.min().item(), output.max().item(), output.mean().item()}"
                )
            )
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False
    ):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        train_loss = criterion(outputs, labels)
        snap_loss_dict = reg.snap_loss()
        snap_loss_acts = (
            snap_loss_dict["activations"] if args.snap_loss_activations else 0
        )
        snap_loss_params = snap_loss_dict["params"] if args.snap_loss_params else 0
        pact_reg_loss = pact_reg()
        (
            train_loss
            + 0.05 * snap_loss_params
            + 0.05 * snap_loss_acts
            + 1 * pact_reg_loss
        ).backward()

        optimizer.step()

        train_loss_acc += train_loss.item() * images.size(0)
        snap_loss_params_acc += snap_loss_params * images.size(0)
        snap_loss_acts_acc += snap_loss_acts * images.size(0)
        pact_reg_loss = pact_reg_loss * images.size(0)

    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss_acc / len(train_loader.dataset):.4f}, Snap Loss Params: {snap_loss_params_acc / len(train_loader.dataset):.4f}, Snap Loss Acts: {snap_loss_acts_acc / len(train_loader.dataset):.4f}, Pact Reg Loss: {pact_reg_loss / len(train_loader.dataset):.4f}"
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
        merge_qat_model(model, inplace=False),
        input_specs=specs,
        weight_specs=weight_specs,
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
    print(f"Epoch {epoch + 1}, Requantized Accuracy: {accuracy:.2f}%")
    torch.save(merge_qat_model(model, inplace=False), "merged_qat_snap_resnet18.pth")
