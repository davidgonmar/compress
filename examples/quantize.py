import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import (
    IntQuantizationSpec,
    to_quantized_online,
    to_quantized_offline,
    get_activations,
    to_quantized_adaround,
)
import argparse
import copy
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="qat_resnet.pth")
parser.add_argument("--print_model", action="store_true")
parser.add_argument("--offline", action="store_true")
parser.add_argument("--adaround", action="store_true")

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

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
loss = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {loss[0]}, Test Accuracy: {loss[1]}")


bit_widths = [4, 8]
signed_options = [True]

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

torch.manual_seed(0)


class OnlyImages:
    def __init__(self, dataset):
        self.dataset = [dataset[i] for i in torch.randperm(len(dataset))[: 512 * 10]]

    def __iter__(self):
        for x, _ in self.dataset:
            yield x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0].to(device)


dataloader = DataLoader(OnlyImages(train_dataset), batch_size=512, shuffle=True)

activations = get_activations(model, dataloader)

for w_linear_bits, w_conv_bits, i_linear_bits, i_conv_bits in product(
    bit_widths, bit_widths, bit_widths, bit_widths
):
    wspecs = {
        "linear": IntQuantizationSpec(w_linear_bits, signed_options[0]),
        "conv2d": IntQuantizationSpec(w_conv_bits, signed_options[0]),
    }

    inpspecs = {
        "linear": IntQuantizationSpec(i_linear_bits, signed_options[0]),
        "conv2d": IntQuantizationSpec(i_conv_bits, signed_options[0]),
    }

    if args.adaround:
        quanted = to_quantized_adaround(
            model,
            inpspecs,
            wspecs,
            data_loader=DataLoader(train_dataset, batch_size=512, shuffle=True),
            inplace=False,
        )
    elif args.offline:
        quanted = to_quantized_offline(
            model,
            inpspecs,
            wspecs,
            inplace=False,
            model_initializer=lambda: copy.deepcopy(model),
            activations=activations,
        )
    else:
        quanted = to_quantized_online(
            model,
            inpspecs,
            wspecs,
            inplace=False,
            model_initializer=lambda: copy.deepcopy(model),
        )
    loss, accuracy = evaluate(quanted, test_loader, criterion, device)
    print(
        f"Weight Linear Bits: {w_linear_bits}, Weight Conv2D Bits: {w_conv_bits}, "
        f"Input Linear Bits: {i_linear_bits}, Input Conv2D Bits: {i_conv_bits} -> "
        f"Test Loss: {loss}, Test Accuracy: {accuracy}"
    )
