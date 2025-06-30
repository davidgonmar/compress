import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import (
    IntAffineQuantizationSpec,
    to_quantized_online,
    get_activations,
    IntAffineQuantizationMode,
    get_quant_dict,
    merge_dicts,
)
import argparse
import copy
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="qat_resnet.pth")
parser.add_argument("--print_model", action="store_true")
parser.add_argument("--offline", action="store_true")
parser.add_argument("--adaround", action="store_true")
parser.add_argument("--kmeans", action="store_true")
parser.add_argument("--leave_edge_layers_8_bits", action="store_true")

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


model = torch.load(args.save_path, weights_only=False)


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


bit_widths = [4]
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

    specs_linear = get_quant_dict(
        model,
        "linear",
        IntAffineQuantizationSpec(
            i_linear_bits,
            signed_options[0],
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=99.5,
        ),
        IntAffineQuantizationSpec(
            w_linear_bits,
            signed_options[0],
            quant_mode=IntAffineQuantizationMode.ENTROPY_SYMMETRIC,
            percentile=99.5,
        ),
    )
    specs_conv2d = get_quant_dict(
        model,
        "conv2d",
        IntAffineQuantizationSpec(
            i_conv_bits,
            signed_options[0],
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=99.5,
        ),
        IntAffineQuantizationSpec(
            w_conv_bits,
            signed_options[0],
            quant_mode=IntAffineQuantizationMode.ENTROPY_SYMMETRIC,
            percentile=99.5,
        ),
    )
    specs = merge_dicts(specs_linear, specs_conv2d)

    if args.leave_edge_layers_8_bits:
        # last layer key is "fc" for resnet18
        specs["fc"]["input"] = IntAffineQuantizationSpec(
            nbits=8,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=99.5,
        )
        # first layer key is "conv1" for resnet18
        specs["conv1"]["input"] = IntAffineQuantizationSpec(
            nbits=8,
            signed=True,
            quant_mode=IntAffineQuantizationMode.SYMMETRIC,
            percentile=99.5,
        )

        if args.leave_edge_layers_8_bits:
            # last layer key is "fc" for resnet18
            specs["fc"]["weight"] = IntAffineQuantizationSpec(
                nbits=8,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ENTROPY_SYMMETRIC,
                percentile=99.5,
            )
            # first layer key is "conv1" for resnet18
            specs["conv1"]["weight"] = IntAffineQuantizationSpec(
                nbits=8,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ENTROPY_SYMMETRIC,
                percentile=99.5,
            )

    quanted = to_quantized_online(
        model,
        specs,
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
    )
    print(
        f"Linear Bits: {w_linear_bits}, Conv2D Bits: {w_conv_bits}, "
        f"Input Linear Bits: {i_linear_bits}, Input Conv2D Bits: {i_conv_bits}"
    )
    loss, accuracy = evaluate(quanted, test_loader, criterion, device)
    print(
        f"Weight Linear Bits: {w_linear_bits}, Weight Conv2D Bits: {w_conv_bits}, "
        f"Input Linear Bits: {i_linear_bits}, Input Conv2D Bits: {i_conv_bits} -> "
        f"Test Loss: {loss}, Test Accuracy: {accuracy}"
    )
