import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import (
    to_quantized_online,
)
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
)
from compress.quantization.recipes import (
    get_recipe_quant,
)
import argparse
from itertools import product

# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name mobilenet_v2 --pretrained_path mobilenetv2.pth
# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet18 --pretrained_path resnet18.pth

# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name mobilenet_v2 --pretrained_path mobilenetv2.pth
# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet18 --pretrained_path resnet18.pth

# python -m examples.quantization.quantize_online --model_name mobilenet_v2 --pretrained_path mobilenetv2.pth
# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet18 --pretrained_path resnet18.pth

# python -m examples.quantization.quantize_online --model_name mobilenet_v2 --pretrained_path mobilenetv2.pth
# python -m examples.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet18 --pretrained_path resnet18.pth


parser = argparse.ArgumentParser()
parser.add_argument("--leave_edge_layers_8_bits", action="store_true")
parser.add_argument("--model_name", type=str, default=None, required=True)
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=512)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
)
model.eval()

model.to(device)
# cifar10 mean and std
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


eval_results = evaluate_vision_model(model, test_loader)

print(f"Test Loss: {eval_results['loss']}, Test Accuracy: {eval_results['accuracy']}")

model.to("cpu")

bit_widths = [2, 4, 8, 16]
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

for w_bits, act_bits in product(bit_widths, bit_widths):

    specs = get_recipe_quant(
        args.model_name,
    )(
        bits_activation=act_bits,
        bits_weight=w_bits,
        clip_percentile=0.995,
        leave_edge_layers_8_bits=args.leave_edge_layers_8_bits,
        symmetric=False,
    )
    quanted = to_quantized_online(
        model.to(device),
        specs,
        inplace=False,
    )
    model.to("cpu")
    quanted.to(device)
    eval_results = evaluate_vision_model(quanted, test_loader)
    print(
        f"Quant: W{w_bits}A{act_bits}, leave_edge_layers_8_bits={args.leave_edge_layers_8_bits}, "
        f"Loss: {eval_results['loss']}, Accuracy: {eval_results['accuracy']}"
    )
