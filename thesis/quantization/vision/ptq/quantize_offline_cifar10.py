import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import to_quantized_offline, get_activations_vision
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean as mean,
    cifar10_std as std,
)
from compress.quantization.recipes import get_recipe_quant, get_quant_keys
from compress.quantization import (
    ConvWeightsPerOutChannel,
    LinearWeightsPerOutChannel,
    PerTensor,
)
from compress.layer_fusion import get_fuse_bn_keys
import argparse
from itertools import product
import json


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, required=True)
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--calibration_batches", type=int, default=10)
parser.add_argument("--output_path", type=str, default="quantization_results.json")
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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# random sample a subset of the dataset for calibration
train_subset_size = 512
train_subset_indices = torch.randperm(len(train_dataset))[:train_subset_size]
train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

eval_results = evaluate_vision_model(model, test_loader)
results = [
    {
        "type": "original",
        "loss": eval_results["loss"],
        "accuracy": eval_results["accuracy"],
    }
]

bit_widths = [2, 4, 8]

activations = get_activations_vision(
    model,
    train_loader,
    keys=get_quant_keys(args.model_name),
)

for per_channel in [True, False]:
    for leave_edge_layers_8_bits in [True, False]:
        for w_bits, act_bits in product(bit_widths, bit_widths):
            specs = get_recipe_quant(args.model_name)(
                bits_activation=act_bits,
                bits_weight=w_bits,
                clip_percentile=0.995,
                leave_edge_layers_8_bits=leave_edge_layers_8_bits,
                symmetric=True,
                linear_weight_grouper=(
                    LinearWeightsPerOutChannel() if per_channel else PerTensor()
                ),
                conv_weight_grouper=(
                    ConvWeightsPerOutChannel() if per_channel else PerTensor()
                ),
            )
            quanted = to_quantized_offline(
                model.to(device),
                specs,
                activations=activations,
                inplace=False,
                fuse_bn_keys=get_fuse_bn_keys(args.model_name),
            )
            quanted.to(device)
            eval_results = evaluate_vision_model(quanted, test_loader)
            print(
                f"Quantized model with {w_bits} bits for weights and {act_bits} bits for activations: "
                f"loss: {eval_results['loss']}, accuracy: {eval_results['accuracy']}"
            )
            results.append(
                {
                    "type": f"W{w_bits}A{act_bits}",
                    "leave_edge_layers_8_bits": leave_edge_layers_8_bits,
                    "loss": eval_results["loss"],
                    "accuracy": eval_results["accuracy"],
                    "weights_per_channel": per_channel,
                }
            )

        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)
