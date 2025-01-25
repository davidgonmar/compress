import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="mnist_model.pth")
args = parser.parse_args()


def plot_weight_distributions(model_path, plots_per_window=6):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    weights = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if module.weight is not None:
                weights.append(module.weight.data.cpu().numpy().flatten())
                layer_names.append(name)

    if len(weights) == 0:
        print("No Linear or Conv2d layers with weights found in the model.")
        return

    total_layers = len(weights)
    num_windows = (total_layers + plots_per_window - 1) // plots_per_window

    for window_idx in range(num_windows):
        start_idx = window_idx * plots_per_window
        end_idx = min((window_idx + 1) * plots_per_window, total_layers)

        fig, axes = plt.subplots(
            end_idx - start_idx, 1, figsize=(10, (end_idx - start_idx) * 3)
        )
        fig.suptitle(f"Weight Distributions (Window {window_idx + 1})", fontsize=16)

        if end_idx - start_idx == 1:
            axes = [axes]

        for ax, weight, name in zip(
            axes, weights[start_idx:end_idx], layer_names[start_idx:end_idx]
        ):
            weight_min, weight_max = np.percentile(weight, [15, 85])
            ax.hist(weight, bins=100, alpha=0.7, range=(weight_min, weight_max))
            ax.set_title(f"Weight Distribution: {name}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show(block=False)


plot_weight_distributions(args.model_path)

# now block
plt.show()
