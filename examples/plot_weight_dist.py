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
            p15, p85 = np.percentile(weight, [15, 85])
            filtered_weights = weight[(weight >= p15) & (weight <= p85)]

            mean, std = np.mean(filtered_weights), np.std(filtered_weights)
            weight_min, weight_max = np.min(filtered_weights), np.max(filtered_weights)
            bin_edges = np.linspace(weight_min, weight_max, 101)  # 100 bins

            hist_data, bins = np.histogram(
                filtered_weights, bins=bin_edges, density=True
            )
            hist_data = hist_data / np.sum(hist_data * (bins[1] - bins[0]))

            ax.bar(
                bins[:-1],
                hist_data,
                width=bins[1] - bins[0],
                edgecolor="black",
                align="edge",
            )
            x = np.linspace(weight_min, weight_max, 500)
            gaussian_data = np.exp(-0.5 * ((x - mean) / std) ** 2) / (
                std * np.sqrt(2 * np.pi)
            )
            ax.plot(x, gaussian_data, label="Gaussian Distribution", color="red")
            laplace_data = np.exp(-np.abs(x - mean) / std) / (2 * std)
            ax.plot(x, laplace_data, label="Laplace Distribution", color="green")

            ax.set_title(f"Weight Distribution (15th-85th Percentile): {name}")
            ax.set_ylabel("Density")
            ax.set_xlabel("Weight Value")
            ax.legend()

            print(
                f"Layer: {name}, Histogram Area: {np.sum(hist_data * (bins[1] - bins[0])):.4f}"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show(block=False)


plot_weight_distributions(args.model_path)

# now block
plt.show()
