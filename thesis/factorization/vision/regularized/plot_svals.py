#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.utils import get_all_convs_and_linears


def compute_singular_value_energies(model: torch.nn.Module) -> np.ndarray:
    energies = []
    for name, layer in get_all_convs_and_linears(model):
        weight = layer.weight.data
        if weight.ndim == 4:
            out_c, in_c, k_h, k_w = weight.shape
            weight_2d = weight.reshape(out_c, in_c * k_h * k_w)
        else:
            weight_2d = weight
        with torch.no_grad():
            s = torch.linalg.svdvals(weight_2d)
        energies.append(s.pow(2).cpu())
    return torch.cat(energies).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, required=True)
    parser.add_argument("--output_plot", type=Path, default=Path("sv_energy.pdf"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_vision_model(
        "resnet20",
        pretrained_path=str(args.pretrained_path),
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet20"),
        modifier_after_load=None,
        model_args={"num_classes": 10},
    ).to(device)
    model.eval()
    energies = compute_singular_value_energies(model)
    energies_sorted = np.sort(energies)[::-1]
    cumulative = np.cumsum(energies_sorted) / energies_sorted.sum()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(energies_sorted, marker="o", linewidth=1, markersize=2)
    ax1.set_xlabel("Singular-value index (sorted)")
    ax1.set_ylabel("Energy (σ²)")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(cumulative, linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative energy")
    ax2.set_ylim(0, 1.0)
    plt.title("Singular-value energies – ResNet-20")
    plt.tight_layout()
    fig.savefig(args.output_plot, dpi=300)
    print(f"Saved plot to {os.path.abspath(args.output_plot)}")


if __name__ == "__main__":
    main()
