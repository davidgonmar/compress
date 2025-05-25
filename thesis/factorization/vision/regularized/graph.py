#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt


def load_curve(json_path: Path, x_key: str):
    """Return x-axis values (ratio) and Top-1 (%) accuracy for a given metric key."""
    with json_path.open() as f:
        data = json.load(f)
    xs = [item[x_key] for item in data]
    ys = [item["accuracy"] * 100 for item in data]
    return xs, ys


def weight_from_filename(path: Path) -> str:
    """
    Extract the numeric weight that appears **just before** the .json extension:
    …_0.005.json   →  0.005
    If no weight can be found, return the whole stem.
    """
    match = re.search(r"_([0-9]*\.?[0-9]+)\.json$", path.name)
    return match.group(1) if match else path.stem


def plot_curves(json_paths, x_key, x_label, out_path):
    plt.figure(figsize=(10, 5))

    for jp in json_paths:
        xs, ys = load_curve(jp, x_key)
        weight = weight_from_filename(jp)
        plt.plot(xs, ys, label=f"Weight {weight}")

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(range(0, 101, 10), fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot FLOPs/Params vs Accuracy from factorisation JSONs"
    )
    parser.add_argument(
        "--jsons",
        nargs="+",
        required=True,
        metavar="FILE",
        help="List of JSON result files (one per model)",
    )
    parser.add_argument("--output_flops", default="flops_vs_acc.pdf")
    parser.add_argument("--output_params", default="params_vs_acc.pdf")
    args = parser.parse_args()

    json_paths = [Path(p) for p in args.jsons]

    plot_curves(
        json_paths,
        x_key="flops_ratio",
        x_label="FLOPs Ratio",
        out_path=args.output_flops,
    )
    plot_curves(
        json_paths,
        x_key="params_ratio",
        x_label="Parameter Count Ratio",
        out_path=args.output_params,
    )
    print(f"Saved plots → {args.output_flops}, {args.output_params}")


if __name__ == "__main__":
    main()
