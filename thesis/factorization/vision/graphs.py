#!/usr/bin/env python
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _to_number(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "total" in v:
            return float(v["total"])
        for x in v.values():
            if isinstance(x, (int, float)):
                return float(x)
    raise ValueError(f"Cannot interpret flops value: {v}")


def load_curve(json_path):
    with Path(json_path).open() as f:
        data = json.load(f)
    pts = []
    for item in data:
        if item.get("type") == "original":
            continue
        flops_val = _to_number(item["flops"])
        acc_val = float(item["accuracy"]) * 100
        pts.append((flops_val, acc_val))
    pts.sort(key=lambda t: t[0])
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_json", required=True)
    parser.add_argument("--manual_flops_json", required=True)
    parser.add_argument("--manual_energy_json", required=True)
    parser.add_argument("--manual_activation_aware_flops_json", required=True)
    parser.add_argument("--manual_activation_aware_energy_json", required=True)
    parser.add_argument("--global_activation_aware_flops_json", required=True)
    parser.add_argument("--output", default="flops_vs_acc.pdf")
    args = parser.parse_args()

    curves = {
        "Global (FLOPs)": load_curve(args.global_json),
        "Manual (FLOPs)": load_curve(args.manual_flops_json),
        "Manual (Energy)": load_curve(args.manual_energy_json),
        "Manual Activation-aware (FLOPs)": load_curve(
            args.manual_activation_aware_flops_json
        ),
        "Manual Activation-aware (Energy)": load_curve(
            args.manual_activation_aware_energy_json
        ),
        "Global Activation-aware (FLOPs)": load_curve(
            args.global_activation_aware_flops_json
        ),
    }
    markers = {
        "Global (FLOPs)": "o",
        "Manual (FLOPs)": "s",
        "Manual (Energy)": "^",
        "Manual Activation-aware (FLOPs)": "D",
        "Manual Activation-aware (Energy)": "v",
        "Global Activation-aware (FLOPs)": "P",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, (xs, ys) in curves.items():
        ax.plot([x / 1e6 for x in xs], ys, marker=markers[label], label=label)

    ax.set_xlabel("FLOPs (Millions)")
    ax.set_ylabel("Top-1 Accuracy (%)")

    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(args.output)
    print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
