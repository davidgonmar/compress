#!/usr/bin/env python
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_curve(json_path):
    with Path(json_path).open() as f:
        data = json.load(f)

    pts = [
        (item["flops"], item["accuracy"])
        for item in data
        if item.get("type") != "original"
    ]
    pts.sort()  # sort by FLOPs
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_json", required=True)
    parser.add_argument("--manual_flops_json", required=True)
    parser.add_argument("--manual_energy_json", required=True)
    parser.add_argument("--output", default="flops_vs_acc.pdf")
    args = parser.parse_args()

    curves = {
        "Global (FLOPs)": load_curve(args.global_json),
        "Manual (FLOPs)": load_curve(args.manual_flops_json),
        "Manual (Energy)": load_curve(args.manual_energy_json),
    }
    markers = {"Global (FLOPs)": "o", "Manual (FLOPs)": "s", "Manual (Energy)": "^"}

    plt.figure()
    for label, (xs, ys) in curves.items():
        plt.plot(
            [x / 1e6 for x in xs],  # convert to millions
            ys,
            marker=markers[label],
            label=label,
        )

    plt.xlabel("FLOPs (Millions)")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
