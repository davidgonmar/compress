#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _accuracy_curve(stats_path: Path):
    """Return a list: [baseline_acc, acc_iter1, acc_iter2, …]."""
    with stats_path.open() as f:
        data = json.load(f)

    curve = [data["baseline"]["accuracy"]]

    for it in data["iterations"]:
        # use accuracy from the **last** finetune epoch;
        # fall back to the ‘after_prune_before_ft’ accuracy if none recorded
        if it.get("finetune_epochs"):
            curve.append(it["finetune_epochs"][-1]["val_accuracy"])
        else:
            curve.append(it["after_prune_before_ft"]["accuracy"])

    return curve


def main():
    p = argparse.ArgumentParser(
        description="Plot pruning-iteration curves and save to PDF"
    )
    p.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing *_stats.json files (taylor, norm_weights, …)",
    )
    p.add_argument("output_pdf", type=Path, help="Filename for the PDF plot")
    args = p.parse_args()

    methods = ["taylor", "norm_weights", "norm_activations"]
    style = {
        "taylor": {"marker": "o"},
        "norm_weights": {"marker": "s"},
        "norm_activations": {"marker": "^"},
    }

    plt.figure(figsize=(8, 5))
    max_x = 0

    for m in methods:
        f = args.results_dir / f"{m}_stats.json"
        if not f.exists():
            print(f"[plot_pruning_results] Warning: {f} not found – skipping")
            continue
        y = _accuracy_curve(f)
        x = range(len(y))
        max_x = max(max_x, x[-1])
        plt.plot(x, y, label=m, **style[m])

    plt.xlabel("Pruning iteration")
    plt.ylabel("Validation Accuracy")
    plt.xlim(0, max_x)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    print(f"[plot_pruning_results] Saved → {args.output_pdf}")


if __name__ == "__main__":
    main()
