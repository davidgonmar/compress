#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _load_curve(stats_path: Path):
    """
    From one stats.json, return:
      - baseline_acc: float
      - prune_pcts:   [pct_pruned1, pct_pruned2, …]
      - prune_accs:   [acc_after_prune1, acc_after_prune2, …]
      - ft_accs:      [acc_after_ft1,   acc_after_ft2,   …]
    """
    with stats_path.open() as f:
        data = json.load(f)

    baseline_acc = data["baseline"]["accuracy"]
    prune_pcts = []
    prune_accs = []
    ft_accs = []

    for it in data["iterations"]:
        s = it["after_prune_before_ft"]["sparsity"]
        pct = s["sparsity_ratio_wrt_prunable"] * 100
        prune_pcts.append(pct)
        prune_accs.append(it["after_prune_before_ft"]["accuracy"])

        ftes = it.get("finetune_epochs", [])
        if ftes:
            ft_accs.append(ftes[-1]["val_accuracy"])
        else:
            # if no fine‐tune happened, repeat prune‐accuracy
            ft_accs.append(prune_accs[-1])

    return baseline_acc, prune_pcts, prune_accs, ft_accs


def main():
    p = argparse.ArgumentParser(description="Plot prune→fine‐tune with sparsity labels")
    p.add_argument("results_dir", type=Path, help="Dir containing *_stats.json")
    p.add_argument("output_pdf", type=Path, help="Path to save the PDF")
    args = p.parse_args()

    methods = ["taylor", "norm_weights", "norm_activations"]
    markers = {"taylor": "o", "norm_weights": "s", "norm_activations": "^"}

    # --- build the x-axis labels once, from the first method ---
    first = args.results_dir / f"{methods[0]}_stats.json"
    if not first.exists():
        print(f"[ERROR] {first} not found")
        return

    baseline, prune_pcts, prune_accs, ft_accs = _load_curve(first)

    # flattened y-axis positions (just indices)
    x = list(range(1 + 2 * len(prune_pcts)))
    # build labels: [baseline, pX%, ftX%, pY%, ftY%, …]
    labels = ["baseline"]
    for pct in prune_pcts:
        labels += [f"p {pct:.1f}%", f"ft {pct:.1f}%"]

    # --- now plot each method ---
    plt.figure(figsize=(10, 5))
    for m in methods:
        f = args.results_dir / f"{m}_stats.json"
        if not f.exists():
            print(f"[Warning] {f} missing, skipping {m}")
            continue

        b, pcts, pr_accs, ft_accs = _load_curve(f)
        y = [b]
        for pr, ft in zip(pr_accs, ft_accs):
            y += [pr, ft]

        plt.plot(x, y, label=m, marker=markers[m])

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Cycle steps (prune → fine‐tune)")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved → {args.output_pdf}")


if __name__ == "__main__":
    main()
