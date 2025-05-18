#!/usr/bin/env python3
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_stats(path):
    """Load JSON stats and return lists of (sparsity, pr_acc, ft_acc)."""
    with open(path, 'r') as f:
        stats = json.load(f)
    out = []
    for it in stats['iterations']:
        # sparsity string from after_prune_before_ft, e.g. '23.45%'
        spar = it['after_prune_before_ft']['sparsity']
        # before-ft accuracy = after_prune_before_ft accuracy
        pr_acc = it['after_prune_before_ft']['accuracy']
        # after-ft accuracy = last finetune epoch val_accuracy
        ft_acc = it['finetune_epochs'][-1]['val_accuracy']
        out.append((spar, pr_acc, ft_acc))
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Plot staged pruning curves from multiple JSON stats files"
    )
    parser.add_argument(
        '--stats_files', '-s',
        nargs='+', required=True,
        help='List of JSON stats files, in same order as --methods'
    )
    parser.add_argument(
        '--methods', '-m',
        nargs='+', required=True,
        help='Names of pruning methods (one per stats file)'
    )
    parser.add_argument(
        '--out_pdf', '-o',
        default='pruning_comparison.pdf',
        help='Output PDF filename'
    )
    args = parser.parse_args()

    if len(args.stats_files) != len(args.methods):
        parser.error("Number of --stats_files must match number of --methods")


    all_data = {}
    for method, path in zip(args.methods, args.stats_files):
        data = load_stats(path)
        all_data[method] = data


    n_iters = len(next(iter(all_data.values())))

    x_pos = np.arange(n_iters * 2)
    sparsities = [d[0] for d in all_data[args.methods[0]]]
    x_labels = []
    for spar in sparsities:
        x_labels += [f"{spar} pr", f"{spar} ft"]

    plt.figure(figsize=(10,5))
    for method, data in all_data.items():
        # data is list of (spar, pr, ft)
        pr_vals = [d[1] for d in data]
        ft_vals = [d[2] for d in data]
        y = np.empty(n_iters * 2)
        y[0::2] = pr_vals
        y[1::2] = ft_vals
        plt.plot(x_pos, y, marker='o', label=method)

    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.ylabel('Validation Accuracy')
    plt.title('Pruning (pr) vs. Finetuning (ft) Accuracy by Sparsity Stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_pdf)
    print(f"Saved plot to {args.out_pdf}")

if __name__ == "__main__":
    main()