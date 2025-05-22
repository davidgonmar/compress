#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

Row = Tuple[str, float, float, float]


def _pretty_method(raw: str) -> str:
    mapping = {
        "norm_weights": "Weight-Norm",
        "norm_activations": "Activation-Norm",
        "taylor_no_bias": "Taylor With No Bias",
        "taylor_bias": "Taylor With Bias",
    }
    return mapping.get(raw, raw.replace("_", " ").title())


def _extract_row(json_path: Path) -> Row:
    with json_path.open() as fp:
        data = json.load(fp)

    cfg = data["config"]
    method = _pretty_method(cfg["method"])
    tgt = float(cfg["target_sparsity"]) * 100.0
    pruned_acc = float(data["pruned"]["accuracy"])

    return (
        method,
        tgt,
        pruned_acc,
        None,
    )  # the fourth element (actual sparsity) not used here


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a sparsity-vs-accuracy plot from pruning JSON stats"
    )
    ap.add_argument("results_dir", help="Directory with *_stats.json files")
    ap.add_argument(
        "--outfile",
        help="Write plot PDF to this path (default: sparsity_vs_accuracy.pdf)",
    )
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    files = sorted(res_dir.glob("*_stats.json"))
    if not files:
        raise SystemExit(f"No *_stats.json files found in {res_dir}")

    # extract rows
    rows: List[Row] = [_extract_row(f) for f in files]

    # sort by target sparsity then method name
    rows.sort(key=lambda r: (round(r[1]), r[0]))

    # group by method
    data = {}
    for method, tgt, acc, _ in rows:
        data.setdefault(method, {"tgt": [], "acc": []})
        data[method]["tgt"].append(tgt)
        data[method]["acc"].append(acc)

    # plot
    plt.figure()
    for method, vals in data.items():
        plt.plot(vals["tgt"], vals["acc"], marker="o", label=method)
    plt.xlabel("Target Sparsity (%)")
    plt.ylabel("Pruned Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # save to PDF
    out_path = Path(args.outfile) if args.outfile else Path("sparsity_vs_accuracy.pdf")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
