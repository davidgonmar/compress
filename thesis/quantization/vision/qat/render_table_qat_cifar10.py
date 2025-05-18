#!/usr/bin/env python
import os
import json
import argparse
import re
import statistics as stats


def render_latex_table(rows, caption, label):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & W bits & A bits & Edge8 & Accuracy (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        method, w, a, edge = r["method"], r["w_bits"], r["a_bits"], r["edge"]
        edge_str = "Yes" if edge else "No"
        mean, std = r["acc_mean"], r["acc_std"]
        lines.append(
            f"{method} & {w} & {a} & {edge_str} & " f"{mean:.2f} $\\pm$ {std:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# qat_w4a4_edge8_s3.json -> groups = (qat, 4, 4, edge8, 3)
PAT = re.compile(r"^(qat|lsq)_w(\d+)a(\d+)_(edge8|noedge)_s(\d+)\.json$")


def parse_filename(fn):
    m = PAT.match(os.path.basename(fn))
    if not m:
        return None
    method, w, a, edge_flag, seed = m.groups()
    return {
        "method": method,
        "w_bits": int(w),
        "a_bits": int(a),
        "edge": edge_flag == "edge8",
        "seed": int(seed),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render LaTeX table (mean±std over seeds) from QAT JSON results"
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory containing result JSON files"
    )
    parser.add_argument("--caption", default="QAT results", help="Table caption")
    parser.add_argument("--label", default="tab:qat_results", help="Table label")
    parser.add_argument(
        "--output_path", help="If given, write LaTeX to this file; otherwise stdout"
    )
    args = parser.parse_args()

    # gather accuracies grouped by hyper‑parameter combo (method, w, a, edge)
    scores = {}
    for fn in os.listdir(args.results_dir):
        if not fn.endswith(".json"):
            continue
        meta = parse_filename(fn)
        if not meta:
            continue
        with open(os.path.join(args.results_dir, fn)) as f:
            data = json.load(f)
        acc = max(d.get("accuracy", 0.0) for d in data) if data else 0.0
        key = (meta["method"], meta["w_bits"], meta["a_bits"], meta["edge"])
        scores.setdefault(key, []).append(acc)

    # compute mean & std for each combo
    rows = []
    for (method, w, a, edge), accs in scores.items():
        mean = stats.mean(accs)
        std = stats.stdev(accs) if len(accs) > 1 else 0.0
        rows.append(
            {
                "method": method,
                "w_bits": w,
                "a_bits": a,
                "edge": edge,
                "acc_mean": mean,
                "acc_std": std,
            }
        )

    # stable ordering
    rows.sort(key=lambda r: (r["method"], r["w_bits"], r["a_bits"], not r["edge"]))

    latex = render_latex_table(rows, args.caption, args.label)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(latex)
    else:
        print(latex)


if __name__ == "__main__":
    main()
