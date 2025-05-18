import argparse
import json
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(
    r"(?P<method>\w+)_p(?P<bits_schedule>[0-9\-]+)_(?P<edge>edge8|noedge)_s(?P<seed>\d+)\.json"
)


def _parse_file(path: Path) -> Dict | None:
    """Return metadata and best/last accuracy at *target* bits."""
    m = NAME_RE.fullmatch(path.name)
    if not m:
        return None  # skip files that do not follow the expected pattern

    meta = m.groupdict()
    try:
        data: List[Dict] = json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] Skipping {path} — cannot read JSON ({exc})", file=sys.stderr)
        return None

    if not data:
        print(f"[warn] Skipping {path} — empty result list", file=sys.stderr)
        return None

    target_bits = data[-1]["bits"]  # final quantization stage

    # filter epochs that use target bits
    target_phase = [d for d in data if d["bits"] == target_bits]
    if not target_phase:
        print(f"[warn] {path}: no epochs at target bits?", file=sys.stderr)
        return None

    best_target = max(target_phase, key=lambda d: d["accuracy"])
    last = data[-1]

    meta.update(
        {
            "target_bits": target_bits,
            "best_target_acc": best_target["accuracy"],
            "last_acc": last["accuracy"],
            "epochs": last["epoch"],
        }
    )
    return meta


# ........................................................................... #
# LaTeX builders
# ........................................................................... #


def _latex_detailed(rows: Iterable[Dict]) -> str:
    header = (
        "\\begin{tabular}{l l l c c c}"
        "\n\\toprule\n"
        "Method & Bits & Edge8 & Seed & Best@Target\\,(\%) & Last\\,(\%)\\\\\n"
        "\\midrule"
    )
    body_lines = [
        f"{r['method']} & {r['bits_schedule']} & {'Yes' if r['edge']=='edge8' else 'No'} & "
        f"{r['seed']} & {r['best_target_acc']:.2f} & {r['last_acc']:.2f}\\\\"
        for r in sorted(
            rows,
            key=lambda x: (x["method"], x["bits_schedule"], x["edge"], int(x["seed"])),
        )
    ]
    footer = "\\bottomrule\n\\end{tabular}"
    return "\n".join([header, *body_lines, footer])


def _latex_summary(rows: Iterable[Dict]) -> str:
    """Return μ ± σ of best accuracy at target bits."""
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r["bits_schedule"], r["edge"])
        groups[key].append(r["best_target_acc"])

    header = (
        "\\begin{tabular}{l l l c}"
        "\n\\toprule\n"
        "Method & Bits & Edge8 & Best@Target\\,(\%$\\mu\\,\\pm\\,\\sigma$)\\\\\n"
        "\\midrule"
    )
    body_lines = []
    for (m, b_sched, e), accs in sorted(groups.items()):
        mean = stats.mean(accs)
        std = stats.stdev(accs) if len(accs) > 1 else 0.0
        body_lines.append(
            f"{m} & {b_sched} & {'Yes' if e=='edge8' else 'No'} & {mean:.2f} $\\pm$ {std:.2f}\\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}"
    return "\n".join([header, *body_lines, footer])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Generate LaTeX tables from JSON results (best acc. at target bits)"
)
parser.add_argument(
    "--results_dir", default="results_prog", help="Directory with *.json run results"
)
parser.add_argument(
    "--output", default="table.tex", help="Output .tex file (ignored with --stdout)"
)
parser.add_argument(
    "--aggregate",
    action="store_true",
    help="Append a summary table with mean ± std across seeds (best@target)",
)
parser.add_argument(
    "--stdout",
    action="store_true",
    help="Print LaTeX to stdout instead of writing a file",
)
args = parser.parse_args()

rows: List[Dict] = []
for json_path in Path(args.results_dir).glob("*.json"):
    parsed = _parse_file(json_path)
    if parsed:
        rows.append(parsed)

if not rows:
    sys.exit(f"No valid result files found in {args.results_dir}")

tex_parts: List[str] = [_latex_detailed(rows)]
if args.aggregate:
    tex_parts.append("\n\n% Aggregated results\n" + _latex_summary(rows))

doc = "\n\n".join(tex_parts)

if args.stdout:
    print(doc)
else:
    Path(args.output).write_text(doc)
    print(f"LaTeX table written to {args.output} (booktabs format)")
