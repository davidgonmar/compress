import json
import argparse


def render_side_by_side_tables(results, caption, label_pt, label_pc):
    # Split results
    per_tensor = [row for row in results if not row.get("weights_per_channel", False)]
    per_channel = [row for row in results if row.get("weights_per_channel", False)]

    def table_block(rows, title, label):
        lines = [
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\centering",
            rf"\caption{{{title}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{l c c}",
            r"\toprule",
            r"Type & Edge8 & Accuracy (\%) \\",
            r"\midrule",
        ]
        for row in rows:
            typ = row.get("type", "")
            leave = row.get("leave_edge_layers_8_bits", "-")
            if isinstance(leave, bool):
                leave = "Yes" if leave else "No"
            acc = row.get("accuracy", 0.0)
            lines.append(f"{typ} & {leave} & {acc:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}"]
        return "\n".join(lines)

    lines = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{caption}}}"]
    # Per-tensor on left
    lines.append(table_block(per_tensor, "Per-tensor quantization results", label_pt))
    lines.append(r"\hfill")
    # Per-channel on right
    lines.append(table_block(per_channel, "Per-channel quantization results", label_pc))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Render side-by-side LaTeX tables from JSON results"
    )
    parser.add_argument("--json_path", required=True, help="Path to JSON results file")
    parser.add_argument(
        "--caption", default="Quantization results", help="Main table caption"
    )
    parser.add_argument(
        "--label_pt", default="tab:quant_pt", help="Label for per-tensor table"
    )
    parser.add_argument(
        "--label_pc", default="tab:quant_pc", help="Label for per-channel table"
    )
    parser.add_argument(
        "--output_path", help="File to write LaTeX tables; stdout if omitted"
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        results = json.load(f)

    latex_tables = render_side_by_side_tables(
        results, args.caption, args.label_pt, args.label_pc
    )

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(latex_tables)
    else:
        print(latex_tables)


if __name__ == "__main__":
    main()
