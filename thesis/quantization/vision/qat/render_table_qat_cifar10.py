import os
import json
import argparse
import re

def render_latex_table(rows, caption, label):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & W bits & A bits & Edge8 & Accuracy (\%) \\",
        r"\midrule"
    ]
    for r in rows:
        method = r['method']
        w = r['w_bits']
        a = r['a_bits']
        edge = "Yes" if r['leave_edge'] else "No"
        acc = r['accuracy']
        lines.append(f"{method} & {w} & {a} & {edge} & {acc:.2f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]
    return "\n".join(lines)

def parse_filename(fn):
    basename = os.path.basename(fn)
    m = re.match(r'^(qat|lsq)_w(\d+)a(\d+)_(edge8|noedge)\.json$', basename)
    if not m:
        return None
    method, w, a, edge = m.groups()
    return {
        'method': method,
        'w_bits': int(w),
        'a_bits': int(a),
        'leave_edge': (edge == 'edge8')
    }

def main():
    parser = argparse.ArgumentParser(description="Render LaTeX table from QAT JSON results")
    parser.add_argument("--results_dir", required=True, help="Directory containing QAT JSON files")
    parser.add_argument("--caption", default="QAT results", help="Table caption")
    parser.add_argument("--label", default="tab:qat_results", help="Table label")
    parser.add_argument("--output_path", help="File to write LaTeX table; stdout if omitted")
    args = parser.parse_args()

    rows = []
    for fn in os.listdir(args.results_dir):
        if fn.endswith('.json'):
            meta = parse_filename(fn)
            if meta:
                path = os.path.join(args.results_dir, fn)
                with open(path) as f:
                    data = json.load(f)
                if data:
                    last = data[-1]
                    acc = last.get('accuracy', 0.0)
                else:
                    acc = 0.0
                meta['accuracy'] = acc
                rows.append(meta)
    # sort rows for consistent order
    rows.sort(key=lambda x: (x['method'], x['w_bits'], x['a_bits'], not x['leave_edge']))
    table = render_latex_table(rows, args.caption, args.label)

    if args.output_path:
        with open(args.output_path, 'w') as f:
            f.write(table)
    else:
        print(table)

if __name__ == "__main__":
    main()

