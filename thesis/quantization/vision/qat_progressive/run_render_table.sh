set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/results_prog}"
OUTPUT="${2:-table.tex}"

if [[ "$OUTPUT" == "-" ]]; then
  python -m quantization.vision.qat_progressive.render_table \
         --results_dir "$RESULTS_DIR" \
         --aggregate \
         --stdout
else
  # If the output path is relative, place it inside RESULTS_DIR
  if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="${RESULTS_DIR%/}/$OUTPUT"
  fi
  python -m quantization.vision.qat_progressive.render_table \
         --results_dir "$RESULTS_DIR" \
         --aggregate \
         --output "$OUTPUT"
  echo "LaTeX table written to $OUTPUT"
fi
