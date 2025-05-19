#!/usr/bin/env bash
set -euo pipefail

# Directory this script lives in
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m sparsity.vision.structured_iterative.plot.py \
  "$DIR/results" \
  "$DIR/results/pruning_accuracy.pdf"
echo "[plot_results.sh] Plot written to results/pruning_accuracy.pdf"
