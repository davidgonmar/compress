#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"



python -m factorization.vision.graphs \
  --global_json "${RESULTS_DIR}/global_flops_results.json" \
  --manual_flops_json "${RESULTS_DIR}/manual_flops_results.json" \
  --manual_energy_json "${RESULTS_DIR}/manual_energy_results.json" \
  --manual_activation_aware_flops_json "${RESULTS_DIR}/manual_activation_aware_flops_results.json" \
  --manual_activation_aware_energy_json "${RESULTS_DIR}/manual_activation_aware_energy_results.json" \
  --global_activation_aware_flops_json "${RESULTS_DIR}/global_activation_aware_flops_results.json" \
  --output "${RESULTS_DIR}/flops_vs_acc.pdf"
