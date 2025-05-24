#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

PRETRAINED_RESNET20="resnet20.pth"


python -m factorization.vision.svd_global_activation_aware \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/global_activation_aware_flops_results.json"

python -m factorization.vision.svd_manual_activation_aware \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric energy \
  --output_file "${RESULTS_DIR}/manual_activation_aware_energy_results.json"

python -m factorization.vision.svd_manual_activation_aware \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/manual_activation_aware_flops_results.json"

python -m factorization.vision.svd_global \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/global_flops_results.json"

python -m factorization.vision.svd_manual \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/manual_flops_results.json"

python -m factorization.vision.svd_manual \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric energy \
  --output_file "${RESULTS_DIR}/manual_energy_results.json"
