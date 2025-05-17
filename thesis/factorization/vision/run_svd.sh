#!/usr/bin/env bash
set -euo pipefail

# Paths to your pretrained weights (edit as needed)
PRETRAINED_RESNET20="resnet20.pth"

# -------------------------------
# 1. Global factorization (metric: FLOPs)
# -------------------------------
python -m factorization.vision.svd_global \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "global_flops_results.json"

# -------------------------------
# 2. Manual factorization (metric: FLOPs)
# -------------------------------
python -m factorization.vision.svd_manual \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_path "manual_flops_results.json"

# -------------------------------
# 3. Manual factorization (metric: Energy)
# -------------------------------
python -m factorization.vision.svd_manual \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric energy \
  --output_path "manual_energy_results.json"