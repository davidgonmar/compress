#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FACTORIZE_SCRIPT="$SCRIPT_DIR/svd_global.py"

for MODEL_PATH in "$RESULTS_DIR"/cifar10_resnet20_hoyer_finetuned_*.pth; do
  MODEL_FILE="$(basename "$MODEL_PATH")"
  BASE_NAME="${MODEL_FILE%.pth}"
  OUTPUT_FILE="$RESULTS_DIR/global_factorization_${BASE_NAME}.json"

  echo "Running factorization on $MODEL_FILE"
  python "$FACTORIZE_SCRIPT" \
    --pretrained_path "$MODEL_PATH" \
    --output_file "$OUTPUT_FILE"
done
