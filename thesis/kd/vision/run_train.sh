#!/usr/bin/env bash
PARALLELISM="${1:-5}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

SEEDS=(0 1 2 3 4)

CMDS=()
for seed in "${SEEDS[@]}"; do
  out="${DIR}/train_s${seed}.json"
  CMDS+=("python \"$CURDIR/train.py\" \
    --nbits 2 \
    --epochs 200 \
    --batch_size 128 \
    --val_batch_size 512 \
    --pretrained_path resnet20.pth \
    --student_batches 1 \
    --matchers_batches 1 \
    --alpha 0.7 \
    --seed \"$seed\" \
    --output_path \"$out\"")
done

printf '%s\n' "${CMDS[@]}" | xargs -n1 -P "$PARALLELISM" bash -c