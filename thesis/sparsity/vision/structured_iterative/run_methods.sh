#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define schedulers and methods
SCHEDULERS=(linear exp)
METHODS=(taylor norm_weights norm_activations)

# Common training settings
MODEL="resnet20"
PRETRAINED="resnet20.pth"
BATCH_SIZE=128
TEST_BATCH=512
WORKERS=4
LR=0.01
MOMENTUM=0.9
WD=5e-4
TARGET_SPARSITY=0.15
N_ITERS=12
EPOCHS_PER_ITER=8
CALIBRATION_SAMPLES=512
CALIBRATION_BS=4
SEED=0

for sched in "${SCHEDULERS[@]}"; do
  echo "=== Scheduler: $sched ==="

  # make a separate folder for this scheduler
  mkdir -p "$DIR/results/$sched"

  for m in "${METHODS[@]}"; do
    echo "--- Running method: $m with scheduler: $sched ---"
    python -m sparsity.vision.structured_iterative.structured_iterative_pruning_cifar10 \
      --model "$MODEL" \
      --pretrained_path "$PRETRAINED" \
      --batch_size "$BATCH_SIZE" \
      --test_batch_size "$TEST_BATCH" \
      --train_workers "$WORKERS" \
      --test_workers "$WORKERS" \
      --lr "$LR" \
      --momentum "$MOMENTUM" \
      --weight_decay "$WD" \
      --target_sparsity "$TARGET_SPARSITY" \
      --n_iters "$N_ITERS" \
      --epochs_per_iter "$EPOCHS_PER_ITER" \
      --scheduler "$sched" \
      --method "$m" \
      --calibration_samples "$CALIBRATION_SAMPLES" \
      --calibration_bs "$CALIBRATION_BS" \
      --seed "$SEED" \
      --stats_file "${DIR}/results/${sched}/${m}_stats.json"
  done
done
