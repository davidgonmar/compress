#!/usr/bin/env bash
set -euo pipefail


DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DIR/results"

MODEL="resnet20"
PRETRAINED="resnet20.pth"
NUM_CLASSES=10
BATCH_SIZE=128
TEST_BATCH=512
WORKERS=4
LR=0.001
MOMENTUM=0.9
WD=5e-4
TARGET_SPARSITY=0.01
N_ITERS=12
EPOCHS_PER_ITER=5
SCHEDULER="linear"
CALIBRATION_SAMPLES=512
CALIBRATION_BS=4
SEED=0

METHODS=(magnitude_weights magnitude_activations taylor)

for m in "${METHODS[@]}"; do
  echo "=== Running method: $m ==="
  python -m sparsity.vision.unstructured_iterative.unstructured_iterative_pruning_cifar10 \
    --model "$MODEL" \
    --pretrained_path "$PRETRAINED" \
    --num_classes "$NUM_CLASSES" \
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
    --scheduler "$SCHEDULER" \
    --method "$m" \
    --calibration_samples "$CALIBRATION_SAMPLES" \
    --calibration_bs "$CALIBRATION_BS" \
    --seed "$SEED" \
    --stats_file "${DIR}/results/${m}_stats.json"
done
