#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/results"

TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
LOG_PATH="$SCRIPT_DIR/results/training_log.json"
SAVE_PATH="$SCRIPT_DIR/results/cifar10_resnet20_hoyer_finetuned.pth"
PRETRAINED_PATH="resnet20.pth"


python "$TRAIN_SCRIPT" \
    --log_path "$LOG_PATH" \
    --save_path "$SAVE_PATH" \
    --pretrained_path "$PRETRAINED_PATH" \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.01 \
    --reg_weight 0.005
