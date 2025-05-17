#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"
python -m quantization.vision.ptq.quantize_offline_cifar10 \
  --model_name resnet20 \
  --pretrained_path resnet20.pth \
  --batch_size 128 \
  --calibration_batches 10 \
  --output_path "$DIR/results/quantization_results.json"