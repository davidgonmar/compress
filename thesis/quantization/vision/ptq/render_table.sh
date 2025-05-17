#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m quantization.vision.ptq.render_table_quantize_offline_cifar10 \
  --json_path "$DIR/results/quantization_results.json" \
  --caption "CIFAR-10 PTQ Results for ResNet-20" \
  --label_pt "tab:resnet20_cifar10_ptq_tensor" \
  --label_pc "tab:resnet20_cifar10_ptq_channel" \
  --output_path "$DIR/results/quantization_results.tex"