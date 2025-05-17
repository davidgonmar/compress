#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
python -m quantization.vision.qat.render_table_qat_cifar10 \
  --results_dir "$DIR" \
  --caption "QAT CIFAR-10 Results (ResNet-20)" \
  --label "tab:qat_resnet20" \
  --output_path "$DIR/qat_results.tex"