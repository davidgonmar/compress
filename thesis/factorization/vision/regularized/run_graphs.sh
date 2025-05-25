#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONS=(
  "$DIR/results/global_factorization_cifar10_resnet20_hoyer_finetuned_0.001.json"
  "$DIR/results/global_factorization_cifar10_resnet20_hoyer_finetuned_0.002.json"
  "$DIR/results/global_factorization_cifar10_resnet20_hoyer_finetuned_0.003.json"
  "$DIR/results/global_factorization_cifar10_resnet20_hoyer_finetuned_0.004.json"
  "$DIR/results/global_factorization_cifar10_resnet20_hoyer_finetuned_0.005.json"
)

python "$DIR/graph.py" \
  --jsons "${JSONS[@]}" \
  --output_flops "$DIR/results/flops_vs_acc.pdf" \
  --output_params "$DIR/results/params_vs_acc.pdf"