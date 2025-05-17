#!/usr/bin/env bash
PARALLELISM="${1:-4}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results_prog"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

METHODS=(qat lsq)
BITS_LISTS=("8 4" "8 4 2")
MILESTONES_LISTS=("40" "25 50")
LEAVE_FLAGS=("--leave_edge_layers_8_bits")
SEEDS=(0 1 2)

CMDS=()

for method in "${METHODS[@]}"; do
  for i in "${!BITS_LISTS[@]}"; do
    bits="${BITS_LISTS[$i]}"
    milestones="${MILESTONES_LISTS[$i]}"
    bits_tag=${bits// /-}
    for leave in "${LEAVE_FLAGS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        tag="${method}_p${bits_tag}"
        [[ -n "$leave" ]] && tag="${tag}_edge8" || tag="${tag}_noedge"
        tag="${tag}_s${seed}"
        out="${DIR}/${tag}.json"
        CMDS+=("python -m quantization.vision.qat_progressive.qat_progressive \
--method $method --bits_list $bits --epoch_milestones $milestones \
$leave --seed $seed --output_path $out")
      done
    done
  done
done

printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD