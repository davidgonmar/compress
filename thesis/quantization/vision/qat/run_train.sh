#!/usr/bin/env bash
PARALLELISM="${1:-4}"

# directory that will hold one JSON file per run
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"

CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p $DIR

METHODS=(qat lsq)
BITS=(2 4 8)
LEAVE_FLAGS=("--leave_edge_layers_8_bits")
SEEDS=(0 1 2 3 4)

CMDS=()

for method in "${METHODS[@]}"; do
  for w in "${BITS[@]}"; do
    for a in "${BITS[@]}"; do
      for leave in "${LEAVE_FLAGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          tag="${method}_w${w}a${a}"
          if [ -n "$leave" ]; then
            tag="${tag}_edge8"
          else
            tag="${tag}_noedge"
          fi
          tag="${tag}_s${seed}"
          out="${DIR}/${tag}.json"

          CMDS+=("python $CURDIR/qat_cifar10.py \
--method $method --nbits_activations $a --nbits_weights $w \
$leave --seed $seed --output_path $out")
        done
      done
    done
  done
done

# launch all jobs in parallel
printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD