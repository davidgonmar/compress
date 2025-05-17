#!/usr/bin/env bash
PARALLELISM="${1:-4}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"

METHODS=(qat lsq)
BITS=(2 4 8)
LEAVE_FLAGS=("" "--leave_edge_layers_8_bits")

# build commands
CMDS=()
for method in "${METHODS[@]}"; do
  for w in "${BITS[@]}"; do
    for a in "${BITS[@]}"; do
      for leave in "${LEAVE_FLAGS[@]}"; do
        tag="${method}_w${w}a${a}"
        if [ -n "$leave" ]; then
          tag="${tag}_edge8"
        else
          tag="${tag}_noedge"
        fi
        out="${DIR}/${tag}.json"
        CMDS+=("python ${DIR}/qat-cifar10.py --method $method --nbits_activations $a \
--nbits_weights $w $leave --output_path $out")
      done
    done
  done
done

# execute in parallel
printf "%s\n" "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
