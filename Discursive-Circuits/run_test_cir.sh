#!/bin/bash

relations=("AB" "A_not" "B_not" "not_not" "all")
srelations=("AB" "A_not" "B_not" "not_not" "all")

for rel in "${relations[@]}"; do
  for srel in "${srelations[@]}"; do
    if [[ "$rel" != "$srel" ]]; then
      echo "Running: rel=$rel, source_relation=$srel"
      python scripts/circuit_discovery.py \
        --model_mode clm \
        --relation "$rel" \
        --source_relation "$srel" \
      	--mode patch \
        --data_inverse
    fi
  done
done
