#!/bin/bash

relations=("A_not" "B_not" "not_not" "all" "AB")
#("A_not" "B_not" "not_not" "all" "AB")
#srelations=("L3_Contingency.Cause.Result")
srelations=("All_union" "All_intersection" "All_top")
#("SUPER_All_union" "SUPER_All_intersection")
#("All_top_causation_ante" "All_intersection_causation_ante")

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
