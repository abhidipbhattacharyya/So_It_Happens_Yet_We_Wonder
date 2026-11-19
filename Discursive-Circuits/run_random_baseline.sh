relations=("A_not" "B_not" "not_not" "all" "AB")
#srelations=("L3_Contingency.Cause.Result")
srelation="all_clm_random_circuits"
seeds=(91 532 438 793 156 284 907 63 419 712 35 668 501 229 874 94 360 781 147 590 18 946 312 655 87 478 821 204 739 42)

for rel in "${relations[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ "$rel" != "$srelation" ]]; then
      echo "Running: rel=$rel, source_relation=$srelation, seed=$seed"
      python scripts/circuit_discovery.py \
        --model_mode clm \
        --relation "$rel" \
        --source_relation "$srelation" \
      	--mode patch \
	    --seed $seed \
        --data_inverse
    fi
  done
done
