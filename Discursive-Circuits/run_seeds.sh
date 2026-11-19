#!/bin/bash

# List of seeds
seeds=(42 91 532 438 793 156 284 907 63 419 712 35 668 501 229 874 94 360 781 147 590 18 946 312 655 87 478 821 204 739)
# List of relations
relations=("all" "AB" "A_not" "B_not" "not_not")
for relation in "${relations[@]}"
do
    echo "===================================="
    echo "Running for relation: $relation"
    echo "===================================="

    # Loop over each seed
    for seed in "${seeds[@]}"
    do
    	echo "Running with seed $seed"
    	python scripts/circuit_discovery.py --relation $relation --source_relation $relation --model_mode clm --seed $seed --mode learn --data_inverse
    done
done

