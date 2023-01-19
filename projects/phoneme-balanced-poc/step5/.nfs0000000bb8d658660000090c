#!/bin/bash

gpu_num=$1
config_name=$2
lr=(0.1 0.01 0.001)
warmup_steps=(2500 5000 10000)
output_feature_size=(128 256)
hidden_feature_size=(128 256)
num_layer=(2 4)
for l in "${lr[@]}"; do
    echo "lr: $l"
    for w in "${warmup_steps[@]}"; do
        echo "warmup_step: $w"
        for o in "${output_feature_size[@]}"; do
            echo "output_feature_size: $o"
            for h in "${hidden_feature_size[@]}"; do
                echo "hidden_feature_size: $h"
                for n in "${num_layer[@]}"; do
                    echo "num_layer: $n"
                    python main.py --config-name $config_name selection.type=random train.optimize.warmup_steps=$w train.optimize.lr=$l model.subsampling.output_feature_size=$o model.transformer.hidden_feature_size=$h model.transformer.num_layer=$n
                done
            done
        done
    done
done