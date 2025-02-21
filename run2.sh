#!/bin/bash

# List of specific combinations to run
# Each combination is a string with the following format:
# "dataname seed encoder_epochs prompt_pool_epochs classifier_epochs prompt_length prompt_top_k prompt_pool_size beta"

device=0
seeds=(2021 2121 2221 2321 2421 2521)
combinations=(
    "FewRel 20 15 200 8 1 8 0.1 1" # Best setting, lk=18
    "FewRel 20 15 200 8 2 4 0.1 1" # Best setting, lk=24
    "FewRel 20 15 200 8 4 2 0.1 1" # Best setting, lk=42
    "FewRel 20 15 200 8 8 1 0.1 1" # Best setting, lk=81
)

# Loop over each combination
for seed in "${seeds[@]}"; do
    for combination in "${combinations[@]}"; do
        # Split the string into individual values
        set -- $combination
        dataname=$1
        encoder_epochs=$2
        prompt_pool_epochs=$3
        classifier_epochs=$4
        prompt_pool_size=$5
        prompt_length=$6
        prompt_top_k=$7
        num_descriptions=$8
        beta=$9

        echo "Running with 
        --dataname=$dataname,
        --seed=$seed,
        --encoder_epochs=$encoder_epochs,
        --prompt_pool_epochs=$prompt_pool_epochs,
        --classifier_epochs=$classifier_epochs,
        --prompt_length=$prompt_length,
        --prompt_top_k=$prompt_top_k,
        --prompt_pool_size=$prompt_pool_size,
        --beta=$beta,
        --num_descriptions=$num_descriptions"

        # Run the Python command with the current combination
        python run.py \
            --gpu $device \
            --max_length 256 \
            --dataname $dataname \
            --encoder_epochs $encoder_epochs \
            --encoder_lr 2e-5 \
            --prompt_pool_epochs $prompt_pool_epochs \
            --prompt_pool_lr 1e-4 \
            --classifier_epochs $classifier_epochs \
            --seed $seed \
            --bert_path bert-base-uncased \
            --data_path datasets \
            --prompt_length $prompt_length \
            --prompt_top_k $prompt_top_k \
            --batch_size 16 \
            --prompt_pool_size $prompt_pool_size \
            --replay_s_e_e 100 \
            --replay_epochs 200 \
            --classifier_lr 5e-5 \
            --prompt_type only_prompt \
            --num_descriptions $num_descriptions \
            --beta $beta \
            --run_name $dataname-$seed-$encoder_epochs-$prompt_pool_epochs-$classifier_epochs-$prompt_pool_size-$prompt_length-$prompt_top_k-$num_descriptions-$beta
    done
done
