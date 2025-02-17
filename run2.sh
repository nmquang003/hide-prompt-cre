#!/bin/bash

# List of specific combinations to run
# Each combination is a string with the following format:
# "dataname seed encoder_epochs prompt_pool_epochs classifier_epochs prompt_length prompt_top_k prompt_pool_size beta"
combinations=(
    "FewRel 2021 15 10 200 8 4 8 0.1" # Best setting
    "FewRel 2021 15 10 200 8 4 8 0" # w/o description
    "FewRel 2021 15 10 200 8 1 1 0" # w/o description and prompt pool
)

# Loop over each combination
for combination in "${combinations[@]}"; do
    # Split the string into individual values
    set -- $combination
    dataname=$1
    seed=$2
    encoder_epochs=$3
    prompt_pool_epochs=$4
    classifier_epochs=$5
    prompt_length=$6
    prompt_top_k=$7
    prompt_pool_size=$8
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
    --beta=$beta"

    # Run the Python command with the current combination
    python run.py \
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
        --num_descriptions 1 \
        --beta $beta
done
