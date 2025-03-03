#!/bin/bash

device=0
seeds=(2021 2121 2221 2321 2421)
combinations=(
    "TACRED 20 15 200 8 4 8 3" # w/o description
    "TACRED 20 15 200 8 4 8 5" # w/o description
    "TACRED 20 15 200 8 4 8 7" # w/o description
    "TACRED 20 15 200 8 4 8 9" # w/o description
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
        prompt_length=$5
        prompt_top_k=$6
        prompt_pool_size=$7
        num_descriptions=$8


        echo "Running with 
        --dataname=$dataname,
        --seed=$seed, 
        --encoder_epochs=$encoder_epochs,
        --prompt_pool_epochs=$prompt_pool_epochs,
        --classifier_epochs=$classifier_epochs,
        --prompt_length=$prompt_length,
        --prompt_top_k=$prompt_top_k,
        --prompt_pool_size=$prompt_pool_size,
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
            --num_descriptions $num_descriptions  \
            --type_ctloss "new" \
            --use_ct_in_encoder "yes" \
            --strategy 3 
    done
done