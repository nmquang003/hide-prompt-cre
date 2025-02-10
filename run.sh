#!/bin/bash

# Mảng chứa các giá trị tham số khác nhau
datanames=(TACRED)
encoder_lrs=(2e-5 5e-5)
prompt_pool_lrs=(1e-4 5e-5)
classifier_lrs=(5e-5 1e-4)
prompt_lengths=(8 16)
prompt_top_ks=(4 8)
num_negs=(2 4 8)
seeds=(2021)
pull_constraint_coeffs=(0.05 0.1 0.2)
contrastive_loss_coeffs=(0.05 0.1 0.2)

# Lặp qua tất cả các kết hợp tham số
for dataname in "${datanames[@]}"; do
  for encoder_lr in "${encoder_lrs[@]}"; do
    for prompt_pool_lr in "${prompt_pool_lrs[@]}"; do
      for classifier_lr in "${classifier_lrs[@]}"; do
        for prompt_length in "${prompt_lengths[@]}"; do
          for prompt_top_k in "${prompt_top_ks[@]}"; do
            for seed in "${seeds[@]}"; do
              for pull_constraint_coeff in "${pull_constraint_coeffs[@]}"; do
                for contrastive_loss_coeff in "${contrastive_loss_coeffs[@]}"; do
                  for num_neg in "${num_negs[@]}"; do
                    echo "Running experiment with dataname=$dataname, encoder_lr=$encoder_lr, prompt_pool_lr=$prompt_pool_lr, classifier_lr=$classifier_lr, prompt_length=$prompt_length, prompt_top_k=$prompt_top_k, seed=$seed, pull_constraint_coeff=$pull_constraint_coeff, contrastive_loss_coeff=$contrastive_loss_coeff"
                    python run.py \
                      --max_length 256 \
                      --dataname "$dataname" \
                      --encoder_epochs 15 \
                      --encoder_lr "$encoder_lr" \
                      --prompt_pool_epochs 15 \
                      --prompt_pool_lr "$prompt_pool_lr" \
                      --classifier_epochs 150 \
                      --seed "$seed" \
                      --bert_path bert-base-uncased \
                      --data_path datasets \
                      --prompt_length "$prompt_length" \
                      --prompt_top_k "$prompt_top_k" \
                      --batch_size 16 \
                      --prompt_pool_size 20 \
                      --replay_s_e_e 100 \
                      --replay_epochs 200 \
                      --classifier_lr "$classifier_lr" \
                      --prompt_type only_prompt \
                      --pull_constraint_coeff "$pull_constraint_coeff" \
                      --contrastive_loss_coeff "$contrastive_loss_coeff" \
                      --num_negs "$num_neg"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done