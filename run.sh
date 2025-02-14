#!/bin/bash

# Mảng chứa các giá trị tham số khác nhau
datanames=(FewRel)
encoder_lrs=(2e-5)
prompt_pool_lrs=(1e-4)
classifier_lrs=(5e-5)
prompt_lengths=(16)
prompt_top_ks=(8)
num_negs=(4)
seeds=(2021)
pull_constraint_coeffs=(0.1)
contrastive_loss_coeffs=(0.1)
encoder_epochs=(15 20)
prompt_pool_epochs=(15 20)
classifier_epochs=(150 200)

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
                    for encoder_epoch in "${encoder_epochs[@]}"; do
                      for prompt_pool_epoch in "${prompt_pool_epochs[@]}"; do
                        for classifier_epoch in "${classifier_epochs[@]}"; do
                          echo "Running experiment with dataname=$dataname, encoder_lr=$encoder_lr, prompt_pool_lr=$prompt_pool_lr, classifier_lr=$classifier_lr, prompt_length=$prompt_length, prompt_top_k=$prompt_top_k, seed=$seed, pull_constraint_coeff=$pull_constraint_coeff, contrastive_loss_coeff=$contrastive_loss_coeff, encoder_epoch=$encoder_epoch, prompt_pool_epoch=$prompt_pool_epoch, classifier_epoch=$classifier_epoch"
                          python run.py \
                            --max_length 256 \
                            --dataname "$dataname" \
                            --encoder_epochs "$encoder_epoch" \
                            --encoder_lr "$encoder_lr" \
                            --prompt_pool_epochs "$prompt_pool_epoch" \
                            --prompt_pool_lr "$prompt_pool_lr" \
                            --classifier_epochs "$classifier_epoch" \
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
    done
  done
done