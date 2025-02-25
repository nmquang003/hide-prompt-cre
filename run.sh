#!/bin/bash

# Danh sách các device
devices=(0 1 2)

# Danh sách seed
seeds=(2021 2121 2221 2321 2421)

# Danh sách các combination (mỗi chuỗi gồm: dataname encoder_epochs prompt_pool_epochs classifier_epochs prompt_pool_size prompt_length prompt_top_k beta num_description use_prompt_in_des use_triple_loss)
combinations=(
    "TACRED 20 15 200 8 1 8 0.1 1 0 0" # Best setting, lk=18
    "TACRED 20 15 200 8 2 4 0.1 1 0 0" # Best setting, lk=24
    "TACRED 20 15 200 8 4 2 0.1 1 0 1" # Best setting, lk=42
    "TACRED 20 15 200 8 8 1 0.1 1 0 1" # Best setting, lk=81
    "FewRel 20 15 200 8 1 8 0.1 1 1 0" # Best setting, lk=18
    "FewRel 20 15 200 8 2 4 0.1 1 1 0" # Best setting, lk=24
    "FewRel 20 15 200 8 4 2 0.1 1 1 1" # Best setting, lk=42
    "FewRel 20 15 200 8 8 1 0.1 1 1 1" # Best setting, lk=81
)

# Biến đếm để chia đều các lần chạy sang device
counter=0
num_devices=${#devices[@]}

# Vòng lặp qua từng seed và từng combination
for seed in "${seeds[@]}"; do
    for combination in "${combinations[@]}"; do

        # Tách chuỗi combination thành các biến riêng
        set -- $combination
        dataname=$1
        encoder_epochs=$2
        prompt_pool_epochs=$3
        classifier_epochs=$4
        prompt_pool_size=$5
        prompt_length=$6
        prompt_top_k=$7
        beta=$8
        num_descriptions=$9
        use_prompt_in_des=${10}
        use_triple_loss=${11}

        # Tính chỉ số device hiện tại theo kiểu round-robin
        current_device=${devices[$(( counter % num_devices ))]}
        echo "Running on GPU $current_device with seed $seed and combination: $combination"

        # Chạy lệnh python trên device được phân bổ
        python run.py \
            --gpu $current_device \
            --max_length 256 \
            --dataname $dataname \
            --seed $seed \
            --encoder_lr 2e-5 \
            --prompt_pool_lr 1e-4 \
            --classifier_lr 5e-5 \
            --encoder_epochs $encoder_epochs \
            --prompt_pool_epochs $prompt_pool_epochs \
            --classifier_epochs $classifier_epochs \
            --bert_path bert-base-uncased \
            --data_path datasets \
            --prompt_pool_size $prompt_pool_size \
            --prompt_length $prompt_length \
            --prompt_top_k $prompt_top_k \
            --batch_size 16 \
            --replay_s_e_e 100 \
            --replay_epochs 200 \
            --num_descriptions $num_descriptions \
            --beta $beta \
            --use_prompt_in_des $use_prompt_in_des \
            --use_triple_loss $use_triple_loss &

        # Tăng biến đếm để chuyển sang device tiếp theo cho lần chạy kế
        counter=$(( counter + 1 ))
    done
done

# Chờ đợi tất cả các tiến trình background hoàn thành
wait
echo "All done!"