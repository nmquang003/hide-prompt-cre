#!/bin/bash

# Danh sách các device (GPU) với các số nguyên bất kỳ, độ dài mảng tùy ý
devices=(0 1)  # Ví dụ: 2 device, bạn có thể thay đổi tùy theo hệ thống của bạn
num_devices=${#devices[@]}

# Danh sách các seed
seeds=(2021 2121 2221 2321 2421)

# Danh sách các combination (mỗi chuỗi gồm:
# "dataname encoder_epochs prompt_pool_epochs classifier_epochs prompt_pool_size prompt_length prompt_top_k beta num_description use_prompt_in_des use_triple_loss")
combinations=(
    "TACRED 20 15 200 8 1 8 0.1 1 0 0"  # Best setting, lk=18
    "TACRED 20 15 200 8 2 4 0.1 1 0 0"  # Best setting, lk=24
    "TACRED 20 15 200 8 4 2 0.1 1 0 1"  # Best setting, lk=42
    "TACRED 20 15 200 8 8 1 0.1 1 0 1"  # Best setting, lk=81
    "FewRel 20 15 200 8 1 8 0.1 1 1 0"   # Best setting, lk=18
    "FewRel 20 15 200 8 2 4 0.1 1 1 0"   # Best setting, lk=24
    "FewRel 20 15 200 8 4 2 0.1 1 1 1"   # Best setting, lk=42
    "FewRel 20 15 200 8 8 1 0.1 1 1 1"   # Best setting, lk=81
)

# Khởi tạo các mảng tasks cho từng device một cách động
# Sử dụng biến tasks_i với i từ 0 đến num_devices-1
for (( i=0; i<num_devices; i++ )); do
    eval "tasks_$i=()"
done

counter=0
# Phân bổ các task theo kiểu round-robin cho từng device
for seed in "${seeds[@]}"; do
    for combination in "${combinations[@]}"; do
        # Tính chỉ số device hiện tại theo round-robin
        device_index=$(( counter % num_devices ))
        
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

        # Tạo lệnh chạy Python cho task hiện tại, gán GPU tương ứng từ mảng devices
        cmd="python run.py \
            --gpu ${devices[$device_index]} \
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
            --use_triple_loss $use_triple_loss"
        
        # Thêm lệnh vào mảng tasks của device tương ứng
        eval "tasks_$device_index+=(\"\$cmd\")"

        counter=$(( counter + 1 ))
    done
done

# Hàm chạy các task theo thứ tự tuần tự trên một device
run_tasks() {
    local device_id=$1
    shift
    local tasks=("$@")
    for t in "${tasks[@]}"; do
        echo "Running on GPU $device_id: $t"
        eval "$t"
    done
}

# Chạy các task cho từng device song song (mỗi device xử lý các task của nó tuần tự)
for (( i=0; i<num_devices; i++ )); do
    device_id=${devices[$i]}
    # Lấy mảng tasks tương ứng với device i
    eval "array=(\"\${tasks_$i[@]}\")"
    run_tasks $device_id "${array[@]}" &
done

# Chờ tất cả các tiến trình background hoàn thành
wait
