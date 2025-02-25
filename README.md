### Requirements

```bash
pip install gpytorch==1.0 transformer wandb python-dotenv --quiet
```


### Run code

```bash
python run.py \
    --gpu 0 \
    --max_length 256 \
    --dataname TACRED \
    --encoder_epochs 20 \
    --prompt_pool_epochs 20 \
    --classifier_epochs 200 \
    --replay_s_e_e 100 \
    --replay_epochs 200 \
    --encoder_lr 2e-5 \
    --classifier_lr 5e-5 \
    --prompt_pool_lr 1e-4 \
    --seed 2021 \
    --bert_path bert-base-uncased \
    --data_path datasets \
    --prompt_length 4 \
    --prompt_top_k 2 \
    --prompt_pool_size 20 \
    --batch_size 16 \
    --num_descriptions 1 \
    --beta 0.1 \
    --use_triplet_loss True \
    --use_prompt_in_des True 
```

### Run many setting on multi device

```bash
bash run.sh
```

Có 5 seed: 2021, 2121, 2221, 2321, 2421
Dataname: TACRED, FewRel

