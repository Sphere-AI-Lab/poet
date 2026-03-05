# source .venv/bin/activate

. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate poet

torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config configs/llama_3b.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 256 \
    --max_length 256 \
    --num_training_steps 999 \
    --warmup_steps 5000 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.01 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 5000 \
    --save_every 10000000 \
    --optimizer q_poet \
    --poet_lr 0.0005 \
    --poet_reset_gap 400 \
    --poet_block_size 256 \
    --init_type normalized \
    --weight_quant \
    --poet_scale_mode 1 \
    --gd_warmup_steps 10000