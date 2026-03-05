# source .venv/bin/activate

. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate poet

torchrun --standalone --nproc_per_node 8 torchrun_main_normalized.py \
    --model_config configs/llama_3b.json \
    --lr 0.0005 \
    --batch_size 32 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 50000 \
    --warmup_steps 2000 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 10000000 \
    --optimizer adamw \
    --init_type same \