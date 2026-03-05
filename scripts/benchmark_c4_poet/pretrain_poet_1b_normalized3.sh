# source .venv/bin/activate

. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate poet

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"


torchrun --standalone --nproc_per_node 8 torchrun_main_normalized.py \
    --model_config configs/llama_1b_oft.json \
    --lr 0.02 \
    --batch_size 64 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 50000 \
    --warmup_steps 2000 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 5000 \
    --save_every 10000000 \
    --optimizer poetv2 \
    --poet_lr 0.001 \
    --poet_reset_gap 100 \
    --poet_block_size 512 \
    --init_type same \
    # --poet_balance_lr \
    # --single_gpu \
    # --use_ngpt \
    # --single_gpu \
    # --profile \
    # --benchmark \


torchrun --standalone --nproc_per_node 8 torchrun_main_normalized.py \
    --model_config configs/llama_1b_oft.json \
    --lr 0.02 \
    --batch_size 64 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 50000 \
    --warmup_steps 2000 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 5000 \
    --save_every 10000000 \
    --optimizer poetv2 \
    --poet_lr 0.002 \
    --poet_reset_gap 100 \
    --poet_block_size 512 \
    --init_type same \
    # --poet_balance_lr \
    # --single_gpu \
    # --use_ngpt \
    # --single_gpu \
    # --profile \
    # --benchmark \