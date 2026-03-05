# source .venv/bin/activate

. /home/wliu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate galore

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"

# export WANDB_MODE="offline"

# export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node 1 torchrun_main_benchmark.py \
    --model_config configs/llama_13b.json \
    --lr 0.001 \
    --batch_size 1 \
    --total_batch_size 1 \
    --max_length 2048 \
    --num_training_steps 2000 \
    --warmup_steps 0 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 500000 \
    --save_every 10000000 \
    --optimizer poet \
    --poet_lr 0.001 \
    --poet_reset_gap 200 \
    --poet_block_size 512 \
    --init_type normalized \
    --single_gpu \
    --benchmark \
    --poet_reset_gap 1000000000 \
    --poet_mem_efficient_mode