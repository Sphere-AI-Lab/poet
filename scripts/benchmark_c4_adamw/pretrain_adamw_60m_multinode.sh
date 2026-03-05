source .venv/bin/activate

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"

export WANDB_MODE="offline"

# Get node rank from command line argument
NODE_RANK=${1:-0} 

# Multi-node settings
export MASTER_ADDR="172.22.8.13"  # IP of node 0
export MASTER_PORT=29500

torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    torchrun_main.py \
    --model_config configs/llama_60m_oft.json \
    --lr 0.001 \
    --batch_size 1 \
    --total_batch_size 512 \
    --num_training_steps 100 \
    --warmup_steps 0 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 10000000 \
    --optimizer adamw \
    --init_type same \

    #     --max_train_tokens 5B \