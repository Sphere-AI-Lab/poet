# source .venv/bin/activate

. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate poet

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"

# Get node rank from command line argument
NODE_RANK=${1:-0} 

# export WANDB_MODE="offline"

# Multi-node settings
export MASTER_ADDR="172.22.8.13"  # IP of node 0
export MASTER_PORT=29500

# export WANDB_MODE="offline"

torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    torchrun_main.py \
    --model_config configs/llama_8b.json \
    --lr 0.005 \
    --batch_size 16 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 100000 \
    --warmup_steps 0 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --save_every 10000000 \
    --optimizer poet \
    --poet_lr 0.001 \
    --poet_reset_gap 200 \
    --poet_block_size 512 \
    --init_type normalized \

# bash pretrain_poet_8b_multinode.sh 0