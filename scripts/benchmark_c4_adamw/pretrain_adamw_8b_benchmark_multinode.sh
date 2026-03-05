# source .venv/bin/activate

. /home/wliu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate galore

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"

export WANDB_MODE="offline"

# Define options arrays
NODE_RANK=${1:-0} 
MAX_LENGTH_OPTIONS=(256 512 1024 2048)

# Get indices (required arguments)
MAX_LENGTH_IDX=$2
MAX_LENGTH=${MAX_LENGTH_OPTIONS[$MAX_LENGTH_IDX]}
BATCH_SIZE=$3
TOTAL_BATCH_SIZE=$4

# Multi-node settings
export MASTER_ADDR="172.22.8.9"  # IP of node 0: 9, 11, 13, 15
export MASTER_PORT=29500

# bash pretrain_adamw_8b_benchmark_multinode.sh


torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    torchrun_main_benchmark.py \
    --model_config configs/llama_8b.json \
    --lr 0.0005 \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --num_training_steps 10000 \
    --warmup_steps 0 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 500000 \
    --save_every 10000000 \
    --optimizer adamw \
    --init_type same \
    --benchmark \

    # --time_budget_hours 1


    #     --max_train_tokens 5B \

    # bash pretrain_adamw_8b_benchmark_multinode.sh 0 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 1 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 2 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 3 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 4 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 5 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 6 2 2 2048
    # bash pretrain_adamw_8b_benchmark_multinode.sh 7 2 2 2048