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
NPROC_OPTIONS=(1 8)
MAX_LENGTH_OPTIONS=(512 1024 2048)

# Get indices (required arguments)
NPROC_IDX=$1
MAX_LENGTH_IDX=$2

# Select values from arrays
NPROC=${NPROC_OPTIONS[$NPROC_IDX]}
MAX_LENGTH=${MAX_LENGTH_OPTIONS[$MAX_LENGTH_IDX]}

# Determine if single GPU mode
if [ "$NPROC" -eq 1 ]; then
    SINGLE_GPU_FLAG="--single_gpu"
else
    SINGLE_GPU_FLAG=""
fi

echo "Running with NPROC=$NPROC (index $NPROC_IDX), MAX_LENGTH=$MAX_LENGTH (index $MAX_LENGTH_IDX)"

torchrun --standalone --nproc_per_node $NPROC torchrun_main_benchmark.py \
    --model_config configs/llama_8b.json \
    --lr 0.0005 \
    --batch_size 1 \
    --total_batch_size 512 \
    --max_length $MAX_LENGTH \
    --num_training_steps 100000 \
    --warmup_steps 0 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 500000 \
    --save_every 10000000 \
    --optimizer adamw \
    --init_type same \
    $SINGLE_GPU_FLAG \
    --benchmark \


    #     --max_train_tokens 5B \