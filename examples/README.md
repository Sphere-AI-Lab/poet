# POET Examples

This directory contains simple examples demonstrating the `poet_torch` API.

## 1. Quick Start


```python
from poet_torch import POETConfig, POETModel, get_poet_optimizer

# 1. Create config
config = POETConfig(block_size=256, merge_interval=200)

# 2. Wrap your model
model = POETModel(your_model, config)

# 3. Create optimizer
optimizer = get_poet_optimizer(model, config)

# 4. Training loop
for step, batch in enumerate(dataloader):
    loss = model(**batch)
    loss.backward()
    optimizer.step()
    model.merge_if_needed(step)  # Automatic merge
```

- Using `POETConfig` for configuration
- Wrapping a model with `POETModel`
- Using `get_poet_optimizer()` for automatic optimizer setup
- Automatic merge with `model.merge_if_needed(step)`

## 2. API Overview

### Configuration

```python
from poet_torch import POETConfig, QPOETConfig

# Standard POET
config = POETConfig(
    block_size=256,            # Block size for transformations
    merge_interval=200,        # Steps between merge operations
    poet_lr=5e-4,              # Learning rate for POET params
    base_lr=1e-3,              # Learning rate for base params
    mem_efficient_mode=False,  # Memory-efficient mode
)

# Quantized POET
qconfig = QPOETConfig(
    block_size=256,
    merge_interval=200,
    weight_bits=8,          # INT8 quantization
    weight_group_size=256,
)
```

### Model Wrapping

```python
from poet_torch import POETModel

# Wrap any PyTorch model
model = POETModel(base_model, config)

# Access model info
model.print_model_info()
info = model.get_model_info()

# Get effective weights
weights = model.get_effective_weights()
```

### Optimizer

```python
from poet_torch import get_poet_optimizer, POETAdamW

# Automatic optimizer setup
optimizer_auto = get_poet_optimizer(model, config)

# Manual optimizer setup
poet_params = get_poet_params(model)
base_params = ...
param_groups = [
    dict(params=base_params, weight_decay=0.0, lr=1e-3, use_poet=False),
    dict(params=poet_params, weight_decay=0.0, lr=5e-4, 
         use_poet=True, poet_merge_interval=200, poet_scale=0.5),
]
optimizer_manual = POETAdamW(param_groups, lr=1e-3)
```

### Training Loop

```python
# Automatic merge check
merged = model.merge_if_needed(step)

# Force immediate merge
model.merge()

# Manual merge (for custom training loops)
from poet_torch import merge_and_reinitialize
merged = merge_and_reinitialize(model, step, merge_interval=200)
```

### More to explore:
- Manual layer replacement with `replace_linear_with_poet()`
- Custom parameter groups
- Memory-efficient mode (`mem_efficient_mode=True`)
- Targeted module replacement
- Manual merge with `merge_and_reinitialize()`
- ...

## 3. Simple POET Example (`1_toy.py`)

A minimal single-GPU training example using the `POETModel` wrapper:

```bash
cd examples
python 1_toy.py --model_config configs/llama_250m.json
```

## 4. Large-scale Pre-training with `torchrun_main.py`

`torchrun_main.py` is a full-featured pretraining script supporting distributed training with DDP. It supports multiple optimizers including POET and QPOET.

### Prerequisites

1. Download and prepare the C4 dataset in `./c4/en/` directory
2. Install dependencies: `pip install torch transformers datasets loguru tqdm`

### Pretrain with POET (3B model on 8 GPUs)

```bash
cd examples
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config configs/llama_3b.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 600000 \
    --warmup_steps 5000 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.0 \
    --grad_clipping 0.1 \
    --dtype bfloat16 \
    --eval_every 20000 \
    --save_every 10000000 \
    --optimizer poet \
    --poet_lr 0.0005 \
    --poet_merge_interval 200 \
    --poet_block_size 512
```

### Pretrain with QPOET (W8 Quantized POET, 3B model on 8 GPUs)

```bash
cd examples
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config configs/llama_3b.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 256 \
    --max_length 256 \
    --num_training_steps 100000 \
    --warmup_steps 5000 \
    --min_lr_ratio 0.01 \
    --weight_decay 0.01 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 5000 \
    --save_every 10000000 \
    --optimizer q_poet \
    --poet_lr 0.0005 \
    --poet_merge_interval 400 \
    --poet_block_size 256 \
    --gd_warmup_steps 10000
```

### Notes

- POET requires dimensions to be divisible by `block_size`
- The `lm_head` layer is excluded by default
- Use `mem_efficient_mode=True` for large models to save memory
- QPOET uses INT8 quantization for additional memory savings
