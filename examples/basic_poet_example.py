"""
Basic POET Training Example (Single GPU)
=========================================

This is a minimal example showing how to use the new poet_torch API
to train a model with POET optimizer.

Usage:
    python examples/basic_poet_example.py --model_config configs/llama_20m.json
"""

import argparse
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoConfig

# Import from new poet_torch API
from poet_torch import POETConfig, QPOETConfig, POETModel, get_poet_optimizer

# For the base model
from peft_pretraining.modeling_llama import LlamaForCausalLM


class DummyDataset(IterableDataset):
    """Dummy dataset that generates random sequences for testing."""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.num_samples):
            random_start = np.random.randint(self.vocab_size - self.seq_length - 10)
            input_ids = torch.arange(random_start, random_start + self.seq_length)
            attention_mask = torch.ones(self.seq_length, dtype=torch.long)
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Basic POET training example")
    parser.add_argument("--model_config", type=str, default="examples/configs/llama_250m.json", help="Path to model config JSON")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--block_size", type=int, default=64, help="POET block size")
    parser.add_argument("--merge_interval", type=int, default=20, help="Merge interval")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_qpoet", action="store_true", help="Whether to use QPOET (quantized POET)")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===============================
    # Step 1: Create QPOET/POET Config
    # ===============================
    if not args.use_qpoet:
        # Create standard POET configuration 
        config = POETConfig(
            block_size=args.block_size,
            merge_interval=args.merge_interval,
            poet_lr=5e-4,
            base_lr=1e-3,
            mem_efficient_mode=True, # Enable memory-efficient mode
        )
        print(f"\nPOET Config: block_size={config.block_size}, merge_interval={config.merge_interval}")
    else:
        # Create QPOET configuration with target modules for quantization
        config = QPOETConfig(
            block_size=args.block_size,
            merge_interval=args.merge_interval,
            poet_lr=5e-4,
            base_lr=1e-3,
            # specify module to adopt POET Linear
            target_module=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'],
            # QPOET-specific settings
            weight_bits=8,
            weight_group_size=256,
        )
        print(f"\nQPOET Config: block_size={config.block_size}, weight_bits={config.weight_bits}, merge_interval={config.merge_interval}")

    # ===============================
    # Step 2: Load Base Model
    # ===============================
    print("\nLoading base model...")
    model_config = AutoConfig.from_pretrained(args.model_config)
    base_model = LlamaForCausalLM(model_config)

    # ===============================
    # Step 3: Wrap with POET
    # ===============================
    print("\nWrapping model with POET...")
    model = POETModel(base_model, config)
    model = model.to(device=device, dtype=torch.bfloat16)

    # ===============================
    # Step 4: Create Optimizer
    # ===============================
    print("\nCreating POET optimizer...")
    optimizer = get_poet_optimizer(model, config)

    # ===============================
    # Step 5: Setup Data
    # ===============================
    print("\nSetting up dataset...")
    dataset = DummyDataset(
        vocab_size=model_config.vocab_size,
        seq_length=128,
        num_samples=args.num_steps * args.batch_size * 2
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    # ===============================
    # Step 6: Training Loop
    # ===============================
    print("\nStarting training...")
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Merge POET transformations if needed
        merged = model.merge_if_needed(step)

        # Logging
        if step % 10 == 0:
            status = " [MERGED]" if merged else ""
            print(f"Step {step:3d} | Loss: {loss.item():.4f}{status}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
