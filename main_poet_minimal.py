"""Minimal training example for POET and QPOET optimizers (Single GPU).

This is a simplified single-GPU training script using a dummy dataset.

Usage:
    python main_poet_minimal.py --model_config configs/llama_9m.json --batch_size 2 --poet_block_size 16
"""

import argparse
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoConfig

from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining import training_utils

from poet_torch import (
    POETAdamW,
    QPOETLinear,
    check_and_merge,
    get_grad_clipping_value,
    prepare_model_for_int8_training_poet,
    replace_linear_with_poet,
)


class DummyDataset(IterableDataset):
    """Dummy dataset that generates random sequences for testing."""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.num_samples):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            attention_mask = torch.ones(self.seq_length, dtype=torch.long)
            yield {"input_ids": input_ids, "attention_mask": attention_mask}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Minimal POET/QPOET training example")
    
    # Model
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config JSON")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_training_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--max_length", type=int, default=256, help="Sequence length")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="poet", choices=["poet", "q_poet"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--poet_lr", type=float, default=1e-3, help="POET learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    
    # POET specific
    parser.add_argument("--poet_reset_gap", type=int, default=20, help="Merge-then-reinitialize gap")
    parser.add_argument("--poet_block_size", type=int, default=64, help="POET block size")
    parser.add_argument("--poet_mem_efficient_mode", action="store_true")
    parser.add_argument("--gd_warmup_steps", type=int, default=50)
    
    # QPOET specific
    parser.add_argument("--weight_quant", action="store_true", help="Use QPOET (INT8)")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--weight_group_size", type=int, default=64)
    
    # Initialization
    parser.add_argument("--init_type", type=str, default="normalized", choices=["normalized"])
    
    # Scheduler
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    
    # System
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=25)
    
    args = parser.parse_args()
    args.total_batch_size = args.batch_size * args.gradient_accumulation
    
    return args


def setup_model_and_optimizer(args, device):
    """Setup model and optimizer."""
    print("Setup Model")
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(model_config)
    
    print("Setup POET")
    if args.optimizer == "poet":
        # Replace linear layers with POET layers
        replace_linear_with_poet(
            model, 
            args.poet_block_size,
            args.init_type,
            device=device,
            dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
            mem_efficient_mode=args.poet_mem_efficient_mode
        )
        
        # Collect parameters
        poet_params = [p for n, p in model.named_parameters() if p.requires_grad and 'oft' in n]
        id_poet = {id(p) for p in poet_params}
        
        decay_params = []
        nodecay_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad or id(p) in id_poet:
                continue
            if p.ndim >= 2 and not n.endswith('bias'):
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr,
                 use_poet=True, poet_reset_gap=args.poet_reset_gap, poet_scale=0.5),
        ]
        
    else:  # q_poet
        # Prepare for INT8 training
        dummy_args = type('Args', (), {
            'poet_block_size': args.poet_block_size,
            'weight_bits': args.weight_bits,
            'weight_group_size': args.weight_group_size,
            'stochastic_round': True,
            'init_type': args.init_type,
        })()
        
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        model = prepare_model_for_int8_training_poet(model, dummy_args, target_modules)
        
        # Collect parameters
        qpoet_params = [p for n, p in model.named_parameters() if p.requires_grad and 'oft' in n]
        id_qpoet = {id(p) for p in qpoet_params}
        
        decay_params = []
        nodecay_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad or id(p) in id_qpoet:
                continue
            if p.ndim >= 2 and not n.endswith('bias'):
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=qpoet_params, weight_decay=0.0, lr=args.poet_lr,
                 use_poet=True, poet_reset_gap=args.poet_reset_gap, poet_scale=0.5),
        ]
    
    print("Setup Optimizer")
    optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                          poet_block_size=args.poet_block_size)
    
    return model, optimizer


def evaluate(model, dataloader, device, max_batches: int = 10):
    """Evaluate model on dummy dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    model.train()
    return avg_loss, perplexity


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Log args
    print("=" * 50)
    print("Minimal POET/QPOET Training Example (Single GPU)")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k:25s}: {v}")
    print("=" * 50)
    
    # Setup model and optimizer
    model, optimizer = setup_model_and_optimizer(args, device)
    
    # Move to device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = model.to(device=device, dtype=dtype)
    
    # Move QPOET buffers
    for module in model.modules():
        if isinstance(module, QPOETLinear):
            module.weight_scales = module.weight_scales.to(device)
            module.weight_zeros = module.weight_zeros.to(device)
    
    # Load checkpoint if continuing
    global_step = 0
    update_step = 0

    # Compile model
    torch.compiler.reset()
    model = torch.compile(model)
    
    # Setup scheduler
    scheduler = training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    
    # Create dummy dataset
    model_config = AutoConfig.from_pretrained(args.model_config)
    dataset = DummyDataset(
        vocab_size=model_config.vocab_size,
        seq_length=args.max_length,
        num_samples=args.num_training_steps * args.batch_size * 10
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    # Training loop
    model.train()
    
    for batch in dataloader:
        if update_step >= args.num_training_steps:
            break
        
        global_step += 1
        
        # Forward
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / args.gradient_accumulation
        loss.backward()
        
        # Skip update if accumulating
        if global_step % args.gradient_accumulation != 0:
            continue
        
        # Gradient clipping
        if args.grad_clipping > 0:
            params = []
            for group in optimizer.param_groups:
                params.extend(group['params'])
            clip_val = get_grad_clipping_value(
                update_step, args.grad_clipping, 10, args.poet_reset_gap, 0.1, args.gd_warmup_steps
            )
            torch.nn.utils.clip_grad_norm_(params, clip_val)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Check and merge
        check_and_merge(model, update_step + 1, args.poet_reset_gap)
        
        update_step += 1
        
        # Log
        lr = optimizer.param_groups[0]["lr"]
        poet_lr = next((pg['lr'] for pg in optimizer.param_groups if pg.get('use_poet')), None)
        
        if update_step % 10 == 0:
            print(
                f"Step {update_step:3d} | Loss: {loss.item() * args.gradient_accumulation:.4f} | "
                f"LR: {lr:.2e} | POET LR: {poet_lr:.2e}"
            )
        
        # Evaluate
        if update_step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate(model, eval_dataloader, device)
            print(f"  Eval @ step {update_step}: loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_loss, final_ppl = evaluate(model, eval_dataloader, device, max_batches=20)
    print(f"Final loss: {final_loss:.4f}, Final perplexity: {final_ppl:.2f}")
    print("Training completed!")


if __name__ == "__main__":
    main()
