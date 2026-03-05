import os
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime
import glob
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
from datasets import DownloadConfig, Features, Value
import datasets.distributed
import wandb
import math

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb

from poet_torch import (
    POETAdamW, 
    replace_linear_with_poet, 
    check_and_merge, 
    get_grad_clipping_value, 
    prepare_model_for_int8_training_poet, 
    QPOETLinear,
)
from MUON.muon_optimized import MuonOptimized

transformers.logging.set_verbosity_error()


def load_local_data(split='train', max_samples=None, seed=42):
    """
    Load local C4 data with reproducible shuffling, loading files one by one until
    reaching the desired number of samples.
    
    Args:
        split: 'train' or 'validation'
        max_samples: Maximum number of samples to load (None for all)
        seed: Random seed for reproducible shuffling
    """
    features = Features({
        'text': Value('string'),
        'timestamp': Value('string'),
        'url': Value('string')
    })
    
    data_dir = "./c4/en"
    cache_dir = "/tmp/c4"
    
    # Get all available files
    all_files = sorted(glob.glob(os.path.join(data_dir, f"c4-{split}.*.json.gz")))
    
    if not all_files:
        logger.warning(f"No files found in {data_dir}, falling back to streaming")
        return None
    
    # Use deterministic file order based on seed
    random.seed(seed)
    random.shuffle(all_files)
    
    # For validation split, load all files regardless of max_samples
    if split == 'validation':
        max_samples = None
    
    # Load files one by one until we have enough samples
    collected_datasets = []
    total_samples = 0
    files_used = 0
    
    for file_path in all_files:
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size/1024/1024:.2f} MB")
            
            # Try to load the dataset with more detailed error handling
            try:
                file_dataset = datasets.load_dataset(
                    "json",
                    data_files=file_path,
                    features=features,
                    streaming=False,
                    cache_dir=cache_dir,  # Use a fresh cache directory
                    num_proc=16
                )
            except Exception as inner_e:
                logger.error(f"Dataset loading error for {file_path}: {str(inner_e)}")
                logger.error(f"Error type: {type(inner_e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
            
            file_samples = len(file_dataset['train'])
            files_used += 1
            
            # Add to our collection
            collected_datasets.append(file_dataset['train'])
            total_samples += file_samples
            
            logger.info(f"Loaded file {files_used}: {file_path} with {file_samples} samples. Total: {total_samples}")
            
            # Check if we have enough samples (only for train split)
            if max_samples is not None and total_samples >= max_samples:
                logger.info(f"Reached target of {max_samples} samples after loading {files_used} files")
                break
                
        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {e}. Skipping.")
    
    # Combine all loaded datasets
    if collected_datasets:
        combined_dataset = datasets.concatenate_datasets(collected_datasets)
        
        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=seed)
        
        # Take exactly max_samples if we have more (only for train split)
        # if max_samples is not None and len(combined_dataset) > max_samples:
        #    combined_dataset = combined_dataset.select(range(max_samples))
            
        logger.info(f"Final dataset has {len(combined_dataset)} samples from {files_used} files")
        return combined_dataset
    else:
        raise ValueError("Failed to load any valid files")

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts", "wsd"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--num_stable_steps", type=int, default=None)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    # beta2 for AdamW
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--init_type", type=str, default="normalized", choices=["normalized", "same", "mup_normalized"])
    parser.add_argument("--mup_alpha", type=float, default=1.0)

    # POET parameters
    parser.add_argument("--poet_lr", type=float, default=1e-4)
    parser.add_argument("--poet_reset_gap", type=int, default=200)
    parser.add_argument("--poet_block_size", type=int, default=256)
    parser.add_argument("--poet_mem_efficient_mode", action="store_true")
    parser.add_argument("--poet_neurips_version", action="store_true")
    parser.add_argument("--gd_warmup_steps", type=int, default=2000)
    parser.add_argument("--poet_balance_lr", action="store_true")
    parser.add_argument("--poet_use_rmsnorm", action="store_true")
    parser.add_argument("--poet_scale_mode", type=int, default=0)

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-iteration benchmark and exit")
    parser.add_argument("--profile", action="store_true", help="Run profiling and exit")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    # val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True) #DGX
    # val_data = val_data.shuffle(seed=42)
    val_data = load_local_data(split='validation', max_samples=10000, seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
        # num_proc=None,
        load_from_cache_file=True
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    # Calculate perplexity
    perplexity = math.exp(total_loss)

    return total_loss, perplexity, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()


    ###########################
    # Initialize wandb
    ###########################

    # initialize wandb with composed names (config is passed later)
    if global_rank == 0:
        model_stem = os.path.splitext(os.path.basename(args.model_config))[0]
        if args.max_train_tokens is not None:
            scale_str = f"{args.max_train_tokens/1e9:.1f}B-tokens"
        else:
            scale_str = f"{args.num_training_steps}steps"

        # allow env/CLI override; fall back to composed name
        project_name = os.environ.get("WANDB_PROJECT", f"sPOET-{model_stem}-{scale_str}-public")
        project_name = getattr(args, "wandb_project", None) or project_name

        group_name = os.environ.get("WANDB_RUN_GROUP", None)
        tags = args.tags.split(",") if args.tags else None

        logger.info(f"[W&B] project={project_name} | run={args.run_name}" + (f" | group={group_name}" if group_name else ""))
        wandb.init(project=project_name, name=args.run_name, group=group_name, tags=tags)
        if wandb.run is not None:
            logger.info(f"[W&B] id={wandb.run.id} | url={wandb.run.url}")
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    ###########################
    # Load data
    ###########################

    # Calculate how many samples we need based on training steps
    samples_needed = args.num_training_steps * args.total_batch_size
    # Add some buffer (10%) to account for filtering, etc.
    samples_needed = int(samples_needed * 1.2)
    logger.info(f"Auto-calculated samples needed: {samples_needed} based on {args.num_training_steps} steps")
    
    seed_for_shuffle = args.seed  # Use the same seed as the rest of the training
    
    # Try to load local data with limited samples
    local_data = load_local_data(split='train', max_samples=samples_needed, seed=seed_for_shuffle)
    
    if local_data is not None:
        # Use the local data we loaded
        data = local_data
    else:
        # Fall back to original streaming approach
        logger.info("Using original streaming approach")
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        data = data.shuffle(seed=seed_for_shuffle)
    
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    # tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(
        "google-t5/t5-base",
        local_files_only=True,
        model_max_length=args.max_length,
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    
    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.weight_quant:
        # Enable INT8 training
        assert args.optimizer.lower() in [
            "q_poet",
            "q_poet_per_layer",
        ]
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        model = prepare_model_for_int8_training_poet(model, args, target_module)
        print('--'*20)
        print('Prepare Model for Int8 Training')
        print('--'*20)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # INT8 training: move the scales and zeros to the same device as the weight
    for _, module in model.named_modules():
        if isinstance(module, QPOETLinear):
            weight_device = module.weight.device
            module.weight_scales = module.weight_scales.to(device=weight_device)
            module.weight_zeros = module.weight_zeros.to(device=weight_device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_int8 = [p for p in model.parameters() if hasattr(p, 'group_size')]

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    if args.optimizer.lower() == "poet":
        replace_linear_with_poet(model, args.poet_block_size, args.init_type, args.mup_alpha, device=model.device, dtype=model.dtype)
        
        poet_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'oft' in name:
                poet_params.append(param)

        id_poet_params = {id(param) for param in poet_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_poet_params:
                continue
            if param.ndim >= 2 and not name.endswith('bias'):
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        if args.poet_scale_mode == 0:
            poet_scale = 1.0
        elif args.poet_scale_mode == 1:
            poet_scale = 1 / 2
        elif args.poet_scale_mode == 2:
            poet_scale = 1 / np.sqrt(2)
        elif args.poet_scale_mode == 3:
            poet_scale = (1 / 2) * (1 / np.sqrt(2))
        elif args.poet_scale_mode == 4:
            poet_scale = 0.1

        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap, poet_scale=poet_scale),
        ]

    elif args.optimizer.lower() == "q_poet":
        qpoet_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'oft' in name:
                qpoet_params.append(param)

        id_qpoet_params = {id(param) for param in qpoet_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_qpoet_params:
                continue
            if param.ndim >= 2 and not name.endswith('bias'):
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        if args.poet_scale_mode == 0:
            poet_scale = 1.0
        elif args.poet_scale_mode == 1:
            poet_scale = 1 / 2
        elif args.poet_scale_mode == 2:
            poet_scale = 1 / np.sqrt(2)
        elif args.poet_scale_mode == 3:
            poet_scale = (1 / 2) * (1 / np.sqrt(2))
        elif args.poet_scale_mode == 4:
            poet_scale = 0.1

        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=qpoet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap, poet_scale=poet_scale),
        ]

    elif args.optimizer.lower() == "muon":
        muon_params = []
        target_modules_list = ["attn", "mlp"]
        # MUON should not be used for bias and embeddings and the final output layer
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable MUON for weights in module: ', module_name)
            muon_params.append(module.weight)
        id_muon_params = [id(p) for p in muon_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_muon_params]
        param_groups = [
            {'params': regular_params, 'use_muon': False, 'lr': args.lr, 'weight_decay': args.weight_decay},           
            {'params': muon_params, 'use_muon': True, 'lr': 0.02, 'weight_decay': args.weight_decay},
        ]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "poet":
        optimizer = POETAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            poet_block_size=args.poet_block_size,
        )
    elif args.optimizer.lower() == "muon":
        muon_params = []
        adamw_params = []
        for group in param_groups:
            if group.get('use_muon'):
                muon_params.extend(group['params'])
            else:
                adamw_params.extend(group['params'])
        optimizer = MuonOptimized(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "q_poet":
        optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, poet_block_size=args.poet_block_size)
    
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            num_stable_steps=args.num_stable_steps,
        )

    # base model is the model without FSDP
    base_model = model
    torch.compiler.reset()
    model = torch.compile(model)

    if not args.single_gpu:
        if any(k in args.optimizer.lower() for k in ('poet', 'muon', 'adamw')):
            model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                # find_unused_parameters=True,
            )
        else:
            mixed_precision_policy = None
            if args.dtype in ["bf16", "bfloat16"]:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            gpus_per_node = torch.cuda.device_count()
            device_mesh = init_device_mesh(
                "cuda", 
                (world_size // gpus_per_node, gpus_per_node),
                mesh_dim_names=("replicate", "shard")
            )

            # Replace the DDP block with FSDP
            model = FSDP(
                model,
                device_id=local_rank,
                mixed_precision=mixed_precision_policy,
                # sharding_strategy=ShardingStrategy.FULL_SHARD,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                device_mesh=device_mesh,
                use_orig_params=True,  # keeps optimizer compatibility and original param views
            )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    optimizer_step_count = 0

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        # padding diagnostics (per-batch)
        # valid_per_sample = batch["attention_mask"].sum(dim=1)  # shape: [batch]
        # avg_valid = valid_per_sample.float().mean().item()
        # max_len = batch["attention_mask"].shape[1]
        # pad_ratio = 1.0 - (avg_valid / max_len)

        # loss = model(**batch, labels=labels).loss
        # scaled_loss = loss / args.gradient_accumulation
        # scaled_loss.backward()
        is_accumulating = global_step % args.gradient_accumulation != 0
        if not args.single_gpu and is_accumulating:
            with model.no_sync():
                loss = model(**batch, labels=labels).loss
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
        else:
            loss = model(**batch, labels=labels).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

        if is_accumulating:
            continue


        # The below code is only executed during the update step
        
        # add grad clipping
        if args.grad_clipping != 0.0: 
            if 'poet' in args.optimizer.lower():
                parameters = []
                for group in param_groups:
                    parameters.extend(group['params'])
                current_grad_clipping_value = get_grad_clipping_value(
                    global_step=optimizer_step_count,
                    grad_clipping=args.grad_clipping,
                    warmup_steps=10,
                    period_T=args.poet_reset_gap,
                    min_ratio=0.1,
                    max_steps=args.gd_warmup_steps,
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, current_grad_clipping_value)
                # total_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
            elif args.optimizer.lower() in ('muon', 'adamw', 'sgd', 'adafactor', 'adam8bit'):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            else:
                if args.single_gpu:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
                else:
                    total_grad_norm = FSDP.clip_grad_norm_(model, args.grad_clipping)

        if global_rank == 0 and args.grad_clipping != 0.0:
            wandb.log({
                "gradients/total_grad_norm": total_grad_norm,
            }, step=global_step)

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step_count += 1
            if args.optimizer.lower() == "poet" or args.optimizer.lower() == "q_poet":
                check_and_merge(model, optimizer_step_count, args.poet_reset_gap)

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            # Save model - handle FSDP vs DDP/single_gpu differently
            if any(k in args.optimizer.lower() for k in ('poet', 'muon', 'adamw')):
                if args.single_gpu:
                    model.save_pretrained(current_model_directory, max_shard_size='100GB')
                else:
                    if global_rank == 0:
                        model.module.save_pretrained(current_model_directory, max_shard_size='100GB')
            else:
                if not args.single_gpu:
                    with FSDP.state_dict_type(
                        model, StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    ):
                        full_state_dict = model.state_dict()
                    if global_rank == 0:
                        base_model.save_pretrained(
                            current_model_directory,
                            state_dict=full_state_dict,
                        )
                else:
                    model.save_pretrained(current_model_directory)

            if global_rank == 0:
                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)
                    
                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, perplexity, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": perplexity,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}, perplexity: {perplexity:.2f}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            if 'poet' in args.optimizer.lower():
                poet_lr = next((pg['lr'] for pg in optimizer.param_groups if pg.get('use_poet')), None)
                wandb.log({
                    "poet_lr": poet_lr,
                    },
                    step=global_step,
                )
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)

        if any(k in args.optimizer.lower() for k in ('poet', 'muon', 'adamw')):
            if args.single_gpu:
                model.save_pretrained(current_model_directory)
            else:
                to_save = model.module
                if global_rank == 0:
                    to_save.save_pretrained(current_model_directory)
        else:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state_dict = model.state_dict()
            if global_rank == 0:
                base_model.save_pretrained(
                    current_model_directory,
                    state_dict=full_state_dict,
                )

        if global_rank == 0:
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, perplexity, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_perplexity": perplexity,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
