import os
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime

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
# from peft_pretraining.modeling_llama_ngpt import LlamaForCausalLM
from peft_pretraining.modeling_llama import LlamaForCausalLM
# from peft_pretraining.ngpt_model import nGPT

from peft import LoraConfig, get_peft_model

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from q_galore_torch import QGaLoreAdamW8bit, QGaLoreAdamW8bit_simulate
from q_galore_torch.utils.quantization import prepare_model_for_int8_training_galore, QGaLoreLinear
from q_galore_torch.utils.simulate_quantization import QLinear
from q_galore_torch.utils.setup import saving_model_weight, load_model_weight

from apollo_torch import APOLLOAdamW, QAPOLLOAdamW #, GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, 
from apollo_torch.utils.fake_quantization import QLinear
from apollo_torch.utils.quantization import QScaleLinear, prepare_model_for_int8_training_apollo

from poet_torch import (
    POETAdamW, 
    POETAdamWContinuous,
    replace_linear_with_poet, 
    check_and_merge, 
    get_grad_clipping_value, 
    mhe_optimized_init, 
    mhe_optimized_init_multi_gpu,
    calculate_total_mhe, 
    mhe_worker_process,
    prepare_model_for_int8_training_poet, 
    QPOETLinear,
    estimate_poet_delta_weff_spec,
    _find_module_by_name_substr,
)
from poet_torch.poet_layerv2 import replace_linear_with_poet_v2, check_and_merge_v2
from poet_torch.poet_layerv3 import replace_linear_with_poet_v3, check_and_merge_v3
from poet_torch.poet_layer_monarch import replace_linear_with_poet_monarch, check_and_merge_monarch
from poet_torch.poet_layer_continuous import replace_linear_with_poet_continuous, check_and_merge_continuous
from poet_torch.q_poet_adamw8bit import AdamW8bit as POETAdamW8bit
from MUON.muon_optimized import MuonOptimized
from poet_torch.sgd import POETSGD
# from MUON.muon_official import MuonWithAuxAdam

from poet_torch.poet_layer import POETLinear

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
    import glob

    # Determine cache directory
    # if os.path.exists("/tmp/c4"):
    #     cache_dir = "/tmp/c4"
    # else:
    #     cache_dir = "/local/reservation/c4"
    
    cache_dir = "/tmp/c4"
    # cache_dir = "/local/reservation/c4"
    
    # Get all available files
    all_files = sorted(glob.glob(os.path.join(data_dir, f"c4-{split}.*.json.gz")))
    
    if not all_files:
        raise ValueError(f"No files found in {data_dir} matching c4-{split}.*.json.gz")
    
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

    parser.add_argument("--init_type", type=str, default="normalized", choices=["normalized", "same", "mhe_optimized", "mup_normalized"])
    parser.add_argument("--mup_alpha", type=float, default=1.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    # Q-GaLore hyperparameters: quantization
    parser.add_argument("--proj_quant", action='store_true')
    parser.add_argument("--proj_bits", type=int, default=8)
    parser.add_argument("--proj_group_size", type=int, default=256)
    parser.add_argument("--weight_quant", action='store_true')
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--weight_group_size", type=int, default=256)
    parser.add_argument("--stochastic_round", action='store_true')
    parser.add_argument("--simulation", action='store_true')
    parser.add_argument("--cos_threshold", type=float, default=1)
    parser.add_argument("--gamma_proj", type=int, default=2)
    parser.add_argument("--queue_size", type=int, default=5)
    # APOLLO hyperparameters
    parser.add_argument("--proj", type=str, default="random") # "random" or "svd"
    parser.add_argument("--scale_type", type=str, default="tensor") # "tensor" or "channel"
    parser.add_argument("--apollo_scale", type=float, default=1.0) # scale for gradient scaling factor
    parser.add_argument("--scale_front", action='store_true') # put the nl before or after scale the gradient with the apollo_scale

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

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

    # MHE Distributed
    parser.add_argument("--mhe_worker_id", type=int, default=None, help="ID of this worker job")
    parser.add_argument("--mhe_total_workers", type=int, default=None, help="Total number of worker jobs")
    parser.add_argument("--mhe_save_dir", type=str, default="mhe_layers", help="Directory to save optimized layers")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-iteration benchmark and exit")
    parser.add_argument("--profile", action="store_true", help="Run profiling and exit")

    # nGPT parameters
    parser.add_argument("--use_ngpt", default=False, action="store_true")
    parser.add_argument("--base_scale", type=float, default=1.0)
    
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
        project_name = os.environ.get("WANDB_PROJECT", f"sPOET-{model_stem}-{scale_str}")
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
        "/lustre/fast/fast/zqiu/hf_models/t5-base",
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

    # # counts "real" (non-pad) tokens after truncation, incl. special tokens
    # total_tokens = 0
    # total_padded_tokens = 0
    # total_sequences = 0


    # if global_rank == 0:
    #     num_examples = len(data)
    #     total_batches = math.ceil(num_examples / args.batch_size)

    #     for batch in tqdm(dataloader, total=total_batches, desc="Counting tokens", ncols=80):
    #         # batch["attention_mask"]: [bs, max_length]
    #         am = batch["attention_mask"]
    #         total_tokens += am.sum().item()
    #         total_padded_tokens += am.numel()
    #         total_sequences += am.shape[0]

    #     logger.info(f"Total sequences: {total_sequences:,}")
    #     logger.info(f"Total batches: {total_batches:,}")
    #     logger.info(f"Batch size: {args.batch_size:,}")
    #     logger.info(f"Total tokens (non-pad): {int(total_tokens):,}")
    #     logger.info(f"Total tokens (with padding): {int(total_padded_tokens):,}")
    #     logger.info(f"Avg non-pad tokens / seq: {total_tokens / total_sequences:.2f}")

    #     exit()
    
    model_config = AutoConfig.from_pretrained(args.model_config)
    # ngpt related parameters
    # model_config.base_scale = 1.0 / math.sqrt(model_config.hidden_size)
    # model_config.use_ngpt = args.use_ngpt
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.weight_quant:
        # Enable INT8 training
        assert args.optimizer.lower() in [
            "q_galore_adamw8bit",
            "q_galore_adamw8bit_per_layer",
            "q_apollo",
            "q_apollo_per_layer",
            "q_poet",
            "q_poet_per_layer",
        ]
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        if 'galore' in args.optimizer.lower():
            model = prepare_model_for_int8_training_galore(model, args, target_module)
        elif 'apollo' in args.optimizer.lower():
            model = prepare_model_for_int8_training_apollo(model, args, target_module)
        elif 'poet' in args.optimizer.lower():
            model = prepare_model_for_int8_training_poet(model, args, target_module)
        print('--'*20)
        print('Prepare Model for Int8 Training')
        print('--'*20)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.init_type == "mhe_optimized" and args.continue_from is None:
        if args.dtype in ["bf16", "bfloat16"]:
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device)

        before_mhe_loss = calculate_total_mhe(model)
        print(f"MHE loss before optimization: {before_mhe_loss:.8f}") 

        # Worker Mode Check
        # if args.mhe_worker_id is not None:
        #     if args.mhe_total_workers is None:
        #         raise ValueError("Must provide --mhe_total_workers when using --mhe_worker_id")
            
        #     logger.info(f"Running in MHE Worker Mode: ID {args.mhe_worker_id}/{args.mhe_total_workers}")
        #     mhe_save_dir = os.path.join(args.mhe_save_dir, model_stem)
        #     print(f"Saving MHE optimized layers to {mhe_save_dir}")
        #     mhe_worker_process(model, args.mhe_worker_id, args.mhe_total_workers, mhe_save_dir)
        #     logger.info("Worker finished processing. Exiting.")
        #     exit(0)

        if not args.single_gpu:
            raise ValueError("mhe_optimized initialization is only supported with single GPU processing")

        model = mhe_optimized_init_multi_gpu(model)
        # mhe_optimized_init(model)
        optimized_mhe_loss = calculate_total_mhe(model)
    
        # Log the comparison
        if global_rank == 0:
            print(f"MHE Loss comparison:")
            print(f"Initial MHE loss: {before_mhe_loss:.4f}")
            print(f"After MHE optimization: {optimized_mhe_loss:.4f}")
            print(f"Improvement: {before_mhe_loss - optimized_mhe_loss:.4f}")

            exit()

            init_dir =os.path.join("mhe_processed_model", model_stem)
            if not os.path.exists(init_dir):
                os.makedirs(init_dir)
            model.save_pretrained(init_dir, safe_serialization=False, max_shard_size='100GB')
            exit()

    if args.continue_from is not None:
        if args.init_type == "mhe_optimized":
            before_mhe_loss = calculate_total_mhe(model)
            logger.info(f"MHE loss before loading: {before_mhe_loss:.8f}") 

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

        if args.init_type == "mhe_optimized":
            after_mhe_loss = calculate_total_mhe(model)
            logger.info(f"MHE loss after loading: {after_mhe_loss:.8f}") 
            logger.info(f"Improvement: {before_mhe_loss - after_mhe_loss:.8f}")

        exit()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # INT8 training: move the scales and zeros to the same device as the weight
    for name, module in model.named_modules():
        if isinstance(module, QGaLoreLinear):
            weight_device = module.weight.device
            module.weight.scales = module.weight.scales.to(device=weight_device)
            module.weight.zeros = module.weight.zeros.to(device=weight_device)

    for _, module in model.named_modules():
        if isinstance(module, QScaleLinear):
            weight_device = module.weight.device
            module.weight.scales = module.weight.scales.to(device=weight_device)
            module.weight.zeros = module.weight.zeros.to(device=weight_device)

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
    
    if 'galore' in args.optimizer.lower() and 'q_' not in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"] #, "lm_head"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

        # then call galore_adamw
        param_groups = [
            {'params': regular_params},           
            {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]
    
    elif 'galore' in args.optimizer.lower() and 'q_' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, QGaLoreLinear) or isinstance(module, QLinear)):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type,
                        "quant": args.proj_quant,'quant_n_bit': args.proj_bits, 'quant_group_size': args.proj_group_size,
                        'cos_threshold': args.cos_threshold, 'gamma_proj': args.gamma_proj, 'queue_size': args.queue_size}]
    
    elif args.optimizer.lower() == "poet":
        replace_linear_with_poet(model, args.poet_block_size, args.init_type, args.mup_alpha, device=model.device, dtype=model.dtype)

        
        # for name, module in model.named_modules():
        #     if isinstance(module, POETLinear):
        #         print('len(module.perm_in)', len(module.perm_in), module.perm_in)
        #         print('len(module.perm_out)', len(module.perm_out), module.perm_out)
        # breakpoint()
        
        poet_params = []
        scale_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'oft' in name:
                poet_params.append(param)
            elif 'scale' in name:
                scale_params.append(param)

        id_poet_params = {id(param) for param in poet_params}
        id_scale_params = {id(param) for param in scale_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_poet_params or id(param) in id_scale_params:
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

    
    elif args.optimizer.lower() == "poet_sgd":
        replace_linear_with_poet(model, args.poet_block_size, args.init_type, args.mup_alpha, device=model.device, dtype=model.dtype)

        poet_params = []
        scale_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'oft' in name:
                poet_params.append(param)
            elif 'scale' in name:
                scale_params.append(param)

        id_poet_params = {id(param) for param in poet_params}
        id_scale_params = {id(param) for param in scale_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_poet_params or id(param) in id_scale_params:
                continue
            if param.ndim >= 2 and not name.endswith('bias'):
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr, use_poet=False),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr, use_poet=False),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap),
        ]


    elif args.optimizer.lower() == "poet_continuous":
        replace_linear_with_poet_continuous(model, args.poet_block_size, args.init_type, device=model.device, dtype=model.dtype)

        poet_params = []
        scale_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'oft' in name:
                poet_params.append(param)
            elif 'scale' in name:
                scale_params.append(param)

        id_poet_params = {id(param) for param in poet_params}
        id_scale_params = {id(param) for param in scale_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_poet_params or id(param) in id_scale_params:
                continue
            if param.ndim >= 2 and not name.endswith('bias'):
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap),
        ]

    elif args.optimizer.lower() == "poet_monarch":
        replace_linear_with_poet_monarch(model, args.poet_block_size, args.init_type, device=model.device, dtype=model.dtype)


        poet_params = []
        scale_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'oft' in name:
                poet_params.append(param)
            elif 'scale' in name:
                scale_params.append(param)

        id_poet_params = {id(param) for param in poet_params}
        id_scale_params = {id(param) for param in scale_params}
        decay_params, nodecay_params = [], []  # they are non-poet parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in id_poet_params or id(param) in id_scale_params:
                continue
            if param.ndim >= 2 and not name.endswith('bias'):
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap),
        ]
    
    elif args.optimizer.lower() == "poetv2":
        replace_linear_with_poet_v2(model, args.poet_block_size, args.init_type, device=model.device, dtype=model.dtype)

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

        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap),
        ]

    elif args.optimizer.lower() == "poetv3":
        replace_linear_with_poet_v3(model, args.poet_block_size, args.init_type, device=model.device, dtype=model.dtype)

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

        # poet params
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0, lr=args.lr),
            dict(params=decay_params, weight_decay=args.weight_decay, lr=args.lr),
            dict(params=poet_params, weight_decay=0.0, lr=args.poet_lr, use_poet=True, poet_reset_gap=args.poet_reset_gap),
        ]

    elif args.optimizer.lower() == "q_poet" or args.optimizer.lower() == "q_poet_4bit":
        # replace_linear_with_qpoet(model, args.poet_block_size, args.init_type, device=model.device, dtype=model.dtype)
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


    elif "apollo" in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        apollo_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, QScaleLinear) or isinstance(module, QLinear)):
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            logger.info(f"Adding {module_name} to APOLLO parameters")
            apollo_params.append(module.weight)

        id_apollo_params = [id(p) for p in apollo_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_apollo_params]
        # then call low rank optimizer

        param_groups = [
            {"params": regular_params},
            {
                "params": apollo_params,
                "rank": args.rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.apollo_scale,
                "proj_type": args.proj_type,
                "proj": args.proj,
                "scale_type": args.scale_type,
            },
        ]
    
    elif args.optimizer.lower() == "lora":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"],
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if 'embed_tokens' in name:
                param.requires_grad = True

        param_groups = [p for p in model.parameters() if p.requires_grad]

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
        # then call galore_adamw
        param_groups = [
            {'params': regular_params, 'use_muon': False, 'lr': args.lr, 'weight_decay': args.weight_decay},           
            {'params': muon_params, 'use_muon': True, 'lr': 0.02, 'weight_decay': args.weight_decay},
        ]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "poet":
        optimizer = POETAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            poet_block_size=args.poet_block_size,
        )
        # Print POET LR scaling ratio per POET param (rank 0 only)
        # if global_rank == 0:
        #     name_map = {id(p): n for n, p in model.named_parameters()}

        #     poet_scalings = []
        #     for pg in optimizer.param_groups:
        #         if not pg.get("use_poet", False):
        #             continue

        #         poet_lr = pg.get("poet_lr", pg["lr"])  # POET groups currently store this in "lr"
        #         for p in pg["params"]:
        #             adj_lr = optimizer.adjust_lr_for_poet(poet_lr, p)
        #             scaling = adj_lr / poet_lr if poet_lr != 0 else float("nan")
        #             poet_scalings.append((name_map.get(id(p), "<unnamed>"), scaling, tuple(p.shape)))

        #     print(f"[POET] num_poet_params={len(poet_scalings)}")
        #     for name, scaling, shape in poet_scalings:
        #         print(f"[POET] scaling={scaling:.6g} shape={shape} name={name}")

        # breakpoint()
    elif args.optimizer.lower() == "poetv2":
        optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "poetv3":
        optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "poet_monarch":
        optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "poet_continuous":
        optimizer = POETAdamWContinuous(param_groups, lr=args.lr, weight_decay=args.weight_decay)
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
        # optimizer = MuonWithAuxAdam(
        #     param_groups
        # )
    elif args.optimizer.lower() == "poet_sgd":
        poet_params = []
        adamw_params = []
        for group in param_groups:
            if group.get('use_poet'):
                poet_params.extend(group['params'])
            else:
                adamw_params.extend(group['params'])
        optimizer = POETSGD(
            lr=args.lr,
            wd=args.weight_decay,
            poet_params=poet_params,
            adamw_params=adamw_params,
            poet_block_size=args.poet_block_size,
        )
        # optimizer = MuonWithAuxAdam(
        #     param_groups
        # )
    elif args.optimizer.lower() == "lora":
        # optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = POETAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    elif args.optimizer.lower() == "apollo_adamw":
        optimizer = APOLLOAdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, scale_front=args.scale_front)
    elif args.optimizer.lower() == "q_apollo":
        optimizer = QAPOLLOAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            # betas=(args.beta1, args.beta2),
            betas=(0.9, 0.999),
            scale_front=args.scale_front,
        )
    elif args.optimizer.lower() == "q_apollo_per_layer":
        optimizer_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params:
                optimizer_dict[p] = QAPOLLOAdamW(
                    [
                        {
                            "params": [p],
                            "rank": args.rank,
                            "update_proj_gap": args.update_proj_gap,
                            "scale": args.apollo_scale,
                            "proj_type": args.proj_type,
                            "proj": args.proj,
                            "scale_type": args.scale_type,
                        }
                    ],
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
            else:
                if p.requires_grad:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params or p.requires_grad:
                scheduler_dict[p] = get_scheduler(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if (not hasattr(p, "float_grad")) and p.grad is None:
                return

            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if id(p) in id_lowrank_params:
                setattr(p, "backward_hook", optimizer_hook)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        layer_wise_flag = True
    # implement adafactor
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
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
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
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        # optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheduler(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True

    elif args.optimizer.lower() == "q_galore_adamw8bit":
        optimizer = QGaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
    
    elif args.optimizer.lower() == "q_poet":
        # optimizer = POETAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
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
        if 'poet' in args.optimizer.lower() or 'lora' in args.optimizer.lower() or 'apollo' in args.optimizer.lower() or 'galore' in args.optimizer.lower() or 'muon' in args.optimizer.lower() or 'adamw' in args.optimizer.lower():
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
    # BENCHMARK: Warmup + 100 iterations
    # ##############################
    # if args.benchmark:
    #     if global_rank == 0:
    #         print("="*60)
    #         print("Starting 100-iteration benchmark...")
    #         print(f"World size: {world_size}")
    #         print(f"Batch size per GPU: {args.batch_size}")
    #         print(f"Gradient accumulation: {args.gradient_accumulation}")
    #         print(f"Effective batch size: {args.total_batch_size}")
    #         print("="*60)
        
    #     # Create iterator from dataloader
    #     dataloader_iter = iter(dataloader)
        
    #     # Warmup iterations (to avoid startup overhead)
    #     warmup_iters = 100
    #     if global_rank == 0:
    #         print(f"Running {warmup_iters} warmup iterations...")
        
    #     for warmup_idx in range(warmup_iters):
    #         batch = next(dataloader_iter)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["input_ids"].clone()
    #         labels[labels == pad_idx] = -100
            
    #         is_accumulating = (warmup_idx + 1) % args.gradient_accumulation != 0
            
    #         # Handle gradient synchronization for DDP/FSDP
    #         if not args.single_gpu and is_accumulating:
    #             if 'poet' in args.optimizer.lower():
    #                 # DDP no_sync
    #                 with model.no_sync():
    #                     loss = model(**batch, labels=labels).loss
    #                     scaled_loss = loss / args.gradient_accumulation
    #                     scaled_loss.backward()
    #             else:
    #                 # FSDP no_sync
    #                 with model.no_sync():
    #                     loss = model(**batch, labels=labels).loss
    #                     scaled_loss = loss / args.gradient_accumulation
    #                     scaled_loss.backward()
    #         else:
    #             loss = model(**batch, labels=labels).loss
    #             scaled_loss = loss / args.gradient_accumulation
    #             scaled_loss.backward()
                
    #             # Apply gradient clipping if enabled
    #             # if args.grad_clipping != 0.0:
    #             #     if 'poet' in args.optimizer.lower():
    #             #         parameters = []
    #             #         for group in param_groups:
    #             #             parameters.extend(group['params'])
    #             #         torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
    #             #     else:
    #             #         if args.single_gpu:
    #             #             torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
    #             #         else:
    #             #             FSDP.clip_grad_norm_(model, args.grad_clipping)
                
    #             # Update step - this is critical!
    #             if not layer_wise_flag:
    #                 optimizer.step()
    #                 # scheduler.step()
    #                 optimizer.zero_grad()
        
    #     # Synchronize all processes and GPU operations
    #     torch.cuda.synchronize()
    #     if not args.single_gpu:
    #         dist.barrier()
        
    #     if global_rank == 0:
    #         print("Warmup complete. Starting benchmark...")
        
    #     # Start benchmark
    #     bench_start = time.perf_counter()
    #     bench_tokens = 0
    #     bench_update_steps = 0
    #     bench_optimizer_step_count = 0  # Track for POET merge checks
        
    #     # Run exactly 100 iterations
    #     for bench_iter in range(100):
    #         batch = next(dataloader_iter)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["input_ids"].clone()
    #         labels[labels == pad_idx] = -100
            
    #         # Count tokens
    #         bench_tokens += (batch["input_ids"] != pad_idx).sum().item()
            
    #         is_accumulating = (bench_iter + 1) % args.gradient_accumulation != 0
            
    #         # Handle gradient synchronization for DDP/FSDP
    #         if not args.single_gpu and is_accumulating:
    #             with model.no_sync():
    #                 loss = model(**batch, labels=labels).loss
    #                 scaled_loss = loss / args.gradient_accumulation
    #                 scaled_loss.backward()
    #         else:
    #             loss = model(**batch, labels=labels).loss
    #             scaled_loss = loss / args.gradient_accumulation
    #             scaled_loss.backward()
                
    #             # Apply gradient clipping if enabled
    #             # if args.grad_clipping != 0.0:
    #             #     if 'poet' in args.optimizer.lower():
    #             #         parameters = []
    #             #         for group in param_groups:
    #             #             parameters.extend(group['params'])
    #             #         torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
    #             #     else:
    #             #         if args.single_gpu:
    #             #             torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
    #             #         else:
    #             #             FSDP.clip_grad_norm_(model, args.grad_clipping)
                
    #             # Update step - match actual training loop
    #             if not layer_wise_flag:
    #                 optimizer.step()
    #                 # scheduler.step()
    #                 optimizer.zero_grad()
    #                 bench_optimizer_step_count += 1
                    
    #                 # POET-specific merge check
    #                 # if 'poet' in args.optimizer.lower():
    #                 #     check_and_merge(model, bench_optimizer_step_count, args.poet_reset_gap)
                
    #             bench_update_steps += 1
        
    #     # Synchronize all processes and GPU operations
    #     torch.cuda.synchronize()
    #     if not args.single_gpu:
    #         dist.barrier()
        
    #     bench_end = time.perf_counter()
        
    #     # Calculate and print results
    #     total_time = bench_end - bench_start
    #     time_per_iter = total_time / 100
    #     time_per_update = total_time / bench_update_steps if bench_update_steps > 0 else 0
        
    #     # Aggregate tokens across all GPUs
    #     bench_tokens_tensor = torch.tensor(bench_tokens, device=device)
    #     if not args.single_gpu:
    #         dist.all_reduce(bench_tokens_tensor, op=dist.ReduceOp.SUM)
    #     total_tokens = bench_tokens_tensor.item()
        
    #     tokens_per_sec = total_tokens / total_time
        
    #     if global_rank == 0:
    #         print(f"\n{'='*60}")
    #         print(f"Benchmark Results (100 iterations):")
    #         print(f"  Total time: {total_time:.4f} seconds")
    #         print(f"  Time per iteration: {time_per_iter*1000:.2f} ms")
    #         print(f"  Time per update step: {time_per_update*1000:.2f} ms")
    #         print(f"  Update steps: {bench_update_steps}")
    #         print(f"  Optimizer steps: {bench_optimizer_step_count}")
    #         print(f"  Total tokens processed: {total_tokens:,}")
    #         print(f"  Throughput: {tokens_per_sec:,.2f} tokens/sec (all GPUs)")
    #         print(f"  Per-GPU throughput: {tokens_per_sec/world_size:,.2f} tokens/sec")
    #         print(f"  World size: {world_size}")
    #         print(f"  Gradient accumulation: {args.gradient_accumulation}")
    #         print(f"  Gradient clipping: {args.grad_clipping}")
    #         print(f"{'='*60}\n")
        
    #     # Exit after benchmark
    #     import sys
    #     sys.exit(0)

    # ##############################
    # PROFILING: Warmup + Trace
    # ##############################
    # if args.profile:
    #     if global_rank == 0:
    #         print("="*60)
    #         print("Starting profiling with warmup and trace...")
    #         print(f"World size: {world_size}")
    #         print(f"Batch size per GPU: {args.batch_size}")
    #         print(f"Gradient accumulation: {args.gradient_accumulation}")
    #         print("="*60)
        
    #     # Create iterator from dataloader
    #     dataloader_iter = iter(dataloader)
        
    #     # Reset peak memory stats
    #     torch.cuda.reset_peak_memory_stats(device)
        
    #     # Warmup iterations (to stabilize before profiling)
    #     warmup_iters = 25
    #     if global_rank == 0:
    #         print(f"Running {warmup_iters} warmup iterations...")
        
    #     profile_optimizer_step_count = 0
    #     for warmup_idx in range(warmup_iters):
    #         batch = next(dataloader_iter)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["input_ids"].clone()
    #         labels[labels == pad_idx] = -100
            
    #         is_accumulating = (warmup_idx + 1) % args.gradient_accumulation != 0
            
    #         # Handle gradient synchronization for DDP/FSDP
    #         if not args.single_gpu and is_accumulating:
    #             with model.no_sync():
    #                 loss = model(**batch, labels=labels).loss
    #                 scaled_loss = loss / args.gradient_accumulation
    #                 scaled_loss.backward()
    #         else:
    #             loss = model(**batch, labels=labels).loss
    #             scaled_loss = loss / args.gradient_accumulation
    #             scaled_loss.backward()
                
    #             # Update step
    #             if not layer_wise_flag:
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 profile_optimizer_step_count += 1
        
    #     # Synchronize before profiling
    #     torch.cuda.synchronize()
    #     if not args.single_gpu:
    #         dist.barrier()
        
    #     if global_rank == 0:
    #         print("Warmup complete. Starting profiling trace...")
        
    #     # Setup profiler with schedule:
    #     # wait=1, warmup=2, active=2, repeat=1 -> total 5 iterations
    #     prof = torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA
    #         ],
    #         schedule=torch.profiler.schedule(
    #             wait=1,      # skip first iteration
    #             warmup=27,    # warmup 2 iterations
    #             active=2,    # actively profile 2 iterations
    #             repeat=1     # do this once
    #         ),
    #         record_shapes=True,
    #         with_stack=True,
    #         on_trace_ready=lambda p: None  # We'll manually export
    #     )
        
    #     prof.start()
        
    #     # Run profiling iterations
    #     profile_total_iters = 30 # wait(1) + warmup(2) + active(2)
    #     for profile_idx in range(profile_total_iters):
    #         batch = next(dataloader_iter)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["input_ids"].clone()
    #         labels[labels == pad_idx] = -100
            
    #         is_accumulating = (profile_idx + 1) % args.gradient_accumulation != 0
            
    #         # Handle gradient synchronization for DDP/FSDP
    #         if not args.single_gpu and is_accumulating:
    #             with model.no_sync():
    #                 loss = model(**batch, labels=labels).loss
    #                 scaled_loss = loss / args.gradient_accumulation
    #                 scaled_loss.backward()
    #         else:
    #             loss = model(**batch, labels=labels).loss
    #             scaled_loss = loss / args.gradient_accumulation
    #             scaled_loss.backward()
                
    #             # Update step
    #             if not layer_wise_flag:
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 profile_optimizer_step_count += 1
            
    #         # Step the profiler
    #         prof.step()
        
    #     # Stop profiling
    #     prof.stop()
        
    #     # Synchronize all processes
    #     torch.cuda.synchronize()
    #     if not args.single_gpu:
    #         dist.barrier()
        
    #     # Export trace
    #     profile_filename = f"trace_{args.optimizer}_{args.name}_rank{global_rank}.json"
    #     prof.export_chrome_trace(profile_filename)
        
    #     # Report results
    #     peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        
    #     if global_rank == 0:
    #         print(f"\n{'='*60}")
    #         print(f"Profiling Complete!")
    #         print(f"  Trace saved to: {profile_filename}")
    #         print(f"  Peak GPU memory: {peak_mem_mb:.2f} MB")
    #         print(f"  Optimizer steps during profiling: {profile_optimizer_step_count}")
    #         print(f"  Total iterations: {warmup_iters + profile_total_iters}")
    #         print(f"\nView trace in Chrome at: chrome://tracing")
    #         print(f"{'='*60}\n")
        
    #     # Exit after profiling
    #     import sys
    #     sys.exit(0)

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
            elif 'lora' in args.optimizer.lower():
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            elif 'galore' in args.optimizer.lower():
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            elif 'apollo' in args.optimizer.lower() and args.grad_clipping != 0.0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            elif 'muon' in args.optimizer.lower():
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            elif 'adamw' in args.optimizer.lower():
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


            # poet_monitor_prev_oft_R = None
            # adam_monitor_prev_W = None

            # poet_monitor_module = [
            #     # "model.layers.5.self_attn.q_proj",
            #     # "model.layers.17.self_attn.q_proj",
            #     # "model.layers.5.self_attn.k_proj",
            #     # "model.layers.17.self_attn.k_proj",
            #     # "model.layers.5.self_attn.v_proj",
            #     # "model.layers.17.self_attn.v_proj", # d_out: 2048 d_in: 2048 / 0.0001
            #     # "model.layers.5.self_attn.o_proj", # d_out: 2048 d_in: 2048 / 0.2 
            #     # "model.layers.17.self_attn.o_proj",
            #     # "model.layers.5.mlp.gate_proj",
            #     # "model.layers.17.mlp.gate_proj",
            #     # "model.layers.5.mlp.down_proj", # d_out: 2048 d_in: 5632 / 0.1
            #     # "model.layers.17.mlp.down_proj", # d_out: 2048 d_in: 5632 / e-05
            #     # "model.layers.5.mlp.up_proj",
            #     # "model.layers.17.mlp.up_proj",
            # ]

            # monitor_substr = poet_monitor_module[0]

            # if global_rank == 0:
            #     # Find the module once (works for POETLinear or nn.Linear)
            #     _, monitor_mod = _find_module_by_name_substr(model, monitor_substr)

            #     if monitor_mod is not None:
            #         opt_name = args.optimizer.lower()

            #         # --- branch A: POET (snapshot oft_R) ---
            #         if "poet" in opt_name and hasattr(monitor_mod, "oft_R"):
            #             poet_monitor_prev_oft_R = monitor_mod.oft_R.detach().clone()

            #         # --- branch B: AdamW / MUON (snapshot weight) ---
            #         elif opt_name in ["adamw", "muon"] and hasattr(monitor_mod, "weight"):
            #             W = monitor_mod.weight
            #             if W is not None and W.ndim == 2:
            #                 adam_monitor_prev_W = W.detach().clone()



            optimizer.step()



            # # --- branch A: POET (your existing logic) ---
            # if poet_monitor_prev_oft_R is not None and global_rank == 0:
            #     dW, sigma_dW = estimate_poet_delta_weff_spec(
            #         monitor_mod,
            #         poet_monitor_prev_oft_R,
            #         compute_dtype=torch.float32,
            #     )
            #     d_out, d_in = monitor_mod.weight.shape
            #     target = math.sqrt(d_out / d_in)
            #     constant = sigma_dW / target
            #     print(f"[poet] delta_W_eff_spec_constant: {constant} d_out: {d_out} d_in: {d_in}")
            #     dW_rms = (dW / args.poet_lr).square().mean().sqrt().item()
            #     print(f"[poet] delta_W_eff_spec_rms: {dW_rms}")

            # # --- branch B: AdamW / MUON (materialize ΔW directly) ---
            # if adam_monitor_prev_W is not None and global_rank == 0:
            #     W_cur = monitor_mod.weight.detach()
            #     dW = (W_cur - adam_monitor_prev_W).to(dtype=torch.float32)

            #     # spectral norm of ΔW
            #     sigma_dW = torch.linalg.matrix_norm(dW, ord=2)

            #     d_out, d_in = W_cur.shape
            #     target = math.sqrt(d_out / d_in)
            #     constant = sigma_dW.item() / target
            #     print(f"[{args.optimizer.lower()}] delta_W_spec_constant: {constant} d_out: {d_out} d_in: {d_in}")
            #     dW_rms = (dW / args.lr).square().mean().sqrt().item()
            #     print(f"[{args.optimizer.lower()}] delta_W_spec_rms: {dW_rms}")



            scheduler.step()


            # lrs = scheduler.get_last_lr()
            # for i, lr in enumerate(lrs):
            #     print(f"lr_next/group{i}: {lr:.6g}")


            optimizer.zero_grad()
            optimizer_step_count += 1
            if args.optimizer.lower() == "poet" or args.optimizer.lower() == "q_poet":
                check_and_merge(model, optimizer_step_count, args.poet_reset_gap)
            elif args.optimizer.lower() == "poet_monarch":
                check_and_merge_monarch(model, optimizer_step_count, args.poet_reset_gap)
            elif args.optimizer.lower() == "poet_continuous":
                check_and_merge_continuous(model, optimizer_step_count, args.poet_reset_gap)


        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            # Save model - handle FSDP vs DDP/single_gpu differently
            if 'poet' in args.optimizer.lower() or 'galore' in args.optimizer.lower() or 'apollo' in args.optimizer.lower() or 'muon' in args.optimizer.lower():
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

        if 'poet' in args.optimizer.lower() or 'galore' in args.optimizer.lower() or 'apollo' in args.optimizer.lower() or 'muon' in args.optimizer.lower():
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
