import os
from datetime import datetime

from loguru import logger


def check_args_torchrun_main(args):
    if args.run_name is None:
        if 'adamw' in args.optimizer.lower() and 'apollo' not in args.optimizer.lower() and 'galore' not in args.optimizer.lower():
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-min_lr_ratio-{args.min_lr_ratio}-wd-{args.weight_decay}-warmup-{args.warmup_steps}-init_type-{args.init_type}-max_length-{args.max_length}-bs-{args.batch_size}"
            )
        elif 'poet' in args.optimizer.lower():
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-poet_lr-{args.poet_lr}-poet_scale_mode-{args.poet_scale_mode}-min_lr_ratio-{args.min_lr_ratio}-wd-{args.weight_decay}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
                f"-init_type-{args.init_type}-block_size-{args.poet_block_size}"
            )
            if args.poet_mem_efficient_mode:
                args.run_name += "-mem_efficient_mode"
        elif args.optimizer.lower() == "lora":
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-lora_r-{args.lora_r}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
                f"-init_type-{args.init_type}"
            )
        elif "apollo" in args.optimizer.lower():
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-apollo_scale-{args.apollo_scale}-rank-{args.rank}-scale_type-{args.scale_type}-proj-{args.proj}-update_proj_gap-{args.update_proj_gap}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
                f"-init_type-{args.init_type}"
            )
        # elif args.optimizer.lower() == "q_apollo":
        #     args.run_name = (
        #         f"q_apollo-lr-{args.lr}-apollo_scale-{args.apollo_scale}-rank-{args.rank}-scale_type-{args.scale_type}-proj-{args.proj}-update_proj_gap-{args.update_proj_gap}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
        #         f"-init_type-{args.init_type}"
        #     )
        elif "galore" in args.optimizer.lower():
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-galore_scale-{args.galore_scale}-rank-{args.rank}-update_proj_gap-{args.update_proj_gap}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
                f"-init_type-{args.init_type}"
            )
        # elif args.optimizer.lower() == "q_galore_adamw8bit":
        #     args.run_name = (
        #         f"q_galore_adamw8bit-lr-{args.lr}-warmup-{args.warmup_steps}-max_length-{args.max_length}-bs-{args.batch_size}"
        #         f"-init_type-{args.init_type}"
        #     )
        elif args.optimizer.lower() == "muon":
            args.run_name = (
                f"{args.optimizer}-lr-{args.lr}-min_lr_ratio-{args.min_lr_ratio}-wd-{args.weight_decay}-warmup-{args.warmup_steps}-init_type-{args.init_type}-max_length-{args.max_length}-bs-{args.batch_size}"
            )
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")

        # if args.poet_balance_lr:
        #     args.run_name += "-balance_lr"

        if args.init_type == 'mup_normalized':
            args.run_name += f"-mup_alpha-{args.mup_alpha}"


        if args.poet_use_rmsnorm:
            args.run_name += "-use_rmsnorm"

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{args.run_name}-{datetime.now().strftime('%Y-%m-%d-%H')}h"

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    return args
