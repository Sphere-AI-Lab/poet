"""Model utilities for POET layer integration.

This module provides functions for replacing standard linear layers with
POET layers and managing the merge-then-reinitialize process.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .poet_layer import POETLinear, QPOETLinear


def replace_linear_with_poet(
    module: nn.Module,
    block_size: int,
    init_type: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    mem_efficient_mode: bool = False,
) -> None:
    """Replace nn.Linear layers with POETLinear layers.
    
    Recursively replaces Linear layers in the module with POETLinear layers.
    The lm_head layer is excluded from replacement.
    
    Args:
        module: Module to modify.
        block_size: Block size for POET transformations.
        init_type: Weight initialization type ('normalized').
        device: Device for new parameters.
        dtype: Data type for new parameters.
        mem_efficient_mode: Whether to use memory-efficient mode.
        
    Raises:
        ValueError: If a layer's dimensions are not divisible by block_size.
    """

    def _convert(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
                in_feat = child.in_features
                out_feat = child.out_features
                
                if block_size and in_feat % block_size == 0 and out_feat % block_size == 0:
                    new_lin = POETLinear(
                        in_features=in_feat,
                        out_features=out_feat,
                        bsz=block_size,
                        bias=(child.bias is not None),
                        device=device,
                        dtype=dtype,
                        mem_efficient_mode=mem_efficient_mode,
                    )
                    with torch.no_grad():
                        if init_type == 'normalized':
                            # [Check 1] Measure Spectral Norm BEFORE normalization
                            # child.weight is the original random initialization
                            # spec_before = torch.linalg.norm(child.weight.data.float(), ord=2).item() / torch.sqrt(torch.tensor(child.weight.data.shape[0]) / torch.tensor(child.weight.data.shape[1]))
                            
                            child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)

                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))

                        # [Check 2] Measure Spectral Norm AFTER normalization & copy
                        # This should be much smaller (close to 2.0 for large square matrices)
                        # spec_after = torch.linalg.norm(new_lin.weight.float(), ord=2).item() / torch.sqrt(torch.tensor(new_lin.weight.data.shape[0]) / torch.tensor(new_lin.weight.data.shape[1]))
                        # print(f"Weight Spectral Norm (Before): {spec_before:.4f}, (After): {spec_after:.4f}")

                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))

                    setattr(m, name, new_lin)
                else:
                    raise ValueError(
                        f"Layer {name} has in_features {in_feat} and "
                        f"out_features {out_feat}, which are not divisible by {block_size}"
                    )
            else:
                _convert(child)

    _convert(module)


def prepare_model_for_int8_training_poet(
    model: nn.Module, args: object, target_module: list
) -> nn.Module:
    """Prepare model for int8 training with POET layers.
    
    Recursively replaces Linear layers with QPOETLinear layers for quantization.
    
    Args:
        model: Model to modify.
        args: Arguments containing quantization settings.
        target_module: List of module names to target for replacement.
        
    Returns:
        Modified model.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_poet(module, args, target_module)

        if isinstance(module, nn.Linear):
            if name not in target_module:
                continue

            bias_data = module.bias.data if module.bias is not None else None
            weight = module.weight.data
            
            if args.init_type == "normalized":
                weight = weight / torch.norm(weight, dim=1, keepdim=True)

            new_layers = QPOETLinear(
                weight,
                bias_data,
                bsz=args.poet_block_size,
                num_bits=args.weight_bits,
                group_size=args.weight_group_size,
                stochastic_round=args.stochastic_round,
            )
            model._modules[name] = new_layers

    return model


def check_and_merge(
    model: nn.Module, iter_count: int = 0, poet_reset_gap: int = 4
) -> None:
    """Check if merge should be performed and execute it.
    
    This function checks if the current iteration count warrants a merge
    of POET transformations. If so, it performs the merge and synchronizes
    across all distributed ranks.
    
    Args:
        model: Model containing POET layers.
        iter_count: Current iteration count.
        poet_reset_gap: Gap between merge operations.
    """
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    with torch.compiler.set_stance("eager_then_compile"):
        for name, module in model.named_modules():
            if isinstance(module, (POETLinear, QPOETLinear)) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        module.merge_then_reinitialize()

                    # Synchronize across ranks
                    torch.distributed.broadcast(module.oft_R.data, src=0)
                    torch.distributed.broadcast(module.weight.data, src=0)
                    
                    if isinstance(module, QPOETLinear):
                        torch.distributed.broadcast(module.weight_scales, src=0)
                        torch.distributed.broadcast(module.weight_zeros, src=0)
                    
                    if module.bias is not None:
                        torch.distributed.broadcast(module.bias.data, src=0)
                    
                    torch.distributed.broadcast(module.perm_in, src=0)
                    torch.distributed.broadcast(module.perm_in_inv, src=0)
                    torch.distributed.broadcast(module.perm_out, src=0)
                    torch.distributed.broadcast(module.perm_out_inv, src=0)

        if is_dist:
            dist.barrier()


def get_grad_clipping_value(
    global_step: int,
    grad_clipping: float,
    warmup_steps: int,
    period_T: int,
    min_ratio: float = 0.1,
    max_steps: int = 2000,
) -> float:
    """Calculate gradient clipping value with warmup.
    
    The clipping value linearly increases from min_ratio * grad_clipping
    to grad_clipping over warmup_steps, repeating every period_T steps.
    
    Args:
        global_step: Current training step.
        grad_clipping: Maximum gradient clipping value.
        warmup_steps: Number of steps for linear warmup.
        period_T: Period for repeating warmup cycle.
        min_ratio: Starting ratio of grad_clipping.
        max_steps: Maximum steps to apply gradient clipping.
        
    Returns:
        Current gradient clipping value.
    """
    if global_step < period_T:
        return grad_clipping

    if global_step > max_steps:
        return grad_clipping

    cycle_position = global_step % period_T
    
    if cycle_position >= warmup_steps:
        return grad_clipping

    warmup_factor = min_ratio + (1.0 - min_ratio) * (cycle_position / max(1, warmup_steps))
    return warmup_factor * grad_clipping

