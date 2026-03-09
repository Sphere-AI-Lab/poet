"""Model utilities for POET integration.

This module provides functions for modifying models to use POET layers,
merging transformations, and analyzing parameter counts.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from poet_torch.layers import POETLinear, QPOETLinear

logger = logging.getLogger(__name__)


def replace_linear_with_poet(
    module: nn.Module,
    block_size: int,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    mem_efficient_mode: bool = False,
    normalize_weights: bool = True,
    bias_requires_grad: bool = False,
) -> Tuple[nn.Module, int]:
    """Replace nn.Linear layers with POETLinear layers.
    
    Recursively replaces Linear layers in the module with POETLinear layers.
    By default, the lm_head layer is excluded from replacement.
    
    Args:
        module: The module to modify.
        block_size: Block size for POET transformations.
        target_modules: List of module names to target. If None, targets all
            Linear layers except those in exclude_modules.
        exclude_modules: List of module names to exclude. Default: ["lm_head"].
        device: Device for new parameters. Default: None (use same as original).
        dtype: Data type for new parameters. Default: None (use same as original).
        mem_efficient_mode: Whether to use memory-efficient mode. Default: False.
        normalize_weights: Whether to normalize weights. Default: True.
        
    Returns:
        Tuple of (modified_module, num_replaced_layers).
        
    Example:
        >>> model = nn.TransformerDecoder(...)
        >>> model, num_replaced = replace_linear_with_poet(
        ...     model, block_size=256, exclude_modules=["lm_head"]
        ... )
        >>> print(f"Replaced {num_replaced} layers")
    """
    if exclude_modules is None:
        exclude_modules = ["lm_head"]
    
    num_replaced = 0
    
    def _should_replace(name: str) -> bool:
        """Check if a module should be replaced."""
        # Check exclude list
        for exclude in exclude_modules:
            if exclude in name.lower():
                return False
        
        # Check target list
        if target_modules is not None:
            return any(target in name for target in target_modules)
        
        return True
    
    def _convert(parent_module: nn.Module, parent_name: str = "") -> None:
        nonlocal num_replaced
        
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear) and _should_replace(full_name):
                in_feat = child.in_features
                out_feat = child.out_features
                
                # Check divisibility
                if in_feat % block_size != 0 or out_feat % block_size != 0:
                    logger.warning(
                        f"Skipping {full_name}: dimensions ({out_feat}, {in_feat}) "
                        f"not divisible by block_size {block_size}"
                    )
                    continue
                
                # Create POET layer
                poet_layer = POETLinear(
                    in_features=in_feat,
                    out_features=out_feat,
                    block_size=block_size,
                    bias=(child.bias is not None),
                    bias_requires_grad=bias_requires_grad,
                    device=device or child.weight.device,
                    dtype=dtype or child.weight.dtype,
                    mem_efficient_mode=mem_efficient_mode,
                )
                
                # Copy and normalize weights
                with torch.no_grad():
                    weight = child.weight.detach().clone()
                    if normalize_weights:
                        weight = weight / torch.norm(weight, dim=1, keepdim=True)
                    poet_layer.weight.copy_(weight.to(poet_layer.weight.dtype))
                    
                    if child.bias is not None and poet_layer.bias is not None:
                        poet_layer.bias.copy_(child.bias.detach().to(poet_layer.bias.dtype))
                
                setattr(parent_module, name, poet_layer)
                num_replaced += 1
                logger.debug(f"Replaced {full_name} with POETLinear")
            else:
                _convert(child, full_name)
    
    _convert(module)
    logger.info(f"Replaced {num_replaced} Linear layers with POETLinear")
    torch.cuda.empty_cache()
    
    return module, num_replaced


def convert_to_qpoet(
    module: nn.Module,
    block_size: int = 256,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    group_size: int = 256,
    num_bits: int = 8,
    normalize_weights: bool = True,
) -> Tuple[nn.Module, int]:
    """Convert nn.Linear layers to quantized QPOETLinear layers.
    
    Recursively replaces Linear layers in the module with QPOETLinear layers.
    By default, the lm_head layer is excluded from replacement.
    
    Args:
        module: The module to modify.
        block_size: Block size for POET transformations.
        target_modules: List of module names to target. If None, targets all
            Linear layers except those in exclude_modules.
        exclude_modules: List of module names to exclude. Default: ["lm_head"].
        group_size: Group size for quantization. Default: 256.
        num_bits: Number of bits for quantization. Default: 8.
        normalize_weights: Whether to normalize_weights weights before quantization. Default: True.
        
    Returns:
        Tuple of (modified_module, num_replaced_layers).
        
    Example:
        >>> model = nn.TransformerDecoder(...)
        >>> model, num_replaced = convert_to_qpoet(
        ...     model, block_size=256, exclude_modules=["lm_head"]
        ... )
        >>> print(f"Converted {num_replaced} layers")
    """
    if exclude_modules is None:
        exclude_modules = ["lm_head"]
    
    num_replaced = 0
    
    def _should_replace(name: str) -> bool:
        """Check if a module should be replaced."""
        # Check exclude list
        for exclude in exclude_modules:
            if exclude in name.lower():
                return False
        
        # Check target list
        if target_modules is not None:
            return any(target in name for target in target_modules)
        
        return True
    
    def _convert(parent_module: nn.Module, parent_name: str = "") -> None:
        nonlocal num_replaced
        
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear) and _should_replace(full_name):
                in_feat = child.in_features
                out_feat = child.out_features
                
                # Check divisibility
                if in_feat % block_size != 0 or out_feat % block_size != 0:
                    logger.warning(
                        f"Skipping {full_name}: dimensions ({out_feat}, {in_feat}) "
                        f"not divisible by block_size {block_size}"
                    )
                    continue
                
                # Create QPOET layer from Linear
                poet_layer = QPOETLinear.from_linear(
                    child,
                    block_size=block_size,
                    group_size=group_size,
                    num_bits=num_bits,
                    normalize_weights=normalize_weights,
                )
                
                setattr(parent_module, name, poet_layer)
                num_replaced += 1
                logger.debug(f"Converted {full_name} to QPOETLinear")
            else:
                _convert(child, full_name)
    
    _convert(module)
    logger.info(f"Converted {num_replaced} Linear layers to QPOETLinear")
    torch.cuda.empty_cache()

    return module, num_replaced


def merge_and_reinitialize(
    model: nn.Module,
    step: int,
    merge_interval: int,
) -> bool:
    """Merge POET transformations if at merge interval.
    
    This function checks if the current step warrants a merge of POET
    transformations and executes it if necessary. In distributed training,
    it synchronizes across all ranks.
    
    Args:
        model: Model containing POET layers.
        step: Current training step.
        merge_interval: Steps between merge and reinitialization.
        
    Returns:
        True if merge was performed, False otherwise.
        
    Example:
        >>> for step, batch in enumerate(dataloader):
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     
        ...     # Merge every 20 steps
        ...     merged = merge_and_reinitialize(model, step, 20)
        ...     if merged:
        ...         print(f"Merged at step {step}")
    """
    if step <= 0 or (step % merge_interval != 0):
        return False
    
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    
    with torch.compiler.set_stance("eager_then_compile"):
        with torch.no_grad():
            # Only rank 0 performs the merge
            if rank == 0:
                for module in model.modules():
                    if isinstance(module, (POETLinear, QPOETLinear)) and module.block_size > 0:
                        module.merge_then_reinitialize()
            
            # Synchronize across ranks
            if is_distributed:
                for module in model.modules():
                    if isinstance(module, (POETLinear, QPOETLinear)) and module.block_size > 0:
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
                
                torch.distributed.barrier()
    
    return True


def calc_poet_grad_clipping_value(
    global_step: int,
    grad_clipping: float,
    warmup_steps: int,
    poet_merge_interval: int,
    min_ratio: float = 0.1,
    max_steps: int = 2000,
) -> float:
    """Calculate gradient clipping value with warmup.
    
    The clipping value linearly increases from min_ratio * grad_clipping
    to grad_clipping over warmup_steps, repeating every poet_merge_interval steps.
    
    Args:
        global_step: Current training step.
        grad_clipping: Maximum gradient clipping value.
        warmup_steps: Number of steps for linear warmup.
        poet_merge_interval: Period for repeating warmup cycle.
        min_ratio: Starting ratio of grad_clipping.
        max_steps: Maximum steps to apply gradient clipping.
        
    Returns:
        Current gradient clipping value.
    """
    if global_step < poet_merge_interval:
        return grad_clipping

    if global_step > max_steps:
        return grad_clipping

    cycle_position = global_step % poet_merge_interval
    
    if cycle_position >= warmup_steps:
        return grad_clipping

    warmup_factor = min_ratio + (1.0 - min_ratio) * (cycle_position / max(1, warmup_steps))
    return warmup_factor * grad_clipping


def get_poet_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """Get all POET parameters from a model.
    
    Args:
        model: The model to search.
        
    Returns:
        List of POET parameters (oft_R parameters).
    """
    poet_params = []
    for name, param in model.named_parameters():
        if "oft" in name and param.requires_grad:
            poet_params.append(param)
    return poet_params


def get_model_info(model: nn.Module) -> dict:
    """Get information about POET parameters in a model.
    
    Args:
        model: The model to analyze.
        
    Returns:
        Dictionary containing:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - poet_params: Number of POET-specific parameters
            - base_params: Number of base (non-POET) parameters
            - poet_layers: Number of POET layers
    """
    total_params = 0
    trainable_params = 0
    poet_params = 0
    poet_layers = 0
    
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        
        if param.requires_grad:
            trainable_params += num
        
        if "oft" in name:
            poet_params += num
    
    for module in model.modules():
        if isinstance(module, (POETLinear, QPOETLinear)):
            poet_layers += 1
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "poet_params": poet_params,
        "base_params": trainable_params - poet_params,
        "poet_layers": poet_layers,
    }


def print_model_info(model: nn.Module) -> None:
    """Print formatted model information.
    
    Args:
        model: The model to analyze.
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("POET Model Information")
    print("=" * 60)
    print(f"Total parameters:        {info['total_params']:,}")
    print(f"Trainable parameters:    {info['trainable_params']:,}")
    print(f"  - Base parameters:     {info['base_params']:,}")
    print(f"  - POET parameters:     {info['poet_params']:,}")
    print(f"POET layers:             {info['poet_layers']}")
    print("=" * 60)
