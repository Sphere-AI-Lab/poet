"""POET Model Wrapper.

This module provides a high-level wrapper for easily applying POET to any model.
"""

import logging
from typing import Union

import torch
import torch.nn as nn

from poet_torch.optim.adamw import POETAdamW
from poet_torch.config import POETConfig, QPOETConfig
from poet_torch.layers import POETLinear, QPOETLinear
from poet_torch.utils.model_utils import (
    replace_linear_with_poet,
    convert_to_qpoet,
    merge_and_reinitialize,
    get_model_info,
    get_poet_params,
    print_model_info,
)

logger = logging.getLogger(__name__)


class POETModel(nn.Module):
    """Wrapper to easily apply POET to any PyTorch model.
    
    This wrapper provides a simple interface for:
    1. Converting a standard model to use POET layers
    2. Managing merge-then-reinitialize cycles
    3. Providing convenient utilities for training
    
    Args:
        model: The base model to wrap.
        config: POET configuration (POETConfig or QPOETConfig).
        
    Example:
        >>> from transformers import AutoModel
        >>> from poet_torch import POETConfig, POETModel
        >>> 
        >>> # Load a standard model
        >>> base_model = AutoModel.from_pretrained("...")
        >>> 
        >>> # Configure POET
        >>> config = POETConfig(
        ...     block_size=256,
        ...     merge_interval=200,
        ...     mem_efficient_mode=False,
        ... )
        >>> 
        >>> # Wrap with POET
        >>> model = POETModel(base_model, config)
        >>> 
        >>> optimizer = get_poet_optimizer(model, config)
        >>> 
        >>> # Training loop
        >>> for step, batch in enumerate(dataloader):
        ...     loss = model(**batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     model.merge_if_needed(step)  # Automatic merge
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[POETConfig, QPOETConfig],
    ):
        super().__init__()
        
        self.config = config
        self.base_model = model
        self.is_quantized = isinstance(config, QPOETConfig)
        
        # Convert model based on config type
        if self.is_quantized:
            self._setup_qpoet()
        else:
            self._setup_poet()
        
        # Log model info
        self.print_model_info()

    def _setup_poet(self) -> None:
        """Setup standard POET model."""
        replace_linear_with_poet(
            self.base_model,
            block_size=self.config.block_size,
            target_modules=self.config.target_modules,
            exclude_modules=self.config.exclude_modules,
            mem_efficient_mode=self.config.mem_efficient_mode,
        )

    def _setup_qpoet(self) -> None:
        """Setup quantized POET model."""
        convert_to_qpoet(
            self.base_model,
            block_size=self.config.block_size,
            target_modules=self.config.target_modules,
            exclude_modules=self.config.exclude_modules,
            group_size=self.config.weight_group_size,
            num_bits=self.config.weight_bits,
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.base_model(*args, **kwargs)

    def merge_if_needed(self, step: int) -> bool:
        """Perform merge-then-reinitialize if at merge interval.
        
        Args:
            step: Current training step.
            
        Returns:
            True if merge was performed, False otherwise.
        """
        return merge_and_reinitialize(
            self.base_model,
            step=step,
            merge_interval=self.config.merge_interval,
        )

    def merge(self) -> None:
        """Force merge-then-reinitialize immediately."""
        merge_and_reinitialize(
            self.base_model,
            step=self.config.merge_interval,  # Force merge
            merge_interval=self.config.merge_interval,
        )

    def get_model_info(self) -> dict:
        """Get information about the wrapped model."""
        return get_model_info(self.base_model)

    def print_model_info(self) -> None:
        """Print formatted model information."""
        print_model_info(self.base_model)

    def get_effective_weights(self) -> dict:
        """Get effective weights for all POET layers.
        
        Returns:
            Dictionary mapping layer names to effective weight tensors.
        """
        weights = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, (POETLinear, QPOETLinear)):
                weights[name] = module.get_effective_weight()
        return weights

    def __getattr__(self, name: str):
        """Forward attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def get_poet_optimizer(model: nn.Module, config: Union[POETConfig, QPOETConfig]) -> POETAdamW:
    """Get a POETAdamW optimizer for the given model and configuration."""
    poet_params = get_poet_params(model)

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
        dict(params=nodecay_params, use_poet=False, weight_decay=0.0, lr=config.base_lr),
        dict(params=decay_params, use_poet=False, weight_decay=config.weight_decay, lr=config.base_lr),
        dict(params=poet_params, use_poet=True, weight_decay=0.0, lr=config.poet_lr),
    ]

    optimizer = POETAdamW(
        param_groups,
        lr=config.base_lr,
        weight_decay=config.weight_decay,
        poet_scale=config.poet_scale,
        poet_merge_interval=config.merge_interval, 
    )

    return optimizer