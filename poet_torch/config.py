"""POET Configuration.

This module provides configuration classes for POET and its variants.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class POETConfig:
    """Configuration for POET (Parameter-Efficient Orthogonal Transformation).
    
    POET reparameterizes weight matrices as W_RP = R * W_0 * P, where W_0 is
    a fixed randomly initialized matrix, and R, P are learnable orthogonal
    matrices. This provides direct control over the weight spectrum throughout
    training.
    
    Args:
        block_size: Block size for POET transformations. Dimensions must be
            divisible by this value. Default: 256.
        mem_efficient_mode: Whether to use memory-efficient mode (POET-X_mem).
            Trades computation for memory by recomputing activations. Default: False.
        merge_interval: Number of steps between merge-then-reinitialize operations.
            Default: 200.
        poet_lr: Learning rate for POET parameters (orthogonal transformations).
            Default: 5e-4.
        base_lr: Learning rate for base parameters (non-POET). Default: 1e-3.
        poet_scale: Scaling factor for POET learning rate adjustment. Default: 0.5.
        weight_decay: Weight decay for base parameters. POET parameters use no
            weight decay. Default: 0.0.
        target_modules: List of module names to replace with POET layers. If None,
            replaces all Linear layers except lm_head. Default: None.
        exclude_modules: List of module names to exclude from replacement.
            Default: ["lm_head"].
    
    Example:
        >>> from poet_torch import POETConfig
        >>> 
        >>> # Standard POET configuration
        >>> config = POETConfig(
        ...     block_size=256,
        ...     merge_interval=20,
        ... )
        >>> 
        >>> # Memory-efficient configuration
        >>> mem_config = POETConfig(
        ...     block_size=512,
        ...     mem_efficient_mode=True,
        ...     merge_interval=50,
        ... )
    """
    
    # Core POET settings
    block_size: int = 256
    
    # Memory and computation
    mem_efficient_mode: bool = False
    
    # Training schedule
    merge_interval: int = 200
    
    # Learning rates and optimization
    poet_lr: float = 5e-4
    base_lr: float = 1e-3
    poet_scale: float = 0.5
    weight_decay: float = 0.0
    
    # Model modification
    target_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        
        if self.merge_interval <= 0:
            raise ValueError(f"merge_interval must be positive, got {self.merge_interval}")
        
        if self.exclude_modules is None:
            self.exclude_modules = ["lm_head"]


@dataclass
class QPOETConfig(POETConfig):
    """Configuration for Quantized POET (QPOET / POET-XQ).
    
    QPOET extends POET with INT8 quantization of base weights for extreme
    memory efficiency. The trainable orthogonal transformations remain in
    full precision.
    
    Args:
        weight_bits: Number of bits for weight quantization. Default: 8.
        weight_group_size: Group size for quantization. Default: 256.
        
    Inherits all arguments from POETConfig.
    
    Example:
        >>> from poet_torch import QPOETConfig
        >>> 
        >>> # 8-bit quantized POET
        >>> config = QPOETConfig(
        ...     block_size=256,
        ...     weight_bits=8,
        ...     weight_group_size=256,
        ... )
    """
    
    # Quantization settings
    weight_bits: int = 8
    weight_group_size: int = 256
    
    def __post_init__(self):
        """Validate quantization configuration."""
        super().__post_init__()
        
        if self.weight_bits != 8:
            raise NotImplementedError("Only 8-bit quantization is currently supported.")
        
        if self.weight_group_size <= 0:
            raise ValueError(f"weight_group_size must be positive, got {self.weight_group_size}")
