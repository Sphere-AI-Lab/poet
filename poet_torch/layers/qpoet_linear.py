"""Quantized POET Linear Layer implementation."""

from typing import Optional

import torch
import torch.nn as nn

from poet_torch.core.ops import (
    block_diag_lr_matmul,
    forward_core_q8,
    get_weight_poet,
    quantize_tensor_int8,
)


class QPOETLinear(nn.Module):
    """Quantized POET linear layer with INT8 weights.
    
    QPOET (also known as POET-XQ) stores the base weight matrix in INT8 format
    for extreme memory efficiency, while keeping the learnable orthogonal
    transformations in full precision.
    
    This is particularly useful for training very large models where memory
    is the primary bottleneck.
    
    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        block_size: Block size for transformations.
        bias: Whether to include bias. Default: False.
        device: Device for parameters. Default: None.
        dtype: Data type for trainable parameters. Default: None.
        num_bits: Number of bits for quantization (currently only 8 supported).
            Default: 8.
        group_size: Group size for quantization. Default: 256.
        mem_efficient_mode: Whether to use memory-efficient mode. Default: False.
        weight: Optional initial weight tensor. If provided, it will be quantized.
        
    Example:
        >>> # Create from scratch
        >>> layer = QPOETLinear(512, 512, block_size=64)
        >>> 
        >>> # Or convert from an existing Linear layer
        >>> linear = nn.Linear(512, 512)
        >>> layer = QPOETLinear.from_linear(linear, block_size=64)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 256,
        bias: bool = False,
        bias_requires_grad: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_bits: int = 8,
        group_size: int = 256,
        mem_efficient_mode: bool = False,
        weight: Optional[torch.Tensor] = None,
        bias_data: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        if num_bits != 8:
            raise NotImplementedError("Only 8-bit weight quantization is currently supported.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.mem_efficient_mode = mem_efficient_mode
        self.weight_group_size = group_size
        self.weight_num_bits = num_bits

        # Handle weight initialization
        if weight is not None:
            # Quantize provided weight
            int8_weight, scales, zeros = quantize_tensor_int8(
                weight.data if isinstance(weight, nn.Parameter) else weight,
                q_group_size=group_size
            )
            self.weight = nn.Parameter(int8_weight, requires_grad=False).to(device=device)
            self.register_buffer("weight_scales", scales.to(device))
            self.register_buffer("weight_zeros", zeros.to(device))
        else:
            # Initialize empty quantized weight
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), device=device, dtype=torch.uint8),
                requires_grad=False
            )
            num_groups = (in_features * out_features) // group_size
            self.register_buffer("weight_scales", torch.ones(num_groups, device=device, dtype=dtype))
            self.register_buffer("weight_zeros", torch.zeros(num_groups, device=device, dtype=dtype))

        # Bias term
        if bias_data is not None:
            self.bias = nn.Parameter(
                bias_data.to(device), device=device, requires_grad=bias_requires_grad
            )
        elif bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, device=device, dtype=dtype, requires_grad=bias_requires_grad
            ))
        else:
            self.register_parameter("bias", None)

        # Calculate number of blocks
        self.num_in_blocks = in_features // block_size
        self.num_out_blocks = out_features // block_size
        n_elements = block_size * (block_size - 1) // 2
        
        # Trainable skew-symmetric parameters
        self.oft_R = nn.Parameter(
            torch.zeros((self.num_in_blocks + self.num_out_blocks, n_elements),
                       device=device, dtype=dtype)
        )

        # Buffers for skew-symmetric construction
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # Permutation buffers
        perm_in = torch.randperm(in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 256,
        group_size: int = 256,
        num_bits: int = 8,
        mem_efficient_mode: bool = False,
        normalize_weights: bool = True,
    ) -> "QPOETLinear":
        """Create a QPOETLinear layer from an existing Linear layer.
        
        Args:
            linear: The Linear layer to convert.
            block_size: Block size for POET transformations.
            group_size: Group size for quantization.
            num_bits: Number of bits for quantization.
            mem_efficient_mode: Whether to use memory-efficient mode.
            normalize_weights: Whether to normalize_weights weights before quantization.
            
        Returns:
            A new QPOETLinear layer with quantized weights.
        """
        weight = linear.weight.data
        if normalize_weights:
            weight = weight / torch.norm(weight, dim=1, keepdim=True)
        
        bias_data = linear.bias.data if linear.bias is not None else None
        
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            block_size=block_size,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            num_bits=num_bits,
            group_size=group_size,
            mem_efficient_mode=mem_efficient_mode,
            weight=weight,
            bias_data=bias_data,
        )

    def _dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize weight to specified dtype.
        
        Args:
            dtype: Target data type.
            
        Returns:
            Dequantized weight tensor.
        """
        w = self.weight.to(dtype).reshape(-1, self.weight_group_size)
        w = (w - self.weight_zeros.to(dtype)) * self.weight_scales.to(dtype)
        return w.reshape(self.weight.shape)

    @torch.no_grad()
    def _requantize_weight(self, w_float: torch.Tensor) -> None:
        """Requantize weight from float tensor.
        
        Args:
            w_float: Float weight tensor.
        """
        q, scales, zeros = quantize_tensor_int8(
            w_float, q_group_size=self.weight_group_size, n_bit=self.weight_num_bits
        )
        self.weight.detach().copy_(q.to(self.weight.device))
        self.weight_scales.copy_(scales.to(self.weight.device))
        self.weight_zeros.copy_(zeros.to(self.weight.device))

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        """Merge POET transformations and reinitialize with quantization."""
        # Compute orthogonal matrices
        R_out, R_in = get_weight_poet(
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.num_out_blocks,
            self.num_in_blocks
        )

        # Dequantize, merge, requantize
        W = self._dequantize_weight(dtype=self.oft_R.dtype)
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        W_new = tmp.t()

        # Generate new permutations
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Apply new permutation
        W_new = W_new.index_select(0, perm_out_inv).index_select(1, perm_in_inv)

        # Requantize
        self._requantize_weight(W_new)

        # Update buffers
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)
        self.oft_R.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying quantized POET transformations.
        """
        return forward_core_q8(
            x,
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.perm_in,
            self.perm_in_inv,
            self.perm_out,
            self.perm_out_inv,
            self.num_in_blocks,
            self.num_out_blocks,
            self.weight,
            self.weight_scales,
            self.weight_zeros,
            self.weight_group_size,
            self.bias,
            self.mem_efficient_mode,
        )

    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective weight matrix.
        
        Args:
            dtype: Data type for computation.
            
        Returns:
            The effective weight matrix.
        """
        R_out, R_in = get_weight_poet(
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.num_out_blocks,
            self.num_in_blocks
        )
        
        W = self._dequantize_weight(self.oft_R.dtype)
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        return tmp.t()

    def extra_repr(self) -> str:
        """String representation."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"block_size={self.block_size}, bias={self.bias is not None}, "
                f"bits={self.weight_num_bits}, group_size={self.weight_group_size}")
