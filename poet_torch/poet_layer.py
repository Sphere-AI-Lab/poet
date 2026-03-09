"""POET linear layer implementations.

This module provides POET linear layers with orthogonal transformations
for parameter-efficient fine-tuning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .poet_core import (
    block_diag_lr_matmul,
    forward_core,
    forward_core_q8,
    get_weight_poet,
    quantize_tensor_int8,
)


class POETLinear(nn.Module):
    """POET linear layer with orthogonal transformations.
    
    This layer applies learnable orthogonal transformations to the input and
    output of a frozen linear layer for parameter-efficient pre-training.
    
    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        bsz: Block size for transformations.
        bias: Whether to include bias.
        device: Device for parameters.
        dtype: Data type for parameters.
        mem_efficient_mode: Whether to use memory-efficient mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bsz: int = 256,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        mem_efficient_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        self.mem_efficient_mode = mem_efficient_mode

        # Base linear layer parameters (frozen)
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # Trainable skew-symmetric parameters per block
        r_in = in_features // bsz
        r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        self.oft_R = nn.Parameter(
            torch.zeros((r_in + r_out, n_elements), device=device, dtype=dtype)
        )
        self.r_in = r_in
        self.r_out = r_out

        # Register buffers for skew-symmetric construction
        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # Register buffers for permutations
        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        """Merge POET transformations into base weight and reinitialize.
        
        This applies the learned orthogonal transformations to the base weight,
        generates new random permutations, and resets the trainable parameters.
        """
        R_out, R_in = get_weight_poet(
            self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in
        )

        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Generate new permutations before applying
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Apply new permutation to weight
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)

        self.weight.detach().copy_(expected)

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
            Output tensor after applying POET transformations.
        """
        return forward_core(
            x,
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.perm_in,
            self.perm_in_inv,
            self.perm_out,
            self.perm_out_inv,
            self.r_in,
            self.r_out,
            self.weight,
            self.bias,
            self.mem_efficient_mode,
        )


class QPOETLinear(nn.Module):
    """Quantized POET linear layer.
    
    This is a quantized version of POETLinear that stores weights in int8
    format for memory efficiency.
    
    Args:
        weight: Initial weight tensor (will be quantized).
        bias: Initial bias tensor or None.
        bsz: Block size for transformations.
        device: Device for parameters.
        dtype: Data type for trainable parameters.
        num_bits: Number of bits for quantization (default 8).
        group_size: Group size for quantization.
        mem_efficient_mode: Whether to use memory-efficient mode.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        bsz: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_bits: int = 8,
        group_size: int = 256,
        mem_efficient_mode: bool = False,
    ) -> None:
        super().__init__()

        if device is None:
            device = weight.device

        # Quantize weight
        int8_weight, scales, zeros = quantize_tensor_int8(
            weight.data, q_group_size=group_size
        )
        torch.cuda.empty_cache()

        self.weight = nn.Parameter(int8_weight, requires_grad=False).to(device)
        self.register_buffer("weight_scales", scales.to(device))
        self.register_buffer("weight_zeros", zeros.to(device))
        self.weight_group_size = group_size
        self.weight_num_bits = num_bits

        if num_bits != 8:
            raise NotImplementedError("Only 8-bit quantization is supported")

        self.bias = (
            nn.Parameter(bias, requires_grad=True).to(device) if bias is not None else None
        )

        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]
        self.block_size = bsz
        self.mem_efficient_mode = mem_efficient_mode

        # Trainable skew-symmetric parameters (same as POETLinear)
        r_in = self.in_features // bsz
        r_out = self.out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        self.oft_R = nn.Parameter(
            torch.zeros((r_in + r_out, n_elements), device=device, dtype=dtype)
        )
        self.r_in = r_in
        self.r_out = r_out

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # Permutation buffers
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @torch.no_grad()
    def _requantize_from_float(self, w_float: torch.Tensor) -> None:
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

    def _dequantize_to(self, dtype: torch.dtype) -> torch.Tensor:
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
    def merge_then_reinitialize(self) -> None:
        """Merge POET transformations and reinitialize with quantization."""
        R_out, R_in = get_weight_poet(
            self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in
        )

        # Dequantize, merge, requantize
        W = self._dequantize_to(dtype=self.oft_R.dtype)
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Generate new permutations
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Apply new permutation
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)

        # Requantize
        self._requantize_from_float(expected)

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
            self.r_in,
            self.r_out,
            self.weight,
            self.weight_scales,
            self.weight_zeros,
            self.weight_group_size,
            self.bias,
            self.mem_efficient_mode,
        )

