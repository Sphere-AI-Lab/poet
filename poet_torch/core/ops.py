"""POET Core Operations.

This module contains the core mathematical operations for POET,
including block-diagonal matrix multiplication, Cayley transform,
and optimized forward passes.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from poet_torch.core.triton_ops import (
    chain_layer_x_checkpoint_mem_o2,
    chain_layer_x_checkpoint_mem_o2_q8,
    chain_layer_x_checkpoint_q8,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Skew-Symmetric Matrix Construction
# =============================================================================

def pytorch_skew_symmetric(
    vec: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor
) -> torch.Tensor:
    """Create skew-symmetric matrices from vector parameters.
    
    Constructs skew-symmetric matrices Q such that Q^T = -Q from
    vectorized upper-triangular parameters.
    
    Args:
        vec: Vector parameters with shape (batch_size, n_elements).
        block_size: Size of the square output matrices.
        rows: Row indices for upper triangular positions.
        cols: Column indices for upper triangular positions.
        
    Returns:
        Skew-symmetric matrices with shape (batch_size, block_size, block_size).
    """
    batch_size = vec.shape[0]
    matrix = vec.new_zeros(batch_size, block_size, block_size)
    matrix[:, rows, cols] = vec
    matrix = matrix - matrix.transpose(-2, -1)
    return matrix


# =============================================================================
# Block-Diagonal Matrix Operations
# =============================================================================

def block_diag_lr_matmul(
    A_blocks: torch.Tensor,
    W: torch.Tensor,
    B_blocks: torch.Tensor
) -> torch.Tensor:
    """Compute block-diagonal left-right matrix multiplication.
    
    Computes (block_diag(A_blocks) @ W @ block_diag(B_blocks)) without
    materializing the full block-diagonal matrices.
    
    Args:
        A_blocks: Left block-diagonal factors with shape (r_m, b, b).
        W: Center matrix with shape (M, N) where M = r_m * b, N = r_n * b.
        B_blocks: Right block-diagonal factors with shape (r_n, b, b).
        
    Returns:
        Result matrix with shape (M, N).
    """
    if A_blocks.ndim != 3 or B_blocks.ndim != 3:
        raise ValueError("A_blocks and B_blocks must be 3D: (r, b, b)")
    
    r_m, b1, b2 = A_blocks.shape
    r_n, b3, b4 = B_blocks.shape
    
    if not (b1 == b2 == b3 == b4):
        raise ValueError("All block sizes must match and be square b x b.")
    
    b = b1
    M = r_m * b
    N = r_n * b
    
    if W.shape != (M, N):
        raise ValueError(f"W must have shape {(M, N)}, got {tuple(W.shape)}")

    # Ensure device/dtype compatibility
    if A_blocks.device != W.device or A_blocks.dtype != W.dtype:
        A_blocks = A_blocks.to(device=W.device, dtype=W.dtype)
    if B_blocks.device != W.device or B_blocks.dtype != W.dtype:
        B_blocks = B_blocks.to(device=W.device, dtype=W.dtype)

    # Reshape W into blocks and apply batched matmuls
    # W_blocks has shape (r_m, r_n, b, b)
    W_blocks = W.view(r_m, b, r_n, b).transpose(1, 2)

    # Left multiply each block-row by corresponding A_blocks[i]
    left = torch.matmul(A_blocks.unsqueeze(1), W_blocks)

    # Right multiply each block-col by corresponding B_blocks[j]
    out_blocks = torch.matmul(left, B_blocks.unsqueeze(0))

    # Fold back to (M, N)
    out = out_blocks.permute(0, 2, 1, 3).contiguous().view(M, N)
    return out


def torch_bmm(x: torch.Tensor, R: torch.Tensor, block_size: int) -> torch.Tensor:
    """Batch matrix multiplication with block-diagonal structure.
    
    Args:
        x: Input tensor with shape (..., features).
        R: Block-diagonal factors with shape (num_blocks, block_size, block_size).
        block_size: Size of each block.
        
    Returns:
        Transformed tensor with same shape as input.
    """
    Bdims = x.shape[:-1]
    xr = x.view(*Bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R)
    x_rot = xr.contiguous().view(*Bdims, -1)
    return x_rot


# =============================================================================
# Permutation Operations
# =============================================================================

class PermutationFunction(torch.autograd.Function):
    """Autograd function for permutations."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, perm: torch.Tensor, inv_perm: torch.Tensor):
        ctx.save_for_backward(inv_perm)
        return x[..., perm]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (inv_perm,) = ctx.saved_tensors
        grad_input = grad_output[..., inv_perm]
        return grad_input, None, None


def permute_x(
    x: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor
) -> torch.Tensor:
    """Apply permutation to input tensor.
    
    Args:
        x: Input tensor.
        perm: Permutation indices.
        inv_perm: Inverse permutation indices.
        
    Returns:
        Permuted tensor.
    """
    return PermutationFunction.apply(x, perm, inv_perm)


# =============================================================================
# Cayley Transform
# =============================================================================

def get_weight_poet(
    R: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    r_out: int,
    r_in: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute POET orthogonal matrices from parameters via Cayley transform.
    
    The Cayley transform maps skew-symmetric matrices to orthogonal matrices:
    R = (I + Q)(I - Q)^(-1) where Q is skew-symmetric.
    
    Args:
        R: POET parameters with shape (r_in + r_out, n_elements).
        block_size: Size of each block.
        rows: Row indices for skew-symmetric construction.
        cols: Column indices for skew-symmetric construction.
        r_out: Number of output blocks.
        r_in: Number of input blocks.
        
    Returns:
        Tuple of (R_out, R_in) orthogonal matrices, each with shape
        (num_blocks, block_size, block_size).
    """
    Q_skew_cat = pytorch_skew_symmetric(R, block_size, rows, cols)
    
    # Use torch.ops.poet.cayley if available (Triton kernel)
    # Fall back to pure PyTorch implementation otherwise
    if hasattr(torch.ops, 'poet') and hasattr(torch.ops.poet, 'cayley'):
        R_cat = torch.ops.poet.cayley(Q_skew_cat)[0]
    else:
        # Pure PyTorch Cayley transform
        R_cat = _cayley_transform_pytorch(Q_skew_cat)
    
    R_out, R_in = R_cat.split([r_out, r_in], dim=0)
    return R_out, R_in


def _cayley_transform_pytorch(Q: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch implementation of Cayley transform.
    
    Computes (I + Q)(I - Q)^(-1) using Neumann series approximation.
    
    Args:
        Q: Skew-symmetric matrices with shape (..., n, n).
        
    Returns:
        Orthogonal matrices with same shape.
    """
    Q2 = Q @ Q
    Yf = 2.0 * (Q + Q2 + Q2 @ Q) + 2.0 * Q2 @ Q2
    Yf.diagonal(dim1=-2, dim2=-1).add_(1.0)
    return Yf


# =============================================================================
# Forward Pass Implementations
# =============================================================================

def chain_layer_x_pytorch(
    x: torch.Tensor,
    Rin: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    Rout: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """PyTorch implementation of POET chain layer forward pass.
    
    Args:
        x: Input tensor.
        Rin: Input rotation matrix.
        weight: Weight matrix.
        bias: Optional bias vector.
        Rout: Output rotation matrix.
        block_size: Block size of POET.
        
    Returns:
        Output tensor.
    """
    x = torch_bmm(x, Rin, block_size)
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    y = torch_bmm(y, Rout, block_size)
    return y


@torch.compile(fullgraph=True)
def forward_core(
    x: torch.Tensor,
    R: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    r_in: int,
    r_out: int,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    mem_efficient_mode: bool = False,
) -> torch.Tensor:
    """Core forward pass for POET linear layer.
    
    Args:
        x: Input tensor.
        R: POET parameters.
        block_size: Block size of POET.
        rows, cols: Indices for skew-symmetric construction.
        perm_in, perm_in_inv: Input permutation and inverse.
        perm_out, perm_out_inv: Output permutation and inverse.
        r_in, r_out: Number of input/output blocks.
        base_weight: Base weight matrix.
        base_bias: Optional base bias.
        mem_efficient_mode: Whether to use memory-efficient mode.
        
    Returns:
        Output tensor with shape.
    """
    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in)

    if not mem_efficient_mode:
        # Standard mode
        x = permute_x(x, perm_in_inv, perm_in)
        y = chain_layer_x_pytorch(x, R_in, base_weight, base_bias, R_out, block_size)
        y = permute_x(y, perm_out, perm_out_inv)
    else:
        # Memory-efficient mode
        y = chain_layer_x_checkpoint_mem_o2(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv, block_size
        )

    return y


@torch.compile(fullgraph=True)
def forward_core_q8(
    x: torch.Tensor,
    R: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    r_in: int,
    r_out: int,
    W_q: torch.Tensor,
    W_scales: torch.Tensor,
    W_zeros: torch.Tensor,
    group_size: int,
    base_bias: Optional[torch.Tensor],
    mem_efficient_mode: bool = False,
) -> torch.Tensor:
    """Core forward pass for quantized POET linear layer.
    
    Args:
        x: Input tensor.
        R: POET parameters.
        block_size: Block size for transformations.
        rows, cols: Indices for skew-symmetric construction.
        perm_in, perm_in_inv: Input permutation and inverse.
        perm_out, perm_out_inv: Output permutation and inverse.
        r_in, r_out: Number of input/output blocks.
        W_q: Quantized weight matrix.
        W_scales: Weight quantization scales.
        W_zeros: Weight quantization zeros.
        group_size: Quantization group size.
        base_bias: Optional base bias.
        mem_efficient_mode: Whether to use memory-efficient mode.
        
    Returns:
        Output tensor.
    """
    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in)

    if not mem_efficient_mode:
        # Standard mode
        x = permute_x(x, perm_in_inv, perm_in)
        y = chain_layer_x_checkpoint_q8(
            x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out, block_size
        )
        y = permute_x(y, perm_out, perm_out_inv)
    else:
        # Memory-efficient mode
        y = chain_layer_x_checkpoint_mem_o2_q8(
            x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv, block_size
        )

    return y


# =============================================================================
# Quantization Utilities
# =============================================================================

def quantize_tensor_int8(
    w: torch.Tensor,
    q_group_size: int = -1,
    n_bit: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor to int8.
    
    Args:
        w: Weight tensor to quantize.
        q_group_size: Group size for quantization. If -1, quantize per row.
        n_bit: Number of bits for quantization (default 8).
        
    Returns:
        Tuple of (quantized weights, scales, zeros).
    """
    assert n_bit == 8, "Only 8-bit quantization is supported."
    org_w_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros
