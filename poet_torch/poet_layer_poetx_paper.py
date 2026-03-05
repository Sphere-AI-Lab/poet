import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from typing import Optional
from .poet_ops import *

import numpy as np
import math
from tqdm import tqdm
import gc
import os
import sys
import logging

logger = logging.getLogger(__name__)

def permute_x(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)

def chain_layer_x_checkpoint_mem_o2(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor,
                                    perm_in_inv: torch.Tensor, perm_in: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2(x, Rin, weight, bias, Rout, perm_in_inv, perm_in, block_size)
    
def chain_layer_x_checkpoint(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint(x, Rin, weight, bias, Rout, block_size)

def chain_layer_x_checkpoint_2lr(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_2lr(x, Rin, weight, bias, Rout, block_size)

def chain_layer_x_checkpoint_q8(x: torch.Tensor, Rin: torch.Tensor, W_q: torch.Tensor, W_scales: torch.Tensor, W_zeros: torch.Tensor, group_size: int, b: Optional[torch.Tensor], Rout: torch.Tensor, bsz: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_q8(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, bsz)

def chain_layer_x_checkpoint_mem_o2_q8(x: torch.Tensor, Rin: torch.Tensor, W_q: torch.Tensor, W_scales: torch.Tensor, W_zeros: torch.Tensor, group_size: int, b: Optional[torch.Tensor], Rout: torch.Tensor, perm_in_inv: torch.Tensor, perm_in: torch.Tensor, bsz: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2_q8(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, perm_in_inv, perm_in, bsz)

def chain_layer_x_checkpoint_4bit(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_4bit: torch.Tensor,
    # quant_state main components:
    qs_absmax: torch.Tensor,
    qs_code: torch.Tensor,
    qs_blocksize: int,
    qs_quant_type: str,
    qs_shape_0: int,
    qs_shape_1: int,
    # quant_state.state2 components (double-quant):
    qs2_absmax: torch.Tensor,
    qs2_code: torch.Tensor,
    qs2_blocksize: int,
    qs_offset: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    bsz: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_4bit(
        x, Rin, W_4bit,
        qs_absmax, qs_code, qs_blocksize, qs_quant_type, qs_shape_0, qs_shape_1,
        qs2_absmax, qs2_code, qs2_blocksize, qs_offset,
        b, Rout, bsz, out_features, in_features
    )

def chain_layer_x_checkpoint_mem_o2_4bit(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_4bit: torch.Tensor,
    # quant_state main components:
    qs_absmax: torch.Tensor,
    qs_code: torch.Tensor,
    qs_blocksize: int,
    qs_quant_type: str,
    qs_shape_0: int,
    qs_shape_1: int,
    # quant_state.state2 components (double-quant):
    qs2_absmax: torch.Tensor,
    qs2_code: torch.Tensor,
    qs2_blocksize: int,
    qs_offset: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    bsz: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2_4bit(
        x, Rin, W_4bit,
        qs_absmax, qs_code, qs_blocksize, qs_quant_type, qs_shape_0, qs_shape_1,
        qs2_absmax, qs2_code, qs2_blocksize, qs_offset,
        b, Rout, perm_in_inv, perm_in, bsz, out_features, in_features
    )


def _quantize_tensor_int8(w, q_group_size=-1, n_bit=8):
    # print(f"Quantizing tensor on device: {w.device}")  # ADD THIS
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

    # assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros


def block_diag_lr_matmul(A_blocks: torch.Tensor, W: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    """
    Compute (block_diag(A_blocks) @ W @ block_diag(B_blocks)) without materializing block-diagonal matrices.

    Args:
      A_blocks: (r_m, b, b) block-diagonal factors for the left (M = r_m * b)
      W:        (M, N) matrix to multiply, where M = r_m * b, N = r_n * b
      B_blocks: (r_n, b, b) block-diagonal factors for the right (N = r_n * b)

    Returns:
      Tensor of shape (M, N)
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

    # Ensure device/dtype compatibility (keeps things simple and safe)
    if A_blocks.device != W.device or A_blocks.dtype != W.dtype:
        A_blocks = A_blocks.to(device=W.device, dtype=W.dtype)
    if B_blocks.device != W.device or B_blocks.dtype != W.dtype:
        B_blocks = B_blocks.to(device=W.device, dtype=W.dtype)

    # Reshape W into blocks and apply batched matmuls:
    # W_ = (r_m, r_n, b, b), where W_[i, j] is the (i, j) b x b block of W
    W_blocks = W.view(r_m, b, r_n, b).transpose(1, 2)  # (r_m, r_n, b, b)

    # Left multiply each block-row by corresponding A_blocks[i]
    # Shapes: (r_m, 1, b, b) @ (r_m, r_n, b, b) -> (r_m, r_n, b, b)
    left = torch.matmul(A_blocks.unsqueeze(1), W_blocks)

    # Right multiply each block-col by corresponding B_blocks[j]
    # Shapes: (r_m, r_n, b, b) @ (1, r_n, b, b) -> (r_m, r_n, b, b)
    out_blocks = torch.matmul(left, B_blocks.unsqueeze(0))

    # Fold back to (M, N)
    out = out_blocks.permute(0, 2, 1, 3).contiguous().view(M, N)
    return out

def pytorch_skew_symmetric(vec, block_size, rows, cols):
    batch_size = vec.shape[0]
    matrix = vec.new_zeros(batch_size, block_size, block_size)  # Inherits requires_grad
    matrix[:, rows, cols] = vec
    matrix = matrix - matrix.transpose(-2, -1)
    return matrix

def cayley_batch(Qf):
    Q2f = Qf @ Qf
    Yf = 2.0 * (Qf + Q2f + Q2f @ Qf) + 2.0 * Q2f @ Q2f
    # Yf = 2.0 * (Qf + Q2f) + Q2f @ (2.0 * Qf + Q2f)
    Yf.diagonal(dim1=-2, dim2=-1).add_(1.0)
    return Yf

def get_weight_poet(R, block_size, rows, cols, r_out, r_in):
    # r_left = Rl.size(0)
    # r_right = Rr.size(0)

    # R = torch.cat([Ro, Ri], dim=0).contiguous()
    # Q_skew_cat = skew_symmetric(R, block_size, rows, cols, idx_ul)
    Q_skew_cat = pytorch_skew_symmetric(R, block_size, rows, cols)
    # Q_skew_cat = torch.ops.poet.skew_symmetric(R, block_size, rows, cols, idx_ul)

    # R_cat = CayleyTritonFn.apply(Q_skew_cat)
    R_cat = torch.ops.poet.cayley(Q_skew_cat)[0]
    # R_cat = cayley_batch(Q_skew_cat)
    R_out, R_in = R_cat.split([r_out, r_in], dim=0)

    return R_out, R_in

def torch_bmm(x, R, block_size):
    Bdims = x.shape[:-1]
    xr = x.view(*Bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R)
    x_rot = xr.contiguous().view(*Bdims, -1)
    return x_rot

def chain_layer_x_pytorch(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor,
                          bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
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
    base_bias: torch.Tensor,
    mem_efficient_mode: bool = False,
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    # base_weight = base_weight * scale.to(dtype=base_weight.dtype)

    # Balance learning rates:
    # If r_out > r_in, scale > 1.
    # R_in (smaller) is multiplied by scale -> gradient boosted by scale.
    # R_out (larger) is divided by scale -> gradient dampened by scale.
    # scale = (r_out / r_in) ** 0.5
    # R_in = R_in * scale
    # R_out = R_out / scale

    # balance mode
    x = permute_x(x, perm_in_inv, perm_in)
    
    # y = chain_layer_x_pytorch(x, R_in, base_weight, base_bias, R_out, block_size)
    y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)
    # lower throughput but less memory (merge permute inside chain layer)
    # y = chain_layer_x_checkpoint_mem_o2(x, R_in, base_weight, base_bias, R_out, perm_in_inv, perm_in, block_size)

    y = permute_x(y, perm_out, perm_out_inv)
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
    base_bias: torch.Tensor,
    mem_efficient_mode: bool = False,
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    x = permute_x(x, perm_in_inv, perm_in)
    y = chain_layer_x_checkpoint_q8(x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out, block_size)
    # y = chain_layer_x_checkpoint_mem_o2_q8(x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out, perm_in_inv, perm_in, block_size)
    y = permute_x(y, perm_out, perm_out_inv)
    return y


# @torch.compile(fullgraph=True)
# def forward_core(
#     x: torch.Tensor, 
#     R: torch.Tensor,
#     block_size: int,
#     rows: torch.Tensor,
#     cols: torch.Tensor,
#     perm_in: torch.Tensor, 
#     perm_in_inv: torch.Tensor,
#     perm_out: torch.Tensor,
#     perm_out_inv: torch.Tensor,
#     r_in: int,
#     r_out: int,
#     base_weight: torch.Tensor,
#     base_bias: torch.Tensor,
#     mem_efficient_mode: bool = False,
# ) -> torch.Tensor:

#     R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

#     # higher throughput but more memory
#     x = permute_x(x, perm_in_inv, perm_in)
#     y = chain_layer_x_pytorch(x, R_in, base_weight, base_bias, R_out, block_size) # y = x @ R @ W @ P
#     y = permute_x(y, perm_out, perm_out_inv)
#     return y


class POETLinear(nn.Module):
    def __init__(self, in_features, out_features, bsz=256, bias=False, device=None, dtype=None, mem_efficient_mode=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        self.mem_efficient_mode = mem_efficient_mode
        # Basic linear layer parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable skew-params per block
        r_in = in_features // bsz
        r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        # Param tensors can be any square; we skew them inside forward
        # self.R_left = nn.Parameter(torch.zeros((r_in, n_elements), **factory_kwargs))
        # self.R_right = nn.Parameter(torch.zeros((r_out, n_elements), **factory_kwargs))
        self.oft_R = nn.Parameter(torch.zeros((r_in + r_out, n_elements), device=device, dtype=dtype))
        self.r_in = r_in
        self.r_out = r_out

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer('rows', rows.to(torch.int32))
        self.register_buffer('cols', cols.to(torch.int32))
        # self.rows = rows.to(torch.int32)
        # self.rows.requires_grad = False
        # self.cols = cols.to(torch.int32)
        # self.cols.requires_grad = False

        group_size = 1
        perm_in = torch.randperm(in_features // group_size, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features // group_size, device=device, dtype=torch.int32)
        self.register_buffer('perm_in', perm_in)
        self.register_buffer('perm_out', perm_out)
        self.register_buffer('perm_in_inv', torch.argsort(perm_in).to(torch.int32))
        self.register_buffer('perm_out_inv', torch.argsort(perm_out).to(torch.int32))
        # self.perm_in = perm_in
        # self.perm_in.requires_grad = False
        # self.perm_out = perm_out
        # self.perm_out.requires_grad = False
        # self.perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        # self.perm_in_inv.requires_grad = False
        # self.perm_out_inv = torch.argsort(perm_out).to(torch.int32)
        # self.perm_out_inv.requires_grad = False

        # self.reset_parameters()

        # self.out_norm = nn.RMSNorm(out_features, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        # self.log_scale = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

    def random_init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.normal_(self.R_left, std=1e-3)
        # nn.init.normal_(self.R_right, std=1e-3)  
        nn.init.normal_(self.oft_R[:self.r_in], std=1e-3)
        nn.init.normal_(self.oft_R[self.r_in:], std=1e-3)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def perform_permutation(self) -> None:
        # Merge the self.linear.weight with permutations to avoid P_in.t() @ W_orig.t() @ P_out in the forward pass
        # W_merged.t() = P_in.t() @ W_orig.t() @ P_out
        # W_merged = P_out.t() @ W_orig @ P_in
        # with torch.no_grad():
        #     W = self.weight
        #     Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        #     W.copy_(Wp)
            # self.linear.weight.data = self.linear.weight.data.index_select(0, self.perm_out_inv)
            # self.linear.weight.data = self.linear.weight.data.index_select(-1, self.perm_in_inv)
        W = self.weight
        Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        self.weight.detach().copy_(Wp)

    def update_permutation(self):
        """Update the permutation of the indices."""
        # with torch.no_grad():
        #     device = self.linear.weight.device
        #     perm_in = torch.randperm(self.in_features, device=device)
        #     self.perm_in.copy_(perm_in)
        #     self.perm_in_inv.copy_(torch.argsort(perm_in))
        #     perm_out = torch.randperm(self.out_features, device=device)
        #     self.perm_out.copy_(perm_out)
        #     self.perm_out_inv.copy_(torch.argsort(perm_out))

        #     self.perform_permutation()

        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in))
        perm_out = torch.randperm(self.out_features, device=device)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out))

        self.perform_permutation()

        # merge the self.linear.weight with permutations to avoid P_in.t() @ W_orig.t() @ P_out in the forward pass
        # W_merged.t() = P_in.t() @ W_orig.t() @ P_out
        # W_merged = P_out.t() @ W_orig @ P_in
        # W_orig = self.linear.weight.data
        # W_orig = W_orig.index_select(0, self.perm_out_inv)
        # W_orig = W_orig.index_select(-1, self.perm_in_inv)
        # self.linear.weight.data.copy_(W_orig)  

    def merge_then_reinitialize_working(self) -> None:
        # with torch.no_grad():
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)
        # R_out = torch.block_diag(*R_out)
        # R_in = torch.block_diag(*R_in)

        # self.undo_permutation()

        # Recover the original weights by undoing permutation
        # W_merged = P_out.t() @ W_orig @ P_in
        # W_orig = P_out @ W_merged @ P_in.t()
        # W_merged = self.linear.weight.data
        # W_orig = W_merged.index_select(-1, self.perm_in)
        # W_orig = W_orig.index_select(0, self.perm_out)

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # W_merged.t() = P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # W_merged.t() = R_in_merged @ W_orig.t() @ R_out_merged
        # R_in_merged = P_in @ R_in @ P_in.t()
        # R_out_merged = P_out @ R_out @ P_out.t()
        # R_in = R_in.index_select(-1, self.perm_in)
        # R_in = R_in.index_select(0, self.perm_in)
        # R_out = R_out.index_select(-1, self.perm_out)
        # R_out = R_out.index_select(0, self.perm_out)
        # W_final = (R_in @ W_orig.t() @ R_out).t()
        # self.linear.weight.data.copy_(W_final)

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # 1) P_in.t() @ W_orig.t() @ P_out
        W = self.weight.detach().clone()
        # W0 = W.detach().clone()
        tmp = W.t()
        # # # 2) R_in @ tmp @ R_out
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # 3) P_in @ tmp @ P_out.t()
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()
        
        # Transpose back to weight shape
        self.weight.detach().copy_(expected)

        # tmp = self.weight.detach().clone()
        # tmp = tmp.t()
        # tmp = tmp.index_select(0, self.perm_in_inv)
        # tmp = tmp.index_select(-1, self.perm_out_inv)
        # tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # tmp = tmp.index_select(0, self.perm_in)
        # tmp = tmp.index_select(-1, self.perm_out)
        # expected = tmp.t()

        # W = self.weight.detach().clone()
        # W = W.t()
        # # P_in.t() @ W_orig.t() @ P_out
        # W = W.index_select(0, self.perm_in_inv)
        # W = W.index_select(1, self.perm_out_inv)
        
        # # R_in @ W.t() @ R_out
        # W = block_diag_lr_matmul(R_in, W, R_out)
        
        # # P_in @ W.t() @ P_out.t()
        # W = W.index_select(0, self.perm_in)
        # W = W.index_select(1, self.perm_out)
        # W = W.t()
        # self.weight.detach().copy_(W)

        self.oft_R.zero_()
        self.update_permutation()

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        # Same math as POETLinear.merge_then_reinitialize, but float compute + requantize
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Generate NEW permutation BEFORE quantizing
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Apply NEW permutation to float weight before quantizing (avoids double quantization)
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)

        self.weight.detach().copy_(expected)

        # Update buffers to match the quantized weight
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)

        self.oft_R.zero_()

    def forward(self, x):
        # scale = torch.exp(torch.tanh(self.log_scale) * 0.2).to(x.device)
        x = forward_core(x, self.oft_R, self.block_size, self.rows, self.cols, 
                self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv, 
                self.r_in, self.r_out, self.weight, self.bias, self.mem_efficient_mode)

        # x = self.out_norm(x)
        return x



class POETLinearNeurips(nn.Module):
    def __init__(self, in_features, out_features, bsz=256, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        # Basic linear layer parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable skew-params per block
        r_in = in_features // bsz
        r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        # Param tensors can be any square; we skew them inside forward
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elements), device=device, dtype=dtype))
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elements), device=device, dtype=dtype))

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer('rows', rows.to(torch.int32))
        self.register_buffer('cols', cols.to(torch.int32))

        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer('perm_in', perm_in)
        self.register_buffer('perm_out', perm_out)
        self.register_buffer('perm_in_inv', torch.argsort(perm_in).to(torch.int32))
        self.register_buffer('perm_out_inv', torch.argsort(perm_out).to(torch.int32))

    def update_permutation(self):
        """Update the permutation of the indices."""
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in))
        perm_out = torch.randperm(self.out_features, device=device)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out))

    def merge_then_reinitialize(self) -> None:
        # with torch.no_grad():
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)
        # R_out = torch.block_diag(*R_out)
        # R_in = torch.block_diag(*R_in)

        # self.undo_permutation()

        # Recover the original weights by undoing permutation
        # W_merged = P_out.t() @ W_orig @ P_in
        # W_orig = P_out @ W_merged @ P_in.t()
        # W_merged = self.linear.weight.data
        # W_orig = W_merged.index_select(-1, self.perm_in)
        # W_orig = W_orig.index_select(0, self.perm_out)

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # W_merged.t() = P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # W_merged.t() = R_in_merged @ W_orig.t() @ R_out_merged
        # R_in_merged = P_in @ R_in @ P_in.t()
        # R_out_merged = P_out @ R_out @ P_out.t()
        # R_in = R_in.index_select(-1, self.perm_in)
        # R_in = R_in.index_select(0, self.perm_in)
        # R_out = R_out.index_select(-1, self.perm_out)
        # R_out = R_out.index_select(0, self.perm_out)
        # W_final = (R_in @ W_orig.t() @ R_out).t()
        # self.linear.weight.data.copy_(W_final)

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # 1) P_in.t() @ W_orig.t() @ P_out
        W = self.weight.detach().clone()
        # W0 = W.detach().clone()
        tmp = W.t()
        # # # 2) R_in @ tmp @ R_out
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # 3) P_in @ tmp @ P_out.t()
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()
        
        # Transpose back to weight shape
        self.weight.detach().copy_(expected)

        # tmp = self.weight.detach().clone()
        # tmp = tmp.t()
        # tmp = tmp.index_select(0, self.perm_in_inv)
        # tmp = tmp.index_select(-1, self.perm_out_inv)
        # tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # tmp = tmp.index_select(0, self.perm_in)
        # tmp = tmp.index_select(-1, self.perm_out)
        # expected = tmp.t()

        # W = self.weight.detach().clone()
        # W = W.t()
        # # P_in.t() @ W_orig.t() @ P_out
        # W = W.index_select(0, self.perm_in_inv)
        # W = W.index_select(1, self.perm_out_inv)
        
        # # R_in @ W.t() @ R_out
        # W = block_diag_lr_matmul(R_in, W, R_out)
        
        # # P_in @ W.t() @ P_out.t()
        # W = W.index_select(0, self.perm_in)
        # W = W.index_select(1, self.perm_out)
        # W = W.t()
        # self.weight.detach().copy_(W)

        self.oft_R.zero_()
        self.update_permutation()

    def get_cayley_transform_neumann_optimized(self, mode='all', num_neumann_terms=5):
        """
        Ultra-optimized version of get_cayley_transform_neumann.
        """
        R_left = None
        R_right = None
        # self.normalize_parameters(rms_norm=1.0)
            
        # Process left transform if needed
        if mode in ['all', 'left']:
            # Initialize result - use existing identity and expand in-place
            # Q_blocks = SkewSymmetricBatched.apply(self.R_out, self.soft_block_size)
            Q_blocks = pytorch_skew_symmetric(self.oft_R_out, self.block_size, self.rows, self.cols)
            R_left = torch.eye(self.block_size, device=self.oft_R_out.device, dtype=self.oft_R_out.dtype).repeat(self.oft_R_out.shape[0], 1, 1)
            
            # For small matrices, unroll the first few iterations
            if num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_left.add_(Q_blocks, alpha=2.0)
                
                if num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_left.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_left.add_(Q_power, alpha=2.0)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            # Initialize result - use existing identity and expand in-place
            # Q_blocks = SkewSymmetricBatched.apply(self.R_in, self.soft_block_size)
            Q_blocks = pytorch_skew_symmetric(self.oft_R_in, self.block_size, self.rows, self.cols)
            R_right = torch.eye(self.block_size, device=self.oft_R_in.device, dtype=self.oft_R_in.dtype).repeat(self.oft_R_in.shape[0], 1, 1)
            
            # For small matrices, unroll the first few iterations
            if num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_right.add_(Q_blocks, alpha=2.0)
                
                if num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_right.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_right.add_(Q_power, alpha=2.0)

        return R_left, R_right

    def forward(self, x):
        R_left, R_right = self.get_cayley_transform_neumann_optimized()

        # y = x @ W_new.t()
        # W_new = P_out @ R_out @ P_out.t() @ W @ P_in @ R_in @ P_in.t()
        # Kernel calculation for Inner = (P_out^T @ W) @ P_in
        temp_W1_kernel = self.weight.index_select(0, self.perm_out_inv)
        Inner = temp_W1_kernel.index_select(1, self.perm_in_inv)
        # temp_W1_kernel = PermuteMatrixFunction.apply(self.weight, self.perm_out_inv, 0) # P_out^T @ W
        # Inner = PermuteMatrixFunction.apply(temp_W1_kernel, self.perm_in_inv, 1) # Inner = (P_out^T @ W) @ P_in

        # tmp =self.matmul_R_left(R_left, Inner.unsqueeze(0)) # R_left @ Inner
        # Outer = self.matmul_R_right(tmp, R_right).squeeze() # Outer = (R_left @ Inner) @ R_right
        # Outer = block_diag_lr_matmul(R_left, Inner, R_right)
        R_left_bs = torch.block_diag(*R_left)
        R_right_bs = torch.block_diag(*R_right)
        Outer = R_left_bs @ Inner @ R_right_bs

        # Kernel calculation for Final = (P_out @ Outer) @ P_in^T
        temp_Outer_kernel = Outer.index_select(0, self.perm_out)
        transformed_weight = temp_Outer_kernel.index_select(1, self.perm_in)
        # temp_Outer_kernel = PermuteMatrixFunction.apply(Outer, self.perm_out, 0) # P_out @ Outer
        # transformed_weight = PermuteMatrixFunction.apply(temp_Outer_kernel, self.perm_in, 1) # (P_out @ Outer) @ P_in^T

        # x = forward_core(x, self.oft_R, self.block_size, self.rows, self.cols, 
        #         self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv, 
        #         self.r_in, self.r_out, self.weight, self.bias)

        return F.linear(x, transformed_weight.squeeze(), self.bias)


class QPOETLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        bsz=256,
        device=None,
        dtype=None,
        num_bits=8,
        group_size=256,
        stochastic_round=True,
        mem_efficient_mode=False,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        int8_weight, scales, zeros = _quantize_tensor_int8(weight.data, q_group_size=group_size)
        torch.cuda.empty_cache()

        self.weight = nn.Parameter(int8_weight, requires_grad=False).to(device) # Only Tensors of floating point and complex dtype can require gradients, using float_gradient to store the gradient
        # self.weight.__setattr__('scales', scales.to(device))
        # self.weight.__setattr__('zeros', zeros.to(device))
        # self.weight.__setattr__('group_size', group_size)
        # self.weight.__setattr__('saved_data_dtype', int8_weight.dtype)
        # self.weight.__setattr__('stochastic_round', stochastic_round)
        # self.weight.__setattr__('num_bits', num_bits)
        self.register_buffer('weight_scales', scales.to(device))
        self.register_buffer('weight_zeros', zeros.to(device))
        self.weight_group_size = group_size
        self.weight_saved_data_dtype = int8_weight.dtype
        self.weight_stochastic_round = stochastic_round
        self.weight_num_bits = num_bits

        if not num_bits == 8:
            raise NotImplementedError

        self.bias = nn.Parameter(bias, requires_grad=True).to(device) if bias is not None else None

        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]
        self.block_size = bsz
        self.mem_efficient_mode = mem_efficient_mode

        # if bias:
        #     self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        # else:
        #     self.register_parameter("bias", None)

        # Trainable skew-params per block (same as POETLinear)
        r_in = self.in_features // bsz
        r_out = self.out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        self.oft_R = nn.Parameter(torch.zeros((r_in + r_out, n_elements), device=device, dtype=dtype))
        self.r_in = r_in
        self.r_out = r_out

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # same perm buffers as POETLinear (perm acts on features/groups, not int8 groups)
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @torch.no_grad()
    def _requantize_from_float(self, w_float: torch.Tensor):
        q, scales, zeros = _quantize_tensor_int8(w_float, q_group_size=self.weight_group_size, n_bit=self.weight_num_bits)
        self.weight.detach().copy_(q.to(self.weight.device))
        self.weight_scales.copy_(scales.to(self.weight.device))
        self.weight_zeros.copy_(zeros.to(self.weight.device))

    def _dequantize_to(self, dtype: torch.dtype):
        w = self.weight.to(dtype).reshape(-1, self.weight_group_size)   
        w = (w - self.weight_zeros.to(dtype)) * self.weight_scales.to(dtype)
        return w.reshape(self.weight.shape)

    # @torch.no_grad()
    # def perform_permutation(self) -> None:
    #     # IMPORTANT: dequantize -> permute in float -> requantize (permutes invalidate old scales/zeros)
    #     #  W = self._dequantize_to(dtype=self.oft_R.dtype)
    #     Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
    #     # self._requantize_from_float(Wp)

    # @torch.no_grad()
    # def update_permutation(self):
    #     device = self.weight.device
    #     perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
    #     self.perm_in.copy_(perm_in)
    #     self.perm_in_inv.copy_(torch.argsort(perm_in).to(torch.int32))
    #     perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
    #     self.perm_out.copy_(perm_out)
    #     self.perm_out_inv.copy_(torch.argsort(perm_out).to(torch.int32))
    #     self.perform_permutation()


    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        # Same math as POETLinear.merge_then_reinitialize, but float compute + requantize
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

        # print(f"1. R_in device: {R_in.device}, R_out device: {R_out.device}")

        # print(f"Starting merge_then_reinitialize with W dtype: {self.weight.dtype}")

        # Step 1-4: Merge adapters (same as before)
        W = self._dequantize_to(dtype=self.oft_R.dtype)

        # print(f"2. After dequantize, W dtype: {W.dtype}")
        # print(f"2. After dequantize, W device: {W.device}")
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # print(f"3. After block_diag_lr_matmul, tmp device: {tmp.device}")
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Step 5: Generate NEW permutation BEFORE quantizing
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Step 6: Apply NEW permutation to the float weight
        # This undoes OLD perm and applies NEW perm in one step
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
        # print(f"4. After permutation, expected device: {expected.device}")
        
        # Step 7: Quantize (only once, with correct NEW perm)
        self._requantize_from_float(expected)
        # print(f"5. After requantize, weight device: {self.weight.device}")
        
        # Step 8: Update buffers to match the quantized weight
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)

        self.oft_R.zero_()

    def forward(self, x):
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



def replace_linear_with_poet(module: nn.Module, block_size: int, init_type: str, mup_alpha: float, device=None, dtype=None, 
                        mem_efficient_mode=False, neurips_version=False, v2: bool=False) -> int:
    def _convert(m: nn.Module, v2: bool=False):
        # nonlocal replaced
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and 'lm_head' not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    if neurips_version:
                        new_lin = POETLinearNeurips(
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bsz=block_size,
                            bias=(child.bias is not None),
                            device=device,
                            dtype=dtype,
                        )
                    else:
                        new_lin = POETLinear(
                            in_features=child.in_features,
                            out_features=child.out_features,
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

                        elif init_type == 'mup_normalized':
                            d_in = torch.tensor(child.weight.data.shape[1])
                            d_out = torch.tensor(child.weight.data.shape[0])

                            # [Check 1] Measure Spectral Norm BEFORE normalization
                            # child.weight is the original random initialization
                            # spec_before = torch.linalg.norm(child.weight.data.float(), ord=2).item() / torch.sqrt(d_out / d_in)
                            
                            normed_weight = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)

                            target_spec = mup_alpha * torch.sqrt(d_out / d_in)
                            current_spec = torch.linalg.norm(normed_weight.float(), ord=2).item()

                            scaling_factor = (target_spec / current_spec).to(dtype=normed_weight.dtype, device=normed_weight.device)
                            final_weight = normed_weight * scaling_factor
                            child.weight.data = final_weight

                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))

                        # [Check 2] Measure Spectral Norm AFTER normalization & copy
                        # This should be much smaller (close to 2.0 for large square matrices)
                        # spec_after = torch.linalg.norm(new_lin.weight.float(), ord=2).item() / torch.sqrt(torch.tensor(new_lin.weight.data.shape[0]) / torch.tensor(new_lin.weight.data.shape[1]))
                        # print(f"Weight Spectral Norm (Before): {spec_before:.4f}, (After): {spec_after:.4f}")

                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))
                    setattr(m, name, new_lin)
                else:
                    # skip non-divisible layers
                    raise ValueError(f"Layer {name} has in_features {child.in_features} and out_features {child.out_features}, which are not divisible by {block_size}")
            else:
                _convert(child)
    _convert(module)


def prepare_model_for_int8_training_poet(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_poet(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module: continue

            bias_data = module.bias.data if module.bias is not None else None
            weight = module.weight.data
            if args.init_type == 'normalized':
                weight = weight / torch.norm(weight, dim=1, keepdim=True)

            new_layers = QPOETLinear(
                weight,
                bias_data,
                bsz=args.poet_block_size,
                num_bits=args.weight_bits,
                group_size=args.weight_group_size,
                stochastic_round=args.stochastic_round
            )
            model._modules[name] = new_layers

    return model

# def prepare_model_for_int8_training_poet(model, args, target_module):
#     is_dist = dist.is_available() and dist.is_initialized()
#     rank = dist.get_rank() if is_dist else 0

#     for name, module in reversed(list(model._modules.items())):
#         if len(list(module.children())) > 0:
#             model._modules[name] = prepare_model_for_int8_training_poet(module, args, target_module)

#         if isinstance(module, nn.Linear):
#             if name not in target_module:
#                 continue

#             # choose a CUDA device for NCCL broadcast
#             bcast_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else module.weight.device

#             # (optional) do init normalization only on rank0, since we broadcast the quantized result
#             if rank == 0:
#                 with torch.no_grad():
#                     if getattr(args, "init_type", None) == "normalized":
#                         module.weight.data = module.weight.data / torch.norm(module.weight.data, dim=1, keepdim=True)
#                     elif getattr(args, "init_type", None) == "mup_normalized":
#                         w = module.weight.data
#                         w = w / torch.norm(w, dim=1, keepdim=True)
#                         module.weight.data = w * float(getattr(args, "mup_alpha", 1.0))

#             # --- rank0 quantizes, others receive ---
#             group_size = args.weight_group_size
#             w_shape = module.weight.shape
#             num_groups = module.weight.numel() // group_size

#             if is_dist:
#                 if rank == 0:
#                     w_float = module.weight.data.to(bcast_device)
#                     q, scales, zeros = _quantize_tensor_int8(w_float, q_group_size=group_size)
#                     q = q.to(bcast_device)
#                     scales = scales.to(bcast_device)
#                     zeros = zeros.to(bcast_device)
#                     bias_t = (module.bias.data.to(bcast_device) if module.bias is not None else torch.empty(0, device=bcast_device, dtype=w_float.dtype))
#                 else:
#                     q = torch.empty(w_shape, device=bcast_device, dtype=torch.uint8)
#                     scales = torch.empty((num_groups, 1), device=bcast_device, dtype=module.weight.dtype)
#                     zeros = torch.empty((num_groups, 1), device=bcast_device, dtype=module.weight.dtype)
#                     bias_t = torch.empty((w_shape[0],), device=bcast_device, dtype=module.weight.dtype) if module.bias is not None else torch.empty(0, device=bcast_device, dtype=module.weight.dtype)

#                 dist.broadcast(q, src=0)
#                 dist.broadcast(scales, src=0)
#                 dist.broadcast(zeros, src=0)
#                 if module.bias is not None:
#                     dist.broadcast(bias_t, src=0)
#             else:
#                 # single-process fallback
#                 q, scales, zeros = _quantize_tensor_int8(module.weight.data, q_group_size=group_size)
#                 bias_t = module.bias.data if module.bias is not None else None

#             new_layers = QPOETLinear(
#                 module.weight,
#                 (bias_t if module.bias is not None else None),
#                 num_bits=args.weight_bits,
#                 group_size=args.weight_group_size,
#                 stochastic_round=args.stochastic_round,
#                 device=bcast_device,                  # important for NCCL broadcast path
#                 dtype=getattr(module.weight, "dtype", None),
#                 prequantized=is_dist,                 # if dist: we already have q/scales/zeros
#                 int8_weight=q if is_dist else None,
#                 scales=scales if is_dist else None,
#                 zeros=zeros if is_dist else None,
#             )
#             model._modules[name] = new_layers

#     return model

# def replace_linear_with_poet(module: nn.Module, block_size: int, init_type: str, device=None, dtype=None) -> int:
#     def _convert(m: nn.Module, parent_name=''):
#         for name, child in list(m.named_children()):
#             full_name = f"{parent_name}.{name}" if parent_name else name
            
#             if isinstance(child, nn.Linear):
#                 # Skip embedding and lm_head layers
#                 if 'embed' in full_name.lower() or 'lm_head' in full_name.lower():
#                     continue
                    
#                 if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
#                     new_lin = POETLinear(
#                         in_features=child.in_features,
#                         out_features=child.out_features,
#                         bsz=block_size,
#                         bias=(child.bias is not None),
#                         device=device,
#                         dtype=dtype,
#                     )
#                     with torch.no_grad():
#                         if init_type == 'normalized':
#                             child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)
#                         new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))
#                         if child.bias is not None and new_lin.bias is not None:
#                             new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))
#                     setattr(m, name, new_lin)
#                 else:
#                     continue
#             else:
#                 _convert(child, full_name)
#     _convert(module)


def check_and_merge(model: nn.Module, iter_count=0, poet_reset_gap=4):
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # with torch.compiler.set_stance("force_eager"):
    with torch.compiler.set_stance("eager_then_compile"):
        for name, module in model.named_modules():
            if isinstance(module, (POETLinear, QPOETLinear)) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        # rank 0 does the merge + permutation update
                        module.merge_then_reinitialize()

                    # ensure all ranks get the exact same state
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


def get_grad_clipping_value(global_step, grad_clipping, warmup_steps, period_T, min_ratio=0.1, max_steps=2000):
    """
    Gradient clipping scheduler that linearly increases from min_ratio * grad_clipping 
    to grad_clipping over warmup_steps, repeating every period_T steps
    
    Args:
        global_step: Current training step
        grad_clipping: Maximum gradient clipping value
        warmup_steps: Number of steps to linearly increase clipping value
        period_T: Period after which the warmup cycle repeats
        min_ratio: Starting ratio of grad_clipping (default: 0.1)
        max_steps: Maximum number of steps to apply gradient clipping
    Returns:
        Current gradient clipping value
    """
    if global_step < period_T:
        return grad_clipping
    
    # Calculate position within the current cycle
    cycle_position = global_step % period_T

    if global_step > max_steps:
        return grad_clipping
    
    if cycle_position >= warmup_steps:
        return grad_clipping
        
    # Linear warmup from min_ratio * grad_clipping to grad_clipping
    warmup_factor = min_ratio + (1.0 - min_ratio) * (cycle_position / max(1, warmup_steps))
    return warmup_factor * grad_clipping



def _find_module_by_name_substr(model: nn.Module, name_substr: str):
    for n, m in model.named_modules():
        if name_substr in n:
            return n, m
    return None, None


def _blockdiag_matvec(blocks: torch.Tensor, vec: torch.Tensor, transpose: bool) -> torch.Tensor:
    # blocks: (r, b, b), vec: (r*b,)
    r, b, _ = blocks.shape
    v = vec.view(r, b)
    A = blocks.transpose(-1, -2) if transpose else blocks
    out = torch.einsum("rbc,rc->rb", A, v)
    return out.reshape(r * b)


@torch.no_grad()
def estimate_poet_delta_weff_spec(
    poet_module: nn.Module,
    oft_R_prev: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> float:
    """
    Estimates ||ΔW_eff||_2 where (row-space) W_eff^T = Rin @ W^T @ Rout.
    We ignore permutations (they are orthogonal and constant between merges).
    """
    device = poet_module.weight.device
    W = poet_module.weight.detach().to(device=device, dtype=compute_dtype)

    # Compute R blocks for prev / cur (keep ops happy by using module's dtype, then cast)
    R_out_prev, R_in_prev = get_weight_poet(
        oft_R_prev.to(device=device, dtype=poet_module.oft_R.dtype),
        poet_module.block_size,
        poet_module.rows,
        poet_module.cols,
        poet_module.r_out,
        poet_module.r_in,
    )
    R_out_cur, R_in_cur = get_weight_poet(
        poet_module.oft_R.detach().to(device=device, dtype=poet_module.oft_R.dtype),
        poet_module.block_size,
        poet_module.rows,
        poet_module.cols,
        poet_module.r_out,
        poet_module.r_in,
    )
    R_out_prev = R_out_prev.to(dtype=compute_dtype)
    R_in_prev = R_in_prev.to(dtype=compute_dtype)
    R_out_cur = R_out_cur.to(dtype=compute_dtype)
    R_in_cur = R_in_cur.to(dtype=compute_dtype)

    # M = Rin @ W^T @ Rout  (shape: in_features x out_features)
    M_prev = block_diag_lr_matmul(R_in_prev, W.t(), R_out_prev)
    # M_prev = M_prev.t()
    M_cur  = block_diag_lr_matmul(R_in_cur,  W.t(), R_out_cur)
    # M_cur = M_cur.t()

    dM = M_cur - M_prev

    # Spectral norm (largest singular value) — SVD-based, can be expensive per-step.
    sigma = torch.linalg.matrix_norm(dM, ord=2).item()

    return dM, float(sigma)