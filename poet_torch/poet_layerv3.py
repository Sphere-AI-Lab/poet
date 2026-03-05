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

def _quantize_tensor_int8(w, q_group_size=-1, n_bit=8):

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

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

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

def cayley_from_oft(oft_vec: torch.Tensor, block_size: int, rows: torch.Tensor, cols: torch.Tensor):
    # oft_vec: (r, n_elements)  ->  (r, b, b) orthogonal blocks
    Q = pytorch_skew_symmetric(oft_vec, block_size, rows, cols)
    return torch.ops.poet.cayley(Q)[0]


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
    scale: float = 1.0,
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    # Balance learning rates:
    # If r_out > r_in, scale > 1.
    # R_in (smaller) is multiplied by scale -> gradient boosted by scale.
    # R_out (larger) is divided by scale -> gradient dampened by scale.
    # scale = (r_out / r_in) ** 0.5
    # R_in = R_in * scale
    # R_out = R_out / scale

    # balance mode
    x = permute_x(x, perm_in_inv, perm_in)
    
    y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)
    # lower throughput but less memory (merge permute inside chain layer)
    # y = chain_layer_x_checkpoint_mem_o2(x, R_in, base_weight, base_bias, R_out, perm_in_inv, perm_in, block_size)
    # y = chain_layer_x_checkpoint_2lr(x, R_in, base_weight, base_bias, R_out, block_size)

    y = permute_x(y, perm_out, perm_out_inv)
    return y


class POETLinearV3(nn.Module):
    """
    POETv2 with an internal buffer `mode_id`:
      - mode_id == 0: train/apply ONLY R_out (left/output), treat R_in as identity (unused)
      - mode_id == 1: train/apply ONLY R_in  (right/input), treat R_out as identity (unused)

    Switching happens ONLY when you call merge (typically via `check_and_merge_v2`).
    """
    def __init__(self, in_features, out_features, bsz=256, bias=False, device=None, dtype=None, init_mode=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        self.register_buffer("mode_id", torch.tensor(init_mode, device=device, dtype=torch.int32))

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.r_in = in_features // bsz
        self.r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2

        self.oft_R_out = nn.Parameter(torch.zeros((self.r_out, n_elements), device=device, dtype=dtype))
        self.oft_R_in  = nn.Parameter(torch.zeros((self.r_in,  n_elements), device=device, dtype=dtype))

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        group_size = 1
        perm_in = torch.randperm(in_features // group_size, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features // group_size, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    def random_init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.oft_R_out, std=1e-3)
        nn.init.normal_(self.oft_R_in, std=1e-3)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def perform_permutation(self) -> None:
        W = self.weight
        Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        self.weight.detach().copy_(Wp)

    def update_permutation(self):
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in))
        perm_out = torch.randperm(self.out_features, device=device)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out))
        self.perform_permutation()

    @torch.no_grad()
    def merge_then_reinitialize(self):
        """
        side="out": merge only R_out into weight, then reset oft_R_out to 0
        side="in" : merge only R_in  into weight, then reset oft_R_in  to 0
        """
        b = self.block_size
        dev, dt = self.weight.device, self.weight.dtype

        if int(self.mode_id.item()) == 0:
            R_out = cayley_from_oft(self.oft_R_out, b, self.rows, self.cols)        # (r_out,b,b)
            R_in = torch.eye(b, device=dev, dtype=dt).repeat(self.r_in, 1, 1)      # identity blocks
        else:
            R_in = cayley_from_oft(self.oft_R_in, b, self.rows, self.cols)         # (r_in,b,b)
            R_out = torch.eye(b, device=dev, dtype=dt).repeat(self.r_out, 1, 1)    # identity blocks

        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        self.weight.detach().copy_(tmp.t())

        if int(self.mode_id.item()) == 0:
            self.oft_R_out.zero_()
        else:
            self.oft_R_in.zero_()

        self.update_permutation()
        self.mode_id.fill_(1 - int(self.mode_id.item()))

    def forward(self, x):
        R = torch.cat([self.oft_R_in, self.oft_R_out], dim=0)
        y = forward_core(
            x=x,
            R=R,
            rows=self.rows,
            cols=self.cols,
            perm_in_inv=self.perm_in_inv,
            perm_in=self.perm_in,
            perm_out=self.perm_out,
            perm_out_inv=self.perm_out_inv,
            block_size=self.block_size,
            r_in=self.r_in,
            r_out=self.r_out,
            base_weight=self.weight,
            base_bias=self.bias,
        )
        return y


@torch.no_grad()
def check_and_merge_v3(model: nn.Module, iter_count: int, gap: int):
    if gap <= 0 or iter_count <= 0 or (iter_count % gap != 0):
        return

    base = model.module if hasattr(model, "module") else model
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    for m in base.modules():
        if isinstance(m, POETLinearV3):
            if rank == 0:
                m.merge_then_reinitialize()

            if is_dist:
                dist.broadcast(m.oft_R_out.data, src=0)
                dist.broadcast(m.oft_R_in.data, src=0)
                dist.broadcast(m.mode_id, src=0)
                dist.broadcast(m.weight.data, src=0)
                if m.bias is not None:
                    dist.broadcast(m.bias.data, src=0)
                dist.broadcast(m.perm_in, src=0)
                dist.broadcast(m.perm_in_inv, src=0)
                dist.broadcast(m.perm_out, src=0)
                dist.broadcast(m.perm_out_inv, src=0)

    if is_dist:
        dist.barrier()



def replace_linear_with_poet_v3(module: nn.Module, block_size: int, init_type: str, device=None, dtype=None,
                            mem_efficient_mode=False, neurips_version=False, v2: bool=False) -> int:
    def _convert(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and 'lm_head' not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    new_lin = POETLinearV3(
                        in_features=child.in_features, 
                        out_features=child.out_features, 
                        bsz=block_size,
                        bias=(child.bias is not None), 
                        device=device, 
                        dtype=dtype,
                    )

                    with torch.no_grad():
                        if init_type == "normalized":
                            child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)
                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))
                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))

                    setattr(m, name, new_lin)
                else:
                    raise ValueError(f"Layer {name} not divisible by block_size={block_size}")
            else:
                _convert(child)

    _convert(module)
    return 0