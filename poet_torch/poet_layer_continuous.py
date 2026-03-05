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

    R_cat = torch.ops.poet.cayley(Q_skew_cat)[0]

    # R_cat = torch.matrix_exp(Q_skew_cat.float()).to(Q_skew_cat.dtype)               # expm per block
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
    
    y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)
    # lower throughput but less memory (merge permute inside chain layer)
    # y = chain_layer_x_checkpoint_mem_o2(x, R_in, base_weight, base_bias, R_out, perm_in_inv, perm_in, block_size)
    # y = chain_layer_x_checkpoint_2lr(x, R_in, base_weight, base_bias, R_out, block_size)

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


@torch.no_grad()
def soft_merge_poet_step(model, alpha=0.01):
    """
    Performs the continuous absorption step.
    Call this every K steps (e.g., K=10) with small alpha (e.g. 0.01).
    """
    for name, module in model.named_modules():
        
        # Detect your Poet layer (customize this check)
        if hasattr(module, "R_in_param") and hasattr(module, "fixed_weight"):
            
            # --- 1. PREPARE PARAMETERS ---
            # Assume shapes:
            # R_in_param:  [Num_Blocks_In,  b, b]  (Skew-symmetric)
            # R_out_param: [Num_Blocks_Out, b, b]  (Skew-symmetric)
            # fixed_weight: [D_out, D_in]          (Dense)
            
            A_in = module.R_in_param
            A_out = module.R_out_param
            W = module.fixed_weight
            
            # --- 2. COMPUTE DELTA ROTATIONS ---
            # We take a tiny slice (alpha) of the current rotation state
            delta_R_in  = cayley_map(alpha * A_in)
            delta_R_out = cayley_map(alpha * A_out)
            
            # --- 3. ABSORB INTO W (LEFT SIDE) ---
            # Operation: W_new = BlockDiag(delta_R_in) @ W
            # Reshape W to match blocks: [D_out, D_in] -> [Num_Blocks, Block_Size, D_in]
            
            N_in, b_in, _ = delta_R_in.shape
            # View W as a stack of blocks
            W_view_in = W.view(N_in, b_in, -1)
            
            # Apply rotation to each block
            # (N, b, b) @ (N, b, rest) -> (N, b, rest)
            W_updated_left = torch.matmul(delta_R_in, W_view_in)
            
            # Flatten back for the next step
            W_temp = W_updated_left.reshape(W.shape)
            
            # --- 4. ABSORB INTO W (RIGHT SIDE) ---
            # Operation: W_new = W_temp @ BlockDiag(delta_R_out)
            # Note: Applying R on the right rotates the COLUMNS (dimension 1).
            
            N_out, b_out, _ = delta_R_out.shape
            # View W as [D_out, Num_Blocks, Block_Size]
            W_view_out = W_temp.view(-1, N_out, b_out)
            
            # We want: row @ R. 
            # In standard matmul (batch @ matrix), this is equivalent to:
            # (Batch, 1, b) @ (Batch, b, b) -> (Batch, 1, b)
            # Or simpler: (..., b) @ (b, b)
            
            # delta_R_out is [N, b, b]. 
            # We need to treat 'D_out' as the batch dimension for this operation? 
            # No, 'N_out' is the batch dimension aligning with delta_R_out.
            
            # Correct logic:
            # Input:  [Rest, N, b_in]
            # Rotation: [N, b_in, b_out] (Technically b_in=b_out=b)
            # We need to broadcast the matmul over 'Rest'.
            # Einstein summation is the safest, bug-free way:
            # r = rest (D_out), n = num_blocks, i = old_dim, j = new_dim
            # W: [r, n, i], R: [n, i, j] -> Output: [r, n, j]
            
            W_final_view = torch.einsum('r n i, n i j -> r n j', W_view_out, delta_R_out)
            
            # Flatten back to [D_out, D_in]
            W_final = W_final_view.reshape(W.shape)
            
            # --- 5. COMMIT UPDATE ---
            module.fixed_weight.copy_(W_final)
            
            # --- 6. DECAY GENERATORS (The "Soft Reset") ---
            # Shrink the A parameters so they don't grow infinitely
            A_in.mul_(1.0 - alpha)
            A_out.mul_(1.0 - alpha)


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

    def merge_then_reinitialize(self) -> None:
        # with torch.no_grad():
        alpha = 0.001
        R_out, R_in = get_weight_poet(self.oft_R * alpha, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

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

        # self.oft_R.zero_()
        self.oft_R.mul_(1.0 - alpha)
        # self.update_permutation()

    def forward(self, x):
        # scale = torch.exp(torch.tanh(self.log_scale) * 0.2).to(x.device)
        x = forward_core(x, self.oft_R, self.block_size, self.rows, self.cols, 
                self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv, 
                self.r_in, self.r_out, self.weight, self.bias, self.mem_efficient_mode)
        return x


def replace_linear_with_poet_continuous(module: nn.Module, block_size: int, init_type: str, device=None, dtype=None, 
                        mem_efficient_mode=False, neurips_version=False, v2: bool=False) -> int:
    def _convert(m: nn.Module, v2: bool=False):
        # nonlocal replaced
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and 'lm_head' not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
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
                            child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)
                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))
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
            new_layers = QPOETLinear(module.weight, bias_data, num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)
            model._modules[name] = new_layers

    return model

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


def check_and_merge_continuous(model: nn.Module, iter_count=0, poet_reset_gap=4):
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # with torch.compiler.set_stance("force_eager"):
    with torch.compiler.set_stance("eager_then_compile"):
        for _, module in model.named_modules():
            if isinstance(module, POETLinear) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        # rank 0 does the merge + permutation update
                        # x = torch.randn(10, module.in_features, device=module.weight.device)
                        # y0 = module.forward(x)
                        module.merge_then_reinitialize()
                        # y1 = module.forward(x)
                        # print("rel_err", (y1 - y0).norm() / (y0.norm() + 1e-12))
                        # assert torch.allclose(y0, y1)

                    # ensure all ranks get the exact same state
                    torch.distributed.broadcast(module.oft_R.data, src=0)
                    torch.distributed.broadcast(module.weight.data, src=0)
                    if module.bias is not None:
                        torch.distributed.broadcast(module.bias.data, src=0)
                    # torch.distributed.broadcast(module.perm_in, src=0)
                    # torch.distributed.broadcast(module.perm_in_inv, src=0)
                    # torch.distributed.broadcast(module.perm_out, src=0)
                    # torch.distributed.broadcast(module.perm_out_inv, src=0)

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


def thomson_random_project_loss_unrolled(weight, pd=40, pn=20, pnd=0):
    """Thomson loss with random projections for intermediate layers"""
    n_input = weight.shape[1] # in_features
    n_filt = weight.shape[0] # out_features
    
    pd1 = pd
    pd2 = n_input
    
    # Calculate number of projections
    if pnd == 0:
        total_p = pn
    else:
        total_p = n_filt // pnd
        
    total_loss = 0
    
    # Generate multiple random projections
    for i in range(total_p):
        # filt = weight.view(-1, n_filt)
        filt = weight.t()
        
        # Create random projection matrix (not learnable)
        p = torch.normal(
            mean=0.0,
            std=1.0,
            size=(pd1, pd2),
            device=filt.device,
            requires_grad=False,
            generator=torch.Generator(device=filt.device).manual_seed(n_input + i),
            dtype=filt.dtype
        )
        
        # Project filters
        projected_filt = torch.mm(p, filt)
        
        # Add negative versions
        filt_neg = -projected_filt
        projected_filt = torch.cat((projected_filt, filt_neg), dim=1)
        n_filt_doubled = 2 * n_filt
        
        # Calculate cosine similarities
        filt_norm = torch.sqrt(torch.sum(projected_filt * projected_filt, dim=0, keepdim=True) + 1e-4)
        norm_mat = torch.mm(filt_norm.t(), filt_norm)
        inner_pro = torch.mm(projected_filt.t(), projected_filt)
        cos_sim = inner_pro / norm_mat
        
        # Calculate repulsion loss
        # cross_terms = 2.0 - 2.0 * cos_sim + torch.eye(n_filt_doubled, device=filt.device)
        cross_terms = 2.0 - 2.0 * cos_sim
        cross_terms.diagonal(dim1=-2, dim2=-1).add_(1.0)
        final = cross_terms.pow(-1)
        final = final.triu(diagonal=1)
        cnt = n_filt_doubled * (n_filt_doubled - 1) / 2.0
        loss = final.sum() / cnt
        
        total_loss += loss
        
    return total_loss / total_p


def thomson_random_project_loss(weight, pd=40, pn=20, pnd=0):
    """Thomson loss with random projections for intermediate layers"""
    n_input = weight.shape[1] # in_features
    n_filt = weight.shape[0] # out_features
    
    pd1 = pd
    pd2 = n_input
    
    # Calculate number of projections
    if pnd == 0:
        total_p = pn
    else:
        total_p = n_filt // pnd
        
    total_loss = 0
    
    # Helper function for checkpointing
    def run_projection_loss(w, seed_val):
        # Re-create generator for deterministic behavior during re-computation
        gen = torch.Generator(device=w.device).manual_seed(seed_val)
        
        filt = w.t() # Use .t() as corrected
        
        # Create random projection matrix
        p = torch.normal(
            mean=0.0,
            std=1.0,
            size=(pd1, pd2),
            device=filt.device,
            requires_grad=False,
            generator=gen,
            dtype=filt.dtype
        )
        
        # Project filters
        projected_filt = torch.mm(p, filt)
        
        # Add negative versions
        filt_neg = -projected_filt
        projected_filt = torch.cat((projected_filt, filt_neg), dim=1)
        n_filt_doubled = 2 * n_filt
        
        # Calculate cosine similarities
        filt_norm = torch.sqrt(torch.sum(projected_filt * projected_filt, dim=0, keepdim=True) + 1e-4)
        norm_mat = torch.mm(filt_norm.t(), filt_norm)
        inner_pro = torch.mm(projected_filt.t(), projected_filt)
        cos_sim = inner_pro / norm_mat
        
        # Calculate repulsion loss
        cross_terms = 2.0 - 2.0 * cos_sim
        cross_terms.diagonal(dim1=-2, dim2=-1).add_(1.0)
        final = cross_terms.pow(-1)
        final = final.triu(diagonal=1)
        cnt = n_filt_doubled * (n_filt_doubled - 1) / 2.0
        loss = final.sum() / cnt
        
        return loss

    # Generate multiple random projections
    for i in range(total_p):
        # Use checkpoint to save memory
        # We pass a seed so the random matrix 'p' is identical in forward and backward passes
        seed = n_input + i
        
        # Checkpointing requires the input (weight) to have requires_grad=True
        if weight.requires_grad:
             curr_loss = checkpoint(run_projection_loss, weight, seed, use_reentrant=False)
        else:
             curr_loss = run_projection_loss(weight, seed)
             
        total_loss = total_loss + curr_loss
        
    return total_loss / total_p


def optimize_layer_mhe(
    layer_weight,
    n_steps=5000, 
    lr=0.1, 
    momentum=0.9,
    print_every=500
):
    with torch.enable_grad():
        weight = nn.Parameter(layer_weight.clone().detach(), requires_grad=True).to(layer_weight.device)

        optimizer = torch.optim.SGD([weight], lr=lr, momentum=momentum)
        
        # Initial loss calculation
        with torch.no_grad():
            train_loss = thomson_random_project_loss(weight)
            init_mhe_loss = mhe_loss(weight)

        for step in range(n_steps):
            optimizer.zero_grad()
            train_loss = thomson_random_project_loss(weight)
            train_loss.backward()
            optimizer.step()
            
            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    val_loss = mhe_loss(weight)
                    logger.info(f'Step [{step+1}/{n_steps}], '
                          f'Train Loss: {train_loss.item():.4f}, '
                          f'MHE Loss: {val_loss.item():.4f}')
            
            # train_losses.append(train_loss.item())

        final_mhe_loss = mhe_loss(weight)
        logger.info(f"Initial MHE loss: {init_mhe_loss:.8f}, Final MHE loss: {final_mhe_loss:.8f}")
    
    result = weight.detach().clone()
    del optimizer, weight, train_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


def calculate_total_mhe(model, target_modules_list=["attn", "mlp"]):
    mhe_losses = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(key in name for key in target_modules_list):
                weight = module.weight.data
                loss = mhe_loss(weight)
                mhe_losses.append(loss.cpu().item())
    return float(np.sum(mhe_losses))


def mhe_loss(filt):
    n_filt, _ = filt.shape
    filt = torch.transpose(filt, 0, 1)
    filt_neg = filt * (-1)
    filt = torch.cat((filt, filt_neg), dim=1)
    n_filt *= 2

    filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
    norm_mat = torch.matmul(filt_norm.t(), filt_norm)
    inner_pro = torch.matmul(filt.t(), filt)
    inner_pro /= norm_mat

    cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.ones(n_filt, device=filt.device)))
    final = torch.pow(cross_terms, -0.5 * torch.ones_like(cross_terms))
    final -= torch.tril(final)
    cnt = n_filt * (n_filt - 1) / 2.0
    MHE_loss = torch.sum(final) / cnt
    return MHE_loss


def mhe_optimized_init(model):    
    with torch.no_grad():
        # First just normalize and calculate MHE loss
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            module.weight.data = module.weight.data / torch.norm(module.weight.data, dim=1, keepdim=True)
        
    # Calculate MHE loss after normalization only
    # normalized_mhe_loss = calculate_total_mhe(model)
    # print(f"MHE loss after normalization: {normalized_mhe_loss:.8f}")

    target_layers = []
    for module_name, module in model.named_modules():
        if "lm_head" in module_name:
            continue
        if isinstance(module, nn.Linear):
            target_layers.append((module_name, module))

    # Optimize each target linear layer in parallel
    import concurrent.futures
    import contextlib

    # Detect available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(num_gpus)]
    else:
        num_gpus = 0
        devices = ['cpu']

    def process_layer(args):
        module_name, module, target_device_str = args
        
        # Target device where optimization will happen
        target_device = torch.device(target_device_str)
        
        # Use a separate stream for each thread to allow kernel overlap on GPU
        if target_device.type == 'cuda':
            stream = torch.cuda.Stream(device=target_device)
            ctx = torch.cuda.stream(stream)
        else:
            stream = None
            ctx = contextlib.nullcontext()

        with ctx:
            # Move weight to target device for optimization
            # We detach to avoid autograd tracking on the original parameter
            weight = module.weight.detach().to(target_device).contiguous()
            weight = weight / weight.norm(dim=1, keepdim=True)
            
            # optimize_layer_mhe runs in the current stream context on target_device
            # Note: optimize_layer_mhe logs to stdout, so output might be interleaved
            optimized_fp32 = optimize_layer_mhe(weight.to(torch.float32)) 

            # Copy result back to the module's original device
            module.weight.data.copy_(optimized_fp32.to(module.weight.device, dtype=module.weight.dtype))
            
            # Cleanup to save memory for other workers
            del weight, optimized_fp32
        
        # Wait for this stream to finish
        if stream:
            stream.synchronize()
        
        return module_name

    # Adjust max_workers based on your VRAM and number of devices. 
    # Each worker creates full-size gradients and optimizer states.
    workers_per_device = 2 if num_gpus > 0 else 4
    max_workers = max(1, num_gpus * workers_per_device)
    if num_gpus == 0: max_workers = 4
    
    print(f"Optimizing layers with {max_workers} parallel workers on {len(devices)} devices ({devices})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks, assigning devices round-robin
        futures = []
        for i, item in enumerate(target_layers):
            assigned_device = devices[i % len(devices)]
            futures.append(executor.submit(process_layer, (*item, assigned_device)))
        
        # Process results as they complete to update progress bar
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(target_layers), desc="Optimizing layers [MHE]"):
            f.result() # Check for exceptions

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Calculate MHE loss after optimization
    # optimized_mhe_loss


def mhe_worker_process(model, worker_id, total_workers, save_dir):
    """
    Worker function for distributed MHE initialization via job scheduler (e.g. Condor).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Moving full model to CPU to free VRAM...")
    model.cpu()

    # Identify all target layers
    target_layers = []
    for module_name, module in model.named_modules():
        if "lm_head" in module_name:
            continue
        if isinstance(module, nn.Linear):
            target_layers.append((module_name, module))
    
    # Determine which layers this worker is responsible for
    my_layers = [item for i, item in enumerate(target_layers) if i % total_workers == worker_id]
    
    print(f"Worker {worker_id}/{total_workers} responsible for {len(my_layers)} layers.")
    
    if len(my_layers) == 0:
        print("No layers assigned to this worker.")
        return

    # Extract the weights we need and detach them to break the graph
    # We store them as (name, tensor_on_cpu)
    layers_to_process = []
    for name, module in my_layers:
        # Detach and move to CPU immediately
        weight_cpu = module.weight.detach().cpu().clone()
        layers_to_process.append((name, weight_cpu))

    # Now aggressively delete the model to free all resources
    print("Deleting full model to free memory...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for module_name, weight_cpu in layers_to_process:
        print(f"Processing layer: {module_name} with shape {weight_cpu.shape}")
        
        # 1. Move to GPU for optimization
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Move to device and normalize
        weight = weight_cpu.to(device).contiguous()
        weight = weight / weight.norm(dim=1, keepdim=True)
        
        # 2. Optimize (in FP32)
        optimized_fp32 = optimize_layer_mhe(weight.to(torch.float32))
        
        # 3. Save
        safe_name = module_name.replace(".", "_")
        save_path = os.path.join(save_dir, f"{safe_name}.pt")
        
        torch.save(optimized_fp32.cpu(), save_path)
        print(f"Saved optimized weights to {save_path}")
        
        # Cleanup loop variables
        del weight, optimized_fp32
        torch.cuda.empty_cache()
        gc.collect()