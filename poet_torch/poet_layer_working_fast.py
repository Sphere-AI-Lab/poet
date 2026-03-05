import torch
from torch import nn
import torch.distributed as dist
from typing import Optional
from .poet_ops import PermutationFunction

import math

def permute_x(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)

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
    # R_cat = torch.ops.poet.cayley(Q_skew_cat)[0]
    R_cat = cayley_batch(Q_skew_cat)
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
    # Ro: torch.Tensor,
    # Ri: torch.Tensor,
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
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    x = permute_x(x, perm_in_inv, perm_in)
    
    # y = chain_layer_x(x, R_in, base_weight, base_bias, R_out, block_size) # y = x @ R @ W @ P
    # y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)
    y = chain_layer_x_pytorch(x, R_in, base_weight, base_bias, R_out, block_size)

    y = permute_x(y, perm_out, perm_out_inv)
    return y



class POETLinear(nn.Module):
    def __init__(self, in_features, out_features, bsz=256, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        # Basic linear layer parameters
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False, 
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable skew-params per block
        r_in = in_features // bsz
        r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        # Param tensors can be any square; we skew them inside forward
        # self.R_left = nn.Parameter(torch.zeros((r_in, n_elements), **factory_kwargs))
        # self.R_right = nn.Parameter(torch.zeros((r_out, n_elements), **factory_kwargs))
        self.oft_R = nn.Parameter(torch.zeros((r_in + r_out, n_elements), **factory_kwargs))
        self.r_in = r_in
        self.r_out = r_out

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        # self.register_buffer('rows', rows.to(torch.int32))
        # self.register_buffer('cols', cols.to(torch.int32))
        self.rows = rows.to(torch.int32)
        self.rows.requires_grad = False
        self.cols = cols.to(torch.int32)
        self.cols.requires_grad = False

        group_size = 1
        perm_in = torch.randperm(in_features // group_size, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features // group_size, device=device, dtype=torch.int32)
        # self.register_buffer('perm_in', perm_in)
        # self.register_buffer('perm_out', perm_out)
        # self.register_buffer('perm_in_inv', torch.argsort(perm_in).to(torch.int32))
        # self.register_buffer('perm_out_inv', torch.argsort(perm_out).to(torch.int32))
        self.perm_in = perm_in
        self.perm_in.requires_grad = False
        self.perm_out = perm_out
        self.perm_out.requires_grad = False
        self.perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        self.perm_in_inv.requires_grad = False
        self.perm_out_inv = torch.argsort(perm_out).to(torch.int32)
        self.perm_out_inv.requires_grad = False

        # self.reset_parameters()

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

    def forward(self, x):
        x = forward_core(x, self.oft_R, self.block_size, self.rows, self.cols, 
                self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv, 
                self.r_in, self.r_out, self.weight, self.bias)
        return x


def replace_linear_with_poet(module: nn.Module, block_size: int, init_type: str, device=None, dtype=None) -> int:
    # replaced = 0
    def _convert(m: nn.Module):
        # nonlocal replaced
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    new_lin = POETLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bsz=block_size,
                        bias=(child.bias is not None),
                        device=device,
                        dtype=dtype,
                    )
                    with torch.no_grad():
                        if init_type == 'normalized':
                            child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)
                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))
                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))
                    setattr(m, name, new_lin)
                    # replaced += 1
                else:
                    # skip non-divisible layers
                    continue
            else:
                _convert(child)
    _convert(module)
    # return replaced


def check_and_merge(model: nn.Module, iter_count=0, poet_reset_gap=4):
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    with torch.compiler.set_stance("force_eager"):
        for _, module in model.named_modules():
            if isinstance(module, POETLinear) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        # rank 0 does the merge + permutation update
                        module.merge_then_reinitialize()

                    # ensure all ranks get the exact same state
                    torch.distributed.broadcast(module.oft_R.data, src=0)
                    torch.distributed.broadcast(module.weight.data, src=0)
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
    # Calculate position within the current cycle
    cycle_position = global_step % period_T

    if global_step > max_steps:
        return grad_clipping
    
    if cycle_position >= warmup_steps:
        return grad_clipping
        
    # Linear warmup from min_ratio * grad_clipping to grad_clipping
    warmup_factor = min_ratio + (1.0 - min_ratio) * (cycle_position / max(1, warmup_steps))
    return warmup_factor * grad_clipping