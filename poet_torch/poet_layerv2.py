import torch
from torch import nn
import math
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

# Reuse core utilities + loads poet ops (torch.ops.poet.*)
from .poet_layer import permute_x, pytorch_skew_symmetric, torch_bmm, block_diag_lr_matmul


def cayley_from_oft(oft_vec: torch.Tensor, block_size: int, rows: torch.Tensor, cols: torch.Tensor):
    # oft_vec: (r, n_elements)  ->  (r, b, b) orthogonal blocks
    Q = pytorch_skew_symmetric(oft_vec, block_size, rows, cols)
    return torch.ops.poet.cayley(Q)[0]


@torch.compile(fullgraph=True)
def forward_core_in_only(
    x: torch.Tensor,
    R: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    block_size: int,
    r_in: int,
    r_out: int,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
):
    R_in = cayley_from_oft(R, block_size, rows, cols)
    # R_in = pytorch_skew_symmetric(R, block_size, rows, cols)
    # x = permute_x(x, perm_in_inv, perm_in)
    # y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)

    B, S, Din = x.shape
    N = B * S
    rin = R.size(0)
    bsz = block_size

    # x @ Rin
    xb = x.view(N, rin, bsz)                 # [N, rin, b]
    xb_r = xb.transpose(0, 1)                # [rin, N, b]
    xR_r = torch.bmm(xb_r, R_in)              # [rin, N, b] @ [rin, b, b] = [rin, N, b]
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # xR @ W^T (+b)
    yb_flat = xR @ base_weight.t()                     # [N, rout*bsz]
    if base_bias is not None:
        yb_flat = yb_flat + base_bias

    y = yb_flat.view(B, S, r_out * bsz)          # [B, S, r_out * bsz]

    # y = permute_x(y, perm_out, perm_out_inv)
    return y


@torch.compile(fullgraph=True)
def forward_core_out_only(
    x: torch.Tensor,
    R: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    block_size: int,
    r_in: int,
    r_out: int,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
):
    R_out = cayley_from_oft(R, block_size, rows, cols)
    # R_out = pytorch_skew_symmetric(R, block_size, rows, cols)
    # x = permute_x(x, perm_in_inv, perm_in)
    # y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)

    B, S, Din = x.shape
    N = B * S
    rout = R.size(0)
    bsz = block_size

    # x @ W^T
    y = x @ base_weight.t()                     # [N, rout*bsz]
    if base_bias is not None:
        y = y + base_bias

    # yb_flat @ Rout
    y = y.view(N, rout, bsz)                 # [N, rout, b]
    y_r = y.transpose(0, 1)                  # [rout, N, b]
    y = torch.bmm(y_r, R_out)                # [rout, N, b] @ [rout, b, b] = [rout, N, b]
    y = y.transpose(0, 1).reshape(B, S, rout * bsz)

    # y = permute_x(y, perm_out, perm_out_inv)
    return y


class POETLinearV2(nn.Module):
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
        # Keep same permute structure as POET
        x = permute_x(x, self.perm_in_inv, self.perm_in)

        if int(self.mode_id.item()) == 0:
            y = forward_core_out_only(
                x=x,
                R=self.oft_R_out,
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

        else:
            y = forward_core_in_only(
                x=x,
                R=self.oft_R_in,
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

        y = permute_x(y, self.perm_out, self.perm_out_inv)
        return y


@torch.no_grad()
def check_and_merge_v2(model: nn.Module, iter_count: int, gap: int):
    if gap <= 0 or iter_count <= 0 or (iter_count % gap != 0):
        return

    base = model.module if hasattr(model, "module") else model
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    for m in base.modules():
        if isinstance(m, POETLinearV2):
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



def replace_linear_with_poet_v2(module: nn.Module, block_size: int, init_type: str, device=None, dtype=None,
                            mem_efficient_mode=False, neurips_version=False, v2: bool=False) -> int:
    def _convert(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and 'lm_head' not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    new_lin = POETLinearV2(
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