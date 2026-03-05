import torch
import triton
import triton.language as tl

from typing import Callable, Optional, Union, Tuple



######################## Chain Layer Checkpoint Slow Quantized ########################

@torch.library.custom_op("poet::chain_layer_checkpoint_mem_o2_q8", mutates_args=())
def chain_layer_checkpoint_mem_o2_q8(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_q: torch.Tensor,        # int8 quantized weight
    W_scales: torch.Tensor,   # scales for dequantization
    W_zeros: torch.Tensor,    # zeros for dequantization
    group_size: int,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    bsz: int,
) -> torch.Tensor:
    x = x[..., perm_in_inv]
    
    B, S, Din = x.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # Dequantize weight on-the-fly
    W = W_q.to(x.dtype).reshape(-1, group_size)
    W = (W - W_zeros.to(x.dtype)) * W_scales.to(x.dtype)
    W = W.reshape(W_q.shape)

    # x @ Rin
    xb = x.view(N, rin, bsz)                 # [N, rin, b]
    xb_r = xb.transpose(0, 1)                # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)              # [rin, N, b]
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # xR @ W^T (+b)
    yb_flat = xR @ W.t()                     # [N, rout*bsz]
    if b is not None:
        yb_flat = yb_flat + b

    # yb_flat @ Rout
    yb = yb_flat.view(N, rout, bsz)          # [N, rout, b]
    yb_r = yb.transpose(0, 1)                # [rout, N, b]
    y = torch.bmm(yb_r, Rout)                # [rout, N, b]
    y = y.transpose(0, 1).reshape(B, S, rout * bsz)

    y = y[..., perm_out]

    return y

def chain_layer_checkpoint_mem_o2_q8_setup_context(ctx, inputs, output):
    x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz = inputs
    # _, yb_r = output
    ctx.save_for_backward(x, Rin, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv)
    ctx.W_q = W_q
    ctx.W_scales = W_scales
    ctx.W_zeros = W_zeros
    ctx.group_size = group_size
    ctx.b = b
    ctx.bsz = bsz

def chain_layer_checkpoint_mem_o2_q8_backward(ctx, g_orig):
    x_orig, Rin, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv = ctx.saved_tensors
    W_q = ctx.W_q
    W_scales = ctx.W_scales
    W_zeros = ctx.W_zeros
    group_size = ctx.group_size
    b = ctx.b
    bsz = ctx.bsz

    B, S, _ = g_orig.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # Dequantize weight on-the-fly
    W = W_q.to(g_orig.dtype).reshape(-1, group_size)
    W = (W - W_zeros.to(g_orig.dtype)) * W_scales.to(g_orig.dtype)
    W = W.reshape(W_q.shape)

    # Un-permute incoming gradient
    g = g_orig[..., perm_out_inv]
    del g_orig

    # Recompute permuted x
    x = x_orig[..., perm_in_inv]
    del x_orig

    # --- Phase 1: grad_Rout ---
    xb_r = x.view(N, rin, bsz).transpose(0, 1)
    xR_r = torch.bmm(xb_r, Rin)
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)
    del xR_r

    if b is not None:
        yb = (xR @ W.t() + b).view(N, rout, bsz)
    else:
        yb = (xR @ W.t()).view(N, rout, bsz)
    del xR

    g_t = g.view(N, rout, bsz).transpose(0, 1)
    grad_Rout = torch.bmm(yb.transpose(0, 1).transpose(1, 2), g_t)
    del yb

    # --- Phase 2: g_xR ---
    g_yb_t = torch.bmm(g_t, Rout.transpose(1, 2))
    del g_t, g
    g_yb = g_yb_t.transpose(0, 1).reshape(N, rout * bsz)
    del g_yb_t
    g_xR = g_yb @ W
    del g_yb

    # --- Phase 3: grad_Rin ---
    g_xR_3 = g_xR.view(N, rin, bsz).transpose(0, 1)
    grad_Rin = torch.bmm(xb_r.transpose(1, 2), g_xR_3)
    del xb_r, x

    # --- Phase 4: grad_x ---
    g_xb_r = torch.bmm(g_xR_3, Rin.transpose(1, 2))
    del g_xR, g_xR_3
    grad_x = g_xb_r.transpose(0, 1).reshape(B, S, rin * bsz)
    del g_xb_r

    grad_x = grad_x[..., perm_in]

    # 13 inputs: x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz
    return grad_x, grad_Rin, None, None, None, None, None, grad_Rout, None, None, None, None, None

@chain_layer_checkpoint_mem_o2_q8.register_fake
def _(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz):
    B, S, Din = x.shape
    # rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S
    Dout, Din = W_q.shape
    shape0 = (rout, N, bsz)
    # shape1 = (N, Dout)
    y = x.new_empty((B, S, Dout), device=x.device, dtype=x.dtype)
    # y0 = x.new_empty(shape0).transpose(0, 1).reshape(B, S, Dout)
    # y1 = x.new_empty(shape1).view(N, rout, bsz).transpose(0, 1)
    return y # y0, y1

torch.library.register_autograd(
    "poet::chain_layer_checkpoint_mem_o2_q8",
    chain_layer_checkpoint_mem_o2_q8_backward,
    setup_context=chain_layer_checkpoint_mem_o2_q8_setup_context
)



######################## Chain Layer Checkpoint Fast ########################

@torch.library.custom_op("poet::chain_layer_checkpoint_q8", mutates_args=())
def chain_layer_checkpoint_q8(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_q: torch.Tensor,        # int8 quantized weight
    W_scales: torch.Tensor,   # scales for dequantization
    W_zeros: torch.Tensor,    # zeros for dequantization
    group_size: int,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    bsz: int,
) -> torch.Tensor:
    B, S, Din = x.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # Dequantize weight on-the-fly
    W = W_q.to(x.dtype).reshape(-1, group_size)
    W = (W - W_zeros.to(x.dtype)) * W_scales.to(x.dtype)
    W = W.reshape(W_q.shape)

    # x @ Rin
    xb = x.view(N, rin, bsz)                 # [N, rin, b]
    xb_r = xb.transpose(0, 1)                # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)              # [rin, N, b]
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # xR @ W^T (+b)
    yb_flat = xR @ W.t()                     # [N, rout*bsz]
    if b is not None:
        yb_flat = yb_flat + b

    # yb_flat @ Rout
    yb = yb_flat.view(N, rout, bsz)          # [N, rout, b]
    yb_r = yb.transpose(0, 1)                # [rout, N, b]
    y = torch.bmm(yb_r, Rout)                # [rout, N, b]
    y = y.transpose(0, 1).reshape(B, S, rout * bsz)

    return y

def chain_layer_checkpoint_q8_setup_context(ctx, inputs, output):
    x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, bsz = inputs
    # _, yb_r = output
    ctx.save_for_backward(x, Rin, Rout)
    ctx.W_q = W_q
    ctx.W_scales = W_scales
    ctx.W_zeros = W_zeros
    ctx.group_size = group_size
    ctx.b = b
    ctx.bsz = bsz

def chain_layer_checkpoint_q8_backward(ctx, g: torch.Tensor) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    x, Rin, Rout = ctx.saved_tensors
    W_q = ctx.W_q
    W_scales = ctx.W_scales
    W_zeros = ctx.W_zeros
    group_size = ctx.group_size
    b = ctx.b
    bsz = ctx.bsz

    B, S, _ = g.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # Dequantize weight on-the-fly in backward (trade compute for memory)
    W = W_q.to(g.dtype).reshape(-1, group_size)
    W = (W - W_zeros.to(g.dtype)) * W_scales.to(g.dtype)
    W = W.reshape(W_q.shape)

    # grad wrt Rout and yb
    g_t = g.view(N, rout, bsz).transpose(0, 1)      # [rout, N, b]
    Rt_t = Rout.transpose(1, 2)                     # [rout, b, b]
    g_yb_t = torch.bmm(g_t, Rt_t)                   # [rout, N, b]
    g_yb = g_yb_t.transpose(0, 1).reshape(N, rout * bsz)

    # recompute xR
    xb   = x.view(N, rin, bsz)
    xb_r = xb.transpose(0, 1)                       # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)                     # [rin, N, b]
    xR   = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # grads
    # grad_W = g_yb.t() @ xR
    # grad_b = g_yb.sum(0) if ctx.b is not None else None
    g_xR   = g_yb @ W                                # [N, rin*bsz]

    # grad Rin
    g_xR_3 = g_xR.view(N, rin, bsz).transpose(0, 1)  # [rin, N, b]
    grad_Rin = torch.bmm(xb_r.transpose(1, 2), g_xR_3)  # [rin, b, b]

    # grad x
    g_xb_r = torch.bmm(g_xR_3, Rin.transpose(1, 2))  # [rin, N, b]
    grad_x = g_xb_r.transpose(0, 1).reshape(B, S, rin * bsz)

    # grad Rout
    yb = None
    if b is not None:
        yb = (xR @ W.t() + b).view(N, rout, bsz)
    else:
        yb = (xR @ W.t()).view(N, rout, bsz)
    yb_t   = yb.transpose(0, 1)                      # [rout, N, b]
    grad_Rout = torch.bmm(yb_t.transpose(1, 2), g_t) # [rout, b, b]

    # Return: x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, bsz
    return grad_x, grad_Rin, None, None, None, None, None, grad_Rout, None

@chain_layer_checkpoint_q8.register_fake
def _(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, bsz):
    B, S, Din = x.shape
    # rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S
    Dout, Din = W_q.shape
    shape0 = (rout, N, bsz)
    # shape1 = (N, Dout)
    y = x.new_empty((B, S, Dout), device=x.device, dtype=x.dtype)
    # y0 = x.new_empty(shape0).transpose(0, 1).reshape(B, S, Dout)
    # y1 = x.new_empty(shape1).view(N, rout, bsz).transpose(0, 1)
    return y # y0, y1

torch.library.register_autograd(
    "poet::chain_layer_checkpoint_q8", 
    chain_layer_checkpoint_q8_backward, 
    setup_context=chain_layer_checkpoint_q8_setup_context
)



########################### Permute_x ###########################
class PermutationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x: torch.Tensor, perm: torch.Tensor, inv_perm: torch.Tensor):
		ctx.save_for_backward(inv_perm)
		return x[..., perm]

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		(inv_perm,) = ctx.saved_tensors
		grad_input = grad_output[..., inv_perm]
		return grad_input, None, None


########################### Triu_assign ###########################
@torch.library.custom_op("poet::triu_assign", mutates_args=())
def triu_assign(
    v: torch.Tensor,
    rows: torch.Tensor,
    cols: Optional[torch.Tensor],
    block_size: int
) -> torch.Tensor:
    matrix = v.new_zeros(v.shape[0], block_size, block_size)
    matrix[:, rows, cols] = v
    # requires_grad after in-place assignment
    matrix.requires_grad_(True)
    return matrix

def triu_assign_setup_context(ctx, inputs, output):
    v, rows, cols, block_size  = inputs
    ctx.save_for_backward(rows, cols)

def triu_assign_backward(ctx, g: torch.Tensor) -> torch.Tensor:
    rows, cols = ctx.saved_tensors
    grad_v = g[:, rows, cols]
    return grad_v, None, None, None

@triu_assign.register_fake
def _(v, rows, cols, block_size):
    batch_size = v.shape[0]
    return v.new_empty(batch_size, block_size, block_size, requires_grad=True)

torch.library.register_autograd(
    "poet::triu_assign", triu_assign_backward, setup_context=triu_assign_setup_context
)


######################## Chain Layer Checkpoint Fast ########################

@torch.library.custom_op("poet::chain_layer_checkpoint", mutates_args=())
def chain_layer_checkpoint(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    bsz: int,
) -> torch.Tensor:
    B, S, Din = x.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # x @ Rin
    xb = x.view(N, rin, bsz)                 # [N, rin, b]
    xb_r = xb.transpose(0, 1)                # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)              # [rin, N, b]
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # xR @ W^T (+b)
    yb_flat = xR @ W.t()                     # [N, rout*bsz]
    if b is not None:
        yb_flat = yb_flat + b

    # yb_flat @ Rout
    yb = yb_flat.view(N, rout, bsz)          # [N, rout, b]
    yb_r = yb.transpose(0, 1)                # [rout, N, b]
    y = torch.bmm(yb_r, Rout)                # [rout, N, b]
    y = y.transpose(0, 1).reshape(B, S, rout * bsz)

    return y

def chain_layer_checkpoint_setup_context(ctx, inputs, output):
    x, Rin, W, b, Rout, bsz = inputs
    # _, yb_r = output
    ctx.save_for_backward(x, Rin, Rout)
    ctx.W = W
    ctx.b = b
    ctx.bsz = bsz

def chain_layer_checkpoint_backward(ctx, g: torch.Tensor) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    x, Rin, Rout = ctx.saved_tensors
    W = ctx.W
    b = ctx.b
    bsz = ctx.bsz

    B, S, _ = g.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # grad wrt Rout and yb
    g_t = g.view(N, rout, bsz).transpose(0, 1)      # [rout, N, b]
    Rt_t = Rout.transpose(1, 2)                     # [rout, b, b]
    g_yb_t = torch.bmm(g_t, Rt_t)                   # [rout, N, b]
    g_yb = g_yb_t.transpose(0, 1).reshape(N, rout * bsz)

    # recompute xR
    xb   = x.view(N, rin, bsz)
    xb_r = xb.transpose(0, 1)                       # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)                     # [rin, N, b]
    xR   = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # grads
    # grad_W = g_yb.t() @ xR
    # grad_b = g_yb.sum(0) if ctx.b is not None else None
    g_xR   = g_yb @ W                                # [N, rin*bsz]

    # grad Rin
    g_xR_3 = g_xR.view(N, rin, bsz).transpose(0, 1)  # [rin, N, b]
    grad_Rin = torch.bmm(xb_r.transpose(1, 2), g_xR_3)  # [rin, b, b]

    # grad x
    g_xb_r = torch.bmm(g_xR_3, Rin.transpose(1, 2))  # [rin, N, b]
    grad_x = g_xb_r.transpose(0, 1).reshape(B, S, rin * bsz)

    # grad Rout
    yb = None
    if b is not None:
        yb = (xR @ W.t() + b).view(N, rout, bsz)
    else:
        yb = (xR @ W.t()).view(N, rout, bsz)
    yb_t   = yb.transpose(0, 1)                      # [rout, N, b]
    grad_Rout = torch.bmm(yb_t.transpose(1, 2), g_t) # [rout, b, b]

    return grad_x, grad_Rin, None, None, grad_Rout, None

@chain_layer_checkpoint.register_fake
def _(x, Rin, W, b, Rout, bsz):
    B, S, Din = x.shape
    # rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S
    Dout, Din = W.shape
    shape0 = (rout, N, bsz)
    # shape1 = (N, Dout)
    y = x.new_empty((B, S, Dout), device=x.device, dtype=x.dtype)
    # y0 = x.new_empty(shape0).transpose(0, 1).reshape(B, S, Dout)
    # y1 = x.new_empty(shape1).view(N, rout, bsz).transpose(0, 1)
    return y # y0, y1

torch.library.register_autograd(
    "poet::chain_layer_checkpoint", chain_layer_checkpoint_backward, setup_context=chain_layer_checkpoint_setup_context
)

######################## Chain Layer Checkpoint Slow ########################

@torch.library.custom_op("poet::chain_layer_checkpoint_mem_o2", mutates_args=())
def chain_layer_checkpoint_mem_o2(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    bsz: int,
) -> torch.Tensor:
    x = x[..., perm_in_inv]

    B, S, Din = x.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # x @ Rin
    xb = x.view(N, rin, bsz)                 # [N, rin, b]
    xb_r = xb.transpose(0, 1)                # [rin, N, b]
    xR_r = torch.bmm(xb_r, Rin)              # [rin, N, b]
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)

    # xR @ W^T (+b)
    yb_flat = xR @ W.t()                     # [N, rout*bsz]
    if b is not None:
        yb_flat = yb_flat + b

    # yb_flat @ Rout
    yb = yb_flat.view(N, rout, bsz)          # [N, rout, b]
    yb_r = yb.transpose(0, 1)                # [rout, N, b]
    y = torch.bmm(yb_r, Rout)                # [rout, N, b]
    y = y.transpose(0, 1).reshape(B, S, rout * bsz)

    y = y[..., perm_out]

    return y

def chain_layer_checkpoint_mem_o2_setup_context(ctx, inputs, output):
    x, Rin, W, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz = inputs
    # _, yb_r = output
    ctx.save_for_backward(x, Rin, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv)
    ctx.W = W
    ctx.b = b
    ctx.bsz = bsz


def chain_layer_checkpoint_mem_o2_backward(ctx, g_orig):
    x_orig, Rin, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv = ctx.saved_tensors
    W = ctx.W
    b = ctx.b
    bsz = ctx.bsz

    B, S, _ = g_orig.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S

    # Recompute the permuted x from saved original
    g = g_orig[..., perm_out_inv]
    del g_orig

    x = x_orig[..., perm_in_inv]
    del x_orig

    # --- Phase 1: grad_Rout (compute and free yb early) ---
    xb_r = x.view(N, rin, bsz).transpose(0, 1)
    xR_r = torch.bmm(xb_r, Rin)
    xR = xR_r.transpose(0, 1).reshape(N, rin * bsz)
    del xR_r

    if b is not None:
        yb = (xR @ W.t() + b).view(N, rout, bsz)
    else:
        yb = (xR @ W.t()).view(N, rout, bsz)
    del xR

    g_t = g.view(N, rout, bsz).transpose(0, 1)
    grad_Rout = torch.bmm(yb.transpose(0, 1).transpose(1, 2), g_t)
    del yb

    # --- Phase 2: g_xR (compute and free g_yb early) ---
    g_yb_t = torch.bmm(g_t, Rout.transpose(1, 2))
    del g_t, g
    g_yb = g_yb_t.transpose(0, 1).reshape(N, rout * bsz)
    del g_yb_t
    g_xR = g_yb @ W
    del g_yb

    # --- Phase 3: grad_Rin ---
    g_xR_3 = g_xR.view(N, rin, bsz).transpose(0, 1)
    grad_Rin = torch.bmm(xb_r.transpose(1, 2), g_xR_3)

    # Free the permuted x copy — no longer needed
    del xb_r, x

    # --- Phase 4: grad_x ---
    g_xb_r = torch.bmm(g_xR_3, Rin.transpose(1, 2))
    del g_xR, g_xR_3
    grad_x = g_xb_r.transpose(0, 1).reshape(B, S, rin * bsz)
    del g_xb_r

    # Un-permute gradient back to original x's coordinate system
    grad_x = grad_x[..., perm_in]

    return grad_x, grad_Rin, None, None, grad_Rout, None, None, None, None, None


@chain_layer_checkpoint_mem_o2.register_fake
def _(x, Rin, W, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz):
    B, S, Din = x.shape
    # rin = Rin.size(0)
    rout = Rout.size(0)
    N = B * S
    Dout, Din = W.shape
    shape0 = (rout, N, bsz)
    # shape1 = (N, Dout)
    y = x.new_empty((B, S, Dout), device=x.device, dtype=x.dtype)
    # y0 = x.new_empty(shape0).transpose(0, 1).reshape(B, S, Dout)
    # y1 = x.new_empty(shape1).view(N, rout, bsz).transpose(0, 1)
    return y # y0, y1

torch.library.register_autograd(
    "poet::chain_layer_checkpoint_mem_o2",
    chain_layer_checkpoint_mem_o2_backward,
    setup_context=chain_layer_checkpoint_mem_o2_setup_context
)

################################# Cayley #################################
def cayley_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_ROW': BM, 'BLOCK_COL': BM, "BLOCK_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
                      num_warps=w, pre_hook=pre_hook)
        for BM in [64, 32, 16]
        for BK in [64, 32, 16]
        for s in ([2, 3, 4])
        for w in [2, 4]
    ]

@triton.autotune(
    configs=cayley_get_configs(),
    key=['N', 'B'],  # cache best config per matrix size
)
@triton.jit
def cayley_forward_kernel(
    Q_ptr, Q2_ptr, Y_ptr,
    N, B,
    q_stride_batch, q_stride_row, q_stride_col,
    q2_stride_batch, q2_stride_row, q2_stride_col,
    y_stride_batch, y_stride_row, y_stride_col,
    # tiling
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    tl.assume(BLOCK_ROW == BLOCK_COL)
    pid_all = tl.program_id(0)
    num_pid_row = tl.cdiv(B, BLOCK_ROW)
    num_pid_col = tl.cdiv(B, BLOCK_COL)
    grid_batch = num_pid_row * num_pid_col
    pid_batch = pid_all // grid_batch
    pid = pid_all % grid_batch

    # pid_row = pid // num_pid_row
    # pid_col = pid % num_pid_row
    num_pid_in_group = GROUP_SIZE_M * num_pid_row
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_col - first_pid_m, GROUP_SIZE_M)
    pid_row = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_col = (pid % num_pid_in_group) // group_size_m

    offs_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    # Base pointers for this batch
    base_q = Q_ptr + pid_batch * q_stride_batch
    base_q2 = Q2_ptr + pid_batch * q2_stride_batch
    base_y = Y_ptr + pid_batch * y_stride_batch

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_ROW, BLOCK_COL), dtype=tl.float32)
    q3_tile = tl.zeros((BLOCK_ROW, BLOCK_COL), dtype=tl.float32)

    # K loop
    for k0 in tl.range(0, B, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = base_q2 + (offs_row[:, None] * q2_stride_row + offs_k[None, :] * q2_stride_col)
        b_ptrs = base_q2 + (offs_k[:, None] * q2_stride_row + offs_col[None, :] * q2_stride_col)
        c_ptrs = base_q + (offs_k[:, None] * q_stride_row + offs_col[None, :] * q_stride_col)

        a = tl.load(a_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
    
        acc += tl.dot(a, b)  # q4_tile_k
        c = tl.load(c_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
        q3_tile += tl.dot(a, c)

    # NOTE(Lixin): I believe we can further optimize this load as we already load q and q2 inside the K loop
    #  but it seems that triton hasn't yet supported local tensor indexing

    q_tile_ptrs = base_q + (offs_row[:, None] * q_stride_row + offs_col[None, :] * q_stride_col)
    q3_tile += tl.load(q_tile_ptrs, mask=(offs_row[:, None] < B) & (offs_col[None, :] < B), other=0.0)

    q2_tile_ptrs = base_q2 + (offs_row[:, None] * q2_stride_row + offs_col[None, :] * q2_stride_col)
    q3_tile += tl.load(q2_tile_ptrs, mask=(offs_row[:, None] < B) & (offs_col[None, :] < B), other=0.0)

    # I + 2 * Q + 2 * (Q @ Q) + 2 * (Q2 @ Q) + (Q2 @ Q2)
    acc = 2.0 * q3_tile + acc

    # Add identity on diagonal
    diag_mask = (offs_row[:, None] == offs_col[None, :])
    acc += tl.where(diag_mask, 1.0, 0.0)

    # y_tile = y_tile.to(tl.float16) if False else y_tile  # left here if you want to force cast
    y = acc.to(base_y.dtype.element_ty)
    tl.store(
        base_y + (offs_row[:, None] * y_stride_row + offs_col[None, :] * y_stride_col),
        y,  # store in original dtype outside if desired
        mask=(offs_row[:, None] < B) & (offs_col[None, :] < B),
    )


@triton.autotune(
    configs=cayley_get_configs(),
    key=['N', 'B'],  # cache best config per matrix size
)
@triton.jit
def cayley_backward_kernel_grad_Q2(
    QT_ptr, dY_ptr, dQ2_ptr,
    N, B,
    qt_stride_batch, qt_stride_row, qt_stride_col,
    dy_stride_batch, dy_stride_row, dy_stride_col,
    dq2_stride_batch, dq2_stride_row, dq2_stride_col,
    # tiling
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    tl.assume(BLOCK_ROW == BLOCK_COL)
    pid_all = tl.program_id(0)
    grid_row = tl.cdiv(B, BLOCK_ROW)
    grid_col = tl.cdiv(B, BLOCK_COL)
    grid_batch = grid_row * grid_col
    pid_batch = pid_all // grid_batch
    pid = pid_all % grid_batch

    num_pid_in_group = GROUP_SIZE_M * grid_row
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_col - first_pid_m, GROUP_SIZE_M)
    pid_row = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_col = (pid % num_pid_in_group) // group_size_m

    offs_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    # Base pointers for this batch
    base_qt = QT_ptr + pid_batch * qt_stride_batch
    base_dy = dY_ptr + pid_batch * dy_stride_batch
    base_dQ2 = dQ2_ptr + pid_batch * dq2_stride_batch

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_ROW, BLOCK_COL), dtype=tl.float32)

    # K loop
    for k0 in tl.range(0, B, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        dy_ptrs = base_dy + (offs_row[:, None] * dy_stride_row + offs_k[None, :] * dy_stride_col)
        qt_ptrs = base_qt + (offs_k[:, None] * qt_stride_row + offs_col[None, :] * qt_stride_col)
        dy = tl.load(dy_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        qt = tl.load(qt_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
        acc += tl.dot(dy, qt)

        qt_ptrs = base_qt + (offs_row[:, None] * qt_stride_row + offs_k[None, :] * qt_stride_col)
        dy_ptrs = base_dy + (offs_k[:, None] * dy_stride_row + offs_col[None, :] * dy_stride_col)
        qt = tl.load(qt_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        dy = tl.load(dy_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
        acc += tl.dot(qt, dy)

    dq2 = acc.to(base_dQ2.dtype.element_ty)
    tl.store(
        base_dQ2 + (offs_row[:, None] * dq2_stride_row + offs_col[None, :] * dq2_stride_col),
        dq2, 
        mask=(offs_row[:, None] < B) & (offs_col[None, :] < B),
    )

@triton.autotune(
    configs=cayley_get_configs(),
    key=['N', 'B'],  # cache best config per matrix size
)
@triton.jit
def cayley_backward_kernel_grad_final(
    QT_ptr, Q2T_ptr, dY_ptr, dQ2_ptr, grad_ptr,
    N, B,
    qt_stride_batch, qt_stride_row, qt_stride_col,
    q2t_stride_batch, q2t_stride_row, q2t_stride_col,
    dy_stride_batch, dy_stride_row, dy_stride_col,
    dq2_stride_batch, dq2_stride_row, dq2_stride_col,
    g_stride_batch, g_stride_row, g_stride_col,
    # tiling
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    tl.assume(BLOCK_ROW == BLOCK_COL)
    pid_all = tl.program_id(0)
    num_pid_row = tl.cdiv(B, BLOCK_ROW)
    num_pid_col = tl.cdiv(B, BLOCK_COL)
    grid_batch = num_pid_row * num_pid_col
    pid_batch = pid_all // grid_batch
    pid = pid_all % grid_batch

    # pid_row = pid // num_pid_row
    # pid_col = pid % num_pid_row
    num_pid_in_group = GROUP_SIZE_M * num_pid_row
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_col - first_pid_m, GROUP_SIZE_M)
    pid_row = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_col = (pid % num_pid_in_group) // group_size_m

    offs_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    # Base pointers for this batch
    base_qt = QT_ptr + pid_batch * qt_stride_batch
    base_q2t = Q2T_ptr + pid_batch * q2t_stride_batch
    base_dy = dY_ptr + pid_batch * dy_stride_batch
    base_dq2 = dQ2_ptr + pid_batch * dq2_stride_batch
    base_grad_ptr = grad_ptr + pid_batch * g_stride_batch

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_ROW, BLOCK_COL), dtype=tl.float32)

    # K loop
    for k0 in tl.range(0, B, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        dq2_ptrs = base_dq2 + (offs_row[:, None] * dq2_stride_row + offs_k[None, :] * dq2_stride_col)
        a = tl.load(dq2_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        dy_ptrs = base_dy + (offs_row[:, None] * dy_stride_row + offs_k[None, :] * dy_stride_col)
        # a += 2 * tl.load(dy_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        tmp = tl.load(dy_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        a += 2 * tmp

        q2t_ptrs = base_q2t + (offs_k[:, None] * q2t_stride_row + offs_col[None, :] * q2t_stride_col)
        b = tl.load(q2t_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
        acc += tl.dot(a, b)

    for k0 in tl.range(0, B, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        q2t_ptrs = base_q2t + (offs_row[:, None] * q2t_stride_row + offs_k[None, :] * q2t_stride_col)
        a = tl.load(q2t_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        qt_ptrs = base_qt + (offs_row[:, None] * qt_stride_row + offs_k[None, :] * qt_stride_col)
        # a += 2 * tl.load(qt_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        tmp = tl.load(qt_ptrs, mask=(offs_row[:, None] < B) & (offs_k[None, :] < B), other=0.0)
        a += 2 * tmp

        dq2_ptrs = base_dq2 + (offs_k[:, None] * dq2_stride_row + offs_col[None, :] * dq2_stride_col)
        b = tl.load(dq2_ptrs, mask=(offs_k[:, None] < B) & (offs_col[None, :] < B), other=0.0)
        acc += tl.dot(a, b)

    dy_ptrs = base_dy + (offs_row[:, None] * dy_stride_row + offs_col[None, :] * dy_stride_col)
    dy = tl.load(dy_ptrs, mask=(offs_row[:, None] < B) & (offs_col[None, :] < B), other=0.0)

    dq2_ptrs = base_dq2 + (offs_row[:, None] * dq2_stride_row + offs_col[None, :] * dq2_stride_col)
    dy += tl.load(dq2_ptrs, mask=(offs_row[:, None] < B) & (offs_col[None, :] < B), other=0.0)

    acc = 2 * dy + acc

    grad = acc.to(base_grad_ptr.dtype.element_ty)
    tl.store(
        base_grad_ptr + (offs_row[:, None] * g_stride_row + offs_col[None, :] * g_stride_col),
        grad,   # store in original dtype outside if desired
        mask=(offs_row[:, None] < B) & (offs_col[None, :] < B),
    )


@torch.library.triton_op("poet::cayley", mutates_args={})
def cayley(Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert Q.ndim == 3 and Q.shape[-1] == Q.shape[-2], "Q must be [N, B, B]"
    N, B, _ = Q.shape
    Q2 = Q @ Q
    Y = torch.empty_like(Q)
    
    # Strides
    q_stride_batch, q_stride_row, q_stride_col = Q.stride()
    q2_stride_batch, q2_stride_row, q2_stride_col = Q2.stride()
    y_stride_batch, y_stride_row, y_stride_col = Y.stride()
    assert q_stride_col == y_stride_col == 1

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_ROW']) * triton.cdiv(B, meta['BLOCK_COL']) * N, )

    torch.library.wrap_triton(cayley_forward_kernel)[grid](
        Q, Q2, Y,
        N, B,
        q_stride_batch, q_stride_row, q_stride_col,
        q2_stride_batch, q2_stride_row, q2_stride_col,
        y_stride_batch, y_stride_row, y_stride_col
    )
    return Y, Q2

@torch.library.triton_op("poet::_cayley_kernel_bwd", mutates_args={})
def cayley_kernel_bwd(Q: torch.Tensor, Q2: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    # each terms:
    # Qf -> grad: dY
    # Qf @ Qf -> grad: QT @ dY + dY @ QT
    # Qf @ Qf @ Qf -> grad: QT @ QT @ dY + QT @ dY @ QT + dY @ QT @ QT == (QT @ dQ2 + dY @ Q2T)
    # Qf @ Qf @ Qf @ Qf -> grad: (QT @ dQ3 + dY @ Q3T) == (Q2T @ dQ2 + dQ2 @ Q2T)

    # grad = 2 * dY + 2 * (QT @ dY + dY @ QT) + 2 * (QT @ dQ2 + dY @ Q2T) + (Q2T @ dQ2 + dQ2 @ Q2T)
    #      = 2 * (dY + dQ2) + (2 * QT + Q2T) @ dQ2 + (2 * dY + dQ2) @ Q2T
    #      = 2 * (dY + dQ2) + (2 * QT + Q2T) @ dQ2 + (2 * dY + dQ2) @ Q2T
    #        {7 read; 2 matmul; 1 write(grad)}
    #   where: dQ2 = QT @ dY + dY @ QT {4 read; 2 matmul; 1 write(dQ2)}
    N, B, _ = dY.shape

    QT: torch.Tensor = Q.transpose(-2, -1)
    Q2T: torch.Tensor = Q2.transpose(-2, -1)

    dQ2 = torch.empty_like(Q)
    # dQ2 = QT @ dY + dY @ QT
    grad = torch.empty_like(Q)
    # grad = 2 * (dY + dQ2) + (2 * QT + Q2T) @ dQ2 + (2 * dY + dQ2) @ Q2T

    qt_stride_batch, qt_stride_row, qt_stride_col = QT.stride()
    q2t_stride_batch, q2t_stride_row, q2t_stride_col = Q2T.stride()
    dy_stride_batch, dy_stride_row, dy_stride_col = dY.stride()
    dq2_stride_batch, dq2_stride_row, dq2_stride_col = dQ2.stride()
    g_stride_batch, g_stride_row, g_stride_col = grad.stride()

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_ROW']) * triton.cdiv(B, meta['BLOCK_COL']) * N, )

    torch.library.wrap_triton(cayley_backward_kernel_grad_Q2)[grid](
        QT, dY, dQ2,
        N, B,
        qt_stride_batch, qt_stride_row, qt_stride_col,
        dy_stride_batch, dy_stride_row, dy_stride_col,
        dq2_stride_batch, dq2_stride_row, dq2_stride_col,
    )

    torch.library.wrap_triton(cayley_backward_kernel_grad_final)[grid](
        QT, Q2T, dY, dQ2, grad,
        N, B,
        qt_stride_batch, qt_stride_row, qt_stride_col,
        q2t_stride_batch, q2t_stride_row, q2t_stride_col,
        dy_stride_batch, dy_stride_row, dy_stride_col,
        dq2_stride_batch, dq2_stride_row, dq2_stride_col,
        g_stride_batch, g_stride_row, g_stride_col,
    )
    
    return grad

# @torch.library.custom_op("poet::cayley", mutates_args=())
# def cayley(Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.poet._cayley_kernel_fwd(Q)

def cayley_setup_context(ctx, inputs, output):
    Q, = inputs
    _, Q2 = output
    ctx.save_for_backward(Q, Q2)

def cayley_backward(ctx, dY: torch.Tensor, dQ2: Optional[torch.Tensor]) -> torch.Tensor:
    Q, Q2 = ctx.saved_tensors
    return torch.ops.poet._cayley_kernel_bwd(Q, Q2, dY)

@cayley.register_fake
def _(x):
    return torch.empty_like(x), torch.empty_like(x)

torch.library.register_autograd(
    "poet::cayley", cayley_backward, setup_context=cayley_setup_context
)

######################### Test #########################
def test_poet_triu_assign():
    device = torch.device('cuda')
    block_size = 16
    rank_in = 8
    rank_out = 4
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
    n_elements = block_size * (block_size - 1) // 2
    Rl = torch.randn((rank_in, n_elements), device=device, requires_grad=True)
    Rr = torch.randn((rank_out, n_elements), device=device, requires_grad=True)
    vec_cat = torch.cat([Rl, Rr], dim=0).contiguous()
    sample_inputs = [
        (vec_cat, rows, cols, block_size),
    ]
    for args in sample_inputs:
        torch.library.opcheck(triu_assign, args)
        torch.library.opcheck(triu_assign, args, test_utils="test_aot_dispatch_static")
    y = torch.ops.poet.triu_assign(*sample_inputs[0])
    y.sum().backward(retain_graph=True)
    print("test triu_assign:")
    print("y:", y.shape)
    print("dRl:", Rl.grad.shape)
    print("dRr:", Rr.grad.shape)

def test_poet_chain_layer():
    device = torch.device('cuda')
    dtype = torch.bfloat16
    B = 2
    S = 256
    Din = 512
    Dout = 1024
    bsz = 256
    rin = Din // bsz    # Number of input blocks
    rout = Dout // bsz  # Number of output blocks
    x = torch.randn(B, S, Din, device=device, dtype=dtype, requires_grad=True)
    Rin = torch.randn(rin, bsz, bsz, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(Dout, Din, device=device, dtype=dtype, requires_grad=False)
    b = torch.randn(Dout, device=device, dtype=dtype, requires_grad=False)
    Rout = torch.randn(rout, bsz, bsz, device=device, dtype=dtype, requires_grad=True)
    
    sample_inputs = [
        (x, Rin, W, b, Rout, bsz),
        (x, Rin, W, None, Rout, bsz),
    ]
    for args in sample_inputs:
        torch.library.opcheck(chain_layer, args)
        torch.library.opcheck(chain_layer, args, test_utils="test_aot_dispatch_static")
    y, _ = torch.ops.poet.chain_layer(*sample_inputs[0])
    y.sum().backward(retain_graph=True)
    print("test chain_layer:")
    print("y:", y.shape)
    print("dX:", x.grad.shape)
    print("dRin:", Rin.grad.shape)
    print("dRout:", Rout.grad.shape)


def test_poet_cayley():
    Q = torch.randn((16, 256, 256), requires_grad=True, device="cuda")
    torch.library.opcheck(cayley, (Q,))
    torch.library.opcheck(cayley, (Q,), test_utils="test_aot_dispatch_static")
    y, _ = torch.ops.poet.cayley(Q)
    y.sum().backward(retain_graph=True)
    print("test cayley:")
    print("y:", y.shape)
    print("dQ:", Q.grad.shape)

if __name__ == "__main__":
    # test_poet_triu_assign()
    # test_poet_chain_layer()
    test_poet_cayley()