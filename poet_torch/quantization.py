import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .poet_ops import *


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


class W8Linear_galore(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)

        def forward_w_float_weight(weight, x, bias):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)

            if bias is not None:
                return x @ float_weight.t() + bias
            else:
                return x @ float_weight.t()

        output = forward_w_float_weight(weight, x, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

        def backward_w_float_weight(weight, grad_output):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)
            grad_input = grad_output @ float_weight
            return grad_input

        grad_input = backward_w_float_weight(weight, grad_output)

        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        out_features, in_features = weight.shape
        # gradient accumulation
        if not hasattr(weight, 'float_grad'):
            weight.__setattr__('float_grad', None)

        if weight.float_grad is not None:
            weight.float_grad += grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features) 
        else:
            weight.float_grad = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)

        if hasattr(weight, 'backward_hook'):
            weight.backward_hook(weight)

        return grad_input, None, grad_bias


class QGaLoreLinear(nn.Module):
    def __init__(self, weight, bias, device=None, dtype=None, num_bits=8, group_size=256, stochastic_round=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        int8_weight, scales, zeros = _quantize_tensor_int8(weight.data, q_group_size=group_size)
        torch.cuda.empty_cache()

        self.weight = Parameter(int8_weight, requires_grad=False).to(device) # Only Tensors of floating point and complex dtype can require gradients, using float_gradient to store the gradient
        self.weight.__setattr__('scales', scales.to(device))
        self.weight.__setattr__('zeros', zeros.to(device))
        self.weight.__setattr__('group_size', group_size)
        self.weight.__setattr__('saved_data_dtype', int8_weight.dtype)
        self.weight.__setattr__('stochastic_round', stochastic_round)

        if not num_bits == 8:
            raise NotImplementedError

        self.bias = Parameter(bias, requires_grad=True).to(device) if bias is not None else None

    def forward(self, input: Tensor) -> Tensor:
        output = W8Linear_galore.apply(input, self.weight, self.bias)
        return output


class W8Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)

        def forward_w_float_weight(weight, x, bias):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)

            if bias is not None:
                return x @ float_weight.t() + bias
            else:
                return x @ float_weight.t()

        output = forward_w_float_weight(weight, x, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

        def backward_w_float_weight(weight, grad_output):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)
            grad_input = grad_output @ float_weight
            return grad_input

        grad_input = backward_w_float_weight(weight, grad_output)

        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        out_features, in_features = weight.shape
        # gradient accumulation
        if not hasattr(weight, 'float_grad'):
            weight.__setattr__('float_grad', None)

        if weight.float_grad is not None:
            weight.float_grad += grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features) 
        else:
            weight.float_grad = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)

        if hasattr(weight, 'backward_hook'):
            weight.backward_hook(weight)

        return grad_input, None, grad_bias


def permute_x(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)


def chain_layer_x_checkpoint(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint(x, Rin, weight, bias, Rout, block_size)


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

# @torch.compile(fullgraph=True)
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

    # balance mode
    x = permute_x(x, perm_in_inv, perm_in)

    y = chain_layer_x_checkpoint(x, R_in, base_weight, base_bias, R_out, block_size)
    # lower throughput but less memory (merge permute inside chain layer)
    # y = chain_layer_x_checkpoint_mem_o2(x, R_in, base_weight, base_bias, R_out, perm_in_inv, perm_in, block_size)
    # y = chain_layer_x_checkpoint_2lr(x, R_in, base_weight, base_bias, R_out, block_size)

    y = permute_x(y, perm_out, perm_out_inv)
    return y


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

        self.weight = Parameter(int8_weight, requires_grad=False).to(device) # Only Tensors of floating point and complex dtype can require gradients, using float_gradient to store the gradient
        self.weight.__setattr__('scales', scales.to(device))
        self.weight.__setattr__('zeros', zeros.to(device))
        self.weight.__setattr__('group_size', group_size)
        self.weight.__setattr__('saved_data_dtype', int8_weight.dtype)
        self.weight.__setattr__('stochastic_round', stochastic_round)

        if not num_bits == 8:
            raise NotImplementedError

        self.bias = Parameter(bias, requires_grad=True).to(device) if bias is not None else None

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
        q, scales, zeros = _quantize_tensor_int8(w_float, q_group_size=self.group_size, n_bit=self.num_bits)
        self.weight.detach().copy_(q.to(self.weight.device))
        self.weight.scales = scales.to(self.weight.device)
        self.weight.zeros = zeros.to(self.weight.device)

    def _dequantize_to(self, dtype: torch.dtype):
        assert self.weight.scales is not None and self.weight.zeros is not None, "QPOETLinear weight not initialized/quantized"
        w = self.weight.to(dtype).reshape(-1, self.weight.group_size)   
        w = (w - self.weight.zeros.to(dtype)) * self.weight.scales.to(dtype)
        return w.reshape(self.weight.shape)

    @torch.no_grad()
    def perform_permutation(self) -> None:
        # IMPORTANT: dequantize -> permute in float -> requantize (permutes invalidate old scales/zeros)
        W = self._dequantize_to(dtype=self.oft_R.dtype)
        Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        self._requantize_from_float(Wp)

    @torch.no_grad()
    def update_permutation(self):
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in).to(torch.int32))
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out).to(torch.int32))
        self.perform_permutation()

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        # Same math as POETLinear.merge_then_reinitialize, but float compute + requantize
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

        W = self._dequantize_to(dtype=self.oft_R.dtype)
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        self._requantize_from_float(expected)

        self.oft_R.zero_()
        self.update_permutation()

    def forward(self, x):
        # dequantize just-in-time for the PoET fused op
        # base_weight = self._dequantize_to(dtype=x.dtype)

        # 1) dequantize to float *and* make it a leaf so autograd gives us grad_w
        w = self.weight.to(x.dtype).reshape(-1, self.weight.group_size)
        w = (w - self.weight.zeros.to(x.dtype)) * self.weight.scales.to(x.dtype)
        float_weight = w.reshape(self.weight.shape).detach().requires_grad_(True)

        x = forward_core(
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
            float_weight,
            self.bias,
            self.mem_efficient_mode,
        )
        # output = W8Linear.apply(x, self.weight, self.bias)
        return x


# --- NEW: replace helper for q_poet ---
def replace_linear_with_qpoet(
    module: nn.Module,
    block_size: int,
    init_type: str,
    device=None,
    dtype=None,
    mem_efficient_mode: bool = False,
    num_bits: int = 8,
    group_size: int = 256,
) -> int:
    def _convert(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    new_lin = QPOETLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bsz=block_size,
                        bias=(child.bias is not None),
                        device=device,
                        dtype=dtype,
                        mem_efficient_mode=mem_efficient_mode,
                        num_bits=num_bits,
                        group_size=group_size,
                    )
                    with torch.no_grad():
                        w = child.weight.detach()
                        if init_type == "normalized":
                            w = w / torch.norm(w, dim=1, keepdim=True)
                        new_lin._requantize_from_float(w.to(device=device, dtype=dtype))
                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(device=device, dtype=dtype))
                    setattr(m, name, new_lin)
                else:
                    raise ValueError(
                        f"Layer {name} has in_features {child.in_features} and out_features {child.out_features}, "
                        f"which are not divisible by {block_size}"
                    )
            else:
                _convert(child)

    _convert(module)
    return 0



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


if __name__ == '__main__':
    GROUP_SIZE=256
    print('*** Memory checking for a single linear layer ***')
    fp16_linear1 = nn.Linear(4096, 4096, bias=False).to('cuda:0').to(torch.bfloat16)
    print('after initial weight for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    mem_weight_float = torch.cuda.memory_allocated('cuda:0')//1024/1024
    x = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:0', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    start = time.time()
    output = fp16_linear1(x)
    print('after forward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    output.sum().backward()
    end = time.time()
    print('after backward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('------------------------------------')

    int8_linear1 = QGaLoreLinear(fp16_linear1.weight, None, device='cuda:1', num_bits=8, group_size=GROUP_SIZE)
    print('after initial weight for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:1')//1024/1024
    x1 = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:1', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    start = time.time()
    output_int8 = int8_linear1(x1)
    print('after forward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    output_int8.sum().backward()
    end = time.time()
    print('after backward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('------------------------------------')

    print('Memory saving for weight: {:.2f} MB, ratio: {:.2f}%'.format(mem_weight_float - mem_weight_int, mem_weight_int / mem_weight_float * 100))
    print('------------------------------------')

