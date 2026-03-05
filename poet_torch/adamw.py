import warnings
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math

class POETAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        threshold: int = 5000,
        poet_reset_gap: int = 0,
        poet_block_size: int = 256,
        # poet_scale: float = 1.0,
        fused: bool = False, # TODO: CHECK IF THIS IS NEEDED
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, 
                   "stochastic": False, "poet_reset_gap": poet_reset_gap}
        super().__init__(params, defaults)

        self.thres = threshold
        self.global_step_counter = 0
        self.poet_block_size = poet_block_size
        # self.poet_scale = poet_scale


    def adjust_lr_for_poet(self, poet_lr: float, p: torch.Tensor) -> float:
        """
        p is POET parameter shaped (r, n_elements), where n_elements is constant for fixed bsz.
        Scale LR by sqrt(r) (number of blocks).
        """
        # num_blocks = p.shape[0]
        # bsz = self.poet_block_size
        # total_dim = num_blocks * bsz
        # scaling = math.sqrt(total_dim)
        # return poet_lr * scaling

        scaling = math.sqrt(p.shape[0] / 2) * self.poet_scale
        return poet_lr * scaling

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Check if reset should be applied based on CayleyLinear's global counter
                if group.get("poet_reset_gap", 0) > 0 and \
                   self.global_step_counter % group["poet_reset_gap"] == 0 and \
                   self.global_step_counter > 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Threshold-based gradient masking
                # TODO: This is used to prevent the gradient from exploding, still need to test it
                # if self.thres != 0 and CayleyLinear.global_step_counter > 1000:
                #     exp_avg_sq1 = exp_avg_sq
                #     mask = (grad**2) > (self.thres * exp_avg_sq1)
                #     if mask.sum() > 0:
                #         grad[mask]=grad[mask].sign()*torch.sqrt(exp_avg_sq1[mask]*self.thres)

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                lr_eff = group["lr"]
                # if group.get("use_poet", False) and group.get("poet_scale") > 0.0:
                #     poet_lr = group.get("poet_lr", group["lr"])
                #     lr_eff = self.adjust_lr_for_poet(poet_lr, p)  # p.shape[0] drives scaling
                if group.get("use_poet", False) and group.get("poet_scale", 1.0) > 0.0:
                    lr_eff = lr_eff * group.get("poet_scale", 1.0)
                step_size = lr_eff

                # group['correct_bias'] = False
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # p.addcdiv_(exp_avg, denom, value=-step_size)
                # compute norm gradient
                norm_grad = exp_avg / denom

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    # p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    p.add_(p, alpha=(-lr_eff * group["weight_decay"]))

                

        self.global_step_counter += 1
        return loss