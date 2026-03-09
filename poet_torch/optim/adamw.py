"""POET AdamW optimizer with support for POET parameter groups."""

import math
import warnings
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


class POETAdamW(Optimizer):
    """Implements Adam algorithm with weight decay fix.
    
    This is an extension of the standard AdamW optimizer that adds support
    for POET (Parameter-Efficient Orthogonal Transformations) parameter groups
    with custom learning rate scaling.
    
    Reference:
        Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
    
    Args:
        params: Iterable of parameters to optimize or dictionaries defining
            parameter groups.
        lr: Learning rate. Default: 1e-3.
        betas: Adam's beta parameters (b1, b2). Default: (0.9, 0.999).
        eps: Adam's epsilon for numerical stability. Default: 1e-6.
        weight_decay: Decoupled weight decay to apply. Default: 0.0.
        correct_bias: Whether to correct bias in Adam. Default: True.
        poet_merge_interval: Steps between resetting optimizer state for POET params.
            Default: 0 (disabled).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        poet_scale: float = 0.5,
        poet_merge_interval: int = 0,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "poet_scale": poet_scale,
            "poet_merge_interval": poet_merge_interval,
        }
        super().__init__(params, defaults)

        self.global_step_counter = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            
        Returns:
            The loss value if closure is provided, None otherwise.
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
                    raise RuntimeError("POETAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Check if reset should be applied based on global counter
                merge_interval = 0
                if group.get("use_poet", False):
                    merge_interval = group.get(
                        "poet_merge_interval", self.defaults["poet_merge_interval"]
                    )
                if (
                    merge_interval > 0
                    and self.global_step_counter % merge_interval == 0
                    and self.global_step_counter > 0
                ):
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Compute bias-corrected denominator
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Calculate effective learning rate
                lr_eff = group["lr"]
                
                # Apply POET-specific learning rate scaling if enabled
                if group.get("use_poet", False):
                    poet_scale = group.get("poet_scale", self.defaults["poet_scale"])
                    if poet_scale > 0.0:
                        lr_eff = lr_eff * poet_scale
                
                step_size = lr_eff

                # Apply bias correction if enabled
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply weight decay (decoupled from adaptive gradients)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-lr_eff * group["weight_decay"]))

        self.global_step_counter += 1
        return loss
