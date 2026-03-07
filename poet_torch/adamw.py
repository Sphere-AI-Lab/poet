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
        threshold: Gradient threshold for numerical stability. Default: 5000.
        poet_reset_gap: Steps between resetting optimizer state for POET params.
            Default: 0 (disabled).
        poet_block_size: Block size for POET transformations. Default: 256.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        threshold: int = 5000,
        poet_reset_gap: int = 0,
        poet_block_size: int = 256
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "stochastic": False,
            "poet_reset_gap": poet_reset_gap,
        }
        super().__init__(params, defaults)

        self.thres = threshold
        self.global_step_counter = 0
        self.poet_block_size = poet_block_size

    def adjust_lr_for_poet(self, poet_lr: float, p: torch.Tensor) -> float:
        """Scale learning rate for POET parameters.
        
        The scaling is based on the number of blocks in the parameter tensor.
        
        Args:
            poet_lr: Base learning rate for POET parameters.
            p: POET parameter tensor shaped (r, n_elements).
            
        Returns:
            Scaled learning rate.
        """
        # Scale by sqrt(r/2) * poet_scale for POET parameters
        scaling = math.sqrt(p.shape[0] / 2)
        return poet_lr * scaling

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.
        
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
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Check if reset should be applied based on global counter
                reset_gap = group.get("poet_reset_gap", 0)
                if (
                    reset_gap > 0
                    and self.global_step_counter % reset_gap == 0
                    and self.global_step_counter > 0
                ):
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Compute bias-corrected denominator
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Calculate effective learning rate
                lr_eff = group["lr"]
                
                # Apply POET-specific learning rate scaling if enabled
                if group.get("use_poet", False):
                    poet_scale = group.get("poet_scale", 1.0)
                    if poet_scale > 0.0:
                        lr_eff = lr_eff * poet_scale
                
                step_size = lr_eff

                # Apply bias correction if enabled
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # Update parameters
                norm_grad = exp_avg / denom
                p.add_(norm_grad, alpha=-step_size)

                # Apply weight decay (decoupled from adaptive gradients)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-lr_eff * group["weight_decay"]))

        self.global_step_counter += 1
        return loss
