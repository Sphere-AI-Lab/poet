import torch
import torch.distributed as dist
import math


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonOptimized(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        # adamw_betas=(0.9, 0.95),
        adamw_betas=(0.9, 0.999),
        # adamw_eps=1e-8,
        adamw_eps=1e-6,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Detect Distributed Context
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            muon_params = [p for p in group["params"] if self.state[p].get("use_muon", False) and p.grad is not None]

            if muon_params:
                # Sort descending by size (optimization for all_gather)
                # Note: If you want to strictly match Imp 1 order, remove this sort, 
                # but sorting helps load balancing in the sharded loop.
                muon_params.sort(key=lambda x: x.numel(), reverse=True)

                lr = group["lr"]
                momentum = group["momentum"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                
                # 2. Pad list for equal distribution
                total_params = len(muon_params)
                params_pad = muon_params
                if total_params % world_size != 0:
                    pad_len = world_size - (total_params % world_size)
                    params_pad = muon_params + [muon_params[-1]] * pad_len

                # 3. Distributed Sharding Loop (The Speed Fix)
                for base_i in range(len(params_pad))[::world_size]:
                    local_idx = base_i + rank
                    
                    if local_idx < len(params_pad):
                        p = params_pad[local_idx]
                        
                        # Only compute if this is a real parameter (not padding)
                        if local_idx < total_params:
                            g = p.grad
                            if g.ndim > 2:
                                g = g.view(g.size(0), -1)

                            state = self.state[p]
                            if "momentum_buffer" not in state:
                                state["momentum_buffer"] = torch.zeros_like(g)
                            
                            # --- Implementation 1 Momentum ---
                            buf = state["momentum_buffer"]
                            buf.mul_(momentum).add_(g)
                            if nesterov:
                                g = g.add(buf, alpha=momentum)
                            else:
                                g = buf

                            # --- Newton-Schulz (Expensive Part) ---
                            u = zeropower_via_newtonschulz5(g, steps=ns_steps)

                            # --- Implementation 1 Scaling ---
                            adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                            # Apply Weight Decay (Note: Imp 1 applies WD to data, separate from update)
                            # If no specific Muon WD was passed, we assume 0 or handled elsewhere, 
                            # but Imp 1 hardcoded `p.data.mul_(1 - lr * wd)`. 
                            # Since Imp 1 didn't have a 'wd' in defaults for Muon (only `wd` arg), 
                            # we assume the user passed `wd` to __init__.
                            # We use group['adamw_wd'] as a fallback if not present, or 0.
                            # Ideally, pass `wd` in defaults. Here we check group keys.
                            wd = group.get("wd", 0.0) # Default to 0 if not specified in group
                            if wd > 0:
                                p.data.mul_(1 - lr * wd)

                            # Apply Update
                            p.data.add_(u, alpha=-adjusted_lr)

                    # 4. Sync Results (All-Gather)
                    if world_size > 1:
                        # Broadcast the updated weights (p.data) to all GPUs
                        current_slice = params_pad[base_i : base_i + world_size]
                        p_owner = params_pad[base_i + rank]
                        dist.all_gather(current_slice, p_owner)

            ############################
            #       AdamW backup       #
            ############################

            adam_params = [p for p in group["params"] if not self.state[p].get("use_muon", False) and p.grad is not None]
            
            if adam_params:
                lr = group['lr']
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                weight_decay = group["wd"]

                for p in adam_params:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g, alpha=-lr / scale)

        return loss
