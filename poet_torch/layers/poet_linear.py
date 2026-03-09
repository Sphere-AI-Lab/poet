"""POET Linear Layer implementation."""

import math
from typing import Optional

import torch
import torch.nn as nn

from poet_torch.core.ops import (
    block_diag_lr_matmul,
    forward_core,
    get_weight_poet,
)


class POETLinear(nn.Module):
    """POET linear layer with orthogonal transformations.
    
    This layer implements the POET reparameterization: W_RP = R_out * W_0 * R_in,
    where W_0 is a fixed base weight matrix, and R_out, R_in are learnable orthogonal
    matrices parameterized via Cayley transform.
    
    The orthogonal transformations are implemented efficiently using block-diagonal
    structures with random permutations.
    
    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        block_size: Block size for transformations. Both in_features and
            out_features must be divisible by block_size.
        bias: Whether to include bias term. Default: False.
        bias_requires_grad: Whether to train the bias term. Default: False.
        device: Device for parameters. Default: None.
        dtype: Data type for parameters. Default: None.
        mem_efficient_mode: Whether to use memory-efficient mode (recomputes
            activations during backward). Default: False.
    
    Example:
        >>> layer = POETLinear(512, 512, block_size=64)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, features)
        >>> y = layer(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 256,
        bias: bool = False,
        bias_requires_grad: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        mem_efficient_mode: bool = False,
    ):
        super().__init__()
        
        # Validate dimensions
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"block_size ({block_size})"
            )
        if out_features % block_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"block_size ({block_size})"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.mem_efficient_mode = mem_efficient_mode

        # Base weight matrix (frozen, not trained)
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        
        # Bias term (frozen, not trained)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=bias_requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        # Calculate number of blocks
        self.num_in_blocks = in_features // block_size
        self.num_out_blocks = out_features // block_size
        n_elements = block_size * (block_size - 1) // 2
        
        # Trainable skew-symmetric parameters for orthogonal matrices
        # Shape: (num_in_blocks + num_out_blocks, n_elements)
        self.oft_R = nn.Parameter(
            torch.zeros((self.num_in_blocks + self.num_out_blocks, n_elements), 
                       device=device, dtype=dtype)
        )

        # Register buffers for skew-symmetric construction
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # Register buffers for permutations (randomly initialized)
        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        """Merge POET transformations into base weight and reinitialize.
        
        This operation:
        1. Computes the current orthogonal matrices R_out and R_in from parameters
        2. Applies them to the base weight: W_new = R_out * W_old * R_in
        3. Generates new random permutations
        4. Resets the trainable parameters
        """
        # Compute orthogonal matrices
        R_out, R_in = get_weight_poet(
            self.oft_R, 
            self.block_size, 
            self.rows, 
            self.cols, 
            self.num_out_blocks, 
            self.num_in_blocks
        )

        # Apply transformations: W = R_out * W_old * R_in
        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        W_new = tmp.t()

        # Generate new permutations
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Apply new permutation to weight
        W_new = W_new.index_select(0, perm_out_inv).index_select(1, perm_in_inv)

        self.weight.detach().copy_(W_new)

        # Update parameters and buffers
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)

        self.oft_R.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying POET transformations.
        """
        return forward_core(
            x,
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.perm_in,
            self.perm_in_inv,
            self.perm_out,
            self.perm_out_inv,
            self.num_in_blocks,
            self.num_out_blocks,
            self.weight,
            self.bias,
            self.mem_efficient_mode,
        )


    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective weight matrix W_eff = R_out * W_0 * R_in.
        
        Returns:
            The effective weight matrix of shape (out_features, in_features).
        """
        R_out, R_in = get_weight_poet(
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.num_out_blocks,
            self.num_in_blocks
        )
        
        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        return tmp.t()

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"block_size={self.block_size}, bias={self.bias is not None}, "
                f"mem_efficient_mode={self.mem_efficient_mode}")
