"""POET: Parameter-Efficient Orthogonal Transformation for PyTorch.

POET is a reparameterized training method that optimizes weight matrices through
Orthogonal Equivalence Transformation (OET), achieving superior generalization
with provably bounded weight spectra.

Quick Start:
    >>> from poet_torch import POETConfig, POETModel, get_poet_optimizer
    >>> 
    >>> # Configure POET
    >>> config = POETConfig(block_size=256, merge_interval=200)
    >>> 
    >>> # Wrap your model
    >>> model = POETModel(your_model, config)
    >>> 
    >>> # Create optimizer
    >>> optimizer = get_poet_optimizer(model, config)
    >>> 
    >>> # Training loop
    >>> for step, batch in enumerate(dataloader):
    ...     loss = model(**batch)
    ...     loss.backward()
    ...     optimizer.step()
    ...     model.merge_if_needed(step)

For more information, visit: https://github.com/Sphere-AI-Lab/poet
"""

__version__ = "0.0.1"

# Core configuration
from poet_torch.config import POETConfig, QPOETConfig

# Model wrapper
from poet_torch.model_wrapper import POETModel, get_poet_optimizer

# Optimizer
from poet_torch.optim import POETAdamW

# Layers (for advanced usage)
from poet_torch.layers import POETLinear, QPOETLinear

# Utilities
from poet_torch.utils import (
    replace_linear_with_poet,
    convert_to_qpoet,
    merge_and_reinitialize,
    calc_poet_grad_clipping_value,
    get_poet_params,
    get_model_info,
    print_model_info,
)

__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "POETConfig",
    "QPOETConfig",
    
    # Main API
    "POETModel",
    "POETAdamW",
    "get_poet_optimizer",
    
    # Layers
    "POETLinear",
    "QPOETLinear",
    
    # Utilities
    "replace_linear_with_poet",
    "convert_to_qpoet",
    "merge_and_reinitialize",
    "calc_poet_grad_clipping_value",
    "get_poet_params",
    "get_model_info",
    "print_model_info",
]


def _check_dependencies():
    """Check for optional dependencies and warn if missing."""
    import warnings
    
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available. POET requires CUDA for optimal performance.",
                UserWarning
            )
    except ImportError:
        raise ImportError("PyTorch is required but not installed.")
    
    try:
        import triton
    except ImportError:
        warnings.warn(
            "Triton is not installed. Some features (memory-efficient mode, "
            "quantized POET) will not be available. Install with: pip install triton",
            UserWarning
        )


# Run dependency check on import
_check_dependencies()
