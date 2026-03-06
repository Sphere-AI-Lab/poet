"""POET: Parameter-Efficient Orthogonal Transformations for Fine-tuning.

This module provides custom layers and optimizers for efficient fine-tuning
of large language models using orthogonal transformations.
"""

from .adamw import POETAdamW
from .poet_layer import POETLinear, QPOETLinear
from .poet_utils import (
    check_and_merge,
    get_grad_clipping_value,
    prepare_model_for_int8_training_poet,
    replace_linear_with_poet,
)

__all__ = [
    "POETAdamW",
    "POETLinear",
    "QPOETLinear",
    "check_and_merge",
    "get_grad_clipping_value",
    "prepare_model_for_int8_training_poet",
    "replace_linear_with_poet",
]
