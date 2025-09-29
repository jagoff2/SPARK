"""Baseline reference models for SPARK experiments."""
from .mlp import (
    BaselineMLP,
    BaselineMLPConfig,
    estimate_mlp_parameters,
    match_mlp_hidden_dim,
)

__all__ = [
    "BaselineMLP",
    "BaselineMLPConfig",
    "estimate_mlp_parameters",
    "match_mlp_hidden_dim",
]
