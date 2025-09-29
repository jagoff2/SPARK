"""Dense MLP baseline models for procedural model comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BaselineMLPConfig:
    """Configuration for a feed-forward language model baseline."""

    input_dim: int
    hidden_dim: int
    vocab_size: int
    num_layers: int = 2


class BaselineMLP(torch.nn.Module):
    """Simple multi-layer perceptron baseline for language modelling."""

    def __init__(self, config: BaselineMLPConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        in_features = config.input_dim
        for _ in range(config.num_layers):
            layers.append(torch.nn.Linear(in_features, config.hidden_dim))
            layers.append(torch.nn.GELU())
            in_features = config.hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.output = torch.nn.Linear(in_features, config.vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run the baseline network."""

        hidden = self.backbone(inputs)
        return self.output(hidden)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


def estimate_mlp_parameters(config: BaselineMLPConfig) -> int:
    """Estimate the number of trainable parameters for the given configuration."""

    dims = [config.input_dim]
    dims.extend([config.hidden_dim] * config.num_layers)
    dims.append(config.vocab_size)
    params = 0
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        params += in_dim * out_dim  # weights
        params += out_dim  # biases
    return params


def match_mlp_hidden_dim(
    reference_params: int,
    input_dim: int,
    vocab_size: int,
    num_layers: int = 2,
    search_bounds: Tuple[int, int] = (16, 4096),
) -> BaselineMLPConfig:
    """Find an MLP width whose parameter count best matches the reference model."""

    lower, upper = search_bounds
    best_config = None
    best_error = float("inf")
    for hidden in range(lower, upper + 1, 8):
        candidate = BaselineMLPConfig(
            input_dim=input_dim,
            hidden_dim=hidden,
            vocab_size=vocab_size,
            num_layers=num_layers,
        )
        params = estimate_mlp_parameters(candidate)
        error = abs(params - reference_params)
        if error < best_error:
            best_error = error
            best_config = candidate
        if error == 0:
            break
    if best_config is None:
        raise ValueError("Failed to match baseline hidden dimension")
    return best_config


__all__ = ["BaselineMLP", "BaselineMLPConfig", "estimate_mlp_parameters", "match_mlp_hidden_dim"]
