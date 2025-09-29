"""Procedural layer generator producing LoRA-style deltas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class GeneratorConfig:
    """Configuration for the layer generator hypernetwork."""

    embed_dim: int
    hidden_dim: int
    rank: int
    num_layers: int = 2


class LayerGenerator(torch.nn.Module):
    """Generates low-rank weight updates conditioned on metadata."""

    def __init__(self, config: GeneratorConfig, metadata_dim: int) -> None:
        super().__init__()
        self.config = config
        layers = [torch.nn.Linear(metadata_dim, config.embed_dim), torch.nn.GELU()]
        for _ in range(config.num_layers - 1):
            layers.append(torch.nn.Linear(config.embed_dim, config.embed_dim))
            layers.append(torch.nn.GELU())
        self.backbone = torch.nn.Sequential(*layers)
        self.up_proj = torch.nn.Linear(config.embed_dim, config.rank * config.embed_dim)
        self.down_proj = torch.nn.Linear(config.embed_dim, config.rank * config.embed_dim)

    def forward(
        self,
        metadata: torch.Tensor,
        input_dim: int,
        output_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(metadata)
        up = self.up_proj(embedding).view(-1, self.config.rank, self.config.embed_dim)
        down = self.down_proj(embedding).view(-1, self.config.rank, self.config.embed_dim)
        a = up.mean(dim=0)[:, :input_dim]
        b = down.mean(dim=0)[:, :output_dim]
        return a, b


def apply_lora_delta(
    base_weight: torch.Tensor,
    delta: Tuple[torch.Tensor, torch.Tensor],
    alpha: float = 1.0,
) -> torch.Tensor:
    """Applies a LoRA-style low-rank update to the base weight."""

    a, b = delta
    update = torch.matmul(b.t(), a)
    return base_weight + alpha * update


__all__ = ["GeneratorConfig", "LayerGenerator", "apply_lora_delta"]
