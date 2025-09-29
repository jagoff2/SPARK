"""Hash-based sparse router kernels implemented in PyTorch."""
from __future__ import annotations

from typing import Tuple

import torch


def route_tokens(
    hidden: torch.Tensor,
    sparsity: float,
    num_neurons: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a sparse subset of neurons per token using random hashes."""


    indices, _, compressed = route_tokens_with_metadata(
        hidden, sparsity=sparsity, num_neurons=num_neurons
    )

    return indices, compressed


def route_tokens_with_metadata(
    hidden: torch.Tensor,
    sparsity: float,
    num_neurons: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Route tokens while also returning dispatch weights."""


    batch, seq_len, _ = hidden.shape
    num_active = max(1, int(num_neurons * sparsity))
    router_scores = torch.randn(batch, seq_len, num_neurons, device=hidden.device)
    topk = torch.topk(router_scores, num_active, dim=-1)
    indices = topk.indices
    weights = torch.softmax(topk.values, dim=-1)
    compressed = hidden.unsqueeze(-2) * weights.unsqueeze(-1)
    return indices, weights, compressed


__all__ = ["route_tokens", "route_tokens_with_metadata"]
