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

    batch, seq_len, _ = hidden.shape
    num_active = max(1, int(num_neurons * sparsity))
    router_scores = torch.randn(batch, seq_len, num_neurons, device=hidden.device)
    topk = torch.topk(router_scores, num_active, dim=-1)
    indices = topk.indices
    # Convert router scores into normalized dispatch weights for the selected
    # neurons and broadcast the token representations to each active slot.
    weights = torch.softmax(topk.values, dim=-1)
    compressed = hidden.unsqueeze(-2) * weights.unsqueeze(-1)
    return indices, compressed


__all__ = ["route_tokens"]
