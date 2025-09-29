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
    mask = torch.zeros_like(router_scores)
    mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=mask.dtype))
    compressed = torch.einsum("bsh,bsnh->bsn", hidden, mask)
    return indices, compressed


__all__ = ["route_tokens"]
