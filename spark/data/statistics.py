"""Dataset statistics and metadata utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

METADATA_DIM: int = 8


def _normalize_divisor(value: int) -> float:
    return float(value) if value else 1.0


def metadata_from_tokens(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute metadata features for each sequence in ``tokens``.

    The returned tensor has ``METADATA_DIM`` features capturing distributional
    properties of the batch that are useful for conditioning the
    :class:`~spark.layer_generator.LayerGenerator` hypernetwork.
    """

    if tokens.ndim != 2:
        raise ValueError("Expected rank-2 token tensor with shape (batch, seq_len)")
    tokens_f = tokens.to(dtype=torch.float32)
    batch, seq_len = tokens_f.shape
    if seq_len == 0:
        raise ValueError("Cannot compute metadata for empty sequences")
    vmax = max(vocab_size - 1, 1)
    mean = tokens_f.mean(dim=1)
    std = tokens_f.std(dim=1, unbiased=False)
    min_vals, _ = tokens_f.min(dim=1)
    max_vals, _ = tokens_f.max(dim=1)
    first_token = tokens_f[:, 0]
    last_token = tokens_f[:, -1]
    length = torch.full((batch,), float(seq_len), dtype=torch.float32, device=tokens_f.device)
    unique_counts = torch.tensor(
        [float(torch.unique(tokens[i]).numel()) for i in range(batch)],
        dtype=torch.float32,
        device=tokens_f.device,
    )
    metadata = torch.stack(
        [
            mean / vmax,
            std / _normalize_divisor(vocab_size),
            min_vals / vmax,
            max_vals / vmax,
            unique_counts / _normalize_divisor(vocab_size),
            first_token / vmax,
            last_token / vmax,
            length / float(seq_len),
        ],
        dim=1,
    )
    return metadata


@dataclass
class DatasetStatistics:
    """Summary statistics required for deterministic instruction seeds."""

    vocab_size: int
    sequence_length: int
    summary: torch.Tensor

    @classmethod
    def from_tokens(cls, tokens: torch.Tensor, vocab_size: int) -> "DatasetStatistics":
        if tokens.ndim != 2:
            raise ValueError("Expected rank-2 tensor with shape (num_samples, seq_len)")
        metadata = metadata_from_tokens(tokens, vocab_size)
        summary = metadata.mean(dim=0)
        return cls(vocab_size=vocab_size, sequence_length=tokens.size(1), summary=summary)

    @classmethod
    def synthetic(cls, vocab_size: int, sequence_length: int) -> "DatasetStatistics":
        base = torch.arange(sequence_length, dtype=torch.int64)
        samples = base.repeat(vocab_size, 1) % max(vocab_size, 1)
        return cls.from_tokens(samples, vocab_size)

    def to_metadata_tensor(self) -> torch.Tensor:
        tensor = self.summary.detach().clone()
        if tensor.numel() != METADATA_DIM:
            raise ValueError(
                f"Summary metadata must contain {METADATA_DIM} values, got {tensor.numel()}"
            )
        return tensor

    def seed_for_layer(self, layer_index: int, embedding: Optional[torch.Tensor] = None) -> int:
        """Derive a reproducible RNG seed for ``layer_index``.

        The seed deterministically combines dataset statistics with an optional
        embedding vector. This keeps procedural weight decoding reproducible
        across training and inference runs.
        """

        if layer_index < 0:
            raise ValueError("Layer index must be non-negative")
        stats_vector = torch.cat(
            [
                torch.tensor(
                    [float(self.vocab_size), float(self.sequence_length)],
                    dtype=torch.float32,
                )
                / _normalize_divisor(self.vocab_size),
                self.to_metadata_tensor(),
            ]
        )
        if embedding is not None:
            embed_vec = embedding.to(dtype=torch.float32).flatten()
            stats_vector = torch.cat([stats_vector, embed_vec])
        hash_value = torch.sum(stats_vector * (layer_index + 1) * 1000.0).item()
        seed = int(abs(hash_value)) % (2**31 - 1)
        if seed == 0:
            seed = layer_index + 1
        return seed

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "summary": self.summary.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetStatistics":
        summary = torch.tensor(payload["summary"], dtype=torch.float32)
        return cls(
            vocab_size=int(payload["vocab_size"]),
            sequence_length=int(payload["sequence_length"]),
            summary=summary,
        )


def serialize_metadata_tensor(metadata: torch.Tensor) -> dict:
    """Serialize a metadata tensor to a portable dictionary."""

    return {
        "shape": list(metadata.shape),
        "values": metadata.detach().cpu().tolist(),
    }


def deserialize_metadata_tensor(payload: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """Inverse of :func:`serialize_metadata_tensor`."""

    tensor = torch.tensor(payload["values"], dtype=torch.float32, device=device)
    shape: Tuple[int, ...] = tuple(payload["shape"])
    if int(torch.tensor(shape).prod().item()) != tensor.numel():
        raise ValueError("Serialized metadata has inconsistent shape")
    return tensor.view(*shape)


__all__ = [
    "DatasetStatistics",
    "METADATA_DIM",
    "metadata_from_tokens",
    "serialize_metadata_tensor",
    "deserialize_metadata_tensor",
]
