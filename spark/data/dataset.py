"""Synthetic data module emitting token, metadata and target tuples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

from .statistics import DatasetStatistics, metadata_from_tokens


@dataclass
class ProceduralDatasetConfig:
    vocab_size: int
    sequence_length: int
    batch_size: int
    num_batches: Optional[int] = None
    seed: int = 0
    device: Optional[torch.device] = None


class ProceduralDataset:
    """Generates reproducible batches of tokens and metadata."""

    def __init__(self, config: ProceduralDatasetConfig, preview_batches: int = 8) -> None:
        self.config = config
        self._device = config.device or torch.device("cpu")
        self._seed = config.seed
        self._preview_batches = max(preview_batches, 1)
        self._statistics = self._bootstrap_statistics()

    @property
    def statistics(self) -> DatasetStatistics:
        return self._statistics

    def iter_batches(self, num_batches: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        limit = num_batches if num_batches is not None else self.config.num_batches
        produced = 0
        generator = torch.Generator(device=self._device).manual_seed(self._seed)
        while limit is None or produced < limit:
            tokens = torch.randint(
                0,
                self.config.vocab_size,
                (self.config.batch_size, self.config.sequence_length),
                generator=generator,
                dtype=torch.int64,
                device=self._device,
            )
            metadata = metadata_from_tokens(tokens, self.config.vocab_size)
            targets = torch.roll(tokens, shifts=-1, dims=1)
            produced += 1
            yield tokens, metadata, targets

    def _bootstrap_statistics(self) -> DatasetStatistics:
        preview = []
        generator = torch.Generator(device=self._device).manual_seed(self._seed)
        for _ in range(self._preview_batches):
            tokens = torch.randint(
                0,
                self.config.vocab_size,
                (self.config.batch_size, self.config.sequence_length),
                generator=generator,
                dtype=torch.int64,
            )
            preview.append(tokens.cpu())
        if not preview:
            raise RuntimeError("Failed to materialize preview batches for statistics")
        sample_tokens = torch.cat(preview, dim=0)
        return DatasetStatistics.from_tokens(sample_tokens, self.config.vocab_size)


__all__ = ["ProceduralDataset", "ProceduralDatasetConfig"]
