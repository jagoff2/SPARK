"""Instruction seed management for deterministic decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from spark.demopack import DecodeInstruction

from .statistics import DatasetStatistics


@dataclass
class InstructionSeedSet:
    """Container tracking per-layer instruction seeds."""

    statistics: DatasetStatistics
    seeds: List[int] = field(default_factory=list)
    embedding: Optional[torch.Tensor] = None

    @classmethod
    def from_statistics(
        cls,
        statistics: DatasetStatistics,
        num_layers: int,
        embedding: Optional[torch.Tensor] = None,
    ) -> "InstructionSeedSet":
        seed_list = [statistics.seed_for_layer(i, embedding) for i in range(num_layers)]
        return cls(statistics=statistics, seeds=seed_list, embedding=embedding)

    def ensure_length(self, num_layers: int) -> None:
        while len(self.seeds) < num_layers:
            layer_index = len(self.seeds)
            self.seeds.append(self.statistics.seed_for_layer(layer_index, self.embedding))

    def to_dict(self) -> dict:
        return {
            "statistics": self.statistics.to_dict(),
            "seeds": list(self.seeds),
            "embedding": None
            if self.embedding is None
            else self.embedding.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InstructionSeedSet":
        statistics = DatasetStatistics.from_dict(payload["statistics"])
        embedding = payload.get("embedding")
        tensor_embed = None
        if embedding is not None:
            tensor_embed = torch.tensor(embedding, dtype=torch.float32)
        return cls(statistics=statistics, seeds=list(payload.get("seeds", [])), embedding=tensor_embed)


def materialize_instructions(
    seed_set: InstructionSeedSet,
    num_layers: int,
    tile_shape: Tuple[int, int],
    codebook_size: int,
    device: Optional[torch.device] = None,
) -> List[DecodeInstruction]:
    """Decode deterministic instruction tensors from ``seed_set``."""

    seed_set.ensure_length(num_layers)
    rows, cols = tile_shape
    instructions: List[DecodeInstruction] = []
    for layer_index in range(num_layers):
        seed = seed_set.seeds[layer_index]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        indices = torch.randint(
            0,
            codebook_size,
            (rows, cols),
            dtype=torch.int32,
            generator=generator,
            device=device,
        )
        instructions.append(DecodeInstruction(codeword_indices=indices, scale=1.0))
    return instructions


__all__ = ["InstructionSeedSet", "materialize_instructions"]
