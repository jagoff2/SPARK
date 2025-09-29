"""High-level integration of SPARK procedural components."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from .attention import AttentionBasisSynthesizer, AtomConfig
from .data import (
    METADATA_DIM,
    DatasetStatistics,
    InstructionSeedSet,
    materialize_instructions,
)
from .demopack import CodebookSpec, DemopackCodebook, DemopackDecoder
from .kv_patcher import KVCachePatcher
from .layer_generator import GeneratorConfig, LayerGenerator, apply_lora_delta
from .opcode_vm import Instruction, OpcodeVM


@dataclass
class ProceduralModelConfig:
    input_dim: int
    hidden_dim: int
    vocab_size: int
    codebook_spec: CodebookSpec
    generator_config: GeneratorConfig
    dataset_statistics: Optional[DatasetStatistics] = None
    instruction_seed_set: Optional[InstructionSeedSet] = None


class ProceduralLanguageModel(torch.nn.Module):
    """Toy language model glueing together procedural components."""

    def __init__(self, config: ProceduralModelConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = DemopackCodebook(config.codebook_spec)
        tile_rows = max(1, min(16, config.codebook_spec.embedding_dim))
        num_tiles = max(1, math.ceil(config.hidden_dim / tile_rows))
        self.dataset_statistics = config.dataset_statistics or DatasetStatistics.synthetic(
            vocab_size=config.vocab_size,
            sequence_length=config.input_dim,
        )
        if config.instruction_seed_set is not None:
            self.instruction_seeds = config.instruction_seed_set
            self.instruction_seeds.ensure_length(num_tiles)
        else:
            self.instruction_seeds = InstructionSeedSet.from_statistics(
                self.dataset_statistics,
                num_layers=num_tiles,
            )
        instructions = materialize_instructions(
            seed_set=self.instruction_seeds,
            num_layers=num_tiles,
            tile_shape=(tile_rows, config.input_dim),
            codebook_size=config.codebook_spec.num_codewords,
        )
        self.decoder = DemopackDecoder(
            codebook=self.codebook,
            out_features=num_tiles * tile_rows,
            in_features=config.input_dim,
            instructions=instructions,
        )
        self.generator = LayerGenerator(config.generator_config, metadata_dim=METADATA_DIM)
        self.lm_head = torch.nn.Linear(num_tiles * tile_rows, config.vocab_size)
        atom_cfgs = [
            AtomConfig(name="sin", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ] + [
            AtomConfig(name="cos", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ]
        self.attention = AttentionBasisSynthesizer(atom_cfgs, max_length=1024)
        self.kv_patcher = KVCachePatcher()
        self.vm = OpcodeVM()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(inputs)
        logits = self.lm_head(hidden)
        return logits

    def apply_generator_delta(self, metadata: torch.Tensor) -> None:
        base = self.lm_head.weight.data.clone()
        delta = self.generator(metadata, base.size(1), base.size(0))
        updated = apply_lora_delta(base, delta)
        self.lm_head.weight.data.copy_(updated)

    def reason(self, instructions: List[Instruction]) -> List[str]:
        state = self.vm.execute(instructions)
        return state.log


__all__ = ["ProceduralModelConfig", "ProceduralLanguageModel"]
