"""High-level integration of SPARK procedural components."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .attention import AttentionBasisSynthesizer, AtomConfig, AttentionPatch
from .data import (
    METADATA_DIM,
    DatasetStatistics,
    InstructionSeedSet,
    materialize_instructions,
)
from .demopack import CodebookSpec, DemopackCodebook, DemopackDecoder
from .hash_router import route_tokens_with_metadata
from .kv_patcher import KVCachePatcher
from .layer_generator import GeneratorConfig, LayerGenerator, apply_lora_delta
from .opcode_vm import Instruction, OpcodeVM


@dataclass
class ProceduralModelConfig:
    """Configuration bundle for :class:`ProceduralLanguageModel`."""

    input_dim: int
    hidden_dim: int
    vocab_size: int
    codebook_spec: CodebookSpec
    generator_config: GeneratorConfig
    dataset_statistics: Optional[DatasetStatistics] = None
    instruction_seed_set: Optional[InstructionSeedSet] = None
    attention_enabled: bool = True
    routing_enabled: bool = True
    kv_cache_enabled: bool = True
    router_sparsity: float = 0.25
    router_num_neurons: int = 64
    metadata_dim: int = METADATA_DIM
    codebook_learnable: bool = False


class ProceduralLanguageModel(torch.nn.Module):
    """Toy language model glueing together procedural components."""

    def __init__(self, config: ProceduralModelConfig) -> None:
        super().__init__()
        self.config = config

        self.codebook = DemopackCodebook(
            config.codebook_spec, learnable=config.codebook_learnable
        )

        # Determine tiling so that the decoder output matches ``hidden_dim``.
        max_rows = max(1, min(config.codebook_spec.embedding_dim, config.hidden_dim))
        tile_rows = math.gcd(config.hidden_dim, max_rows)
        if tile_rows == 0:
            tile_rows = 1
        num_tiles = max(1, config.hidden_dim // tile_rows)

        # Prepare statistics and instruction seeds.
        if config.dataset_statistics is not None:
            self.dataset_statistics = config.dataset_statistics
        else:
            self.dataset_statistics = DatasetStatistics.synthetic(
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
            out_features=config.hidden_dim,
            in_features=config.input_dim,
            instructions=instructions,
        )

        self.metadata_dim = config.metadata_dim
        self.generator = LayerGenerator(config.generator_config, metadata_dim=self.metadata_dim)

        self.lm_head = torch.nn.Linear(config.hidden_dim, config.vocab_size)
        self.token_proj = torch.nn.Linear(config.input_dim, config.hidden_dim)
        self.query_proj = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn_output_proj = torch.nn.Linear(config.hidden_dim, config.input_dim)

        atom_cfgs = [
            AtomConfig(name="sin", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ] + [
            AtomConfig(name="cos", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ]
        self.attention = AttentionBasisSynthesizer(atom_cfgs, max_length=1024)
        self.kv_patcher = KVCachePatcher()
        self.vm = OpcodeVM()
        self.latest_routing_metadata: List[Optional[Dict[str, torch.Tensor]]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        elif inputs.ndim != 3:
            raise ValueError("Inputs must be rank-2 or rank-3 tensor")

        _, seq_len, _ = inputs.shape
        device = inputs.device

        tokens = self.token_proj(inputs)
        queries = self.query_proj(tokens)
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)

        decoder_inputs: List[torch.Tensor] = []
        routing_metadata: List[Optional[Dict[str, torch.Tensor]]] = []

        for step in range(seq_len):
            query = queries[:, step : step + 1, :]
            if self.config.kv_cache_enabled:
                segment_id = step
                self.kv_patcher.mark(segment_id)
                try:
                    cached_keys, cached_values = self.kv_patcher.gather([segment_id])
                except KeyError:
                    cached_keys = keys[:, : step + 1, :]
                    cached_values = values[:, : step + 1, :]
                else:
                    if cached_keys.size(1) < step + 1:
                        new_keys = keys[:, cached_keys.size(1) : step + 1, :]
                        new_values = values[:, cached_keys.size(1) : step + 1, :]
                        cached_keys = torch.cat([cached_keys, new_keys], dim=1)
                        cached_values = torch.cat([cached_values, new_values], dim=1)
                self.kv_patcher.update(segment_id, cached_keys, cached_values)
            else:
                cached_keys = keys[:, : step + 1, :]
                cached_values = values[:, : step + 1, :]

            if self.config.attention_enabled:
                patch = self._build_attention_patch(query.size(1), device)
                attn_out = self.attention.apply_attention(
                    patch, query, cached_keys, cached_values
                )
            else:
                attn_out = query

            if self.config.routing_enabled:
                sparse, dense, metadata = self._compile_routing_metadata(attn_out)
                combined = sparse.sum(dim=-2)
                fallback = dense.squeeze(1)
                combined = combined.squeeze(1)
                combined = combined + (fallback - fallback.detach())
                routing_metadata.append(metadata)
            else:
                combined = attn_out.squeeze(1)
                routing_metadata.append(None)

            projected = self.attn_output_proj(combined)
            decoder_inputs.append(projected)

        if self.config.kv_cache_enabled:
            self.kv_patcher.sweep()

        self.latest_routing_metadata = routing_metadata

        stacked = torch.stack(decoder_inputs, dim=1)
        decoder_input = stacked.mean(dim=1)
        hidden = self.decoder(decoder_input)
        logits = self.lm_head(hidden)
        return logits

    def apply_generator_delta(self, metadata: torch.Tensor) -> None:
        if metadata.ndim == 1:
            metadata = metadata.unsqueeze(0)
        elif metadata.ndim != 2:
            raise ValueError("Metadata tensor must be rank-1 or rank-2")

        base = self.lm_head.weight.data.clone()
        delta = self.generator(metadata, base.size(1), base.size(0))
        updated = apply_lora_delta(base, delta)
        self.lm_head.weight.data.copy_(updated)

    def reason(self, instructions: List[Instruction]) -> List[str]:
        state = self.vm.execute(instructions)
        return state.log

    def _build_attention_patch(self, seq_len: int, device: torch.device) -> AttentionPatch:
        atom_indices = torch.arange(len(self.attention.atoms), device=device, dtype=torch.long)
        gains = torch.ones(atom_indices.size(0), device=device, dtype=torch.float32)
        shifts = torch.zeros_like(atom_indices)
        window = torch.ones(seq_len, device=device, dtype=torch.float32)
        return AttentionPatch(atom_indices=atom_indices, gains=gains, shifts=shifts, window=window)

    def _compile_routing_metadata(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        sparsity = self.config.router_sparsity
        num_neurons = self.config.router_num_neurons
        indices, weights, compressed = route_tokens_with_metadata(
            hidden,
            sparsity=sparsity,
            num_neurons=num_neurons,
        )
        metadata: Dict[str, torch.Tensor] = {"indices": indices, "weights": weights}
        return compressed, hidden, metadata


__all__ = ["ProceduralModelConfig", "ProceduralLanguageModel"]
