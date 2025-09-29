"""High-level integration of SPARK procedural components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .attention import AttentionBasisSynthesizer, AttentionPatch, AtomConfig
from .demopack import CodebookSpec, DemopackCodebook, DemopackDecoder, build_random_instructions
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
    metadata_dim: int = 8
    attention_max_length: int = 1024
    enable_demopack: bool = True
    enable_lora: bool = True
    enable_attention_basis: bool = True
    enable_kv_patching: bool = True


class ProceduralLanguageModel(torch.nn.Module):
    """Toy language model glueing together procedural components."""

    def __init__(self, config: ProceduralModelConfig) -> None:
        super().__init__()
        self.config = config
        tile_rows = 16
        num_tiles = max(1, config.hidden_dim // tile_rows)
        decoder_out = num_tiles * tile_rows
        if config.enable_demopack:
            self.codebook = DemopackCodebook(config.codebook_spec)
            instructions = build_random_instructions(
                num_tiles=num_tiles,
                tile_shape=(tile_rows, config.input_dim),
                codebook_size=config.codebook_spec.num_codewords,
            )
            self.decoder: torch.nn.Module = DemopackDecoder(
                codebook=self.codebook,
                out_features=decoder_out,
                in_features=config.input_dim,
                instructions=instructions,
            )
        else:
            self.codebook = None
            decoder_out = config.hidden_dim
            self.decoder = torch.nn.Linear(config.input_dim, decoder_out)
        self.decoder_out = decoder_out
        if config.enable_lora:
            self.generator: Optional[LayerGenerator] = LayerGenerator(
                config.generator_config, metadata_dim=config.metadata_dim
            )
        else:
            self.generator = None
        self.lm_head = torch.nn.Linear(self.decoder_out, config.vocab_size)
        atom_cfgs = [
            AtomConfig(name="sin", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ] + [
            AtomConfig(name="cos", frequency=freq) for freq in (1.0, 2.0, 4.0)
        ]
        if config.enable_attention_basis:
            self.attention: Optional[AttentionBasisSynthesizer] = AttentionBasisSynthesizer(
                atom_cfgs, max_length=config.attention_max_length
            )
            self._attention_patch = AttentionPatch(
                atom_indices=torch.arange(len(atom_cfgs), dtype=torch.long),
                gains=torch.ones(len(atom_cfgs)),
                shifts=torch.zeros(len(atom_cfgs), dtype=torch.long),
                window=torch.ones(self.decoder_out),
            )
        else:
            self.attention = None
            self._attention_patch = None
        self.kv_patcher = KVCachePatcher() if config.enable_kv_patching else None
        self.vm = OpcodeVM()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(inputs)
        if self.attention is not None and self._attention_patch is not None:
            patch = self._attention_patch
            attention_patch = AttentionPatch(
                atom_indices=patch.atom_indices.to(inputs.device),
                gains=patch.gains.to(inputs.device),
                shifts=patch.shifts.to(inputs.device),
                window=patch.window.to(inputs.device),
            )
            qkv = hidden.unsqueeze(1)
            attended = self.attention.apply_attention(attention_patch, qkv, qkv, qkv)
            hidden = hidden + attended.squeeze(1)
        if self.kv_patcher is not None:
            segment_id = 0
            try:
                _, cached_values = self.kv_patcher.gather([segment_id])
                cached = cached_values.mean(dim=-2)
                hidden = hidden + cached.unsqueeze(0)
            except KeyError:
                pass
            cache_value = hidden.mean(dim=0, keepdim=True).detach()
            self.kv_patcher.update(segment_id, cache_value, cache_value)
        logits = self.lm_head(hidden)
        return logits

    def apply_generator_delta(self, metadata: torch.Tensor) -> None:
        if not self.config.enable_lora or self.generator is None:
            return
        base = self.lm_head.weight.data.clone()
        delta = self.generator(metadata, base.size(1), base.size(0))
        a, b = delta
        if a.size(1) < base.size(1):
            pad = base.size(1) - a.size(1)
            a = torch.nn.functional.pad(a, (0, pad))
        if b.size(1) < base.size(0):
            pad = base.size(0) - b.size(1)
            b = torch.nn.functional.pad(b, (0, pad))
        updated = apply_lora_delta(base, (a, b))
        self.lm_head.weight.data.copy_(updated)

    def reason(self, instructions: List[Instruction]) -> List[str]:
        state = self.vm.execute(instructions)
        return state.log


__all__ = ["ProceduralModelConfig", "ProceduralLanguageModel"]
