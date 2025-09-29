"""Demopack weight synthesis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


@dataclass
class CodebookSpec:
    num_codewords: int
    embedding_dim: int

    @property
    def size_bytes(self) -> int:
        return self.num_codewords * self.embedding_dim * 2


class DemopackCodebook(torch.nn.Module):
    def __init__(self, spec: CodebookSpec, init_scale: float = 0.02) -> None:
        super().__init__()
        self.spec = spec
        codewords = torch.empty(spec.num_codewords, spec.embedding_dim)
        torch.nn.init.trunc_normal_(codewords, std=init_scale)
        self.register_buffer("codewords", codewords)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype not in (torch.int16, torch.int32, torch.int64):
            raise TypeError("Codebook indices must be an integer tensor")
        codewords = self.codewords.to(indices.device)
        return torch.nn.functional.embedding(indices, codewords)


@dataclass
class DecodeInstruction:
    codeword_indices: torch.Tensor
    scale: float = 1.0
    rotation: Optional[torch.Tensor] = None


class DemopackDecoder(torch.nn.Module):
    def __init__(
        self,
        codebook: DemopackCodebook,
        out_features: int,
        in_features: int,
        instructions: Iterable[DecodeInstruction],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.codebook = codebook
        self.out_features = out_features
        self.in_features = in_features
        self.instructions = torch.nn.ModuleList()
        for inst in instructions:
            module = _InstructionBuffer(inst, in_features)
            self.instructions.append(module)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return f"out_features={self.out_features}, in_features={self.in_features}, tiles={len(self.instructions)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.in_features}, got {x.size(-1)}"
            )
        weight = self._decode_weight(x.device)
        out = torch.nn.functional.linear(x, weight, self.bias)
        return out

    def _decode_weight(self, device: torch.device) -> torch.Tensor:
        decoded_tiles = []
        for module in self.instructions:
            inst = module.to_instruction(device)
            decoded = self.codebook(inst.codeword_indices)
            if decoded.size(-1) != self.in_features:
                raise RuntimeError("Decoded tile width does not match in_features")
            if inst.rotation is not None:
                decoded = torch.matmul(decoded, inst.rotation)
            decoded = decoded.reshape(-1, self.in_features)
            decoded_tiles.append(decoded * inst.scale)
        weight = torch.cat(decoded_tiles, dim=0)
        if weight.size(0) != self.out_features:
            raise RuntimeError("Decoded weight rows do not match out_features")
        return weight


class _InstructionBuffer(torch.nn.Module):
    def __init__(self, instruction: DecodeInstruction, in_features: int) -> None:
        super().__init__()
        if instruction.rotation is not None:
            if instruction.rotation.ndim != 2 or instruction.rotation.size(1) != in_features:
                raise ValueError("Rotation matrix must align with in_features")
            self.register_buffer("rotation", instruction.rotation)
        else:
            self.register_buffer("rotation", None)
        self.register_buffer("indices", instruction.codeword_indices)
        self.scale = instruction.scale
        self.in_features = in_features

    def to_instruction(self, device: torch.device) -> DecodeInstruction:
        rotation = self.rotation.to(device) if self.rotation is not None else None
        return DecodeInstruction(
            codeword_indices=self.indices.to(device),
            scale=self.scale,
            rotation=rotation,
        )


def build_random_instructions(
    num_tiles: int,
    tile_shape: Tuple[int, int],
    codebook_size: int,
    device: Optional[torch.device] = None,
) -> Iterable[DecodeInstruction]:
    rows, cols = tile_shape
    indices = torch.randint(0, codebook_size, (num_tiles, rows, cols), dtype=torch.int32, device=device)
    instructions = []
    for tile in indices:
        instructions.append(DecodeInstruction(codeword_indices=tile, scale=1.0))
    return instructions


__all__ = [
    "CodebookSpec",
    "DemopackCodebook",
    "DemopackDecoder",
    "DecodeInstruction",
    "build_random_instructions",
]
