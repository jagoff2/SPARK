"""Procedural attention builders for SPARK."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass
class AtomConfig:
    name: str
    frequency: float
    phase: float = 0.0


@dataclass
class AttentionPatch:
    atom_indices: torch.Tensor
    gains: torch.Tensor
    shifts: torch.Tensor
    window: torch.Tensor


class AttentionBasisSynthesizer(torch.nn.Module):
    """Constructs linear-time attention kernels from atom instructions."""

    def __init__(self, atoms: List[AtomConfig], max_length: int) -> None:
        super().__init__()
        self.atoms = atoms
        self.max_length = max_length
        waveforms = torch.stack([self._atom_waveform(atom) for atom in atoms])
        self.register_buffer("waveforms", waveforms)

    def _atom_waveform(self, atom: AtomConfig) -> torch.Tensor:
        x = torch.linspace(0, 1, self.max_length)
        if atom.name == "sin":
            return torch.sin(2 * math.pi * atom.frequency * x + atom.phase)
        if atom.name == "cos":
            return torch.cos(2 * math.pi * atom.frequency * x + atom.phase)
        raise ValueError(f"Unknown atom {atom.name}")

    def compile_patch(self, patch: AttentionPatch, seq_len: int) -> torch.Tensor:
        if seq_len > self.max_length:
            raise ValueError("Sequence length exceeds synthesizer maximum")
        atoms = self.waveforms.to(patch.atom_indices.device)[patch.atom_indices]
        window = torch.nn.functional.pad(
            patch.window, (0, max(0, seq_len - patch.window.size(-1))), value=0.0
        )[:seq_len]
        bases = atoms[:, :seq_len] * patch.gains.unsqueeze(-1)
        shifts = patch.shifts.to(torch.long)
        shifted = torch.stack([
            torch.roll(bases[i], shifts[i].item(), dims=0) for i in range(bases.size(0))
        ])
        return shifted * window

    def apply_attention(
        self,
        patch: AttentionPatch,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = queries.size(1)
        basis = self.compile_patch(patch, seq_len)
        kernel = basis.sum(dim=0)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scores = scores + kernel
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, values)


__all__ = ["AtomConfig", "AttentionPatch", "AttentionBasisSynthesizer"]
