"""KV cache patcher utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


@dataclass
class CacheEntry:
    key: torch.Tensor
    value: torch.Tensor
    dirty: bool = False


class KVCachePatcher:
    """Reuses KV cache segments across multi-pass decoding."""

    def __init__(self) -> None:
        self.cache: Dict[int, CacheEntry] = {}

    def mark(self, segment_id: int) -> None:
        entry = self.cache.get(segment_id)
        if entry:
            entry.dirty = True

    def update(self, segment_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.cache[segment_id] = CacheEntry(key=key, value=value, dirty=False)

    def gather(self, segment_ids: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = []
        values = []
        for seg_id in segment_ids:
            entry = self.cache.get(seg_id)
            if entry is None:
                raise KeyError(f"Missing segment {seg_id}")
            keys.append(entry.key)
            values.append(entry.value)
            entry.dirty = False
        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2)

    def sweep(self) -> None:
        for seg_id in list(self.cache.keys()):
            if self.cache[seg_id].dirty:
                del self.cache[seg_id]


__all__ = ["KVCachePatcher", "CacheEntry"]
