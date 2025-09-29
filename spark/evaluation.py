"""Evaluation utilities for comparing procedural and dense models."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from .opcode_vm import Instruction
from .procedural_model import ProceduralLanguageModel, ProceduralModelConfig


@dataclass
class EvaluationBatch:
    """Container for paired model inputs and targets."""

    inputs: torch.Tensor
    targets: torch.Tensor

    def __post_init__(self) -> None:
        if self.targets is None:
            raise ValueError("Targets must be provided for evaluation")
        if self.inputs.ndim not in (2, 3):
            raise ValueError("Inputs must be rank-2 or rank-3 tensor")
        if self.targets.ndim != 1:
            raise ValueError("Targets must be a rank-1 tensor")
        if self.inputs.size(0) != self.targets.size(0):
            raise ValueError(
                "Batch size mismatch between inputs and targets"
            )

    @property
    def token_count(self) -> int:
        if self.inputs.ndim == 3:
            return int(self.inputs.size(0) * self.inputs.size(1))
        if self.inputs.ndim == 2:
            return int(self.inputs.size(0))
        raise ValueError("Inputs must be rank-2 or rank-3 tensor")


class DenseBaseline(torch.nn.Module):
    """Simple dense network for baseline comparisons."""

    def __init__(self, config: ProceduralModelConfig) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.input_dim, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, config.vocab_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.mean(dim=1)
        elif inputs.ndim != 2:
            raise ValueError("Inputs must be rank-2 or rank-3 tensor")
        return self.layers(inputs)


def build_dense_baseline(config: ProceduralModelConfig) -> DenseBaseline:
    """Create a dense baseline with similar capacity to the procedural model."""

    return DenseBaseline(config)


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def estimate_model_size_bytes(model: torch.nn.Module) -> int:
    """Approximate parameter memory footprint in bytes."""

    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    return total


def compute_perplexity(model: torch.nn.Module, batch: EvaluationBatch) -> Dict[str, float]:
    """Compute cross-entropy loss and perplexity on ``batch``."""

    model.eval()
    with torch.no_grad():
        logits = model(batch.inputs)
        targets = batch.targets
        if targets.ndim != 1:
            targets = targets.view(-1)
        if logits.size(0) != targets.size(0):
            raise ValueError("Targets must align with model output batch size")
        loss = torch.nn.functional.cross_entropy(logits, targets.long(), reduction="mean")
    return {"loss": float(loss.item()), "perplexity": float(torch.exp(loss).item())}


def benchmark_throughput(
    model: torch.nn.Module,
    batch: EvaluationBatch,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> float:
    """Measure tokens/sec processed by ``model`` on ``batch``."""

    model.eval()
    inputs = batch.inputs
    device = inputs.device
    if hasattr(model, "kv_patcher"):
        model.kv_patcher.cache.clear()  # type: ignore[attr-defined]

    with torch.no_grad():
        for _ in range(max(0, num_warmup)):
            _ = model(inputs)
    _synchronize_if_needed(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, num_iters)):
            _ = model(inputs)
    _synchronize_if_needed(device)
    duration = time.perf_counter() - start
    tokens = batch.token_count * max(1, num_iters)
    return float(tokens / max(duration, 1e-6))


def evaluate_model(
    model: torch.nn.Module,
    batch: EvaluationBatch,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> Dict[str, float]:
    """Return perplexity, throughput, and memory metrics for ``model``."""

    perplexity_stats = compute_perplexity(model, batch)
    throughput = benchmark_throughput(model, batch, num_warmup=num_warmup, num_iters=num_iters)
    size_bytes = estimate_model_size_bytes(model)
    return {
        **perplexity_stats,
        "throughput_tps": throughput,
        "parameter_bytes": float(size_bytes),
    }


def compare_models(
    procedural: ProceduralLanguageModel,
    dense: torch.nn.Module,
    batch: EvaluationBatch,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ``procedural`` and ``dense`` models side-by-side."""

    proc_stats = evaluate_model(procedural, batch, num_warmup=num_warmup, num_iters=num_iters)
    dense_stats = evaluate_model(dense, batch, num_warmup=num_warmup, num_iters=num_iters)
    comparison = {
        "perplexity_delta": proc_stats["perplexity"] - dense_stats["perplexity"],
        "throughput_speedup": proc_stats["throughput_tps"] / max(dense_stats["throughput_tps"], 1e-6),
        "memory_reduction": 1.0 - proc_stats["parameter_bytes"] / max(dense_stats["parameter_bytes"], 1e-6),
    }
    return {"procedural": proc_stats, "dense": dense_stats, "comparison": comparison}


def demonstrate_reasoning(
    model: ProceduralLanguageModel,
    instructions: Iterable[Instruction],
) -> Iterable[str]:
    """Proxy to :meth:`ProceduralLanguageModel.reason` for discovery notebooks."""

    return model.reason(list(instructions))


def demonstrate_cache_reuse(
    model: ProceduralLanguageModel,
    batch: EvaluationBatch,
) -> Dict[str, float]:
    """Showcase KV-cache reuse by timing consecutive forward passes."""

    if not model.config.kv_cache_enabled:
        raise ValueError("Model must have KV cache enabled for this demonstration")
    model.kv_patcher.cache.clear()
    model.eval()
    inputs = batch.inputs
    device = inputs.device
    with torch.no_grad():
        start = time.perf_counter()
        logits_first = model(inputs)
        _synchronize_if_needed(device)
        first_time = time.perf_counter() - start

        start = time.perf_counter()
        logits_second = model(inputs)
        _synchronize_if_needed(device)
        second_time = time.perf_counter() - start

    tokens = float(batch.token_count)
    return {
        "segments_cached": float(len(model.kv_patcher.cache)),
        "first_pass_s": float(first_time),
        "second_pass_s": float(second_time),
        "throughput_speedup": float(first_time / max(second_time, 1e-6)),
        "outputs_consistent": bool(torch.allclose(logits_first, logits_second, atol=1e-5, rtol=1e-5)),
        "tokens": tokens,
    }


__all__ = [
    "EvaluationBatch",
    "DenseBaseline",
    "build_dense_baseline",
    "estimate_model_size_bytes",
    "compute_perplexity",
    "benchmark_throughput",
    "evaluate_model",
    "compare_models",
    "demonstrate_reasoning",
    "demonstrate_cache_reuse",
]
