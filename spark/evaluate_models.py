"""Evaluation utilities comparing procedural and baseline models."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import BaselineMLP, match_mlp_hidden_dim
from .demopack import CodebookSpec
from .layer_generator import GeneratorConfig
from .procedural_model import ProceduralLanguageModel, ProceduralModelConfig

LOGGER = logging.getLogger(__name__)


def generate_validation_data(
    num_samples: int, input_dim: int, vocab_size: int, seed: int
) -> Tuple[TensorDataset, DataLoader]:
    """Create a synthetic validation set shared by all models."""

    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(num_samples, input_dim, generator=g)
    targets = torch.randint(0, vocab_size, (num_samples,), generator=g)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataset, loader


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    apply_generator: bool = False,
    metadata_dim: int = 8,
) -> Dict[str, float]:
    """Compute perplexity, accuracy, latency, and memory usage."""

    model.eval()
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    latencies: List[float] = []
    if apply_generator and hasattr(model, "apply_generator_delta"):
        metadata = torch.randn(1, metadata_dim, device=device)
        model.apply_generator_delta(metadata)
    tracemalloc.start()
    start_time = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_start = time.perf_counter()
            logits = model(inputs)
            latencies.append(time.perf_counter() - batch_start)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.numel()
    total_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    perplexity = math.exp(total_loss / total_samples)
    accuracy = total_correct / total_samples
    avg_latency = sum(latencies) / len(latencies)
    return {
        "perplexity": float(perplexity),
        "accuracy": float(accuracy),
        "latency_ms": float(avg_latency * 1000.0),
        "total_time_s": float(total_time),
        "peak_mem_mb": float(peak / (1024 * 1024)),
    }


def build_procedural_config(args: argparse.Namespace) -> ProceduralModelConfig:
    """Create the procedural model configuration from CLI arguments."""

    codebook = CodebookSpec(num_codewords=args.codebook_size, embedding_dim=args.input_dim)
    generator = GeneratorConfig(
        embed_dim=args.generator_embed,
        hidden_dim=args.generator_hidden,
        rank=args.generator_rank,
        num_layers=args.generator_layers,
    )
    return ProceduralModelConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        codebook_spec=codebook,
        generator_config=generator,
        metadata_dim=args.metadata_dim,
        attention_max_length=args.attention_max_length,
    )


def instantiate_procedural_model(config: ProceduralModelConfig) -> ProceduralLanguageModel:
    model = ProceduralLanguageModel(config)
    return model


def run_ablation(
    base_config: ProceduralModelConfig,
    data_loader: DataLoader,
    metadata_dim: int,
) -> List[Dict[str, object]]:
    """Evaluate ablations that toggle individual procedural features."""

    experiments = []
    feature_flags = {
        "Procedural-Full": base_config,
        "No-Demopack": replace(base_config, enable_demopack=False),
        "No-LoRA": replace(base_config, enable_lora=False),
        "No-Attention": replace(base_config, enable_attention_basis=False),
        "No-KV-Patching": replace(base_config, enable_kv_patching=False),
    }
    for name, config in feature_flags.items():
        model = instantiate_procedural_model(config)
        params = sum(p.numel() for p in model.parameters())
        metrics = evaluate_model(
            model,
            data_loader,
            apply_generator=config.enable_lora,
            metadata_dim=metadata_dim,
        )
        experiments.append({
            "model": name,
            "params": params,
            **metrics,
        })
    return experiments


def evaluate_baseline(
    reference_params: int,
    input_dim: int,
    vocab_size: int,
    data_loader: DataLoader,
    num_layers: int,
) -> Dict[str, object]:
    """Construct and evaluate the MLP baseline with matched parameter count."""

    baseline_config = match_mlp_hidden_dim(
        reference_params=reference_params,
        input_dim=input_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )
    baseline = BaselineMLP(baseline_config)
    params = sum(p.numel() for p in baseline.parameters())
    metrics = evaluate_model(baseline, data_loader, apply_generator=False)
    return {
        "model": f"Baseline-MLP-{baseline_config.hidden_dim}",
        "params": params,
        **metrics,
    }


def render_markdown_table(results: Iterable[Dict[str, object]]) -> str:
    headers = [
        "Model",
        "Params (M)",
        "Perplexity",
        "Accuracy",
        "Latency (ms)",
        "Peak Memory (MB)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in results:
        params_m = row["params"] / 1_000_000
        line = "| {model} | {params:.3f} | {perplexity:.3f} | {accuracy:.3f} | {latency:.2f} | {mem:.2f} |".format(
            model=row["model"],
            params=params_m,
            perplexity=row["perplexity"],
            accuracy=row["accuracy"],
            latency=row["latency_ms"],
            mem=row["peak_mem_mb"],
        )
        lines.append(line)
    return "\n".join(lines)


def plot_metrics(results: List[Dict[str, object]], report_dir: Path) -> None:
    names = [row["model"] for row in results]
    perplexities = [row["perplexity"] for row in results]
    accuracies = [row["accuracy"] for row in results]
    latencies = [row["latency_ms"] for row in results]
    memory = [row["peak_mem_mb"] for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, perplexities, color="#3366cc")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_title("Perplexity across models")

    axes[1].bar(names, accuracies, color="#dc3912")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy across models")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(report_dir / "perplexity_accuracy.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(latencies, memory, c="#109618")
    for name, x, y in zip(names, latencies, memory):
        ax2.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Latency vs memory trade-off")
    fig2.tight_layout()
    fig2.savefig(report_dir / "latency_memory.png", dpi=200)
    plt.close(fig2)


def save_reports(results: List[Dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    markdown = [
        "# Procedural vs Baseline Evaluation",
        "",
        "The table below summarises the quantitative comparison between the procedural model, its ablations, and the parameter-matched baseline.",
        "",
        render_markdown_table(results),
        "",
        "Generated plots:",
        "- `perplexity_accuracy.png`: perplexity and accuracy comparison.",
        "- `latency_memory.png`: latency versus peak memory usage.",
    ]
    (report_dir / "poc_report.md").write_text("\n".join(markdown))
    plot_metrics(results, report_dir)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--generator-embed", type=int, default=16)
    parser.add_argument("--generator-hidden", type=int, default=32)
    parser.add_argument("--generator-rank", type=int, default=4)
    parser.add_argument("--generator-layers", type=int, default=2)
    parser.add_argument("--metadata-dim", type=int, default=8)
    parser.add_argument("--attention-max-length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--baseline-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    _, loader = generate_validation_data(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    config = build_procedural_config(args)
    procedural_results = run_ablation(config, loader, metadata_dim=args.metadata_dim)
    reference_params = procedural_results[0]["params"]
    baseline_result = evaluate_baseline(
        reference_params=reference_params,
        input_dim=args.input_dim,
        vocab_size=args.vocab_size,
        data_loader=loader,
        num_layers=args.baseline_layers,
    )
    all_results = procedural_results + [baseline_result]
    for result in all_results:
        LOGGER.info(
            "%s :: perplexity=%.3f accuracy=%.3f latency=%.2fms peak_mem=%.2fMB params=%d",
            result["model"],
            result["perplexity"],
            result["accuracy"],
            result["latency_ms"],
            result["peak_mem_mb"],
            result["params"],
        )
    save_reports(all_results, args.report_dir)
    LOGGER.info("Reports written to %s", args.report_dir)


if __name__ == "__main__":
    main()
