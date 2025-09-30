"""Command-line interface for SPARK utilities.

This module exposes two convenience entry points:

``spark eval``
    Runs automated evaluation between the procedural model and a dense
    baseline.  Results are printed to stdout and optionally persisted as JSON
    for downstream dashboards.

``spark chat``
    Launches a tiny interactive loop that demonstrates how the procedural
    model can be used from a conversational interface.  The chat is intentionally
    lightweight – it exercises the opcode VM for reasoning traces and projects
    user text into the model input space to generate pseudo tokens.

Both commands share a common configuration surface so they can be invoked with a
single script entry point which is especially useful for CI/CD jobs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from .demopack import CodebookSpec
from .evaluation import EvaluationBatch, build_dense_baseline, compare_models
from .layer_generator import GeneratorConfig
from .opcode_vm import Instruction, Opcode
from .procedural_model import ProceduralLanguageModel, ProceduralModelConfig
from .training import TrainingConfig, TrainingEpochReport, run_training


# ---------------------------------------------------------------------------
# Shared configuration helpers


@dataclass
class ModelCLIConfig:
    """Container describing the high level model hyper-parameters."""

    input_dim: int
    hidden_dim: int
    vocab_size: int
    codebook_size: int
    generator_embed_dim: int
    generator_hidden_dim: int
    generator_rank: int
    metadata_dim: int
    codebook_learnable: bool


def _build_model_config(cfg: ModelCLIConfig) -> ProceduralModelConfig:
    codebook_spec = CodebookSpec(num_codewords=cfg.codebook_size, embedding_dim=cfg.input_dim)
    generator_cfg = GeneratorConfig(
        embed_dim=cfg.generator_embed_dim,
        hidden_dim=cfg.generator_hidden_dim,
        rank=cfg.generator_rank,
    )
    return ProceduralModelConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        codebook_spec=codebook_spec,
        generator_config=generator_cfg,
        metadata_dim=cfg.metadata_dim,
        codebook_learnable=cfg.codebook_learnable,
    )


def _model_config_from_training(cfg: TrainingConfig) -> ModelCLIConfig:
    return ModelCLIConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        codebook_size=cfg.codebook_size,
        generator_embed_dim=cfg.generator_embed_dim,
        generator_hidden_dim=cfg.generator_hidden_dim,
        generator_rank=cfg.generator_rank,
        metadata_dim=cfg.metadata_dim,
        codebook_learnable=cfg.codebook_learnable,
    )


def _device_from_args(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# ``spark eval`` implementation


def _encode_eval_batch(
    batch_size: int,
    input_dim: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> EvaluationBatch:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = torch.randn(batch_size, input_dim, generator=generator, device=device)
    targets = torch.randint(
        0, vocab_size, (batch_size,), generator=generator, device=device
    )
    return EvaluationBatch(inputs=inputs, targets=targets)


def _aggregate_metrics(reports: List[dict]) -> dict:
    if not reports:
        return {}

    def _mean(values: Iterable[float]) -> float:
        seq = list(values)
        if not seq:
            return math.nan
        return float(sum(seq) / len(seq))

    summary = {}
    # Each report record stores {"seed": int, "metrics": {...}}
    sample = reports[0]["metrics"]
    for section, metrics in sample.items():
        section_summary = {}
        for key in metrics.keys():
            section_summary[key] = _mean(r["metrics"][section][key] for r in reports)
        summary[section] = section_summary
    return summary


def _load_training_config(path: Optional[str]) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    payload = json.loads(Path(path).read_text())
    return TrainingConfig(**payload)


def _apply_override(cfg: TrainingConfig, field: str, value: Optional[int | float | str | bool]) -> None:
    if value is not None:
        setattr(cfg, field, value)


def run_eval_command(args: argparse.Namespace) -> int:
    device = _device_from_args(args.device)
    _set_seed(args.seed, device)

    cfg = ModelCLIConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        codebook_size=args.codebook_size,
        generator_embed_dim=args.generator_embed_dim,
        generator_hidden_dim=args.generator_hidden_dim,
        generator_rank=args.generator_rank,
        metadata_dim=args.metadata_dim,
        codebook_learnable=args.codebook_learnable,
    )
    model_cfg = _build_model_config(cfg)

    model = ProceduralLanguageModel(model_cfg).to(device)
    dense = build_dense_baseline(model_cfg).to(device)

    reports: List[dict] = []
    for run in range(args.runs):
        batch_seed = args.seed + run
        batch = _encode_eval_batch(
            args.batch_size, args.input_dim, args.vocab_size, batch_seed, device
        )
        metrics = compare_models(
            procedural=model,
            dense=dense,
            batch=batch,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )
        reports.append({"seed": batch_seed, "metrics": metrics})

    summary = _aggregate_metrics(reports)
    payload = {
        "config": cfg.__dict__,
        "runs": reports,
        "summary": summary,
        "device": str(device),
    }

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


def run_train_command(args: argparse.Namespace) -> int:
    device = _device_from_args(args.device)
    cfg = _load_training_config(args.config)

    _apply_override(cfg, "input_dim", args.input_dim)
    _apply_override(cfg, "hidden_dim", args.hidden_dim)
    _apply_override(cfg, "vocab_size", args.vocab_size)
    _apply_override(cfg, "codebook_size", args.codebook_size)
    _apply_override(cfg, "generator_embed_dim", args.generator_embed_dim)
    _apply_override(cfg, "generator_hidden_dim", args.generator_hidden_dim)
    _apply_override(cfg, "generator_rank", args.generator_rank)
    _apply_override(cfg, "metadata_dim", args.metadata_dim)
    if args.codebook_learnable is not None:
        cfg.codebook_learnable = bool(args.codebook_learnable)
    _apply_override(cfg, "batch_size", args.batch_size)
    _apply_override(cfg, "epochs", args.epochs)
    _apply_override(cfg, "steps_per_epoch", args.steps_per_epoch)
    _apply_override(cfg, "eval_steps", args.eval_steps)
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_amp is not None:
        cfg.use_amp = bool(args.use_amp)
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.resume is not None:
        cfg.resume_from = args.resume

    total_epochs = cfg.epochs

    def _log_progress(report: TrainingEpochReport) -> None:
        print(
            f"Epoch {report.epoch}/{total_epochs} - train_loss: {report.train_loss:.4f} - eval_loss: {report.eval_loss:.4f}",
            flush=True,
        )

    summary = run_training(cfg, device=device, progress_callback=_log_progress)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# ``spark chat`` implementation


def _encode_prompt_as_tensor(prompt: str, dim: int, device: torch.device) -> torch.Tensor:
    encoded = torch.zeros(dim, dtype=torch.float32)
    data = prompt.encode("utf-8")
    length = min(len(data), dim)
    if length > 0:
        values = torch.tensor(list(data[:length]), dtype=torch.float32)
        encoded[:length] = values / 127.5 - 1.0
    else:
        encoded = torch.randn(dim)
    return encoded.unsqueeze(0).to(device)


def _encode_prompt_metadata(prompt: str, dim: int, device: torch.device) -> torch.Tensor:
    histogram = torch.zeros(dim, dtype=torch.float32)
    if prompt:
        for index, char in enumerate(prompt.lower()):
            bucket = index % dim
            histogram[bucket] += (ord(char) % 97) / 96.0
    return histogram.unsqueeze(0).to(device)


def _instructions_from_prompt(prompt: str) -> List[Instruction]:
    return [
        Instruction(Opcode.PLAN, prompt or "idle"),
        Instruction(Opcode.RETR, "recall prior context"),
        Instruction(Opcode.EXEC, "synthesise reply"),
        Instruction(Opcode.CHECK, "deliver response"),
    ]


def _decode_tokens(token_ids: List[int]) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    if not token_ids:
        return ""
    return "".join(alphabet[token % len(alphabet)] for token in token_ids)


def run_chat_command(args: argparse.Namespace) -> int:
    device = _device_from_args(args.device)
    _set_seed(args.seed, device)

    cfg = ModelCLIConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        codebook_size=args.codebook_size,
        generator_embed_dim=args.generator_embed_dim,
        generator_hidden_dim=args.generator_hidden_dim,
        generator_rank=args.generator_rank,
        metadata_dim=args.metadata_dim,
        codebook_learnable=args.codebook_learnable,
    )
    model_cfg = _build_model_config(cfg)
    model = ProceduralLanguageModel(model_cfg).to(device)

    print("SPARK chat interface – type /exit to quit.\n")

    base_weight = model.lm_head.weight
    generator_embed = getattr(model.generator.config, "embed_dim", base_weight.size(1))
    supports_generator_delta = generator_embed >= base_weight.size(1) and generator_embed >= base_weight.size(0)
    if not supports_generator_delta and args.show_trace:
        print("(Generator delta skipped – embedding dimension too small for LM head.)")

    turns = 0
    while args.max_turns is None or turns < args.max_turns:
        try:
            prompt = input("You: ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in {"/exit", "exit", "quit", "q"}:
            break

        metadata = _encode_prompt_metadata(prompt, cfg.metadata_dim, device)
        if supports_generator_delta:
            try:
                model.apply_generator_delta(metadata)
            except RuntimeError:
                supports_generator_delta = False
                if args.show_trace:
                    print("(Generator delta disabled after runtime mismatch.)")

        tape = _instructions_from_prompt(prompt)
        trace = model.reason(tape)

        inputs = _encode_prompt_as_tensor(prompt, cfg.input_dim, device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=min(args.top_k, probs.size(-1)), dim=-1)
        decoded = _decode_tokens(topk.indices[0].tolist())

        response_lines = [f"TRACE: {line}" for line in trace] if args.show_trace else []
        response_lines.append(f"TOKENS: {decoded}" if decoded else "TOKENS: <none>")
        print("Model: ")
        for line in response_lines:
            print(f"  {line}")

        turns += 1

    print("Session ended.")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing


def _model_default(value: ModelCLIConfig | None, field: str, fallback: int) -> int:
    return getattr(value, field) if value is not None else fallback


def _bool_default(value: ModelCLIConfig | None, field: str, fallback: bool) -> bool:
    return getattr(value, field) if value is not None else fallback


def _add_model_arguments(
    parser: argparse.ArgumentParser,
    defaults: ModelCLIConfig | None = None,
    *,
    as_overrides: bool = False,
) -> None:
    def numeric_argument(name: str, field: str, fallback: int, help_text: str) -> None:
        default_value = None if as_overrides else _model_default(defaults, field, fallback)
        extra = f" (default: {_model_default(defaults, field, fallback)})" if as_overrides else ""
        parser.add_argument(name, type=int, default=default_value, help=help_text + extra)

    numeric_argument("--input-dim", "input_dim", 32, "Dimension of model inputs")
    numeric_argument("--hidden-dim", "hidden_dim", 64, "Hidden size of the model")
    numeric_argument("--vocab-size", "vocab_size", 320, "Vocabulary size for logits")
    numeric_argument("--codebook-size", "codebook_size", 128, "Number of codebook entries")
    numeric_argument("--generator-embed-dim", "generator_embed_dim", 32, "Generator embedding dimension")
    numeric_argument("--generator-hidden-dim", "generator_hidden_dim", 64, "Generator hidden dimension")
    numeric_argument("--generator-rank", "generator_rank", 4, "Generator low-rank size")
    numeric_argument("--metadata-dim", "metadata_dim", 16, "Metadata vector dimension")

    bool_default = None if as_overrides else _bool_default(defaults, "codebook_learnable", False)
    bool_extra = (
        f" (default: {_bool_default(defaults, 'codebook_learnable', False)})" if as_overrides else ""
    )
    parser.add_argument(
        "--codebook-learnable",
        action=argparse.BooleanOptionalAction,
        default=bool_default,
        help="Promote the codebook to a learnable parameter" + bool_extra,
    )
    device_default = None if as_overrides else None
    parser.add_argument("--device", type=str, default=device_default, help="Device to run on (cpu/cuda)")
    seed_default = None if as_overrides else 42
    seed_extra = " (default: 42)" if as_overrides else ""
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed" + seed_extra)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPARK command-line toolkit")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Run automated evaluations")
    _add_model_arguments(eval_parser)
    eval_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    eval_parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    eval_parser.add_argument("--num-warmup", type=int, default=2, help="Warmup iterations")
    eval_parser.add_argument("--num-iters", type=int, default=5, help="Benchmark iterations")
    eval_parser.add_argument(
        "--output", type=str, default=None, help="Optional JSON file to store the results"
    )
    eval_parser.set_defaults(func=run_eval_command)

    chat_parser = subparsers.add_parser("chat", help="Launch the interactive chat demo")
    _add_model_arguments(chat_parser)
    chat_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of tokens to display from the model output"
    )
    chat_parser.add_argument(
        "--show-trace", action="store_true", help="Display opcode reasoning traces"
    )
    chat_parser.add_argument(
        "--max-turns", type=int, default=None, help="Maximum number of turns before exit"
    )
    chat_parser.set_defaults(func=run_chat_command)

    training_defaults = TrainingConfig()
    train_parser = subparsers.add_parser("train", help="Run a training experiment")
    train_parser.add_argument(
        "--config", type=str, default=None, help="Optional JSON config containing TrainingConfig fields"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path produced by a previous training run",
    )
    train_parser.add_argument(
        "--output", type=str, default=None, help="Optional JSON file to store the training summary"
    )
    _add_model_arguments(
        train_parser,
        defaults=_model_config_from_training(training_defaults),
        as_overrides=True,
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Mini-batch size (default: {training_defaults.batch_size})",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of epochs to train (default: {training_defaults.epochs})",
    )
    train_parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=f"Synthetic steps per epoch (default: {training_defaults.steps_per_epoch})",
    )
    train_parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=f"Number of evaluation steps (default: {training_defaults.eval_steps})",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Optimizer learning rate (default: {training_defaults.learning_rate})",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help=f"Optimizer weight decay (default: {training_defaults.weight_decay})",
    )
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help=f"Gradient clipping value (default: {training_defaults.grad_clip})",
    )
    train_parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Enable automatic mixed precision (default: {training_defaults.use_amp})",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"Directory to store checkpoints (default: {training_defaults.checkpoint_dir})",
    )
    train_parser.set_defaults(func=run_train_command)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

