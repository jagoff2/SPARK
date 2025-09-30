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


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dim", type=int, default=32, help="Dimension of model inputs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size of the model")
    parser.add_argument("--vocab-size", type=int, default=320, help="Vocabulary size for logits")
    parser.add_argument("--codebook-size", type=int, default=128, help="Number of codebook entries")
    parser.add_argument(
        "--generator-embed-dim", type=int, default=32, help="Generator embedding dimension"
    )
    parser.add_argument(
        "--generator-hidden-dim", type=int, default=64, help="Generator hidden dimension"
    )
    parser.add_argument("--generator-rank", type=int, default=4, help="Generator low-rank size")
    parser.add_argument("--metadata-dim", type=int, default=16, help="Metadata vector dimension")
    parser.add_argument(
        "--codebook-learnable",
        action="store_true",
        help="Promote the codebook to a learnable parameter",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


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

