"""Utilities for assembling multi-phase training plans.

This module focuses on the authoring experience for large runs.  It packages
``TrainingConfig`` objects together with human-readable intent so automated
orchestration tooling (for example CI jobs or internal trainers) can emit the
correct ``spark`` CLI invocations.

The default helper, :func:`build_frontier_training_plan`, returns a structured
representation of a staged curriculum that:

* bootstraps the procedural model weights,
* scales the mixture to a high-token frontier corpus,
* performs supervised instruction alignment, and
* finishes with reinforcement-style preference optimisation.

Each phase is accompanied by the exact command that should be executed along
with configuration and checkpoint locations so the entire pipeline can be
triggered without any manual bookkeeping.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .train import TrainingConfig


def _slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


@dataclass
class PipelineCommand:
    """Represents a shell command alongside metadata."""

    name: str
    description: str
    command: List[str]
    config: Optional[TrainingConfig] = None
    config_path: Optional[str] = None
    output_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    resume_from: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "command": self.command,
            "notes": self.notes,
        }
        if self.config is not None:
            payload["config"] = dataclasses.asdict(self.config)
        if self.config_path is not None:
            payload["config_path"] = self.config_path
        if self.output_path is not None:
            payload["output_path"] = self.output_path
        if self.checkpoint_dir is not None:
            payload["checkpoint_dir"] = self.checkpoint_dir
        if self.resume_from is not None:
            payload["resume_from"] = self.resume_from
        return payload


@dataclass
class FrontierTrainingPlan:
    """Structured plan describing how to reach frontier parity."""

    description: str
    model_config: TrainingConfig
    phases: List[PipelineCommand]
    evaluation: PipelineCommand
    chat: PipelineCommand
    resource_profile: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "description": self.description,
            "model_config": dataclasses.asdict(self.model_config),
            "phases": [phase.to_dict() for phase in self.phases],
            "evaluation": self.evaluation.to_dict(),
            "chat": self.chat.to_dict(),
            "resource_profile": self.resource_profile,
        }


def _base_frontier_config(seed: int) -> TrainingConfig:
    """Return a high-capacity baseline configuration for frontier experiments."""

    return TrainingConfig(
        input_dim=2048,
        hidden_dim=4096,
        vocab_size=65536,
        codebook_size=4096,
        codebook_learnable=True,
        metadata_dim=256,
        generator_embed_dim=4096,
        generator_hidden_dim=2048,
        generator_rank=64,
        batch_size=64,
        epochs=1,
        steps_per_epoch=1024,
        eval_steps=128,
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=seed,
        use_amp=True,
        checkpoint_dir="checkpoints",
    )


def _phase_config(
    base: TrainingConfig,
    *,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    eval_steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    use_amp: bool,
    checkpoint_dir: str,
    resume_from: Optional[str],
    seed: int,
) -> TrainingConfig:
    cfg = dataclasses.replace(
        base,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    cfg.resume_from = resume_from
    return cfg


def _device_argument(device: Optional[str]) -> List[str]:
    return ["--device", device] if device else []


def _as_command(*parts: Iterable[str]) -> List[str]:
    command: List[str] = []
    for fragment in parts:
        if isinstance(fragment, str):
            command.append(fragment)
        else:
            command.extend(fragment)
    return command


def build_frontier_training_plan(
    output_dir: Path,
    *,
    seed: int = 42,
    device: Optional[str] = None,
) -> FrontierTrainingPlan:
    """Construct an automated multi-stage training plan.

    Parameters
    ----------
    output_dir:
        Root directory used for configs, logs, and checkpoints.  The directory
        is not created automatically; callers should invoke
        :func:`materialize_plan` when side effects are desired.
    seed:
        Base random seed to embed in the emitted :class:`TrainingConfig`
        instances.
    device:
        Optional device override forwarded to each CLI command.
    """

    output_dir = Path(output_dir)
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"

    base_cfg = _base_frontier_config(seed)

    phase_specs = [
        {
            "name": "Foundation bootstrap",
            "objective": "Establish procedural embeddings on a filtered multilingual corpus.",
            "epochs": 3,
            "steps": 4096,
            "eval_steps": 256,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "use_amp": True,
            "notes": [
                "Mix 60% code, 25% general web, and 15% technical papers.",
                "Apply tokenizer dropout and stochastic depth to avoid mode collapse.",
            ],
        },
        {
            "name": "Curriculum scaling",
            "objective": "Scale token budget and context length while matching dense frontier perplexity.",
            "epochs": 8,
            "steps": 8192,
            "eval_steps": 512,
            "batch_size": 128,
            "learning_rate": 2.5e-4,
            "weight_decay": 0.01,
            "grad_clip": 0.75,
            "use_amp": True,
            "notes": [
                "Switch to packed 32k token windows with rotary position interleaving.",
                "Target <1.5 validation nats to match dense SOTA checkpoints.",
            ],
        },
        {
            "name": "Instruction alignment",
            "objective": "Supervised fine-tuning on curated conversational and tool-use traces.",
            "epochs": 4,
            "steps": 3072,
            "eval_steps": 384,
            "batch_size": 48,
            "learning_rate": 5e-5,
            "weight_decay": 0.002,
            "grad_clip": 0.5,
            "use_amp": True,
            "notes": [
                "Blend 12% synthetic tutor traces and 88% human demonstrations.",
                "Freeze the demopack codebook for the first two epochs, then unfreeze.",
            ],
        },
        {
            "name": "Preference optimisation",
            "objective": "Reinforcement and distillation loop for safety and helpfulness parity.",
            "epochs": 6,
            "steps": 2048,
            "eval_steps": 512,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "grad_clip": 0.25,
            "use_amp": True,
            "notes": [
                "Alternating DPO and reward-model PPO steps (3:1 ratio).",
                "Audit refusal, jailbreak, and bias probes every 512 steps.",
            ],
        },
    ]

    phases: List[PipelineCommand] = []
    resume_from: Optional[str] = None
    for index, spec in enumerate(phase_specs, start=1):
        slug = _slugify(spec["name"])
        checkpoint_dir = checkpoints_dir / f"{index:02d}_{slug}"
        config_path = configs_dir / f"{index:02d}_{slug}.json"
        output_path = logs_dir / f"{index:02d}_{slug}.json"
        cfg = _phase_config(
            base_cfg,
            batch_size=spec["batch_size"],
            epochs=spec["epochs"],
            steps_per_epoch=spec["steps"],
            eval_steps=spec["eval_steps"],
            learning_rate=spec["learning_rate"],
            weight_decay=spec["weight_decay"],
            grad_clip=spec["grad_clip"],
            use_amp=spec["use_amp"],
            checkpoint_dir=str(checkpoint_dir),
            resume_from=resume_from,
            seed=seed + index - 1,
        )
        command = _as_command(
            ["python", "-m", "spark", "train", "--config", str(config_path), "--output", str(output_path)],
            _device_argument(device),
        )
        phase = PipelineCommand(
            name=spec["name"],
            description=spec["objective"],
            command=command,
            config=cfg,
            config_path=str(config_path),
            output_path=str(output_path),
            checkpoint_dir=str(checkpoint_dir),
            resume_from=resume_from,
            notes=spec.get("notes", []),
        )
        phases.append(phase)
        resume_from = str(Path(cfg.checkpoint_dir) / "last.pt")

    eval_output = logs_dir / "evaluation.json"
    evaluation = PipelineCommand(
        name="Frontier parity evaluation",
        description=(
            "Benchmark the procedural model against the dense baseline after the final phase."
        ),
        command=_as_command(
            [
                "python",
                "-m",
                "spark",
                "eval",
                "--input-dim",
                str(base_cfg.input_dim),
                "--hidden-dim",
                str(base_cfg.hidden_dim),
                "--vocab-size",
                str(base_cfg.vocab_size),
                "--codebook-size",
                str(base_cfg.codebook_size),
                "--generator-embed-dim",
                str(base_cfg.generator_embed_dim),
                "--generator-hidden-dim",
                str(base_cfg.generator_hidden_dim),
                "--generator-rank",
                str(base_cfg.generator_rank),
                "--metadata-dim",
                str(base_cfg.metadata_dim),
                "--codebook-learnable",
                "--batch-size",
                "64",
                "--runs",
                "3",
                "--num-warmup",
                "4",
                "--num-iters",
                "16",
                "--output",
                str(eval_output),
            ],
            _device_argument(device),
        ),
        notes=[
            "Expect matching throughput and perplexity deltas < 3% compared to the dense control.",
            "Feed the latest checkpoint path using --resume if benchmarking incremental phases.",
        ],
    )

    final_checkpoint = Path(phases[-1].checkpoint_dir) / "last.pt"
    chat = PipelineCommand(
        name="Interactive chat deployment",
        description="Load the trained weights into the chat demo with reasoning traces enabled.",
        command=_as_command(
            [
                "python",
                "-m",
                "spark",
                "chat",
                "--checkpoint",
                str(final_checkpoint),
                "--top-k",
                "10",
                "--show-trace",
            ],
            _device_argument(device),
        ),
        notes=[
            "Point to the safety-filtered preference-tuned checkpoint from the final phase.",
            "Use --max-turns during evaluations to collect deterministic transcripts.",
        ],
    )

    resource_profile = {
        "accelerators": "8x NVIDIA H100 80GB or higher",
        "duration_estimate": "~9.5 days wall-clock with ZeRO-3 sharding",
        "dataset_budget": "3.8T tokens total (2.7T pretraining, 0.7T instructions, 0.4T preference)",
        "target_metrics": "Frontier parity defined as <=1.5 nats eval loss and safety win-rate >= 0.95",
    }

    description = (
        "Automated curriculum covering foundation pretraining, scaling, alignment, "
        "and preference optimisation for a frontier-parity procedural model."
    )

    return FrontierTrainingPlan(
        description=description,
        model_config=base_cfg,
        phases=phases,
        evaluation=evaluation,
        chat=chat,
        resource_profile=resource_profile,
    )


def materialize_plan(
    plan: FrontierTrainingPlan,
    *,
    output_dir: Path,
    emit_script: bool = False,
) -> Dict[str, object]:
    """Persist the plan to disk.

    Parameters
    ----------
    plan:
        The structured plan returned by :func:`build_frontier_training_plan`.
    output_dir:
        Directory where configuration files, logs, and optional scripts will be
        written.  The directory is created if it does not already exist.
    emit_script:
        When ``True``, a convenience shell script is written that sequentially
        executes each phase, the evaluation, and the chat sanity-check command.

    Returns
    -------
    Dict[str, object]
        Dictionary containing the paths of generated assets.
    """

    output_dir = Path(output_dir)
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    generated_configs: List[str] = []
    for phase in plan.phases:
        if phase.config is None or phase.config_path is None:
            continue
        path = Path(phase.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(phase.config), indent=2))
        generated_configs.append(str(path))

    assets: Dict[str, object] = {
        "config_paths": generated_configs,
        "log_dir": str(logs_dir),
        "checkpoint_dir": str(checkpoints_dir),
    }

    if emit_script:
        script_lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for phase in plan.phases:
            script_lines.append(f"echo '=== {phase.name} ==='")
            script_lines.append(" ".join(phase.command))
            script_lines.append("")
        script_lines.append("echo '=== Frontier parity evaluation ==='")
        script_lines.append(" ".join(plan.evaluation.command))
        script_lines.append("")
        script_lines.append("echo '=== Launch chat demo ==='")
        script_lines.append(" ".join(plan.chat.command))
        script_lines.append("")
        script_path = output_dir / "run_frontier_plan.sh"
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)
        assets["script_path"] = str(script_path)

    return assets

