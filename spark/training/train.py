"""Training entry point for the SPARK procedural language model."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from spark.demopack import CodebookSpec
from spark.procedural_model import ProceduralLanguageModel, ProceduralModelConfig
from spark.layer_generator import GeneratorConfig


@dataclasses.dataclass
class TrainingConfig:
    """Serializable configuration for a training experiment."""

    input_dim: int = 32
    hidden_dim: int = 64
    vocab_size: int = 128
    codebook_size: int = 64
    codebook_learnable: bool = True
    metadata_dim: int = 8
    generator_embed_dim: int = 128
    generator_hidden_dim: int = 32
    generator_rank: int = 4
    batch_size: int = 16
    epochs: int = 5
    steps_per_epoch: int = 100
    eval_steps: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    use_amp: bool = True
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None


@dataclasses.dataclass
class TrainingEpochReport:
    """Lightweight summary for a single epoch."""

    epoch: int
    train_loss: float
    eval_loss: float
    global_step: int


class SyntheticSequenceDataset(Dataset):
    """Simple dataset producing random inputs, metadata, and targets."""

    def __init__(
        self,
        size: int,
        input_dim: int,
        metadata_dim: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.inputs = torch.randn(size, input_dim, generator=generator)
        self.metadata = torch.randn(size, metadata_dim, generator=generator)
        self.targets = torch.randint(0, vocab_size, (size,), generator=generator)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "metadata": self.metadata[idx],
            "targets": self.targets[idx],
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainingConfig, device: torch.device) -> ProceduralLanguageModel:
    codebook_spec = CodebookSpec(num_codewords=cfg.codebook_size, embedding_dim=cfg.input_dim)
    generator_cfg = GeneratorConfig(
        embed_dim=cfg.generator_embed_dim,
        hidden_dim=cfg.generator_hidden_dim,
        rank=cfg.generator_rank,
    )
    model_cfg = ProceduralModelConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        codebook_spec=codebook_spec,
        generator_config=generator_cfg,
        metadata_dim=cfg.metadata_dim,
        codebook_learnable=cfg.codebook_learnable,
    )
    model = ProceduralLanguageModel(model_cfg)
    return model.to(device)


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticSequenceDataset(
        size=cfg.batch_size * cfg.steps_per_epoch,
        input_dim=cfg.input_dim,
        metadata_dim=cfg.metadata_dim,
        vocab_size=cfg.vocab_size,
        seed=cfg.seed,
    )
    eval_dataset = SyntheticSequenceDataset(
        size=cfg.batch_size * cfg.eval_steps,
        input_dim=cfg.input_dim,
        metadata_dim=cfg.metadata_dim,
        vocab_size=cfg.vocab_size,
        seed=cfg.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, eval_loader


def configure_optimizer(
    model: ProceduralLanguageModel, cfg: TrainingConfig
) -> torch.optim.Optimizer:
    # Always include generator and LM head parameters.
    params: List[Dict[str, torch.Tensor]] = [
        {"params": model.generator.parameters()},
        {"params": model.lm_head.parameters()},
    ]
    # Optionally include the codebook if it has been promoted to a parameter.
    if isinstance(model.codebook.codewords, torch.nn.Parameter):
        params.append({"params": [model.codebook.codewords]})
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return optimizer


def save_checkpoint(
    path: Path,
    model: ProceduralLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    step: int,
    cfg: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": dataclasses.asdict(cfg),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: ProceduralLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[int, int, TrainingConfig]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    epoch = payload.get("epoch", 0)
    step = payload.get("step", 0)
    cfg_dict = payload.get("config")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing serialized configuration")
    cfg = TrainingConfig(**cfg_dict)
    return epoch, step, cfg


def run_epoch(
    model: ProceduralLanguageModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    train: bool,
) -> Tuple[float, int]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_batches = 0
    original_weight = None
    if not train:
        original_weight = model.lm_head.weight.detach().clone()
    base_weight = model.lm_head.weight
    generator_embed = getattr(getattr(model.generator, "config", None), "embed_dim", base_weight.size(1))
    supports_generator_delta = generator_embed >= base_weight.size(1) and generator_embed >= base_weight.size(0)
    generator_warning_emitted = False
    def autocast_cm():
        if device.type == "cuda" and use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return contextlib.nullcontext()

    grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
    with grad_ctx:
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            metadata = batch["metadata"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_cm():
                if supports_generator_delta:
                    try:
                        model.apply_generator_delta(metadata)
                    except RuntimeError:
                        supports_generator_delta = False
                        if not generator_warning_emitted:
                            print(
                                "(Generator delta disabled after runtime mismatch during training.)",
                                flush=True,
                            )
                            generator_warning_emitted = True
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.detach().item()
            total_batches += 1
    mean_loss = total_loss / max(total_batches, 1)
    if original_weight is not None:
        model.lm_head.weight.data.copy_(original_weight)
    return mean_loss, total_batches


def run_training(
    cfg: TrainingConfig,
    *,
    device: torch.device | None = None,
    progress_callback: Callable[[TrainingEpochReport], None] | None = None,
) -> Dict[str, object]:
    """Train the procedural model according to ``cfg``.

    Parameters
    ----------
    cfg:
        Dataclass describing the training experiment.
    device:
        Optional explicit device.  When omitted a CUDA device is selected when
        available, otherwise CPU is used.
    progress_callback:
        Optional callable invoked after each epoch with a
        :class:`TrainingEpochReport`.

    Returns
    -------
    Dict[str, object]
        Dictionary containing the serialized configuration, device string,
        checkpoint path (if any), and a history of epoch level metrics.
    """

    cfg = dataclasses.replace(cfg)
    set_seed(cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model = build_model(cfg, device)
    optimizer = configure_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    resume_path = cfg.resume_from
    if resume_path:
        epoch, step, saved_cfg = load_checkpoint(Path(resume_path), model, optimizer, scaler)
        cfg = saved_cfg
        cfg.resume_from = resume_path
        start_epoch = epoch
        global_step = step
        set_seed(cfg.seed)

    checkpoint_path: Path | None = None
    if cfg.checkpoint_dir:
        checkpoint_path = Path(cfg.checkpoint_dir) / "last.pt"

    train_loader, eval_loader = build_dataloaders(cfg)

    history: List[TrainingEpochReport] = []
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, num_batches = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg.grad_clip,
            cfg.use_amp and device.type == "cuda",
            train=True,
        )
        global_step += num_batches
        eval_loss, _ = run_epoch(
            model,
            eval_loader,
            optimizer,
            scaler,
            device,
            cfg.grad_clip,
            cfg.use_amp and device.type == "cuda",
            train=False,
        )
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch + 1, global_step, cfg)
        report = TrainingEpochReport(
            epoch=epoch + 1,
            train_loss=train_loss,
            eval_loss=eval_loss,
            global_step=global_step,
        )
        history.append(report)
        if progress_callback is not None:
            progress_callback(report)

    return {
        "config": dataclasses.asdict(cfg),
        "device": str(device),
        "history": [dataclasses.asdict(r) for r in history],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path")
    return parser.parse_args()


def load_config(path: str | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    cfg_dict = json.loads(Path(path).read_text())
    return TrainingConfig(**cfg_dict)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume:
        cfg.resume_from = args.resume
    total_epochs = cfg.epochs

    def _log_progress(report: TrainingEpochReport) -> None:
        print(
            f"Epoch {report.epoch}/{total_epochs} - train_loss: {report.train_loss:.4f} - eval_loss: {report.eval_loss:.4f}",
            flush=True,
        )

    run_training(cfg, progress_callback=_log_progress)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
