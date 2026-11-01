"""Training pipeline and loop for Granita models."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from granita.config import TrainConfig
from granita.data.dataset import load_dataset_from_path
from granita.eval.evaluate import EvalMetrics, evaluate_model
from granita.models.backbone import create_model

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def resolve_device(device: str) -> str:
    """``auto`` -> ``cuda`` when available, else the given device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step linear warmup over ``warmup_epochs`` then cosine decay to zero."""
    total = max(1, epochs * steps_per_epoch)
    warmup = min(total, max(0, warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    """Execute one training epoch (``scheduler`` is stepped per batch)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and torch.cuda.is_available() and "cuda" in str(device)

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item()) * targets.size(0)
        _, preds = outputs.max(1)
        correct += int(preds.eq(targets).sum().item())
        total += targets.size(0)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def config_dict(config: TrainConfig) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    classes: list[str],
    epoch: int,
    metrics: EvalMetrics,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write a checkpoint loadable with ``torch.load(..., weights_only=True)``.

    The optimizer state is stored only when given (the ``last`` checkpoint),
    which keeps ``best_model.pt`` a third of the size.
    """
    data: dict[str, Any] = {
        "epoch": epoch,
        "model_name": config.model_name,
        "classes": list(classes),
        "model_state_dict": model.state_dict(),
        "val_top1": metrics.top1_accuracy,
        "val_macro_f1": metrics.macro_f1,
        "val_source_accuracy": dict(metrics.source_accuracy),
        "config": config_dict(config),
    }
    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(data, path)


def _save_safetensors(
    path: Path, model: nn.Module, config: TrainConfig, classes: list[str]
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        return
    metadata = {"model_name": config.model_name, "classes": json.dumps(list(classes))}
    save_file(model.state_dict(), str(path), metadata=metadata)


def run_training(config: TrainConfig) -> Path:
    """Train one model; returns the path of the best (val macro F1) checkpoint.

    Writes ``last_model.pt`` (with optimizer state), ``best_model.pt``,
    ``best_model.safetensors`` and a per-epoch ``history.json`` into
    ``config.output_dir``.
    """
    device = resolve_device(config.device)
    print(f"[*] Training on device: {device}")
    torch.manual_seed(config.seed)

    # 1. Load data
    train_loader, val_loader, _, classes = load_dataset_from_path(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    ds_train = train_loader.dataset
    ds_val = val_loader.dataset
    n_train = len(ds_train) if isinstance(ds_train, Sized) else len(train_loader)
    n_val = len(ds_val) if isinstance(ds_val, Sized) else len(val_loader)
    print(f"[*] Loaded dataset: {n_train} train, {n_val} val, {len(classes)} classes")

    # 2. Build model
    model = create_model(model_name=config.model_name, num_classes=len(classes))
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Created {config.model_name} with {num_params:,} trainable parameters")

    # 3. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, config.epochs, max(1, len(train_loader)), config.warmup_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    scaler = torch.amp.GradScaler("cuda") if "cuda" in str(device) else None

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    best_val_f1 = -1.0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
        )

        # Validation
        val_metrics = evaluate_model(model, val_loader, classes=classes, device=device)
        elapsed = time.time() - epoch_start

        sources = " ".join(
            f"{name}={acc * 100:.1f}%" for name, acc in val_metrics.source_accuracy.items()
        )
        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_metrics.top1_accuracy * 100:.2f}% "
            f"Top3: {val_metrics.top3_accuracy * 100:.2f}% "
            f"Macro F1: {val_metrics.macro_f1 * 100:.2f}%"
            + (f" [{sources}]" if sources else "")
            + f" ({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": val_metrics.top1_accuracy,
                "val_top3": val_metrics.top3_accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "val_source_accuracy": dict(val_metrics.source_accuracy),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": elapsed,
            }
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        save_checkpoint(
            output_dir / "last_model.pt", model, config, classes, epoch, val_metrics, optimizer
        )
        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            save_checkpoint(best_checkpoint_path, model, config, classes, epoch, val_metrics)
            if config.save_safetensors:
                _save_safetensors(output_dir / "best_model.safetensors", model, config, classes)

    total_time = time.time() - start_time
    print(
        f"[*] Training finished in {total_time:.1f}s. Best Val Macro F1: {best_val_f1 * 100:.2f}%"
    )
    print(f"[*] Saved best model checkpoint to {best_checkpoint_path}")
    return best_checkpoint_path


def main() -> None:
    from granita.cli import cmd_train, parser_train

    parser = argparse.ArgumentParser(description="Train a Granita music glyph classifier")
    parser_train(parser)
    cmd_train(parser.parse_args())


if __name__ == "__main__":
    main()
