"""Training pipeline and loop for Granita models."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from granita.config import TrainConfig
from granita.data.dataset import load_dataset_from_path
from granita.eval.evaluate import evaluate_model
from granita.models.backbone import create_model

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Execute one training epoch."""
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

        total_loss += float(loss.item()) * targets.size(0)
        _, preds = outputs.max(1)
        correct += int(preds.eq(targets).sum().item())
        total += targets.size(0)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def run_training(config: TrainConfig) -> Path:
    """Main training execution routine given a TrainConfig."""
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    scaler = torch.amp.GradScaler("cuda") if "cuda" in str(device) else None

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    best_val_f1 = 0.0
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
        )
        scheduler.step()

        # Validation
        val_metrics = evaluate_model(model, val_loader, classes=classes, device=device)
        elapsed = time.time() - epoch_start

        top1_pct = val_metrics.top1_accuracy * 100
        top3_pct = val_metrics.top3_accuracy * 100
        f1_pct = val_metrics.macro_f1 * 100
        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {top1_pct:.2f}% Top3: {top3_pct:.2f}% "
            f"Macro F1: {f1_pct:.2f}% ({elapsed:.1f}s)"
        )

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "model_name": config.model_name,
            "classes": classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_top1": val_metrics.top1_accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
        }

        # Save latest
        torch.save(checkpoint_data, output_dir / "last_model.pt")

        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            torch.save(checkpoint_data, best_checkpoint_path)

            if config.save_safetensors:
                try:
                    from safetensors.torch import save_file

                    save_file(model.state_dict(), str(output_dir / "best_model.safetensors"))
                except ImportError:
                    pass

    total_time = time.time() - start_time
    best_f1_pct = best_val_f1 * 100
    print(f"[*] Training finished in {total_time:.1f}s. Best Val Macro F1: {best_f1_pct:.2f}%")
    print(f"[*] Saved best model checkpoint to {best_checkpoint_path}")
    return best_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Granita music glyph classifier")
    parser.add_argument("--data", type=str, default="data/symbols/dataset.npz", help="Dataset path")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--model", type=str, default="granitanet", choices=["granitanet", "tiny"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
