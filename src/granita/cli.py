"""Unified command-line interface for Granita."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tabulate import tabulate

from granita.config import CLASS_GROUPS, DEFAULT_CLASSES, TrainConfig
from granita.data.dataset import load_dataset_from_path
from granita.eval.evaluate import evaluate_model, format_metrics_table
from granita.infer.predictor import GranitaClassifier
from granita.models.backbone import (
    AVAILABLE_MODELS,
    MODEL_DESCRIPTIONS,
    count_parameters,
    create_model,
)
from granita.models.export import export_onnx, export_torchscript
from granita.train.ablation import DEFAULT_ABLATION_MODELS, run_ablation
from granita.train.trainer import run_training


def cmd_info(_: argparse.Namespace) -> None:
    """Show information about Granita configuration, models, supported classes, and hardware."""
    print("=" * 68)
    print(" Granita: Music Glyph Recognition System")
    print("=" * 68)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:           {vram_gb:.2f} GB")
    print(f"Total Classes:  {len(DEFAULT_CLASSES)}")

    # Models breakdown
    model_rows = []
    for name in (
        "tiny",
        "granitanet",
        "regla_t",
        "regla_s",
        "iformer_t",
        "iformer_s",
        "lsnet_t",
        "lsnet_s",
    ):
        m = create_model(name, num_classes=len(DEFAULT_CLASSES))
        params = count_parameters(m)
        desc = MODEL_DESCRIPTIONS.get(name, "")
        model_rows.append([name, f"{params:,}", desc])

    print("\nAvailable Model Backbones (for Training & Ablation):")
    print(tabulate(model_rows, headers=["Model", "Parameters", "Description"], tablefmt="github"))

    # Groups breakdown
    groups: dict[str, list[str]] = {}
    for cls_name, grp in CLASS_GROUPS.items():
        groups.setdefault(grp, []).append(cls_name)

    rows = []
    for grp, cls_list in sorted(groups.items()):
        ex = ", ".join(cls_list[:5]) + ("..." if len(cls_list) > 5 else "")
        rows.append([grp.capitalize(), len(cls_list), ex])

    print("\nSupported Glyph Categories:")
    print(tabulate(rows, headers=["Category", "Count", "Examples"], tablefmt="github"))


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Granita model."""
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


def cmd_ablate(args: argparse.Namespace) -> None:
    """Run an ablation study across multiple model architectures."""
    cfg = TrainConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    run_ablation(models=list(args.models), base_config=cfg)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model."""
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    _, val_l, test_l, classes = load_dataset_from_path(args.data, batch_size=args.batch_size)
    loader = test_l if args.split == "test" and test_l is not None else val_l

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "granitanet")
    model = create_model(model_name=model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = evaluate_model(model, loader, classes=classes, device=device)
    print(format_metrics_table(metrics, classes))


def cmd_infer(args: argparse.Namespace) -> None:
    """Classify a single glyph image."""
    from PIL import Image

    classifier = GranitaClassifier.from_checkpoint(args.checkpoint, device=args.device)
    with Image.open(args.image) as img:
        pred = classifier.predict(img, top_k=args.top_k)

    conf_pct = pred.confidence * 100
    print(f"Prediction: {pred.label} (confidence: {conf_pct:.2f}%, group: {pred.group})")
    print("\nTop Candidates:")
    for label, conf in pred.top_k:
        print(f"  - {label:<24}: {conf * 100:.2f}%")


def cmd_export(args: argparse.Namespace) -> None:
    """Export a checkpoint to TorchScript or ONNX."""
    device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    model_name = ckpt.get("model_name", "granitanet")

    model = create_model(model_name=model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    fmt = args.format.lower()
    out = Path(args.output)
    if fmt in ("torchscript", "ts", "pt"):
        export_torchscript(model, out)
        print(f"[*] Exported TorchScript model to {out}")
    elif fmt in ("onnx",):
        export_onnx(model, out)
        print(f"[*] Exported ONNX model to {out}")
    else:
        raise ValueError(f"Unknown format: {fmt}. Choose 'torchscript' or 'onnx'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Granita: Music Glyph Classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Info
    p_info = subparsers.add_parser("info", help="Show system, model, and class information")
    p_info.set_defaults(func=cmd_info)

    # Train
    p_train = subparsers.add_parser("train", help="Train a Granita classifier")
    p_train.add_argument("--data", type=str, default="data/symbols/dataset.npz")
    p_train.add_argument("--output-dir", type=str, default="checkpoints")
    p_train.add_argument(
        "--model",
        type=str,
        default="granitanet",
        choices=list(AVAILABLE_MODELS),
    )
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--device", type=str, default="auto")
    p_train.set_defaults(func=cmd_train)

    # Ablate
    p_ablate = subparsers.add_parser(
        "ablate",
        help="Run an ablation study across multiple model architectures",
    )
    p_ablate.add_argument("--data", type=str, default="data/symbols/dataset.npz")
    p_ablate.add_argument("--output-dir", type=str, default="checkpoints/ablation")
    p_ablate.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_ABLATION_MODELS),
        choices=list(AVAILABLE_MODELS),
        help="List of model architectures to compare",
    )
    p_ablate.add_argument("--batch-size", type=int, default=128)
    p_ablate.add_argument("--epochs", type=int, default=15)
    p_ablate.add_argument("--lr", type=float, default=1e-3)
    p_ablate.add_argument("--weight-decay", type=float, default=1e-4)
    p_ablate.add_argument("--num-workers", type=int, default=2)
    p_ablate.add_argument("--seed", type=int, default=42)
    p_ablate.add_argument("--device", type=str, default="auto")
    p_ablate.set_defaults(func=cmd_ablate)

    # Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a model checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--data", type=str, default="data/symbols/dataset.npz")
    p_eval.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--device", type=str, default="auto")
    p_eval.set_defaults(func=cmd_eval)

    # Infer
    p_infer = subparsers.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--image", type=str, required=True)
    p_infer.add_argument("--device", type=str, default="auto")
    p_infer.add_argument("--top-k", type=int, default=5)
    p_infer.set_defaults(func=cmd_infer)

    # Export
    p_export = subparsers.add_parser("export", help="Export model to TorchScript or ONNX")
    p_export.add_argument("--checkpoint", type=str, required=True)
    p_export.add_argument("--output", type=str, required=True)
    p_export.add_argument(
        "--format", type=str, default="torchscript", choices=["torchscript", "onnx"]
    )
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
