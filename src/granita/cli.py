"""Unified command-line interface for Granita."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tabulate import tabulate

from granita.config import CLASS_GROUPS, DEFAULT_CLASSES, TrainConfig
from granita.data.dataset import SPLITS, load_split_dataset, make_loader
from granita.eval.evaluate import evaluate_model, format_metrics_table
from granita.infer.predictor import (
    DEFAULT_PEN_WIDTH,
    DEFAULT_STROKE_FRACTION,
    GranitaClassifier,
    load_strokes_json,
)
from granita.models.backbone import (
    AVAILABLE_MODELS,
    MODEL_DESCRIPTIONS,
    count_parameters,
    create_model,
)
from granita.models.export import export_onnx, export_pt2, export_torchscript
from granita.train.ablation import DEFAULT_ABLATION_MODELS, run_ablation
from granita.train.trainer import resolve_device, run_training

DEFAULT_DATA = "data/symbols/dataset.npz"


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


def _training_args(p: argparse.ArgumentParser, epochs: int, output_dir: str) -> None:
    p.add_argument("--data", type=str, default=DEFAULT_DATA, help="npz archive or ImageFolder root")
    p.add_argument("--output-dir", type=str, default=output_dir)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument(
        "--warmup-epochs", type=int, default=2, help="linear LR warmup before the cosine decay"
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")


def _train_config(args: argparse.Namespace, model_name: str) -> TrainConfig:
    return TrainConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        model_name=model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def parser_train(p: argparse.ArgumentParser) -> None:
    _training_args(p, epochs=25, output_dir="checkpoints")
    p.add_argument("--model", type=str, default="granitanet", choices=list(AVAILABLE_MODELS))


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Granita model."""
    run_training(_train_config(args, args.model))


def parser_ablate(p: argparse.ArgumentParser) -> None:
    _training_args(p, epochs=15, output_dir="checkpoints/ablation")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_ABLATION_MODELS),
        choices=list(AVAILABLE_MODELS),
        help="List of model architectures to compare",
    )
    p.add_argument(
        "--force", action="store_true", help="retrain arms whose best_model.pt already exists"
    )


def cmd_ablate(args: argparse.Namespace) -> None:
    """Run an ablation study across multiple model architectures."""
    cfg = _train_config(args, "granitanet")
    run_ablation(models=list(args.models), base_config=cfg, force=args.force)


def parser_eval(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pt)")
    p.add_argument("--data", type=str, default=DEFAULT_DATA, help="npz archive or ImageFolder root")
    p.add_argument("--split", type=str, default="test", choices=list(SPLITS))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--confusions", type=int, default=10, help="confusion pairs to list")
    p.add_argument("--device", type=str, default="auto")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model on one split (train is evaluated without augmentation)."""
    device = resolve_device(args.device)
    dataset = load_split_dataset(args.data, args.split)
    if dataset is None:
        raise SystemExit(f"{args.data} has no {args.split!r} split")
    loader = make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    classes = dataset.classes

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = ckpt.get("model_name", "granitanet")
    model = create_model(model_name=model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = evaluate_model(model, loader, classes=classes, device=device)
    print(f"# {model_name} on the {args.split} split of {args.data} ({len(dataset)} samples)\n")
    print(format_metrics_table(metrics, classes, confusions=args.confusions))


def parser_infer(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str, help="isolated glyph image")
    source.add_argument(
        "--strokes", type=str, help="JSON pen strokes [[[x, y], ...], ...] from an editor"
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_STROKE_FRACTION,
        help="strokes only: glyph span as a fraction of the canvas (small dot ~0.3, dynamic ~0.8)",
    )
    p.add_argument(
        "--pen-width",
        type=float,
        default=DEFAULT_PEN_WIDTH,
        help="strokes only: pen width as a fraction of the canvas",
    )
    p.add_argument(
        "--brush", action="store_true", help="strokes only: pressure-like brush instead of a pen"
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--top-k", type=int, default=5)


def cmd_infer(args: argparse.Namespace) -> None:
    """Classify a single glyph image or a set of pen strokes."""
    from PIL import Image

    classifier = GranitaClassifier.from_checkpoint(args.checkpoint, device=args.device)
    if args.strokes:
        pred = classifier.predict_strokes(
            load_strokes_json(args.strokes),
            top_k=args.top_k,
            fraction=args.fraction,
            pen_width=args.pen_width,
            brush=args.brush,
        )
    else:
        with Image.open(args.image) as img:
            pred = classifier.predict(img, top_k=args.top_k)

    conf_pct = pred.confidence * 100
    print(f"Prediction: {pred.label} (confidence: {conf_pct:.2f}%, group: {pred.group})")
    print("\nTop Candidates:")
    for label, conf in pred.top_k:
        print(f"  - {label:<24}: {conf * 100:.2f}%")


def parser_export(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "pt2", "torchscript"],
        help="onnx (ONNX Runtime, browsers), pt2 (torch.export), torchscript (deprecated)",
    )
    p.add_argument(
        "--static-batch", action="store_true", help="fix the batch size to 1 instead of dynamic"
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export a checkpoint to ONNX, torch.export (.pt2) or TorchScript."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    model_name = ckpt.get("model_name", "granitanet")

    model = create_model(model_name=model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])

    out = Path(args.output)
    dynamic = not args.static_batch
    if args.format == "onnx":
        export_onnx(model, out, dynamic_batch=dynamic)
    elif args.format == "pt2":
        export_pt2(model, out, dynamic_batch=dynamic)
    else:
        export_torchscript(model, out)
    print(f"[*] Exported {model_name} as {args.format} to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Granita: Music Glyph Classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_info = subparsers.add_parser("info", help="Show system, model, and class information")
    p_info.set_defaults(func=cmd_info)

    p_train = subparsers.add_parser("train", help="Train a Granita classifier")
    parser_train(p_train)
    p_train.set_defaults(func=cmd_train)

    p_ablate = subparsers.add_parser(
        "ablate", help="Run an ablation study across multiple model architectures"
    )
    parser_ablate(p_ablate)
    p_ablate.set_defaults(func=cmd_ablate)

    p_eval = subparsers.add_parser("eval", help="Evaluate a model checkpoint")
    parser_eval(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_infer = subparsers.add_parser("infer", help="Classify a glyph image or pen strokes")
    parser_infer(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    p_export = subparsers.add_parser("export", help="Export a model to ONNX, pt2 or TorchScript")
    parser_export(p_export)
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
