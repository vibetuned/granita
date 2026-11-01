"""Multi-model ablation study runner and benchmarking for Granita."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tabulate import tabulate

from granita.config import SOURCE_NAMES, TrainConfig
from granita.data.dataset import load_dataset_from_path
from granita.eval.evaluate import evaluate_model
from granita.models.backbone import count_parameters, create_model
from granita.train.trainer import resolve_device, run_training

if TYPE_CHECKING:
    import torch.nn as nn

DEFAULT_ABLATION_MODELS: tuple[str, ...] = (
    "tiny",
    "granitanet",
    "regla",
    "iformer",
    "lsnet",
)


@dataclass
class AblationResult:
    """Results and efficiency metrics for a single model in an ablation study."""

    model_name: str
    num_params: int
    train_time_sec: float
    latency_ms_batch128: float
    val_top1: float
    val_macro_f1: float
    test_top1: float
    test_top3: float
    test_macro_f1: float
    group_f1: dict[str, float]
    checkpoint_path: str
    eval_split: str = "test"
    """Split the ``test_*`` numbers come from: ``test``, or ``val`` without a test split."""
    source_accuracy: dict[str, float] = field(default_factory=dict)
    """Top-1 on the eval split per sample source (font / stroke / crop)."""
    epochs: int = 0
    """Epochs the best checkpoint was trained for."""


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    device: str | torch.device,
    batch_size: int = 128,
    warmup: int = 10,
    runs: int = 40,
) -> float:
    """Measure average forward-pass latency (ms) for a batch of 32x32 glyphs."""
    model.eval()
    dummy = torch.randn(batch_size, 1, 32, 32, device=device)
    is_cuda = "cuda" in str(device) and torch.cuda.is_available()

    for _ in range(warmup):
        _ = model(dummy)
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def format_ablation_report(results: list[AblationResult]) -> str:
    """Format ablation results into Markdown comparison tables."""
    split = results[0].eval_split.capitalize() if results else "Test"
    summary_headers = [
        "Model",
        "Params",
        "Epochs",
        "Train Time",
        "Batch-128 Latency",
        "Val F1",
        f"{split} Top-1",
        f"{split} Top-3",
        f"{split} Macro F1",
    ]
    summary_rows = []
    for r in results:
        summary_rows.append(
            [
                r.model_name,
                f"{r.num_params:,}",
                r.epochs,
                f"{r.train_time_sec:.1f}s",
                f"{r.latency_ms_batch128:.2f} ms",
                f"{r.val_macro_f1 * 100:.2f}%",
                f"{r.test_top1 * 100:.2f}%",
                f"{r.test_top3 * 100:.2f}%",
                f"{r.test_macro_f1 * 100:.2f}%",
            ]
        )
    summary_table = tabulate(summary_rows, headers=summary_headers, tablefmt="github")

    # Collect all group names
    all_groups = sorted({grp for r in results for grp in r.group_f1})
    group_headers = ["Model", *[g.capitalize() for g in all_groups]]
    group_rows = []
    for r in results:
        row = [r.model_name]
        for g in all_groups:
            score = r.group_f1.get(g, 0.0)
            row.append(f"{score * 100:.2f}%")
        group_rows.append(row)
    group_table = tabulate(group_rows, headers=group_headers, tablefmt="github")

    report = (
        "# Granita Model Ablation Report\n\n"
        "## Overall Performance & Efficiency\n\n"
        f"{summary_table}\n\n"
        "## Semantic Group Macro F1 Breakdown\n\n"
        f"{group_table}\n"
    )

    sources = [s for s in SOURCE_NAMES if any(s in r.source_accuracy for r in results)]
    if sources:
        source_headers = ["Model", *[f"{s.capitalize()} Top-1" for s in sources]]
        source_rows = [
            [r.model_name, *[f"{r.source_accuracy.get(s, 0.0) * 100:.2f}%" for s in sources]]
            for r in results
        ]
        report += (
            f"\n## {split} Top-1 by Sample Source\n\n"
            "`stroke` is the stylus-drawn subset, the input a score editor sends.\n\n"
            f"{tabulate(source_rows, headers=source_headers, tablefmt='github')}\n"
        )
    return report


def _trained_seconds(arm_out: Path) -> float:
    history = arm_out / "history.json"
    if not history.is_file():
        return 0.0
    return float(sum(h.get("seconds", 0.0) for h in json.loads(history.read_text())))


def run_ablation(
    models: list[str],
    base_config: TrainConfig,
    force: bool = False,
) -> list[AblationResult]:
    """Train and evaluate multiple model architectures under identical conditions.

    An arm whose ``best_model.pt`` already exists under ``base_config.output_dir``
    is evaluated without retraining unless ``force`` is set.
    """
    device = resolve_device(base_config.device)

    _, val_loader, test_loader, classes = load_dataset_from_path(
        data_path=base_config.data_path,
        batch_size=base_config.batch_size,
        num_workers=base_config.num_workers,
    )
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split = "test" if test_loader is not None else "val"
    if test_loader is None:
        print("[!] Dataset has no test split: reporting the val split as the eval split")

    root_out = Path(base_config.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    results: list[AblationResult] = []

    for model_name in models:
        print("\n" + "=" * 68)
        print(f" Ablation Arm: {model_name}")
        print("=" * 68)

        arm_out = root_out / model_name
        best_ckpt_path = arm_out / "best_model.pt"
        if best_ckpt_path.is_file() and not force:
            print(f"[*] Reusing {best_ckpt_path} (pass --force to retrain)")
            train_time = _trained_seconds(arm_out)
        else:
            arm_cfg = TrainConfig(
                data_path=base_config.data_path,
                output_dir=arm_out,
                model_name=model_name,
                batch_size=base_config.batch_size,
                epochs=base_config.epochs,
                lr=base_config.lr,
                weight_decay=base_config.weight_decay,
                label_smoothing=base_config.label_smoothing,
                warmup_epochs=base_config.warmup_epochs,
                context_dim=base_config.context_dim,
                context_dropout=base_config.context_dropout,
                num_workers=base_config.num_workers,
                seed=base_config.seed,
                device=device,
                save_safetensors=base_config.save_safetensors,
            )
            t0 = time.time()
            best_ckpt_path = run_training(arm_cfg)
            train_time = time.time() - t0

        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model = create_model(
            model_name=model_name,
            num_classes=len(classes),
            context_dim=int(ckpt.get("context_dim", 0)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        n_params = count_parameters(model)
        latency_ms = benchmark_latency(model, device=device, batch_size=128)
        test_metrics = evaluate_model(model, eval_loader, classes=classes, device=device)

        results.append(
            AblationResult(
                model_name=model_name,
                num_params=n_params,
                train_time_sec=train_time,
                latency_ms_batch128=latency_ms,
                val_top1=float(ckpt.get("val_top1", 0.0)),
                val_macro_f1=float(ckpt.get("val_macro_f1", 0.0)),
                test_top1=test_metrics.top1_accuracy,
                test_top3=test_metrics.top3_accuracy,
                test_macro_f1=test_metrics.macro_f1,
                group_f1=test_metrics.group_f1,
                checkpoint_path=str(best_ckpt_path),
                eval_split=eval_split,
                source_accuracy=dict(test_metrics.source_accuracy),
                epochs=int(ckpt.get("epoch", 0)),
            )
        )

    report_md = format_ablation_report(results)
    print("\n" + report_md)

    (root_out / "ablation_report.md").write_text(report_md, encoding="utf-8")
    (root_out / "ablation_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"[*] Saved ablation report to {root_out / 'ablation_report.md'}")
    return results
