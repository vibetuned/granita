"""Evaluation metrics and assessment routines for Granita models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

from granita.config import CLASS_GROUPS
from granita.data.dataset import load_dataset_from_path
from granita.models.backbone import create_model

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass
class EvalMetrics:
    """Summary evaluation metrics."""

    top1_accuracy: float
    top3_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    group_f1: dict[str, float]
    confusion_matrix: np.ndarray


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    classes: list[str],
    device: str | torch.device = "cpu",
) -> EvalMetrics:
    """Compute comprehensive classification metrics on a dataset."""
    model.eval()
    num_classes = len(classes)

    all_preds: list[int] = []
    all_targets: list[int] = []
    top3_correct = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        _, preds = logits.max(1)

        # Top-3 accuracy
        k = min(3, num_classes)
        _, topk_preds = logits.topk(k, dim=1)
        top3_correct += int(topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds)).sum().item())

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        total_samples += targets.size(0)

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    # Confusion matrix: rows = true, cols = pred
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        conf_matrix[t, p] += 1

    per_class_f1: dict[str, float] = {}
    f1_list: list[float] = []
    support_list: list[int] = []

    for i, cls_name in enumerate(classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_f1[cls_name] = f1
        f1_list.append(f1)
        support_list.append(int(support))

    total = sum(support_list)
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    weighted_f1 = (
        float(sum(f1 * sup for f1, sup in zip(f1_list, support_list, strict=True)) / total)
        if total > 0
        else 0.0
    )
    top1_acc = float((y_true == y_pred).mean()) if total_samples > 0 else 0.0
    top3_acc = float(top3_correct / total_samples) if total_samples > 0 else 0.0

    # Group metrics
    group_f1: dict[str, float] = {}
    group_scores: dict[str, list[float]] = {}
    for cls_name, f1 in per_class_f1.items():
        grp = CLASS_GROUPS.get(cls_name, "other")
        group_scores.setdefault(grp, []).append(f1)
    for grp, scores in group_scores.items():
        group_f1[grp] = float(np.mean(scores))

    return EvalMetrics(
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        group_f1=group_f1,
        confusion_matrix=conf_matrix,
    )


def format_metrics_table(metrics: EvalMetrics, classes: list[str]) -> str:
    """Format evaluation metrics into a human-readable table."""
    headers = ["Class", "Group", "F1 Score"]
    rows = []
    for cls_name in classes:
        grp = CLASS_GROUPS.get(cls_name, "other")
        score = metrics.per_class_f1.get(cls_name, 0.0)
        rows.append([cls_name, grp, f"{score * 100:.2f}%"])

    table_str = tabulate(rows, headers=headers, tablefmt="github")

    summary_rows = [
        ["Top-1 Accuracy", f"{metrics.top1_accuracy * 100:.2f}%"],
        ["Top-3 Accuracy", f"{metrics.top3_accuracy * 100:.2f}%"],
        ["Macro F1", f"{metrics.macro_f1 * 100:.2f}%"],
        ["Weighted F1", f"{metrics.weighted_f1 * 100:.2f}%"],
    ]
    summary_table = tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="github")

    group_rows = [[grp, f"{score * 100:.2f}%"] for grp, score in sorted(metrics.group_f1.items())]
    group_table = tabulate(group_rows, headers=["Group", "Macro F1"], tablefmt="github")

    parts = [
        "## Summary",
        summary_table,
        "## Group Performance",
        group_table,
        "## Per-Class Breakdown",
        table_str,
    ]
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Granita glyph model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="data/symbols/dataset.npz", help="Dataset path")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
