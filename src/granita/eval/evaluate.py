"""Evaluation metrics and assessment routines for Granita models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate
from torch.utils.data import RandomSampler

from granita.config import CLASS_GROUPS, SOURCE_NAMES

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
    source_accuracy: dict[str, float] = field(default_factory=dict)
    """Top-1 accuracy per sample source (font / stroke / crop) when the dataset records it."""
    source_count: dict[str, int] = field(default_factory=dict)


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    top3_correct: int,
    sources: np.ndarray | None = None,
) -> EvalMetrics:
    """Compute every metric from label arrays (``sources`` aligned with ``y_true``)."""
    num_classes = len(classes)
    total_samples = len(y_true)

    # Confusion matrix: rows = true, cols = pred
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf_matrix, (y_true, y_pred), 1)

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

        per_class_f1[cls_name] = float(f1)
        f1_list.append(float(f1))
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
    group_scores: dict[str, list[float]] = {}
    for cls_name, f1 in per_class_f1.items():
        group_scores.setdefault(CLASS_GROUPS.get(cls_name, "other"), []).append(f1)
    group_f1 = {grp: float(np.mean(scores)) for grp, scores in group_scores.items()}

    # Source metrics (font / stroke / crop)
    source_accuracy: dict[str, float] = {}
    source_count: dict[str, int] = {}
    if sources is not None and len(sources) == total_samples:
        correct = y_true == y_pred
        for code, name in enumerate(SOURCE_NAMES):
            mask = sources == code
            n = int(mask.sum())
            if n:
                source_count[name] = n
                source_accuracy[name] = float(correct[mask].mean())

    return EvalMetrics(
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        group_f1=group_f1,
        confusion_matrix=conf_matrix,
        source_accuracy=source_accuracy,
        source_count=source_count,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    classes: list[str],
    device: str | torch.device = "cpu",
) -> EvalMetrics:
    """Compute comprehensive classification metrics on a dataset.

    Per-source accuracy is reported when the loader's dataset carries a
    ``sources`` array and iterates in order (no random sampler).
    """
    model.eval()
    num_classes = len(classes)

    all_preds: list[int] = []
    all_targets: list[int] = []
    top3_correct = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        _, preds = logits.max(1)

        k = min(3, num_classes)
        _, topk_preds = logits.topk(k, dim=1)
        top3_correct += int(topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds)).sum().item())

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    sources = getattr(dataloader.dataset, "sources", None)
    if isinstance(dataloader.sampler, RandomSampler):
        sources = None
    return metrics_from_predictions(y_true, y_pred, classes, top3_correct, sources=sources)


def top_confusions(
    metrics: EvalMetrics, classes: list[str], n: int = 10
) -> list[tuple[str, str, int]]:
    """The ``n`` most frequent off-diagonal ``(true, predicted, count)`` cells."""
    cm = metrics.confusion_matrix.copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1]
    out: list[tuple[str, str, int]] = []
    for idx in flat[:n]:
        t, p = np.unravel_index(int(idx), cm.shape)
        if cm[t, p] == 0:
            break
        out.append((classes[int(t)], classes[int(p)], int(cm[t, p])))
    return out


def format_metrics_table(metrics: EvalMetrics, classes: list[str], confusions: int = 10) -> str:
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

    parts = ["## Summary", summary_table, "## Group Performance", group_table]

    if metrics.source_accuracy:
        source_rows = [
            [name, metrics.source_count[name], f"{acc * 100:.2f}%"]
            for name, acc in metrics.source_accuracy.items()
        ]
        parts += [
            "## Per-Source Accuracy",
            tabulate(source_rows, headers=["Source", "Samples", "Top-1"], tablefmt="github"),
        ]

    conf_rows = [[t, p, c] for t, p, c in top_confusions(metrics, classes, confusions)]
    if conf_rows:
        parts += [
            "## Top Confusions",
            tabulate(conf_rows, headers=["True", "Predicted", "Count"], tablefmt="github"),
        ]

    parts += ["## Per-Class Breakdown", table_str]
    return "\n\n".join(parts)


def main() -> None:
    from granita.cli import cmd_eval, parser_eval

    parser = argparse.ArgumentParser(description="Evaluate a Granita glyph model")
    parser_eval(parser)
    cmd_eval(parser.parse_args())


if __name__ == "__main__":
    main()
