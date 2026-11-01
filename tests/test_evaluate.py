"""Unit tests for Granita evaluation metrics."""

import numpy as np
import torch

from granita.data.dataset import SymbolNpzDataset, make_loader
from granita.eval.evaluate import (
    evaluate_model,
    format_metrics_table,
    metrics_from_predictions,
    top_confusions,
)
from granita.models.backbone import GranitaTiny


def test_metrics_math_and_sources() -> None:
    classes = ["fing1", "fing2", "trill"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    sources = np.array([0, 0, 1, 1, 2, 2])
    m = metrics_from_predictions(y_true, y_pred, classes, top3_correct=6, sources=sources)

    assert m.top1_accuracy == 4 / 6
    assert m.top3_accuracy == 1.0
    assert m.confusion_matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # fing1: p=1/2 r=1/2 f1=0.5; fing2: p=2/3 r=1 f1=0.8; trill: p=1 r=1/2 f1=2/3
    assert np.isclose(m.per_class_f1["fing1"], 0.5)
    assert np.isclose(m.per_class_f1["fing2"], 0.8)
    assert np.isclose(m.per_class_f1["trill"], 2 / 3)
    assert np.isclose(m.macro_f1, (0.5 + 0.8 + 2 / 3) / 3)
    assert np.isclose(m.weighted_f1, (0.5 * 2 + 0.8 * 2 + 2 / 3 * 2) / 6)
    assert np.isclose(m.group_f1["fingering"], 0.65)
    assert np.isclose(m.group_f1["ornament"], 2 / 3)
    assert m.source_accuracy == {"font": 0.5, "stroke": 1.0, "crop": 0.5}
    assert m.source_count == {"font": 2, "stroke": 2, "crop": 2}
    assert top_confusions(m, classes)[0] in (("fing1", "fing2", 1), ("trill", "fing1", 1))

    table = format_metrics_table(m, classes)
    assert "Per-Source Accuracy" in table and "Top Confusions" in table


def test_metrics_without_sources_or_samples() -> None:
    m = metrics_from_predictions(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), ["a"], 0
    )
    assert m.top1_accuracy == 0.0 and m.source_accuracy == {}
    assert top_confusions(m, ["a"]) == []


def test_evaluate_model_reports_sources_only_in_order() -> None:
    classes = ["a", "b"]
    images = np.full((8, 32, 32), 255, dtype=np.uint8)
    labels = np.array([0, 1] * 4, dtype=np.int64)
    sources = np.array([1] * 8, dtype=np.uint8)
    from granita.data.transforms import get_eval_transforms

    ds = SymbolNpzDataset(images, labels, classes, transform=get_eval_transforms(), sources=sources)
    model = GranitaTiny(num_classes=2)
    ordered = evaluate_model(model, make_loader(ds, 4, shuffle=False, num_workers=0), classes)
    assert ordered.source_count == {"stroke": 8}
    shuffled = evaluate_model(model, make_loader(ds, 4, shuffle=True, num_workers=0), classes)
    assert shuffled.source_accuracy == {}
    assert isinstance(ordered.confusion_matrix, np.ndarray)
    assert torch.is_tensor(torch.tensor(ordered.top1_accuracy))
