"""Unit tests for Granita classifier predictor and inference pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from granita.infer.predictor import (
    GranitaClassifier,
    crop_glyph,
    load_strokes_json,
    pad_to_square,
    render_strokes,
    to_gray_uint8,
)
from granita.models.backbone import GranitaNet, GranitaTiny


def _classifier(classes: list[str]) -> GranitaClassifier:
    model = GranitaNet(num_classes=len(classes), in_channels=1)
    return GranitaClassifier(model=model, classes=classes, device="cpu")


def test_predictor_with_pil_image() -> None:
    classes = ["fing1", "fing2", "trill"]
    classifier = _classifier(classes)

    img = Image.new("L", (32, 32), color=255)
    for y in range(10, 22):
        for x in range(10, 22):
            img.putpixel((x, y), 0)

    pred = classifier.predict(img, top_k=3)
    assert pred.label in classes
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) == 3


def test_predictor_with_numpy_array() -> None:
    classes = ["accidentalFlat", "accidentalSharp"]
    classifier = _classifier(classes)

    arr = np.full((32, 32), 255, dtype=np.uint8)
    arr[12:20, 12:20] = 0

    pred = classifier.predict(arr)
    assert pred.label in classes
    assert pred.group == "accidental"


def test_predictor_batch_mixed_inputs() -> None:
    classes = ["fing1", "fing2", "trill"]
    classifier = _classifier(classes)

    imgs = [
        Image.new("RGB", (32, 32), color=(255, 255, 255)),
        np.full((32, 32, 3), 255, dtype=np.uint8),  # RGB numpy (H, W, C)
        torch.full((3, 32, 32), 255, dtype=torch.uint8),  # RGB tensor (C, H, W)
        torch.full((1, 32, 32), 1.0),  # float [0, 1] tensor
        np.full((48, 16), 255, dtype=np.uint8),  # non-square
    ]

    preds = classifier.predict_batch(imgs, top_k=2)
    assert len(preds) == 5
    for p in preds:
        assert p.label in classes
        assert len(p.top_k) == 2
    assert classifier.predict_batch([]) == []


def test_to_gray_and_pad_to_square() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    gray = to_gray_uint8(rgb)
    assert gray.shape == (4, 6) and int(gray[0, 0]) == 150
    assert to_gray_uint8(np.full((2, 2), 0.5)).tolist() == [[128, 128], [128, 128]]
    assert to_gray_uint8(torch.zeros(1, 3, 5)).shape == (3, 5)
    with pytest.raises(ValueError, match="2-D"):
        to_gray_uint8(np.zeros((2, 2, 2, 2)))

    square = pad_to_square(np.zeros((4, 10), dtype=np.uint8))
    assert square.shape == (10, 10)
    assert int(square[0, 0]) == 255 and int(square[5, 5]) == 0  # centred on paper
    assert pad_to_square(np.zeros((10, 10), dtype=np.uint8), margin=0.5).shape == (20, 20)


def test_prepare_tensor_pads_instead_of_squashing() -> None:
    classifier = _classifier(["a", "b"])
    wide = np.full((16, 64), 255, dtype=np.uint8)
    wide[6:10, 2:62] = 0  # a long horizontal bar
    t = classifier._prepare_tensor(wide)
    assert t.shape == (1, 1, 32, 32)
    ink_rows = (t[0, 0] < 0).any(dim=1).nonzero().flatten().tolist()
    assert ink_rows and min(ink_rows) >= 12 and max(ink_rows) <= 19  # stays a thin centred bar


def test_crop_glyph_matches_musicset_protocol() -> None:
    page = np.full((200, 200), 255, dtype=np.uint8)
    page[40:60, 100:110] = 0  # 10x20 glyph
    patch = crop_glyph(page, (100, 40, 110, 60), 32)
    assert patch.shape == (32, 32) and patch.dtype == np.uint8
    # 18% margin on the 20px side -> ~27px square window -> glyph fills ~74% of height, ~37% width
    ink = patch < 128
    rows = ink.any(axis=1).nonzero()[0]
    cols = ink.any(axis=0).nonzero()[0]
    assert 20 <= rows.max() - rows.min() + 1 <= 26
    assert 9 <= cols.max() - cols.min() + 1 <= 14
    assert int(patch[0, 0]) == 255
    # near the page edge the missing margin is filled with paper tone
    edge = crop_glyph(page, (0, 0, 10, 20), 32)
    assert edge.shape == (32, 32) and int(edge[0, 0]) == 255
    with pytest.raises(ValueError, match="degenerate"):
        crop_glyph(page, (10, 10, 11, 11))

    clf = _classifier(["fing1", "fing2"])
    assert clf.predict_box(page, (100, 40, 110, 60)).label in ("fing1", "fing2")


def test_render_strokes_like_musicset_templates() -> None:
    # A vertical bar drawn in editor pixel coordinates far from the origin
    strokes = [[(500.0, 300.0), (502.0, 420.0)]]
    glyph = render_strokes(strokes, 32, fraction=0.7)
    assert glyph.shape == (32, 32) and glyph.dtype == np.uint8
    ink = glyph < 128
    rows = ink.any(axis=1).nonzero()[0]
    cols = ink.any(axis=0).nonzero()[0]
    assert 20 <= rows.max() - rows.min() + 1 <= 26  # ~70% of 32 plus round caps
    assert cols.max() - cols.min() + 1 <= 6  # pen width, not squashed to a square
    assert abs((rows.max() + rows.min()) / 2 - 15.5) <= 1.5  # centred
    assert int(glyph[0, 0]) == 255

    dot = render_strokes([[(3.0, 3.0)]], 32, fraction=0.3)
    assert (dot < 128).sum() > 0  # a single point still draws a pen dot
    with pytest.raises(ValueError, match="at least one point"):
        render_strokes([[]], 32)

    clf = _classifier(["articStaccatoAbove", "fing1"])
    pred = clf.predict_strokes(strokes, top_k=2, fraction=0.7)
    assert pred.label in ("articStaccatoAbove", "fing1")


def test_load_strokes_json(tmp_path: Path) -> None:
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps([[[0, 0], [1, 2.5]], [[3, 3]]]))
    assert load_strokes_json(plain) == [[(0.0, 0.0), (1.0, 2.5)], [(3.0, 3.0)]]
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"strokes": [[[1, 1]]]}))
    assert load_strokes_json(wrapped) == [[(1.0, 1.0)]]


def test_from_checkpoint_round_trip(tmp_path: Path) -> None:
    model = GranitaTiny(num_classes=3)
    ckpt = tmp_path / "m.pt"
    torch.save(
        {"model_name": "tiny", "classes": ["a", "b", "c"], "model_state_dict": model.state_dict()},
        ckpt,
    )
    clf = GranitaClassifier.from_checkpoint(ckpt, device="cpu")
    assert clf.classes == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        GranitaClassifier.from_checkpoint(tmp_path / "missing.pt", device="cpu")
