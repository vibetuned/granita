"""Unit tests for Granita classifier predictor and inference pipeline."""

import numpy as np
import torch
from PIL import Image

from granita.infer.predictor import GranitaClassifier
from granita.models.backbone import GranitaNet


def test_predictor_with_pil_image() -> None:
    classes = ["fing1", "fing2", "trill"]
    model = GranitaNet(num_classes=len(classes), in_channels=1)
    classifier = GranitaClassifier(model=model, classes=classes, device="cpu")

    # White 32x32 image with black square in center
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
    model = GranitaNet(num_classes=len(classes), in_channels=1)
    classifier = GranitaClassifier(model=model, classes=classes, device="cpu")

    arr = np.full((32, 32), 255, dtype=np.uint8)
    arr[12:20, 12:20] = 0

    pred = classifier.predict(arr)
    assert pred.label in classes
    assert pred.group == "accidental"


def test_predictor_batch() -> None:
    classes = ["fing1", "fing2", "trill"]
    model = GranitaNet(num_classes=len(classes), in_channels=1)
    classifier = GranitaClassifier(model=model, classes=classes, device="cpu")

    imgs = [
        Image.new("L", (32, 32), color=255),
        np.full((32, 32), 255, dtype=np.uint8),
        torch.full((1, 32, 32), 255, dtype=torch.uint8),
    ]

    preds = classifier.predict_batch(imgs, top_k=2)
    assert len(preds) == 3
    for p in preds:
        assert p.label in classes
        assert len(p.top_k) == 2
