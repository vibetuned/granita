"""Unit tests for Granita dataset loaders and transforms."""

import numpy as np
import pytest
import torch

from granita.data.dataset import SymbolNpzDataset
from granita.data.transforms import get_eval_transforms, get_train_transforms


def test_symbol_npz_dataset() -> None:
    images = np.full((10, 32, 32), 255, dtype=np.uint8)
    # Add fake ink
    images[:, 10:20, 10:20] = 0
    labels = np.arange(10, dtype=np.int64)
    classes = [f"c{i}" for i in range(10)]

    tf = get_train_transforms((32, 32))
    dataset = SymbolNpzDataset(images, labels, classes, transform=tf)

    assert len(dataset) == 10
    img, label = dataset[0]
    assert img.shape == (1, 32, 32)
    assert isinstance(label, int)
    assert label == 0


def test_eval_transforms_shape() -> None:
    eval_tf = get_eval_transforms((32, 32))
    raw_tensor = torch.full((1, 32, 32), 255, dtype=torch.uint8)
    out = eval_tf(raw_tensor)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32


def test_dataset_mismatch_raises() -> None:
    images = np.zeros((5, 32, 32), dtype=np.uint8)
    labels = np.zeros(4, dtype=np.int64)
    classes = ["c0"]

    with pytest.raises(ValueError, match="Mismatch"):
        SymbolNpzDataset(images, labels, classes)
