"""Unit tests for Granita dataset loaders and transforms."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from granita.data.dataset import (
    SymbolFolderDataset,
    SymbolNpzDataset,
    load_dataset_from_path,
    load_split_dataset,
    make_loader,
)
from granita.data.transforms import get_eval_transforms, get_train_transforms


def _glyphs(n: int) -> np.ndarray:
    images = np.full((n, 32, 32), 255, dtype=np.uint8)
    images[:, 10:20, 10:20] = 0
    return images


def test_symbol_npz_dataset() -> None:
    images = _glyphs(10)
    labels = np.arange(10, dtype=np.int64)
    classes = [f"c{i}" for i in range(10)]

    tf = get_train_transforms((32, 32))
    dataset = SymbolNpzDataset(images, labels, classes, transform=tf)

    assert len(dataset) == 10
    img, label = dataset[0]
    assert img.shape == (1, 32, 32)
    assert isinstance(label, int)
    assert label == 0
    assert dataset.sources is None


def test_eval_transforms_shape() -> None:
    eval_tf = get_eval_transforms((32, 32))
    raw_tensor = torch.full((1, 32, 32), 255, dtype=torch.uint8)
    out = eval_tf(raw_tensor)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32
    assert float(out.max()) == 1.0  # paper maps to +1 after mean/std 0.5


def test_dataset_mismatch_raises() -> None:
    images = np.zeros((5, 32, 32), dtype=np.uint8)
    labels = np.zeros(4, dtype=np.int64)
    classes = ["c0"]

    with pytest.raises(ValueError, match="Mismatch"):
        SymbolNpzDataset(images, labels, classes)
    with pytest.raises(ValueError, match="sources"):
        SymbolNpzDataset(images, np.zeros(5, dtype=np.int64), classes, sources=np.zeros(3))


def _write_npz(folder: Path, with_sources: bool, with_test: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "x_train": _glyphs(12),
        "y_train": np.arange(12, dtype=np.int64) % 3,
        "x_val": _glyphs(6),
        "y_val": np.arange(6, dtype=np.int64) % 3,
        "classes": np.array(["fing1", "fing2", "trill"]),
    }
    if with_test:
        arrays["x_test"] = _glyphs(3)
        arrays["y_test"] = np.arange(3, dtype=np.int64)
    if with_sources:
        arrays["s_train"] = np.array([0, 1, 2] * 4, dtype=np.uint8)
        arrays["s_val"] = np.array([0, 1, 2] * 2, dtype=np.uint8)
        if with_test:
            arrays["s_test"] = np.array([1, 1, 2], dtype=np.uint8)
    np.savez(folder / "dataset.npz", **arrays)  # ty: ignore[invalid-argument-type]
    return folder / "dataset.npz"


def test_load_npz_splits_with_sources(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "npz", with_sources=True)
    train, _val, test, classes = load_dataset_from_path(npz, batch_size=4, num_workers=0)
    assert classes == ["fing1", "fing2", "trill"]
    test_ds = load_split_dataset(npz, "test")
    assert test is not None and test_ds is not None and len(test_ds) == 3
    val_ds = load_split_dataset(npz, "val")
    assert val_ds is not None and val_ds.sources is not None
    assert val_ds.sources.tolist() == [0, 1, 2, 0, 1, 2]
    images, labels = next(iter(train))
    assert images.shape == (4, 1, 32, 32) and labels.shape == (4,)


def test_load_npz_without_sources_or_test(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "npz", with_sources=False, with_test=False)
    _, _, test, _ = load_dataset_from_path(npz, batch_size=4, num_workers=0)
    assert test is None
    assert load_split_dataset(npz, "test") is None
    train_ds = load_split_dataset(npz, "train")
    assert train_ds is not None and train_ds.sources is None
    with pytest.raises(ValueError, match="unknown split"):
        load_split_dataset(npz, "holdout")


def test_load_image_folder(tmp_path: Path) -> None:
    root = tmp_path / "folder"
    for split, n in (("train", 4), ("val", 2)):
        for cls in ("fing1", "trill"):
            d = root / split / cls
            d.mkdir(parents=True)
            for k in range(n):
                Image.fromarray(_glyphs(1)[0], mode="L").save(d / f"{k:03d}.png")
    (root / "labels.json").write_text(json.dumps({"classes": ["fing1", "trill"], "invert": False}))

    ds = SymbolFolderDataset(root / "train", classes=["fing1", "trill"])
    assert len(ds) == 8 and ds[0][0].shape == (1, 32, 32)
    _train, _val, test, classes = load_dataset_from_path(root, batch_size=8, num_workers=0)
    assert classes == ["fing1", "trill"] and test is None
    val_ds = load_split_dataset(root, "val")
    assert val_ds is not None and len(val_ds) == 4
    loader = make_loader(ds, batch_size=8, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    assert images.shape == (8, 1, 32, 32) and labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_from_path(tmp_path / "nope.npz")
