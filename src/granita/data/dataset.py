"""Dataset loaders for Granita music glyph datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from granita.config import CONTEXT_DIM, DEFAULT_CLASSES, PAPER
from granita.data.transforms import get_eval_transforms, get_train_transforms

if TYPE_CHECKING:
    import torchvision.transforms.v2 as T

SPLITS: tuple[str, ...] = ("train", "val", "test")
Item = tuple[torch.Tensor, torch.Tensor, int]
"""``(image, context vector, label)``; the context is all zeros when unknown."""
GlyphLoader = DataLoader[Item]


class SymbolNpzDataset(Dataset[Item]):
    """PyTorch Dataset backed by arrays from a ``musicset symbols`` npz archive.

    ``sources`` (optional, one uint8 code per sample indexing
    ``granita.config.SOURCE_NAMES``) is kept for per-source evaluation;
    ``contexts`` (optional, ``(N, CONTEXT_DIM)`` float32) is the conditioning
    vector, zeros when the archive has none.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        classes: list[str],
        transform: T.Compose | Callable[[torch.Tensor], torch.Tensor] | None = None,
        sources: np.ndarray | None = None,
        contexts: np.ndarray | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels")
        if sources is not None and len(sources) != len(labels):
            raise ValueError(f"Mismatch: {len(sources)} sources vs {len(labels)} labels")
        if contexts is not None and contexts.shape != (len(labels), CONTEXT_DIM):
            raise ValueError(f"contexts must be ({len(labels)}, {CONTEXT_DIM}), got {contexts.shape}")

        self.images = images
        self.labels = labels.astype(np.int64)
        self.classes = classes
        self.transform = transform
        self.sources = None if sources is None else sources.astype(np.int64)
        self.contexts = None if contexts is None else contexts.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Item:
        img_np = self.images[idx]
        if img_np.ndim == 2:
            # (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        elif img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_tensor = torch.from_numpy(img_np)
        else:
            # (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        label = int(self.labels[idx])

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        context = (
            torch.zeros(CONTEXT_DIM) if self.contexts is None else torch.from_numpy(self.contexts[idx])
        )
        return img_tensor, context, label


class SymbolFolderDataset(Dataset[Item]):
    """PyTorch Dataset loading glyph images from a directory organized by class name."""

    def __init__(
        self,
        root_dir: str | Path,
        classes: list[str] | None = None,
        transform: T.Compose | Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root_dir)
        self.transform = transform
        self.sources = None
        self.contexts = None

        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as pil_img:
            gray = pil_img.convert("L")
            img_arr = np.array(gray, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.zeros(CONTEXT_DIM), label


def read_labels_json(data_path: Path) -> dict[str, object]:
    """``labels.json`` written next to a ``musicset symbols`` build, or ``{}``."""
    folder = data_path if data_path.is_dir() else data_path.parent
    labels_json = folder / "labels.json"
    if not labels_json.is_file():
        return {}
    with labels_json.open(encoding="utf-8") as f:
        return json.load(f)


def _paper_level(meta: dict[str, object]) -> int:
    return 0 if meta.get("invert") else PAPER


def load_split_dataset(
    data_path: str | Path,
    split: str,
    train_augment: bool = False,
) -> SymbolNpzDataset | SymbolFolderDataset | None:
    """One split of an npz archive or ``ImageFolder`` root, or ``None`` if it is absent.

    Evaluation transforms unless ``train_augment``; ``classes`` and ``sources``
    are available as dataset attributes.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}: {SPLITS}")
    path = Path(data_path)
    meta = read_labels_json(path)
    transform = (
        get_train_transforms(paper=_paper_level(meta)) if train_augment else get_eval_transforms()
    )

    if path.is_file() and path.suffix == ".npz":
        with np.load(path) as data:
            if f"x_{split}" not in data:
                return None
            classes = (
                [str(c) for c in data["classes"]] if "classes" in data else list(DEFAULT_CLASSES)
            )
            sources = data[f"s_{split}"] if f"s_{split}" in data else None
            contexts = data[f"c_{split}"] if f"c_{split}" in data else None
            return SymbolNpzDataset(
                data[f"x_{split}"],
                data[f"y_{split}"],
                classes,
                transform=transform,
                sources=sources,
                contexts=contexts,
            )
    if path.is_dir():
        split_dir = path / split
        if not split_dir.is_dir():
            return None
        classes = list(meta.get("classes", DEFAULT_CLASSES))
        return SymbolFolderDataset(split_dir, classes=classes, transform=transform)
    raise FileNotFoundError(f"Dataset path not found: {path}")


def make_loader(
    dataset: Dataset[Item],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> GlyphLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def load_dataset_from_path(
    data_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[GlyphLoader, GlyphLoader, GlyphLoader | None, list[str]]:
    """Train (augmented, shuffled), val and optional test DataLoaders plus the class list."""
    train_ds = load_split_dataset(data_path, "train", train_augment=True)
    val_ds = load_split_dataset(data_path, "val")
    test_ds = load_split_dataset(data_path, "test")
    if train_ds is None or val_ds is None:
        raise FileNotFoundError(f"{data_path} needs both a train and a val split")

    train_loader = make_loader(train_ds, batch_size, True, num_workers, pin_memory)
    val_loader = make_loader(val_ds, batch_size, False, num_workers, pin_memory)
    test_loader = (
        make_loader(test_ds, batch_size, False, num_workers, pin_memory)
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader, train_ds.classes
