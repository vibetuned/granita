"""Image transforms and augmentations for music glyphs."""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T

from granita.config import DEFAULT_INPUT_SIZE, PAPER

GLYPH_MEAN = (0.5,)
GLYPH_STD = (0.5,)


def get_train_transforms(
    image_size: tuple[int, int] = DEFAULT_INPUT_SIZE, paper: int = PAPER
) -> T.Compose:
    """Training pipeline: mild affine jitter on top of musicset's build-time augmentation.

    ``paper`` is the grey level uncovered by the affine warp: 255 for the default
    black-on-white data, 0 for a dataset built with ``musicset symbols --invert``.
    """
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(image_size, antialias=True),
            T.RandomAffine(
                degrees=(-7, 7),
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                fill=paper,
            ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=GLYPH_MEAN, std=GLYPH_STD),
        ]
    )


def get_eval_transforms(image_size: tuple[int, int] = DEFAULT_INPUT_SIZE) -> T.Compose:
    """Transform pipeline for evaluation and inference."""
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(image_size, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=GLYPH_MEAN, std=GLYPH_STD),
        ]
    )
