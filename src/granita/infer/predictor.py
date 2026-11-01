"""Inference engine and high-level Python API for Granita glyph classification.

Every input is brought to the form ``musicset symbols`` gives its training
samples: black ink on white paper, the glyph centred in a square canvas, then
resized to 32x32. One entry point per sample source:

* :meth:`GranitaClassifier.predict` / :meth:`~GranitaClassifier.predict_batch`:
  an already isolated glyph image, padded to square (never squashed);
* :meth:`GranitaClassifier.predict_strokes`: stylus strokes from an editor,
  rasterised like musicset's stroke templates (:func:`render_strokes`);
* :meth:`GranitaClassifier.predict_box`: a bounding box on a page image,
  cropped like musicset's score crops (:func:`crop_glyph`).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from granita.config import CLASS_GROUPS, DEFAULT_CLASSES, DEFAULT_INPUT_SIZE, PAPER
from granita.data.transforms import get_eval_transforms
from granita.models.backbone import create_model

if TYPE_CHECKING:
    import torch.nn as nn

ImageLike = Image.Image | np.ndarray | torch.Tensor
Stroke = Sequence[tuple[float, float]]
"""One pen stroke as ``(x, y)`` points, in any consistent unit (pixels, mm, ...)."""

# musicset draws stroke templates spanning 0.55-0.88 of the canvas with a pen
# 0.055-0.115 canvas widths wide; these are the midpoints.
DEFAULT_STROKE_FRACTION = 0.7
DEFAULT_PEN_WIDTH = 0.085


def to_gray_uint8(image: ImageLike) -> np.ndarray:
    """Any image-like input as an ``(H, W)`` uint8 array, 255 = paper.

    Colour inputs are converted to luminance; float arrays and tensors in
    ``[0, 1]`` are scaled to ``[0, 255]``.
    """
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):  # (C, H, W)
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            rgb = arr[..., :3].astype(np.float32)
            arr = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {tuple(arr.shape)}")
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0.0, 255.0).round()
    return arr.astype(np.uint8)


def pad_to_square(gray: np.ndarray, paper: int = PAPER, margin: float = 0.0) -> np.ndarray:
    """Centre ``gray`` on a square paper canvas ``(1 + 2 * margin)`` times its longer side."""
    h, w = gray.shape
    side = max(1, math.ceil(max(h, w) * (1.0 + 2.0 * margin)))
    canvas = np.full((side, side), paper, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = gray
    return canvas


def crop_glyph(
    page: ImageLike,
    bbox: Sequence[float],
    size: int = DEFAULT_INPUT_SIZE[0],
    *,
    margin: float = 0.18,
    min_span_px: float = 14.0,
) -> np.ndarray:
    """Crop ``bbox`` (``x1, y1, x2, y2`` pixels) from a page the way musicset builds score crops.

    The box is padded to a square spanning ``(1 + 2 * margin)`` times its longer
    side (at least ``min_span_px``), out-of-page margins are filled with the local
    paper tone, and the patch is resized to ``size`` with bilinear filtering.
    Mirrors ``musicset.symbols._crop_box_to_patch``; keep the two in sync.
    """
    image = to_gray_uint8(page)
    h, w = image.shape
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1, x2 = max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))
    y1, y2 = max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2.0 or bh < 2.0:
        raise ValueError(f"bbox {tuple(bbox)} is degenerate on a {w}x{h} page")

    span = max(min_span_px, max(bw, bh) * (1.0 + 2.0 * margin))
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half = 0.5 * span
    ix0, iy0 = math.floor(cx - half), math.floor(cy - half)
    ix1, iy1 = math.ceil(cx + half), math.ceil(cy + half)
    side = max(1, ix1 - ix0, iy1 - iy0)
    ix1, iy1 = ix0 + side, iy0 + side

    sx0, sy0 = max(0, ix0), max(0, iy0)
    sx1, sy1 = min(w, ix1), min(h, iy1)
    if sx1 <= sx0 or sy1 <= sy0:
        raise ValueError(f"bbox {tuple(bbox)} falls outside the {w}x{h} page")
    sub = image[sy0:sy1, sx0:sx1]
    bg = int(np.percentile(sub, 85)) if sub.size else PAPER
    square = np.full((side, side), bg, dtype=np.uint8)
    square[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = sub
    return np.asarray(
        Image.fromarray(square, mode="L").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def render_strokes(
    strokes: Sequence[Stroke],
    size: int = DEFAULT_INPUT_SIZE[0],
    *,
    fraction: float = DEFAULT_STROKE_FRACTION,
    pen_width: float = DEFAULT_PEN_WIDTH,
    canvas: int | None = None,
    paper: int = PAPER,
) -> np.ndarray:
    """Rasterise pen strokes the way musicset renders its stylus stroke templates.

    The strokes' joint bounding box is centred and scaled, aspect preserved, so
    its longer side spans ``fraction`` of the canvas; the pen is ``pen_width``
    canvas widths wide with round caps. Drawn at ``canvas`` pixels (default
    ``max(96, 3 * size)`` like musicset) and downsampled to ``size``.

    Absolute size is lost when normalising by the bounding box, so pass the
    glyph's size relative to its surroundings as ``fraction`` when the editor
    knows it (a staccato dot is small, a dynamic is large).
    """
    points = [(float(x), float(y)) for stroke in strokes for x, y in stroke]
    if not points:
        raise ValueError("render_strokes needs at least one point")
    canvas_px = canvas or max(96, size * 3)
    xs, ys = zip(*points, strict=True)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin)
    scale = (canvas_px * fraction) / span if span > 1e-9 else 0.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    width = max(2, round(canvas_px * pen_width))
    r = width * 0.45

    img = Image.new("L", (canvas_px, canvas_px), paper)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = [
            (canvas_px * 0.5 + (float(x) - cx) * scale, canvas_px * 0.5 + (float(y) - cy) * scale)
            for x, y in stroke
        ]
        if not pts:
            continue
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=width, joint="curve")
        for px, py in (pts[0], pts[-1]):
            draw.ellipse((px - r, py - r, px + r, py + r), fill=0)
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


@dataclass
class Prediction:
    """Inference result for a single music glyph."""

    label: str
    confidence: float
    group: str
    class_idx: int
    top_k: list[tuple[str, float]]

    def __repr__(self) -> str:
        return (
            f"Prediction(label='{self.label}', "
            f"confidence={self.confidence:.4f}, group='{self.group}')"
        )


class GranitaClassifier:
    """High-level classifier for predicting musical glyphs and symbols."""

    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        device: str | torch.device = "cpu",
        paper: int = PAPER,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.paper = paper
        self.transform = get_eval_transforms()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> GranitaClassifier:
        """Load a Granita classifier from a checkpoint file (.pt)."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        target_device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        ckpt = torch.load(path, map_location=target_device, weights_only=True)
        classes = list(ckpt.get("classes", DEFAULT_CLASSES))
        model_name = ckpt.get("model_name", "granitanet")

        model = create_model(model_name=model_name, num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model=model, classes=classes, device=target_device)

    def _prepare_tensor(self, image: ImageLike) -> torch.Tensor:
        """Convert an isolated glyph into a normalized ``(1, 1, H, W)`` tensor.

        A 4-D tensor is taken as an already preprocessed batch and only moved
        to the device.
        """
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return image.to(self.device)
        gray = pad_to_square(to_gray_uint8(image), paper=self.paper)
        tensor = torch.from_numpy(gray).unsqueeze(0)
        return self.transform(tensor).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _predict_tensor(self, batch: torch.Tensor, top_k: int) -> list[Prediction]:
        logits = self.model(batch)
        probs = F.softmax(logits, dim=-1)
        k = min(top_k, len(self.classes))
        top_probs, top_indices = probs.topk(k, dim=-1)

        results: list[Prediction] = []
        for i in range(batch.shape[0]):
            best_idx = int(top_indices[i, 0].item())
            best_label = self.classes[best_idx]
            results.append(
                Prediction(
                    label=best_label,
                    confidence=float(top_probs[i, 0].item()),
                    group=CLASS_GROUPS.get(best_label, "other"),
                    class_idx=best_idx,
                    top_k=[
                        (self.classes[int(idx.item())], float(p.item()))
                        for idx, p in zip(top_indices[i], top_probs[i], strict=True)
                    ],
                )
            )
        return results

    def predict(self, image: ImageLike, top_k: int = 5) -> Prediction:
        """Predict the symbol class for a single isolated glyph image."""
        return self._predict_tensor(self._prepare_tensor(image), top_k)[0]

    def predict_batch(self, images: Sequence[ImageLike], top_k: int = 5) -> list[Prediction]:
        """Predict classes for a batch of isolated glyph images."""
        if not images:
            return []
        batch = torch.cat([self._prepare_tensor(img) for img in images], dim=0)
        return self._predict_tensor(batch, top_k)

    def predict_strokes(
        self,
        strokes: Sequence[Stroke],
        top_k: int = 5,
        *,
        fraction: float = DEFAULT_STROKE_FRACTION,
        pen_width: float = DEFAULT_PEN_WIDTH,
    ) -> Prediction:
        """Classify stylus strokes drawn in an editor (see :func:`render_strokes`)."""
        size = DEFAULT_INPUT_SIZE[0]
        glyph = render_strokes(
            strokes, size, fraction=fraction, pen_width=pen_width, paper=self.paper
        )
        return self.predict(glyph, top_k=top_k)

    def predict_box(self, page: ImageLike, bbox: Sequence[float], top_k: int = 5) -> Prediction:
        """Classify the glyph inside ``bbox`` on a page image (see :func:`crop_glyph`)."""
        return self.predict(crop_glyph(page, bbox, DEFAULT_INPUT_SIZE[0]), top_k=top_k)


def load_strokes_json(path: str | Path) -> list[list[tuple[float, float]]]:
    """Strokes from a JSON file: ``[[[x, y], ...], ...]`` or ``{"strokes": [...]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["strokes"] if isinstance(data, dict) else data
    return [[(float(p[0]), float(p[1])) for p in stroke] for stroke in raw]


def main() -> None:
    from granita.cli import cmd_infer, parser_infer

    parser = argparse.ArgumentParser(description="Classify a music symbol with Granita")
    parser_infer(parser)
    cmd_infer(parser.parse_args())


if __name__ == "__main__":
    main()
