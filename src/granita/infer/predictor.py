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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from granita.config import CLASS_GROUPS, CONTEXT_DIM, DEFAULT_CLASSES, DEFAULT_INPUT_SIZE, PAPER
from granita.context import (
    NOTE_RANGE,
    SymbolContext,
    context_from_mapping,
    context_from_strokes,
    context_vector,
)
from granita.data.transforms import get_eval_transforms
from granita.models.backbone import create_model

if TYPE_CHECKING:
    import torch.nn as nn

ImageLike = Image.Image | np.ndarray | torch.Tensor
Stroke = Sequence[tuple[float, float]]
ContextLike = SymbolContext | Mapping[str, Any] | np.ndarray | None
"""Where the symbol sits: a ``SymbolContext``, its field dict, a ready vector, or unknown."""
"""One pen stroke as ``(x, y)`` points, in any consistent unit (pixels, mm, ...)."""

# 0.05 (~1.6px on 32x32) scored best on held-out handwriting in the stroke-style sweep.
DEFAULT_STROKE_FRACTION = 0.7
DEFAULT_PEN_WIDTH = 0.05
CANVAS_STAFF_SPACES = 3.0
"""Staff spaces that fill the glyph canvas when an editor sends its staff space.

Calibrated by hand on an iPad and a Galaxy Tab with the test shells (``apps/``,
2026-11-01): at 3.0 every symbol written at its natural size on the staff was read,
while musicset's HOMUS value (8.8, a 150-unit HOMUS canvas taken as ~8.8 staff spaces,
``musicset.homus.CANVAS_STAFF_SPACES``) left fingerings and dynamics a few pixels
tall, far below the 0.52-0.90 their training renders use. The HOMUS constant stays
what it is, a property of that dataset; this one is the editor's rule."""
FRACTION_RANGE = (0.08, 0.90)


def fraction_from_staff_space(strokes: Sequence[Stroke], staff_space: float) -> float:
    """``fraction`` for :func:`render_strokes` from the editor's staff space (distance
    between two staff lines in the strokes' units), so a drawing keeps its natural size."""
    if staff_space <= 0:
        raise ValueError(f"staff_space must be positive, got {staff_space}")
    points = [(float(x), float(y)) for stroke in strokes for x, y in stroke]
    if not points:
        raise ValueError("fraction_from_staff_space needs at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    lo, hi = FRACTION_RANGE
    return float(min(hi, max(lo, span / (CANVAS_STAFF_SPACES * staff_space))))


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
    paper: int = PAPER,
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
    bg = int(np.percentile(sub, 85)) if sub.size else paper
    square = np.full((side, side), bg, dtype=np.uint8)
    square[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = sub
    return np.asarray(
        Image.fromarray(square, mode="L").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _segment_widths(pts: Sequence[tuple[float, float]], width: int) -> list[int]:
    """Pressure-like pen widths per segment: wide where the pen moves slowly, narrow where fast."""
    lengths = [math.dist(a, b) for a, b in pairwise(pts)]
    ref = sorted(lengths)[len(lengths) // 2] if lengths else 0.0
    if ref <= 1e-9:
        return [width] * len(lengths)
    return [max(1, round(width * min(1.3, max(0.55, 1.3 - 0.5 * (ln / ref))))) for ln in lengths]


def render_strokes(
    strokes: Sequence[Stroke],
    size: int = DEFAULT_INPUT_SIZE[0],
    *,
    fraction: float = DEFAULT_STROKE_FRACTION,
    pen_width: float = DEFAULT_PEN_WIDTH,
    brush: bool = False,
    canvas: int | None = None,
    paper: int = PAPER,
) -> np.ndarray:
    """Rasterise pen strokes the way musicset renders its stylus stroke templates.

    The strokes' joint bounding box is centred and scaled, aspect preserved, so
    its longer side spans ``fraction`` of the canvas; the pen is ``pen_width``
    canvas widths wide with round caps. Drawn at ``canvas`` pixels (default
    ``max(96, 3 * size)`` like musicset) and downsampled to ``size``.

    ``brush`` imitates a pressure brush instead of a uniform pen: each segment's
    width follows the pen speed (slow = wide) and the result gets a soft edge.

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

    img = Image.new("L", (canvas_px, canvas_px), paper)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = [
            (canvas_px * 0.5 + (float(x) - cx) * scale, canvas_px * 0.5 + (float(y) - cy) * scale)
            for x, y in stroke
        ]
        if not pts:
            continue
        if brush and len(pts) >= 2:
            widths = _segment_widths(pts, width)
            for (a, b), w in zip(pairwise(pts), widths, strict=True):
                draw.line([a, b], fill=0, width=w)
                rr = w * 0.5
                draw.ellipse((b[0] - rr, b[1] - rr, b[0] + rr, b[1] + rr), fill=0)
            r = widths[0] * 0.5
        else:
            if len(pts) >= 2:
                draw.line(pts, fill=0, width=width, joint="curve")
            r = width * 0.45
        for px, py in (pts[0], pts[-1]):
            draw.ellipse((px - r, py - r, px + r, py + r), fill=0)
    if brush:
        img = img.filter(ImageFilter.GaussianBlur(radius=canvas_px * 0.012))
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
        self.context_dim = int(getattr(model, "context_dim", 0))
        """0 when the model ignores the conditioning vector."""
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

        model = create_model(
            model_name=model_name,
            num_classes=len(classes),
            context_dim=int(ckpt.get("context_dim", 0)),
        )
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model=model, classes=classes, device=target_device)

    def _context_tensor(self, contexts: Sequence[ContextLike] | None, n: int) -> torch.Tensor:
        rows = []
        for ctx in contexts if contexts is not None else [None] * n:
            if ctx is None or isinstance(ctx, SymbolContext):
                rows.append(context_vector(ctx))
            elif isinstance(ctx, np.ndarray):
                rows.append(np.asarray(ctx, dtype=np.float32).reshape(CONTEXT_DIM))
            else:
                rows.append(context_vector(context_from_mapping(ctx)))
        return torch.from_numpy(np.stack(rows)).to(self.device)

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
    def _predict_tensor(
        self, batch: torch.Tensor, top_k: int, contexts: Sequence[ContextLike] | None = None
    ) -> list[Prediction]:
        logits = self.model(batch, self._context_tensor(contexts, batch.shape[0]))
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

    def predict(
        self, image: ImageLike, top_k: int = 5, *, context: ContextLike = None
    ) -> Prediction:
        """Predict the symbol class for a single isolated glyph image.

        ``context`` says where in the score the symbol sits (see
        :mod:`granita.context`); models trained without one ignore it.
        """
        return self._predict_tensor(self._prepare_tensor(image), top_k, [context])[0]

    def predict_batch(
        self,
        images: Sequence[ImageLike],
        top_k: int = 5,
        *,
        contexts: Sequence[ContextLike] | None = None,
    ) -> list[Prediction]:
        """Predict classes for a batch of isolated glyph images."""
        if not images:
            return []
        batch = torch.cat([self._prepare_tensor(img) for img in images], dim=0)
        return self._predict_tensor(batch, top_k, contexts)

    def predict_strokes(
        self,
        strokes: Sequence[Stroke],
        top_k: int = 5,
        *,
        fraction: float | None = DEFAULT_STROKE_FRACTION,
        staff_space: float | None = None,
        pen_width: float = DEFAULT_PEN_WIDTH,
        brush: bool = False,
        context: ContextLike = None,
    ) -> Prediction:
        """Classify stylus strokes drawn in an editor (see :func:`render_strokes`).

        Pass ``staff_space`` (pixels between two staff lines) so the drawing
        keeps its natural size; ``fraction`` is used only when it is absent.
        ``context`` is where the mark was drawn (:func:`granita.context.context_from_strokes`).
        """
        size = DEFAULT_INPUT_SIZE[0]
        if staff_space is not None:
            fraction = fraction_from_staff_space(strokes, staff_space)
        if fraction is None:
            fraction = DEFAULT_STROKE_FRACTION
        if isinstance(context, Mapping) and ("width" not in context or "height" not in context):
            # Size the context from the strokes: by the staff space when known, else by the
            # staff space the fraction implies (span = fraction * CANVAS_STAFF_SPACES spaces)
            if "y_staff" not in context:
                raise ValueError("a stroke context needs at least 'y_staff'")
            if staff_space is None:
                points = [(float(x), float(y)) for stroke in strokes for x, y in stroke]
                xs = [pt[0] for pt in points]
                ys = [pt[1] for pt in points]
                span = max(max(xs) - min(xs), max(ys) - min(ys)) if points else 0.0
                staff_space = max(span, 1e-6) / (CANVAS_STAFF_SPACES * fraction)
            dx, dy = context.get("note_dx"), context.get("note_dy")
            nw, nh = context.get("note_w"), context.get("note_h")
            context = context_from_strokes(
                strokes,
                staff_space,
                y_staff=float(context["y_staff"]),
                x_measure=float(context.get("x_measure", 0.5)),
                barline=float(context.get("barline", NOTE_RANGE)),
                note_dx=None if dx is None else float(dx),
                note_dy=None if dy is None else float(dy),
                note_w=None if nw is None else float(nw),
                note_h=None if nh is None else float(nh),
            )
        glyph = render_strokes(
            strokes, size, fraction=fraction, pen_width=pen_width, brush=brush, paper=self.paper
        )
        return self.predict(glyph, top_k=top_k, context=context)

    def predict_box(
        self,
        page: ImageLike,
        bbox: Sequence[float],
        top_k: int = 5,
        *,
        context: ContextLike = None,
    ) -> Prediction:
        """Classify the glyph inside ``bbox`` on a page image (see :func:`crop_glyph`)."""
        return self.predict(
            crop_glyph(page, bbox, DEFAULT_INPUT_SIZE[0], paper=self.paper),
            top_k=top_k,
            context=context,
        )


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
