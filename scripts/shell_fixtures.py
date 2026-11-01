"""Fixtures for the mobile test shells (``apps/ios``, ``apps/android``).

A few stroke sets drawn on a staff, rendered and classified by the released
PyTorch checkpoint, so each shell can check its own rasteriser and its exported
model (Core ML, LiteRT) against the reference:

    uv run python scripts/shell_fixtures.py        # -> models/granitanet/fixtures.json

Each fixture carries the strokes in page pixels, the staff they were drawn on
(staff space, middle line, barlines), the context the shells derive from them
(raw facts and the 9-float vector), the 32 x 32 image ``render_strokes`` makes
(uniform pen, natural size from the staff space) and the checkpoint's logits,
with and without the context. The shapes are polylines, not handwriting: the
fixtures pin numbers, not accuracy, and ``drawn_as`` is only what was meant.

The 96 -> 32 downsample is also re-implemented here (``downsample_like_pil``)
and checked against Pillow, so the shells can copy that exact filter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from granita.context import SymbolContext, context_vector
from granita.infer.predictor import (
    GranitaClassifier,
    Stroke,
    fraction_from_staff_space,
    render_strokes,
)

CANVAS_PX = 96
SIZE_PX = 32

# The staff the strokes are drawn on, in page pixels: five lines a staff space
# apart around the middle line, a barline at each end.
STAFF_SPACE = 24.0
MIDDLE_Y = 300.0
LEFT_X = 100.0
RIGHT_X = 700.0


def _loop(cx: float, cy: float, rx: float, ry: float, n: int = 16) -> Stroke:
    """A closed polyline ellipse, drawn clockwise from the left."""
    return [
        (cx - rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n + 1)
    ]


def _arc(cx: float, cy: float, r: float, n: int = 12) -> Stroke:
    """The upper half of a circle, left to right."""
    return [
        (cx - r * math.cos(math.pi * i / n), cy - r * math.sin(math.pi * i / n))
        for i in range(n + 1)
    ]


FIXTURES: list[tuple[str, list[Stroke]]] = [
    ("barline", [[(400.0, 252.0), (400.5, 348.0)]]),
    ("dot in the staff", [[(420.0, 288.0), (421.0, 289.0), (420.0, 290.0)]]),
    ("dot above the staff", [[(400.0, 230.0), (401.0, 231.0), (400.0, 232.0)]]),
    (
        "sharp",
        [
            [(392.0, 270.0), (392.0, 330.0)],
            [(408.0, 266.0), (408.0, 326.0)],
            [(380.0, 292.0), (420.0, 284.0)],
            [(380.0, 312.0), (420.0, 304.0)],
        ],
    ),
    (
        "flat",
        [
            [(390.0, 250.0), (390.0, 330.0)],
            [(390.0, 300.0), (405.0, 296.0), (410.0, 310.0), (400.0, 326.0), (390.0, 330.0)],
        ],
    ),
    (
        "natural",
        [
            [(392.0, 260.0), (392.0, 320.0)],
            [(408.0, 276.0), (408.0, 340.0)],
            [(392.0, 296.0), (408.0, 292.0)],
            [(392.0, 320.0), (408.0, 316.0)],
        ],
    ),
    (
        "quarter note",
        [
            _loop(400.0, 324.0, 11.0, 8.0),
            _loop(400.0, 324.0, 7.0, 5.0),
            _loop(400.0, 324.0, 3.0, 2.0),
            [(411.0, 324.0), (411.0, 250.0)],
        ],
    ),
    ("half note", [_loop(400.0, 324.0, 11.0, 8.0), [(411.0, 324.0), (411.0, 250.0)]]),
    ("whole note", [_loop(400.0, 300.0, 13.0, 8.0)]),
    (
        "quarter rest",
        [
            [
                (395.0, 255.0),
                (410.0, 275.0),
                (395.0, 295.0),
                (412.0, 318.0),
                (400.0, 312.0),
                (392.0, 330.0),
                (404.0, 340.0),
            ]
        ],
    ),
    ("half rest", [[(388.0, 296.0), (412.0, 296.0)], [(388.0, 300.0), (412.0, 300.0)]]),
    ("fermata", [_arc(400.0, 240.0, 18.0), [(400.0, 240.0), (401.0, 241.0)]]),
    (
        "fingering 3",
        [[(392.0, 262.0), (410.0, 262.0), (400.0, 276.0), (410.0, 290.0), (392.0, 292.0)]],
    ),
]


def symbol_context(strokes: list[Stroke]) -> SymbolContext:
    """The context a shell derives from strokes on the fixture staff: no notehead near."""
    xs = [x for s in strokes for x, _ in s]
    ys = [y for s in strokes for _, y in s]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    return SymbolContext(
        y_staff=(cy - MIDDLE_Y) / STAFF_SPACE,
        x_measure=(cx - LEFT_X) / (RIGHT_X - LEFT_X),
        barline=min(cx - LEFT_X, RIGHT_X - cx) / STAFF_SPACE,
        note_dx=None,
        note_dy=None,
        width=(max(xs) - min(xs)) / STAFF_SPACE,
        height=(max(ys) - min(ys)) / STAFF_SPACE,
    )


def _pil_taps(in_size: int, out_size: int) -> list[tuple[int, list[float]]]:
    """Pillow's bilinear coefficients for a reduction: for each output pixel, the first
    input pixel and the normalised triangle weights (``precompute_coeffs`` in Resample.c)."""
    scale = in_size / out_size
    support = 1.0 * scale
    taps = []
    for out in range(out_size):
        centre = (out + 0.5) * scale
        lo = max(0, int(centre - support + 0.5))
        hi = min(in_size, int(centre + support + 0.5))
        weights = [max(0.0, 1.0 - abs((x - centre + 0.5) / scale)) for x in range(lo, hi)]
        total = sum(weights)
        taps.append((lo, [w / total for w in weights]))
    return taps


def downsample_like_pil(img: np.ndarray, size: int = SIZE_PX) -> np.ndarray:
    """``img`` (uint8, square) reduced to ``size`` the way ``Image.resize(BILINEAR)`` does:
    a triangle filter ``in/out`` pixels wide, rows first, each pass rounded to 8 bits."""
    taps = _pil_taps(img.shape[0], size)

    def pass_rows(src: np.ndarray) -> np.ndarray:  # reduce along the last axis
        out = np.empty((src.shape[0], size), dtype=np.float64)
        for j, (lo, ws) in enumerate(taps):
            out[:, j] = src[:, lo : lo + len(ws)].astype(np.float64) @ np.asarray(ws)
        return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)

    return pass_rows(pass_rows(img).T).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", default="models/granitanet/best_model.pt")
    parser.add_argument("--output", default="models/granitanet/fixtures.json")
    args = parser.parse_args()

    clf = GranitaClassifier.from_checkpoint(args.checkpoint, device="cpu")
    zero_ctx = torch.zeros(1, clf.context_dim)
    fixtures = []
    worst = 0
    for drawn_as, strokes in FIXTURES:
        fraction = fraction_from_staff_space(strokes, STAFF_SPACE)
        big = render_strokes(strokes, CANVAS_PX, fraction=fraction, canvas=CANVAS_PX)
        glyph = render_strokes(strokes, SIZE_PX, fraction=fraction, canvas=CANVAS_PX)
        mine = downsample_like_pil(big)
        worst = max(worst, int(np.abs(mine.astype(int) - glyph.astype(int)).max()))
        ctx = symbol_context(strokes)
        vec = context_vector(ctx)
        x = ((torch.from_numpy(glyph).float() / 255.0 - 0.5) / 0.5).view(1, 1, SIZE_PX, SIZE_PX)
        with torch.no_grad():
            logits = clf.model(x, torch.from_numpy(vec)[None])[0]
            plain = clf.model(x, zero_ctx)[0]
        probs = torch.softmax(logits, -1)
        best = int(probs.argmax())
        fixtures.append(
            {
                "drawn_as": drawn_as,
                "strokes": [[[x, y] for x, y in s] for s in strokes],
                "fraction": round(fraction, 6),
                "context": {
                    "y_staff": ctx.y_staff,
                    "x_measure": ctx.x_measure,
                    "barline": ctx.barline,
                    "note_dx": None,
                    "note_dy": None,
                    "width": ctx.width,
                    "height": ctx.height,
                },
                "context_vector": [round(float(v), 6) for v in vec],
                "pixels": glyph.flatten().tolist(),
                "logits": [round(float(v), 5) for v in logits],
                "logits_no_context": [round(float(v), 5) for v in plain],
                "top": {
                    "label": clf.classes[best],
                    "index": best,
                    "confidence": round(float(probs[best]), 5),
                },
            }
        )
        label, confidence = clf.classes[best], float(probs[best])
        print(f"{drawn_as:22s} -> {label:24s} {confidence:.3f}   fraction {fraction:.3f}")
    print(f"downsample_like_pil vs Pillow: max |Δ| {worst} grey level(s)")
    if worst > 1:
        raise SystemExit("the re-implemented downsample drifted from Pillow's")

    out = {
        "model": "granitanet",
        "checkpoint": args.checkpoint,
        "staff": {
            "staff_space": STAFF_SPACE,
            "middle_y": MIDDLE_Y,
            "left_x": LEFT_X,
            "right_x": RIGHT_X,
        },
        "render": {
            "canvas_px": CANVAS_PX,
            "size_px": SIZE_PX,
            "pen_px": max(2, round(CANVAS_PX * 0.05)),
            "paper": 255,
        },
        "classes": clf.classes,
        "fixtures": fixtures,
    }
    Path(args.output).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {args.output}: {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
