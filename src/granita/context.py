"""Conditioning vector: where a pen mark sits in the score. Mirror of ``musicset.context``.

Facts are in staff spaces (the distance between two staff lines) so they do
not depend on the device; ``context_vector`` maps them to ``CONTEXT_DIM``
floats roughly in [-1, 1], all zero with ``valid = 0`` when unknown. Keep the
two implementations identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from granita.config import CONTEXT_DIM, CONTEXT_FEATURES

NOTE_RANGE = 8.0
"""Staff spaces within which a note's body counts as the symbol's neighbour."""
STAFF_HALF_HEIGHT = 2.0
"""Staff spaces from the middle line to the outer lines: the ``in_staff`` zone."""
NOTE_MARGIN = 0.5
"""Staff spaces a symbol may stick out past a body's sides and still count as above or
below it (or past its top and bottom and still count as left or right of it)."""


@dataclass(frozen=True, slots=True)
class SymbolContext:
    """Raw facts about a symbol's place in the score, in staff spaces."""

    y_staff: float
    """Symbol centre from the staff's middle line, positive downwards."""
    x_measure: float
    """0 at the left barline, 1 at the right (0.5 when no measure is known)."""
    barline: float
    """Distance to the nearest barline."""
    note_dx: float | None
    note_dy: float | None
    """Symbol centre minus the centre of the nearest note's body; ``None`` without one.

    A note's body is the bounding box of its notehead, stem and flag, or of every
    notehead plus the stem and flag of its chord, without accidentals, dots or
    articulations: the note as the editor shows it before the mark is added."""
    width: float
    height: float
    note_w: float | None = None
    note_h: float | None = None
    """Width and height of that body (``None`` or 0: a point)."""


def context_vector(ctx: SymbolContext | None) -> np.ndarray:
    """``CONTEXT_DIM`` float32 features for ``ctx`` (see ``CONTEXT_FEATURES``); all zero
    when ``None``. Flags: the staff zone (in / above / below the lines) and where the
    symbol is relative to the nearest note's body (above its top, below its bottom, left
    or right of its sides)."""
    if ctx is None:
        return np.zeros(CONTEXT_DIM, dtype=np.float32)
    has_note = ctx.note_dx is not None and ctx.note_dy is not None
    dx = ctx.note_dx if ctx.note_dx is not None else 0.0
    dy = ctx.note_dy if ctx.note_dy is not None else 0.0
    hw = 0.5 * (ctx.note_w or 0.0)
    hh = 0.5 * (ctx.note_h or 0.0)
    over = has_note and abs(dx) <= hw + NOTE_MARGIN
    beside = has_note and abs(dy) <= hh + NOTE_MARGIN
    return np.array(
        [
            1.0,
            1.0 if abs(ctx.y_staff) <= STAFF_HALF_HEIGHT else 0.0,
            1.0 if ctx.y_staff < -STAFF_HALF_HEIGHT else 0.0,
            1.0 if ctx.y_staff > STAFF_HALF_HEIGHT else 0.0,
            max(-2.0, min(2.0, ctx.y_staff / 6.0)),
            max(-1.0, min(1.0, ctx.x_measure * 2.0 - 1.0)),
            max(0.0, min(1.0, ctx.barline / 8.0)),
            1.0 if has_note else 0.0,
            1.0 if over and dy < -hh else 0.0,
            1.0 if over and dy > hh else 0.0,
            1.0 if beside and dx < -hw else 0.0,
            1.0 if beside and dx > hw else 0.0,
            max(-1.0, min(1.0, dx / NOTE_RANGE)),
            max(-1.0, min(1.0, dy / NOTE_RANGE)),
            max(0.0, min(1.5, ctx.width / 8.0)),
            max(0.0, min(1.5, ctx.height / 8.0)),
        ],
        dtype=np.float32,
    )


def context_from_strokes(
    strokes: Sequence[Sequence[tuple[float, float]]],
    staff_space: float,
    *,
    y_staff: float,
    x_measure: float = 0.5,
    barline: float = NOTE_RANGE,
    note_dx: float | None = None,
    note_dy: float | None = None,
    note_w: float | None = None,
    note_h: float | None = None,
) -> SymbolContext:
    """Context of an editor drawing: its size from the strokes, its place from the editor.

    ``y_staff`` is the strokes' centre in staff spaces from the middle line
    (down is positive); ``note_dx`` / ``note_dy`` the centre minus the centre of the
    nearest existing note's body (``SymbolContext.note_dx``), and ``note_w`` /
    ``note_h`` that body's size, or ``None`` when no note is within ``NOTE_RANGE``.
    """
    if staff_space <= 0:
        raise ValueError(f"staff_space must be positive, got {staff_space}")
    points = [(float(x), float(y)) for stroke in strokes for x, y in stroke]
    if not points:
        raise ValueError("context_from_strokes needs at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return SymbolContext(
        y_staff=y_staff,
        x_measure=x_measure,
        barline=barline,
        note_dx=note_dx,
        note_dy=note_dy,
        width=(max(xs) - min(xs)) / staff_space,
        height=(max(ys) - min(ys)) / staff_space,
        note_w=note_w if note_dx is not None else None,
        note_h=note_h if note_dx is not None else None,
    )


def context_from_mapping(raw: Mapping[str, Any]) -> SymbolContext:
    """Context from a dict with the ``SymbolContext`` field names (``note_dx`` may be null,
    ``note_w`` / ``note_h`` absent)."""
    dx, dy = raw.get("note_dx"), raw.get("note_dy")
    near = dx is not None and dy is not None
    return SymbolContext(
        y_staff=float(raw["y_staff"]),
        x_measure=float(raw.get("x_measure", 0.5)),
        barline=float(raw.get("barline", NOTE_RANGE)),
        note_dx=None if dx is None else float(dx),
        note_dy=None if dy is None else float(dy),
        width=float(raw["width"]),
        height=float(raw["height"]),
        note_w=float(raw.get("note_w") or 0.0) if near else None,
        note_h=float(raw.get("note_h") or 0.0) if near else None,
    )


__all__ = [
    "CONTEXT_DIM",
    "CONTEXT_FEATURES",
    "NOTE_MARGIN",
    "NOTE_RANGE",
    "STAFF_HALF_HEIGHT",
    "SymbolContext",
    "context_from_mapping",
    "context_from_strokes",
    "context_vector",
]
