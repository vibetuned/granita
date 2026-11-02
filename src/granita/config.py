"""Configuration, class definitions, and metadata for granita."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Canonical music glyph classes, in musicset's ``symbols`` catalog order. Two variants,
# one per consumer (musicset ``symbols --profile``):
#   kakigori: every class, engraved only (fonts and score crops), for the OMR pipeline;
#   battuta: the stylus set of the Battuta editor, trained with real handwriting.
KAKIGORI_CLASSES: Final[tuple[str, ...]] = (
    "fing1",
    "fing2",
    "fing3",
    "fing4",
    "fing5",
    "trill",
    "mordent",
    "turn",
    "fermata",
    "arpeg",
    "articStaccatoAbove",
    "articTenutoAbove",
    "articAccentAbove",
    "articStaccatissimoAbove",
    "articMarcatoAbove",
    "accidentalFlat",
    "accidentalNatural",
    "accidentalSharp",
    "accidentalDoubleSharp",
    "accidentalDoubleFlat",
    "dynamicPiano",
    "dynamicMezzo",
    "dynamicForte",
    "dynamicSforzando",
    "dynamicZ",
    "dynamicPP",
    "dynamicMP",
    "dynamicMF",
    "dynamicFF",
    "dynamicFFF",
    "dynamicForzando",
    "dynamicSforzato",
    "pedal",
    "noteWhole",
    "noteHalf",
    "noteQuarter",
    "note8th",
    "note16th",
    "note32nd",
    "note64th",
    "augmentationDot",
    "restWholeHalf",
    "restQuarter",
    "rest8th",
    "rest16th",
    "rest32nd",
    "rest64th",
    "gClef",
    "fClef",
    "cClef",
    "barlineSingle",
    "dalSegno",
    "daCapo",
    "fine",
    "segno",
    "coda",
)

BATTUTA_CLASSES: Final[tuple[str, ...]] = (
    "fing1",
    "fing2",
    "fing3",
    "fing4",
    "fing5",
    "trill",
    "mordent",
    "turn",
    "fermata",
    "arpeg",
    "articStaccatoAbove",
    "articTenutoAbove",
    "articAccentAbove",
    "articStaccatissimoAbove",
    "articMarcatoAbove",
    "accidentalFlat",
    "accidentalNatural",
    "accidentalSharp",
    "dynamicPiano",
    "dynamicForte",
    "dynamicSforzando",
    "dynamicPP",
    "dynamicMP",
    "dynamicMF",
    "dynamicFF",
    "dynamicFFF",
    "dynamicForzando",
    "dynamicSforzato",
    "noteWhole",
    "noteHalf",
    "noteQuarter",
    "note8th",
    "note16th",
    "augmentationDot",
    "restQuarter",
    "gClef",
    "fClef",
    "cClef",
    "dalSegno",
    "daCapo",
    "fine",
    "segno",
    "coda",
)
"""The editor's stylus classes: the catalog without the double accidentals (rare),
``dynamicMezzo`` and ``dynamicZ`` (only parts of mp / mf / sfz), ``pedal``, ``note32nd`` and
``note64th``, every rest but the quarter (the editor infers a rest's duration from the
measure) and ``barlineSingle`` (the editor draws barlines itself)."""

VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "battuta": BATTUTA_CLASSES,
    "kakigori": KAKIGORI_CLASSES,
}

DEFAULT_CLASSES: Final[tuple[str, ...]] = KAKIGORI_CLASSES
"""Fallback when a dataset or checkpoint carries no class list: the full catalog."""

CLASS_TO_IDX: Final[dict[str, int]] = {name: i for i, name in enumerate(DEFAULT_CLASSES)}
IDX_TO_CLASS: Final[dict[int, str]] = {i: name for i, name in enumerate(DEFAULT_CLASSES)}

CLASS_GROUPS: Final[dict[str, str]] = {
    "fing1": "fingering",
    "fing2": "fingering",
    "fing3": "fingering",
    "fing4": "fingering",
    "fing5": "fingering",
    "trill": "ornament",
    "mordent": "ornament",
    "turn": "ornament",
    "fermata": "ornament",
    "arpeg": "ornament",
    "articStaccatoAbove": "articulation",
    "articTenutoAbove": "articulation",
    "articAccentAbove": "articulation",
    "articStaccatissimoAbove": "articulation",
    "articMarcatoAbove": "articulation",
    "accidentalFlat": "accidental",
    "accidentalNatural": "accidental",
    "accidentalSharp": "accidental",
    "accidentalDoubleSharp": "accidental",
    "accidentalDoubleFlat": "accidental",
    "dynamicPiano": "dynamic",
    "dynamicMezzo": "dynamic",
    "dynamicForte": "dynamic",
    "dynamicSforzando": "dynamic",
    "dynamicZ": "dynamic",
    "dynamicPP": "dynamic",
    "dynamicMP": "dynamic",
    "dynamicMF": "dynamic",
    "dynamicFF": "dynamic",
    "dynamicFFF": "dynamic",
    "dynamicForzando": "dynamic",
    "dynamicSforzato": "dynamic",
    "pedal": "pedal",
    "noteWhole": "note",
    "noteHalf": "note",
    "noteQuarter": "note",
    "note8th": "note",
    "note16th": "note",
    "note32nd": "note",
    "note64th": "note",
    "augmentationDot": "note",
    "restWholeHalf": "rest",
    "restQuarter": "rest",
    "rest8th": "rest",
    "rest16th": "rest",
    "rest32nd": "rest",
    "rest64th": "rest",
    "gClef": "clef",
    "fClef": "clef",
    "cClef": "clef",
    "barlineSingle": "barline",
    "dalSegno": "navigation",
    "daCapo": "navigation",
    "fine": "navigation",
    "segno": "navigation",
    "coda": "navigation",
}

DEFAULT_INPUT_SIZE: Final[tuple[int, int]] = (32, 32)
DEFAULT_IN_CHANNELS: Final[int] = 1
PAPER: Final[int] = 255
"""Background grey level of every training image (black ink on white paper)."""

# Conditioning vector: where the symbol sits in the score, mirrored from musicset.context
CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "valid",
    "in_staff",
    "above_staff",
    "below_staff",
    "y_staff",
    "x_measure",
    "barline",
    "has_note",
    "note_above",
    "note_below",
    "note_left",
    "note_right",
    "note_dx",
    "note_dy",
    "width",
    "height",
)
CONTEXT_DIM: Final[int] = len(CONTEXT_FEATURES)

# Sample provenance codes written by ``musicset symbols`` as ``s_<split>`` in dataset.npz
SOURCE_NAMES: Final[tuple[str, ...]] = ("font", "stroke", "crop", "hand", "emnist", "fornes")
"""``font``: SMuFL glyph render, ``stroke``: stylus stroke template, ``crop``: real score
crop, ``hand``: real pen strokes (HOMUS writers, never shared between splits, and codas
composed from each writer's own strokes), ``emnist``: real handwritten digits and letters
(EMNIST) composed into fingerings, dynamics, D.S., D.C., Fine, segno, coda and turn,
``fornes``: scanned handwritten clefs and accidentals (Fornés writers, never shared
between splits)."""


@dataclass
class TrainConfig:
    """Hyperparameters and file paths for training granita models."""

    data_path: Path = field(default_factory=lambda: Path("data/symbols/dataset.npz"))
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    model_name: str = "granitanet"
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    context_dim: int = 0
    """0 trains without the conditioning vector; CONTEXT_DIM uses it."""
    context_dropout: float = 0.25
    """Share of training samples whose context is zeroed so the model also works without one."""
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    save_safetensors: bool = True
