"""Configuration, class definitions, and metadata for granita."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Canonical music glyph classes, in musicset's `symbols` catalog order (the first 33 are
# the original marks, the rest were added with the HOMUS pen-stroke data)
DEFAULT_CLASSES: Final[tuple[str, ...]] = (
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
    "timeSigCommon",
    "timeSigCutCommon",
    "timeSig2over2",
    "timeSig2over4",
    "timeSig3over4",
    "timeSig3over8",
    "timeSig4over4",
    "timeSig6over8",
    "timeSig9over8",
    "timeSig12over8",
    "barlineSingle",
)

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
    "timeSigCommon": "timesig",
    "timeSigCutCommon": "timesig",
    "timeSig2over2": "timesig",
    "timeSig2over4": "timesig",
    "timeSig3over4": "timesig",
    "timeSig3over8": "timesig",
    "timeSig4over4": "timesig",
    "timeSig6over8": "timesig",
    "timeSig9over8": "timesig",
    "timeSig12over8": "timesig",
    "barlineSingle": "barline",
}

DEFAULT_INPUT_SIZE: Final[tuple[int, int]] = (32, 32)
DEFAULT_IN_CHANNELS: Final[int] = 1
PAPER: Final[int] = 255
"""Background grey level of every training image (black ink on white paper)."""

# Conditioning vector: where the symbol sits in the score, mirrored from musicset.context
CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "valid",
    "y_staff",
    "x_measure",
    "barline",
    "has_note",
    "note_dx",
    "note_dy",
    "width",
    "height",
)
CONTEXT_DIM: Final[int] = len(CONTEXT_FEATURES)

# Sample provenance codes written by ``musicset symbols`` as ``s_<split>`` in dataset.npz
SOURCE_NAMES: Final[tuple[str, ...]] = ("font", "stroke", "crop", "hand")
"""``font``: SMuFL glyph render, ``stroke``: stylus stroke template, ``crop``: real score
crop, ``hand``: real pen strokes (HOMUS writers, never shared between splits)."""


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
