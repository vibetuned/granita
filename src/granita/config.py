"""Configuration, class definitions, and metadata for granita."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Canonical 33 music glyph classes matching musicset isolated symbols
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
}

DEFAULT_INPUT_SIZE: Final[tuple[int, int]] = (32, 32)
DEFAULT_IN_CHANNELS: Final[int] = 1
PAPER: Final[int] = 255
"""Background grey level of every training image (black ink on white paper)."""

# Sample provenance codes written by ``musicset symbols`` as ``s_<split>`` in dataset.npz
SOURCE_NAMES: Final[tuple[str, ...]] = ("font", "stroke", "crop")
"""``font``: SMuFL glyph render, ``stroke``: stylus stroke template, ``crop``: real score crop."""


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
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    save_safetensors: bool = True
