"""Granita: Lightweight Music Glyph and Symbol Classifier for OMR."""

from granita.config import (
    CLASS_GROUPS,
    CLASS_TO_IDX,
    DEFAULT_CLASSES,
    IDX_TO_CLASS,
    TrainConfig,
)
from granita.infer.predictor import GranitaClassifier, Prediction
from granita.models.backbone import GranitaNet, GranitaTiny, create_model

__version__ = "0.1.0"

__all__ = [
    "CLASS_GROUPS",
    "CLASS_TO_IDX",
    "DEFAULT_CLASSES",
    "IDX_TO_CLASS",
    "GranitaClassifier",
    "GranitaNet",
    "GranitaTiny",
    "Prediction",
    "TrainConfig",
    "create_model",
]
