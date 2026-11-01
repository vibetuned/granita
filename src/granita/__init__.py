"""Granita: Lightweight Music Glyph and Symbol Classifier for OMR."""

from granita.config import (
    CLASS_GROUPS,
    CLASS_TO_IDX,
    DEFAULT_CLASSES,
    IDX_TO_CLASS,
    TrainConfig,
)
from granita.infer.predictor import GranitaClassifier, Prediction
from granita.models.backbone import (
    AVAILABLE_MODELS,
    MODEL_DESCRIPTIONS,
    GranitaNet,
    GranitaTiny,
    count_parameters,
    create_model,
)
from granita.models.iformer import GranitaIFormer
from granita.models.lsnet import GranitaLSNet
from granita.models.regla import GranitaReGLA
from granita.train.ablation import AblationResult, run_ablation

__version__ = "0.1.0"

__all__ = [
    "AVAILABLE_MODELS",
    "CLASS_GROUPS",
    "CLASS_TO_IDX",
    "DEFAULT_CLASSES",
    "IDX_TO_CLASS",
    "MODEL_DESCRIPTIONS",
    "AblationResult",
    "GranitaClassifier",
    "GranitaIFormer",
    "GranitaLSNet",
    "GranitaNet",
    "GranitaReGLA",
    "GranitaTiny",
    "Prediction",
    "TrainConfig",
    "count_parameters",
    "create_model",
    "run_ablation",
]
