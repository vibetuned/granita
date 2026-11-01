"""Unit tests for Granita model architectures."""

import pytest
import torch

from granita.config import DEFAULT_CLASSES
from granita.models.backbone import GranitaNet, GranitaTiny, create_model


def test_granitanet_forward() -> None:
    model = GranitaNet(num_classes=33, in_channels=1)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 33)


def test_granitanet_custom_classes() -> None:
    model = GranitaNet(num_classes=10, in_channels=1)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_granitatiny_forward() -> None:
    model = GranitaTiny(num_classes=33, in_channels=1)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 33)


def test_create_model_factory() -> None:
    model = create_model("granitanet", num_classes=len(DEFAULT_CLASSES))
    assert isinstance(model, GranitaNet)

    tiny = create_model("tiny", num_classes=len(DEFAULT_CLASSES))
    assert isinstance(tiny, GranitaTiny)

    with pytest.raises(ValueError, match="Unknown model architecture"):
        create_model("invalid_model")
