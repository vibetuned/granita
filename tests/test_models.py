"""Unit tests for Granita model architectures."""

import pytest
import torch

from granita.config import DEFAULT_CLASSES
from granita.models.backbone import (
    AVAILABLE_MODELS,
    GranitaNet,
    GranitaTiny,
    count_parameters,
    create_model,
)
from granita.models.iformer import GranitaIFormer
from granita.models.lsnet import GranitaLSNet
from granita.models.regla import GranitaReGLA


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


@pytest.mark.parametrize("variant", ["regla", "regla_t", "regla_s"])
def test_regla_forward_and_backward(variant: str) -> None:
    model = GranitaReGLA(variant=variant, num_classes=33, in_channels=1)
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 33)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("variant", ["iformer", "iformer_t", "iformer_s"])
def test_iformer_forward_and_backward(variant: str) -> None:
    model = GranitaIFormer(variant=variant, num_classes=33, in_channels=1)
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 33)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("variant", ["lsnet", "lsnet_t", "lsnet_s"])
def test_lsnet_forward_and_backward(variant: str) -> None:
    model = GranitaLSNet(variant=variant, num_classes=33, in_channels=1)
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 33)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
def test_create_all_available_models(model_name: str) -> None:
    model = create_model(model_name, num_classes=len(DEFAULT_CLASSES))
    assert count_parameters(model) > 0
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, len(DEFAULT_CLASSES))


def test_create_model_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown model architecture"):
        create_model("invalid_model")
