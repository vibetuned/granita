"""Unit tests for exporting Granita models to TorchScript."""

from pathlib import Path

import pytest
import torch

from granita.models.backbone import create_model
from granita.models.export import export_torchscript


@pytest.mark.parametrize("model_name", ["tiny", "granitanet", "regla", "iformer", "lsnet"])
def test_export_torchscript_all_families(tmp_path: Path, model_name: str) -> None:
    model = create_model(model_name, num_classes=10, in_channels=1)
    out_file = tmp_path / f"{model_name}.pt"

    export_torchscript(model, out_file)
    assert out_file.is_file()

    # Load back traced model and test inference
    traced = torch.jit.load(str(out_file))  # type: ignore[attr-defined]
    x = torch.randn(1, 1, 32, 32)
    out = traced(x)
    assert out.shape == (1, 10)
