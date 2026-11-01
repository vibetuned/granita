"""Unit tests for exporting Granita models to TorchScript."""

from pathlib import Path

import torch

from granita.models.backbone import GranitaNet
from granita.models.export import export_torchscript


def test_export_torchscript(tmp_path: Path) -> None:
    model = GranitaNet(num_classes=10, in_channels=1)
    out_file = tmp_path / "model.pt"

    export_torchscript(model, out_file)
    assert out_file.is_file()

    # Load back traced model and test inference
    traced = torch.jit.load(str(out_file))  # type: ignore[attr-defined]
    x = torch.randn(1, 1, 32, 32)
    out = traced(x)
    assert out.shape == (1, 10)
