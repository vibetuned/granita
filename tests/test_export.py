"""Unit tests for exporting Granita models to ONNX, torch.export and TorchScript."""

from pathlib import Path

import numpy as np
import pytest
import torch

from granita.models.backbone import create_model
from granita.models.export import export_onnx, export_pt2, export_torchscript

FAMILIES = ["tiny", "granitanet", "regla", "iformer", "lsnet"]


@pytest.mark.parametrize("model_name", FAMILIES)
def test_export_torchscript_all_families(tmp_path: Path, model_name: str) -> None:
    model = create_model(model_name, num_classes=10, in_channels=1)
    out_file = tmp_path / f"{model_name}.pt"

    export_torchscript(model, out_file)
    assert out_file.is_file()

    traced = torch.jit.load(str(out_file))  # type: ignore[attr-defined]
    x = torch.randn(1, 1, 32, 32)
    out = traced(x)
    assert out.shape == (1, 10)


@pytest.mark.parametrize("model_name", FAMILIES)
def test_export_pt2_dynamic_batch(tmp_path: Path, model_name: str) -> None:
    model = create_model(model_name, num_classes=7, in_channels=1).eval()
    out_file = export_pt2(model, tmp_path / f"{model_name}.pt2")
    program = torch.export.load(str(out_file)).module()
    x = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        expected = model(x)
        got = program(x)
    assert got.shape == (3, 7)
    assert torch.allclose(got, expected, atol=1e-5)
    assert program(torch.randn(1, 1, 32, 32)).shape == (1, 7)


@pytest.mark.parametrize("model_name", FAMILIES)
def test_export_onnx_matches_torch(tmp_path: Path, model_name: str) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = create_model(model_name, num_classes=5, in_channels=1).eval()
    out_file = export_onnx(model, tmp_path / f"{model_name}.onnx")
    assert out_file.is_file()

    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["input"]
    assert [o.name for o in session.get_outputs()] == ["logits"]
    for batch in (1, 4):
        x = torch.randn(batch, 1, 32, 32)
        with torch.no_grad():
            expected = model(x).numpy()
        (got,) = session.run(None, {"input": x.numpy()})
        assert got.shape == (batch, 5)
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-3)
