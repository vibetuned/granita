"""Unit tests for exporting Granita models to ONNX, Core ML, LiteRT, pt2 and TorchScript."""

from pathlib import Path

import numpy as np
import pytest
import torch

from granita.models.backbone import create_model
from granita.models.export import (
    export_coreml,
    export_litert,
    export_onnx,
    export_pt2,
    export_torchscript,
)

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


def test_export_with_context_input(tmp_path: Path) -> None:
    from granita.models.backbone import GranitaNet
    from granita.models.export import write_sidecar

    model = GranitaNet(num_classes=4, context_dim=9).eval()
    program = torch.export.load(str(export_pt2(model, tmp_path / "ctx.pt2"))).module()
    x, ctx = torch.randn(3, 1, 32, 32), torch.randn(3, 9)
    with torch.no_grad():
        assert torch.allclose(program(x, ctx), model(x, ctx), atol=1e-5)
    ort = pytest.importorskip("onnxruntime")
    out_file = export_onnx(model, tmp_path / "ctx.onnx")
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["input", "context"]
    (got,) = session.run(None, {"input": x.numpy(), "context": ctx.numpy()})
    with torch.no_grad():
        np.testing.assert_allclose(got, model(x, ctx).numpy(), atol=1e-4, rtol=1e-3)
    sidecar = write_sidecar(out_file, ["a", "b", "c", "d"], "granitanet", context_dim=9)
    import json

    meta = json.loads(sidecar.read_text())
    assert meta["context"]["features"][0] == "valid" and meta["classes"] == ["a", "b", "c", "d"]
    assert (
        write_sidecar(out_file, ["a"], "tiny")
        and json.loads(sidecar.read_text())["context"] is None
    )


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


@pytest.mark.parametrize("model_name", FAMILIES)
def test_export_coreml_all_families(tmp_path: Path, model_name: str) -> None:
    ct = pytest.importorskip("coremltools")
    model = create_model(model_name, num_classes=5, in_channels=1).eval()
    out_dir = export_coreml(model, tmp_path / f"{model_name}.mlpackage")
    assert out_dir.is_dir() and out_dir.suffix == ".mlpackage"

    loaded = ct.models.MLModel(str(out_dir))
    spec = loaded.get_spec()
    assert [i.name for i in spec.description.input] == ["input"]
    assert [o.name for o in spec.description.output] == ["logits"]


@pytest.mark.parametrize("model_name", FAMILIES)
def test_export_litert_matches_torch(tmp_path: Path, model_name: str) -> None:
    pytest.importorskip("litert_torch")
    litert_interp = pytest.importorskip("ai_edge_litert.interpreter")
    model = create_model(model_name, num_classes=5, in_channels=1).eval()
    out_file = export_litert(model, tmp_path / f"{model_name}.tflite")
    assert out_file.is_file()

    interp = litert_interp.Interpreter(model_path=str(out_file))
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    assert len(in_details) == 1

    x = torch.randn(1, 1, 32, 32)
    interp.set_tensor(in_details[0]["index"], x.numpy())
    interp.invoke()
    got = interp.get_tensor(out_details[0]["index"])
    with torch.no_grad():
        expected = model(x).numpy()
    assert got.shape == (1, 5)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-3)


def test_export_mobile_with_context_input(tmp_path: Path) -> None:
    ct = pytest.importorskip("coremltools")
    pytest.importorskip("litert_torch")
    litert_interp = pytest.importorskip("ai_edge_litert.interpreter")
    from granita.models.backbone import GranitaNet

    model = GranitaNet(num_classes=4, context_dim=9).eval()

    # Core ML with context
    coreml_dir = export_coreml(model, tmp_path / "ctx")
    assert coreml_dir.suffix == ".mlpackage" and coreml_dir.is_dir()
    spec = ct.models.MLModel(str(coreml_dir)).get_spec()
    assert [i.name for i in spec.description.input] == ["input", "context"]
    assert [o.name for o in spec.description.output] == ["logits"]

    # LiteRT with context
    tflite_file = export_litert(model, tmp_path / "ctx.tflite")
    interp = litert_interp.Interpreter(model_path=str(tflite_file))
    interp.allocate_tensors()
    x, ctx = torch.randn(1, 1, 32, 32), torch.randn(1, 9)
    for detail in interp.get_input_details():
        shape = tuple(detail["shape"])
        if len(shape) == 4:
            interp.set_tensor(detail["index"], x.numpy())
        elif len(shape) == 2:
            interp.set_tensor(detail["index"], ctx.numpy())
    interp.invoke()
    got = interp.get_tensor(interp.get_output_details()[0]["index"])
    with torch.no_grad():
        np.testing.assert_allclose(got, model(x, ctx).numpy(), atol=1e-4, rtol=1e-3)
