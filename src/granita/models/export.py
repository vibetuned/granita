"""Export utilities for deploying Granita models: ONNX, Core ML, LiteRT, pt2 and TorchScript."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.nn as nn

DEFAULT_INPUT_SHAPE: tuple[int, int, int, int] = (1, 1, 32, 32)
MAX_BATCH = 4096
PERSONAL_FUNCTION = "personal"
"""The Core ML function / LiteRT signature that takes the last layer as inputs."""


class HeadInputs(torch.nn.Module):
    """A ``GranitaNet`` whose last layer comes in as inputs: ``forward(input, context,
    head_w, head_b)`` gives ``(logits, features)``, the features being the 128 values that
    layer reads. An app can then train that one layer on the device from its users'
    replies and still run the whole network — GELU, squeeze-excitation and all — on the
    accelerator (the user's design, 2026-11-02). ``head_w`` is ``(classes, 128)``,
    ``head_b`` ``(classes,)``; fed the trained layer's own weights (``head_weights``) it
    computes exactly what the model does."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net
        self.context_dim = int(getattr(net, "context_dim", 0))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, head_w: torch.Tensor, head_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net.features(x, context)
        return torch.nn.functional.linear(features, head_w, head_b), features


def head_weights(net: torch.nn.Module) -> tuple[Any, Any]:
    """The last layer's weight ``(classes, 128)`` and bias ``(classes,)``, float32 arrays."""
    import numpy as np

    last = net.head[-1]
    return (
        last.weight.detach().cpu().numpy().astype(np.float32),
        last.bias.detach().cpu().numpy().astype(np.float32),
    )


def write_sidecar(
    output_path: str | Path,
    classes: list[str],
    model_name: str,
    *,
    context_dim: int = 0,
    pen_width: float | None = None,
    extra: dict[str, object] | None = None,
    head: tuple[Any, Any] | None = None,
) -> Path:
    """Write ``<output>.json`` next to an exported model: everything an editor needs to
    feed it the way granita does (class list, input tensor, stroke rasterisation) — and,
    with ``head`` (``head_weights``), the last layer's weights for the ``personal``
    function, so an app can start its own copy from them."""
    from granita.config import CLASS_GROUPS, CONTEXT_FEATURES, DEFAULT_INPUT_SIZE, PAPER
    from granita.data.transforms import GLYPH_MEAN, GLYPH_STD
    from granita.infer.predictor import CANVAS_STAFF_SPACES, DEFAULT_PEN_WIDTH, FRACTION_RANGE

    if pen_width is None:
        pen_width = DEFAULT_PEN_WIDTH

    path = Path(output_path).with_suffix(Path(output_path).suffix + ".json")
    sidecar: dict[str, object] = {
        "model": model_name,
        "classes": list(classes),
        "context": {
            "input": "context",
            "features": list(CONTEXT_FEATURES),
            "units": "staff spaces, scaled by granita.context.context_vector",
            "note": "note_* features describe the nearest note's body: the bounding box of "
            "its notehead, stem and flag, or of every notehead plus the stem and flag of its "
            "chord, without accidentals, dots or articulations; note_dx / note_dy are the "
            "symbol centre minus that box's centre, note_above / note_below mean the centre "
            "is above its top / below its bottom within 0.5 spaces of its sides, note_left / "
            "note_right left / right of its sides within 0.5 spaces of its top and bottom",
            "unknown": "all zeros (valid = 0)",
        }
        if context_dim
        else None,
        "groups": {c: CLASS_GROUPS.get(c, "other") for c in classes},
        "input": {
            "name": "input",
            "shape": [1, 1, *DEFAULT_INPUT_SIZE],
            "dtype": "float32",
            "layout": "NCHW, one grey channel, black ink on white paper",
            "preprocess": f"x = (pixel / 255 - {GLYPH_MEAN[0]}) / {GLYPH_STD[0]}",
            "paper": PAPER,
            "resize": "pad to square with paper, then bilinear resize",
        },
        "output": {
            "name": "logits",
            "shape": [1, len(classes)],
            "softmax": "apply for confidences",
        },
        "strokes": {
            "canvas_px": max(96, DEFAULT_INPUT_SIZE[0] * 3),
            "size_px": DEFAULT_INPUT_SIZE[0],
            "pen_width": pen_width,
            "pen_width_note": "fraction of the canvas; uniform width, round caps, no blur; "
            "0.05 scored best on held-out writers, training covered 0.055-0.115",
            "fraction": "longer side of the strokes' bounding box as a fraction of the canvas: "
            f"min({FRACTION_RANGE[1]}, max({FRACTION_RANGE[0]}, "
            f"span / ({CANVAS_STAFF_SPACES} * staff_space)))",
            "staff_space": "pixels between two staff lines in the strokes' coordinates",
            "centre": "bounding-box centre at the canvas centre, aspect preserved",
            "downsample": "bilinear to size_px",
        },
    }
    if head is not None:
        w, b = head
        sidecar["head"] = {
            "note": "the last layer, Linear(128 -> classes), with its trained weights. The "
            f"`{PERSONAL_FUNCTION}` Core ML function / LiteRT signature takes a copy as inputs "
            "head_w (classes x 128) and head_b (classes) and gives back the features that "
            "layer reads, so an app can train the layer on the device from its users' "
            "replies; fed these weights it computes exactly what the default function does",
            "function": PERSONAL_FUNCTION,
            "inputs": {"weight": "head_w", "bias": "head_b"},
            "outputs": {"logits": "logits", "features": "features"},
            "shape": [int(w.shape[0]), int(w.shape[1])],
            "weight": [[round(float(v), 6) for v in row] for row in w],
            "bias": [round(float(v), 6) for v in b],
        }
    if extra:
        sidecar.update(extra)
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def _context_dim(model: nn.Module) -> int:
    return int(getattr(model, "context_dim", 0))


def _example(
    input_shape: tuple[int, int, int, int], dynamic_batch: bool, context_dim: int
) -> tuple[torch.Tensor, ...]:
    # torch.export specialises a size-1 batch, so trace with 2 when the batch is dynamic
    batch = max(2, input_shape[0]) if dynamic_batch else input_shape[0]
    x = torch.zeros((batch, *input_shape[1:]))
    return (x, torch.zeros((batch, context_dim))) if context_dim else (x,)


def _dynamic_shapes(dynamic_batch: bool, context_dim: int) -> tuple[dict[int, object], ...] | None:
    if not dynamic_batch:
        return None
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    return ({0: batch}, {0: batch}) if context_dim else ({0: batch},)


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export to a single ONNX file through ``torch.export`` (inputs ``input`` and, for a
    model with a conditioning vector, ``context``; output ``logits``).

    Needs the ``export`` extra (``onnx``, ``onnxscript``). The batch axis is
    dynamic unless ``dynamic_batch`` is false.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    context_dim = _context_dim(model)
    torch.onnx.export(
        model,
        _example(input_shape, dynamic_batch, context_dim),
        str(output_path),
        input_names=["input", "context"] if context_dim else ["input"],
        output_names=["logits"],
        dynamic_shapes=_dynamic_shapes(dynamic_batch, context_dim),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
        optimize=True,
    )
    return output_path


def export_pt2(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    dynamic_batch: bool = True,
) -> Path:
    """Save a ``torch.export`` program (``.pt2``), the successor of TorchScript.

    Load with ``torch.export.load(path).module()``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    context_dim = _context_dim(model)
    program = torch.export.export(
        model,
        _example(input_shape, dynamic_batch, context_dim),
        dynamic_shapes=_dynamic_shapes(dynamic_batch, context_dim),
    )
    torch.export.save(program, str(output_path))
    return output_path


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    device: str = "cpu",
) -> Path:
    """Trace and export to TorchScript (.pt).

    TorchScript is deprecated upstream since torch 2.14; prefer
    :func:`export_pt2` or :func:`export_onnx` for new integrations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()
    example = tuple(t.to(device) for t in _example(input_shape, False, _context_dim(model)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        traced = torch.jit.trace(model, example)  # type: ignore[attr-defined]
        traced.save(str(output_path))
    return output_path


def export_coreml(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    dynamic_batch: bool = True,
    personal: bool = False,
) -> Path:
    """Export to a Core ML ``.mlpackage`` through ``torch.export`` and ``coremltools``
    (inputs ``input`` and, for a model with a conditioning vector, ``context``; output ``logits``).

    Needs the ``export`` extra (``coremltools``). If ``output_path`` does not end with
    ``.mlpackage``, ``.mlpackage`` is used as its suffix. When ``dynamic_batch`` is true,
    a ``RangeDim(1..MAX_BATCH)`` batch axis is requested and falls back to static batch
    for models whose ops require static batch unfolding (e.g. ``lsnet``'s ``im2col``).

    With ``personal`` (a ``GranitaNet``), a multi-function package (iOS 18): the default
    function ``infer`` as above, and ``personal`` — ``input``, ``context``, ``head_w``,
    ``head_b`` in, ``logits`` and ``features`` out (``HeadInputs``) — for an app that
    trains the last layer itself; an app that knows only the default keeps working.
    """
    import coremltools as ct  # type: ignore[import-untyped]
    import numpy as np

    output_path = Path(output_path)
    if output_path.suffix != ".mlpackage":
        output_path = output_path.with_suffix(".mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    context_dim = _context_dim(model)
    target = ct.target.iOS18 if personal else ct.target.iOS16

    def _tensor(name: str, shape: tuple[Any, ...] | None = None) -> Any:
        # An output names itself, its shape the model's; an input states its shape.
        return (
            ct.TensorType(name=name, dtype=np.float32)
            if shape is None
            else ct.TensorType(name=name, shape=shape, dtype=np.float32)
        )

    def _convert(dyn: bool) -> Any:
        program = torch.export.export(
            model,
            _example(input_shape, dyn, context_dim),
            dynamic_shapes=_dynamic_shapes(dyn, context_dim),
        ).run_decompositions({})
        batch_dim: Any = (
            ct.RangeDim(lower_bound=1, upper_bound=MAX_BATCH, default=input_shape[0])
            if dyn
            else input_shape[0]
        )
        inputs = [_tensor("input", (batch_dim, *input_shape[1:]))]
        if context_dim:
            inputs.append(_tensor("context", (batch_dim, context_dim)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ct.convert(
                program,
                inputs=inputs,
                outputs=[_tensor("logits")],
                convert_to="mlprogram",
                minimum_deployment_target=target,
            )

    def _convert_personal() -> Any:
        wrapper = HeadInputs(model).eval()
        w, b = head_weights(model)
        example = (
            *_example(input_shape, False, context_dim),
            torch.from_numpy(w),
            torch.from_numpy(b),
        )
        program = torch.export.export(wrapper, example).run_decompositions({})
        inputs = [
            _tensor("input", (input_shape[0], *input_shape[1:])),
            _tensor("context", (input_shape[0], context_dim)),
            _tensor("head_w", tuple(int(n) for n in w.shape)),
            _tensor("head_b", tuple(int(n) for n in b.shape)),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ct.convert(
                program,
                inputs=inputs,
                outputs=[_tensor("logits"), _tensor("features")],
                convert_to="mlprogram",
                minimum_deployment_target=target,
            )

    try:
        mlmodel = _convert(dynamic_batch)
    except ValueError:
        if not dynamic_batch:
            raise
        mlmodel = _convert(False)
    if not personal:
        mlmodel.save(str(output_path))
        return output_path

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        infer_path = Path(tmp) / "infer.mlpackage"
        personal_path = Path(tmp) / f"{PERSONAL_FUNCTION}.mlpackage"
        mlmodel.save(str(infer_path))
        _convert_personal().save(str(personal_path))
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(str(infer_path), src_function_name="main", target_function_name="infer")
        desc.add_function(
            str(personal_path), src_function_name="main", target_function_name=PERSONAL_FUNCTION
        )
        desc.default_function_name = "infer"
        if output_path.exists():
            shutil.rmtree(output_path)
        ct.utils.save_multifunction(desc, str(output_path))
    return output_path


def export_litert(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    personal: bool = False,
) -> Path:
    """Export to a LiteRT (``.tflite``) flatbuffer through ``litert-torch``.

    Needs the ``export`` extra (``litert-torch``). LiteRT's JAX lowering bridge
    traces concrete static shapes (default ``(1, 1, 32, 32)``, plus ``(1, context_dim)``
    when the model takes a conditioning vector). With ``personal`` (a ``GranitaNet``) two
    signatures: ``serving_default`` as before and ``personal``, the last layer as inputs
    and the features out (``HeadInputs``), as the Core ML function of that name.
    """
    import litert_torch  # type: ignore[import-untyped]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    context_dim = _context_dim(model)
    example = _example(input_shape, False, context_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if personal:
            w, b = head_weights(model)
            wrapper = HeadInputs(model).eval()
            edge_model = (
                litert_torch.signature("serving_default", model, example)
                .signature(
                    PERSONAL_FUNCTION, wrapper, (*example, torch.from_numpy(w), torch.from_numpy(b))
                )
                .convert()
            )
        else:
            edge_model = litert_torch.convert(model, example)
    edge_model.export(str(output_path))
    return output_path
