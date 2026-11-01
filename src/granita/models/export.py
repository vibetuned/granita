"""Export utilities for deploying Granita models: ONNX, ``torch.export`` and TorchScript."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn

DEFAULT_INPUT_SHAPE: tuple[int, int, int, int] = (1, 1, 32, 32)
MAX_BATCH = 4096


def _example(input_shape: tuple[int, int, int, int], dynamic_batch: bool) -> torch.Tensor:
    # torch.export specialises a size-1 batch, so trace with 2 when the batch is dynamic
    batch = max(2, input_shape[0]) if dynamic_batch else input_shape[0]
    return torch.zeros((batch, *input_shape[1:]))


def _dynamic_shapes(dynamic_batch: bool) -> tuple[dict[int, torch.export.Dim]] | None:
    if not dynamic_batch:
        return None
    return ({0: torch.export.Dim("batch", min=1, max=MAX_BATCH)},)


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export to a single ONNX file through ``torch.export`` (input ``input``, output ``logits``).

    Needs the ``export`` extra (``onnx``, ``onnxscript``). The batch axis is
    dynamic unless ``dynamic_batch`` is false.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        (_example(input_shape, dynamic_batch),),
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=_dynamic_shapes(dynamic_batch),
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
    program = torch.export.export(
        model,
        (_example(input_shape, dynamic_batch),),
        dynamic_shapes=_dynamic_shapes(dynamic_batch),
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
    example = torch.zeros(input_shape, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        traced = torch.jit.trace(model, example)  # type: ignore[attr-defined]
        traced.save(str(output_path))
    return output_path
