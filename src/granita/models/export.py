"""Export utilities for deploying Granita models to TorchScript and ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 1, 32, 32),
    device: str = "cpu",
) -> Path:
    """Trace and export a Granita model to TorchScript (.pt format)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()
    example = torch.zeros(input_shape, device=device)
    traced = torch.jit.trace(model, example)  # type: ignore[attr-defined]
    traced.save(str(output_path))
    return output_path


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 1, 32, 32),
    opset_version: int = 17,
) -> Path:
    """Export a Granita model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    dummy_input = torch.randn(*input_shape, device="cpu")

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return output_path
