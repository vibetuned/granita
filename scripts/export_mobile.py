"""Export a trained Granita checkpoint to Core ML (.mlpackage) and LiteRT (.tflite).

Exports the checkpoint to Apple Core ML and Google LiteRT (formerly TFLite) alongside
the ``.json`` editor sidecar, verifies LiteRT numerical parity against PyTorch, and
prints an artifact summary table.

    uv run python scripts/export_mobile.py \\
        --checkpoint checkpoints/granitanet_v5_ctx/best_model.pt \\
        --output-dir checkpoints/granitanet_v5_ctx/export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from granita.config import DEFAULT_CLASSES
from granita.models.backbone import create_model
from granita.models.export import (
    DEFAULT_INPUT_SHAPE,
    export_coreml,
    export_litert,
    export_onnx,
    export_pt2,
    write_sidecar,
)


def _artifact_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def _verify_litert(
    tflite_path: Path,
    model: torch.nn.Module,
    x: torch.Tensor,
    ctx: torch.Tensor | None,
) -> float:
    from ai_edge_litert import interpreter as litert_interp  # type: ignore[import-untyped]

    interp = litert_interp.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    for detail in interp.get_input_details():
        shape = tuple(detail["shape"])
        if len(shape) == 4:
            interp.set_tensor(detail["index"], x.numpy())
        elif ctx is not None and len(shape) == 2:
            interp.set_tensor(detail["index"], ctx.numpy())
    interp.invoke()
    got = interp.get_tensor(interp.get_output_details()[0]["index"])
    with torch.no_grad():
        expected = (model(x, ctx) if ctx is not None else model(x)).numpy()
    return float(np.max(np.abs(got - expected)))


def _litert_names(tflite_path: Path, context_dim: int) -> dict[str, object]:
    """The names an app feeds the tflite by: litert-torch's (``args_0``, ``args_1``,
    ``output_0`` in ``serving_default``), not the sidecar's ``input`` and ``context``."""
    from ai_edge_litert import interpreter as litert_interp  # type: ignore[import-untyped]

    interp = litert_interp.Interpreter(model_path=str(tflite_path))
    ((signature, io),) = interp.get_signature_list().items()
    names = list(io["inputs"])
    inputs = {"input": names[0]}
    if context_dim:
        inputs["context"] = names[1]
    return {"signature": signature, "inputs": inputs, "output": io["outputs"][0]}


def _verify_onnx(
    onnx_path: Path,
    model: torch.nn.Module,
    x: torch.Tensor,
    ctx: torch.Tensor | None,
) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {"input": x.numpy()}
    if ctx is not None:
        feeds["context"] = ctx.numpy()
    (got,) = session.run(None, feeds)
    with torch.no_grad():
        expected = (model(x, ctx) if ctx is not None else model(x)).numpy()
    return float(np.max(np.abs(got - expected)))


def _verify_pt2(
    pt2_path: Path,
    model: torch.nn.Module,
    x: torch.Tensor,
    ctx: torch.Tensor | None,
) -> float:
    prog = torch.export.load(str(pt2_path)).module()
    with torch.no_grad():
        got = (prog(x, ctx) if ctx is not None else prog(x)).numpy()
        expected = (model(x, ctx) if ctx is not None else model(x)).numpy()
    return float(np.max(np.abs(got - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/granitanet_v5_ctx/best_model.pt",
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported files (defaults to checkpoint directory)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename stem (defaults to the checkpoint's model_name)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["coreml", "litert"],
        choices=["coreml", "litert", "onnx", "pt2"],
        help="Target formats to export",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Fix batch size to 1 for Core ML / ONNX / pt2 instead of dynamic RangeDim",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = list(ckpt.get("classes", DEFAULT_CLASSES))
    model_name = str(ckpt.get("model_name", "granitanet"))
    context_dim = int(ckpt.get("context_dim", 0))

    model = create_model(
        model_name=model_name, num_classes=len(classes), context_dim=context_dim
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or model_name
    dynamic = not args.static_batch

    torch.manual_seed(0)
    x = torch.randn(DEFAULT_INPUT_SHAPE)
    ctx = torch.randn(1, context_dim) if context_dim else None

    rows: list[list[str]] = []
    last_out: Path | None = None
    litert_names: dict[str, object] | None = None

    for fmt in args.formats:
        if fmt == "coreml":
            out = export_coreml(model, out_dir / f"{stem}.mlpackage", dynamic_batch=dynamic)
            parity = "n/a (macOS runtime)"
        elif fmt == "litert":
            out = export_litert(model, out_dir / f"{stem}.tflite")
            diff = _verify_litert(out, model, x, ctx)
            parity = f"{diff:.2e}"
            litert_names = _litert_names(out, context_dim)
        elif fmt == "onnx":
            out = export_onnx(model, out_dir / f"{stem}.onnx", dynamic_batch=dynamic)
            diff = _verify_onnx(out, model, x, ctx)
            parity = f"{diff:.2e}"
        elif fmt == "pt2":
            out = export_pt2(model, out_dir / f"{stem}.pt2", dynamic_batch=dynamic)
            diff = _verify_pt2(out, model, x, ctx)
            parity = f"{diff:.2e}"
        else:
            raise ValueError(f"Unknown format: {fmt}")

        last_out = out
        inputs_desc = "input (1,1,32,32)" + (f" + context (1,{context_dim})" if context_dim else "")
        rows.append(
            [
                fmt,
                str(out),
                _format_size(_artifact_bytes(out)),
                inputs_desc,
                f"logits (1,{len(classes)})",
                parity,
            ]
        )

    if last_out is not None:
        sidecar = write_sidecar(
            out_dir / f"{stem}.model",
            classes,
            model_name,
            context_dim=context_dim,
            extra={
                "checkpoint": str(ckpt_path),
                "formats": list(args.formats),
                **({"litert": litert_names} if litert_names else {}),
            },
        )
        # Rename <stem>.model.json -> <stem>.json for a clean shared sidecar
        clean_sidecar = out_dir / f"{stem}.json"
        sidecar.replace(clean_sidecar)
        print(f"[*] Wrote sidecar metadata to {clean_sidecar}\n")

    print(
        tabulate(
            rows,
            headers=["Format", "Artifact", "Size", "Inputs", "Output", "Max |Δ| vs PyTorch"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
