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
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabulate import tabulate

from granita.config import DEFAULT_CLASSES
from granita.models.backbone import create_model
from granita.models.export import (
    DEFAULT_INPUT_SHAPE,
    PERSONAL_FUNCTION,
    export_coreml,
    export_litert,
    export_onnx,
    export_pt2,
    head_weights,
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


def _by_shape(table: dict[str, tuple[int, ...]], want: Any) -> str:
    """The one name in ``table`` whose shape ``want`` accepts."""
    (name,) = [n for n, s in table.items() if want(s)]
    return name


def _verify_litert(
    tflite_path: Path,
    model: torch.nn.Module,
    x: torch.Tensor,
    ctx: torch.Tensor | None,
) -> float:
    """Max |Δ| against PyTorch over every signature: the logits of each, the features of
    ``personal`` — its inputs matched by shape, the last layer's weights fed as they are."""
    from ai_edge_litert import interpreter as litert_interp  # type: ignore[import-untyped]

    interp = litert_interp.Interpreter(model_path=str(tflite_path))
    arrays: list[np.ndarray] = [x.numpy()]
    if ctx is not None:
        arrays.append(ctx.numpy())
    if hasattr(model, "features"):
        arrays.extend(head_weights(model))
    with torch.no_grad():
        logits = (model(x, ctx) if ctx is not None else model(x)).numpy()
        feats = model.features(x, ctx).numpy() if hasattr(model, "features") else None
    worst = 0.0
    for signature in interp.get_signature_list():
        runner = interp.get_signature_runner(signature)
        feeds = {}
        for name, detail in runner.get_input_details().items():
            feeds[name] = next(a for a in arrays if tuple(a.shape) == tuple(detail["shape"]))
        for got in runner(**feeds).values():
            expected = logits if got.shape[-1] == logits.shape[-1] else feats
            if expected is not None:
                worst = max(worst, float(np.max(np.abs(got - expected))))
    return worst


def _litert_names(tflite_path: Path, context_dim: int, classes: int) -> dict[str, object]:
    """The names an app feeds the tflite by: litert-torch's (``args_0``, ``args_1``,
    ``output_0``…), not the sidecar's — per signature, each tensor told by its shape; the
    keys of the single-signature days (``signature``, ``inputs``, ``output``) kept for
    the readers that have them."""
    from ai_edge_litert import interpreter as litert_interp  # type: ignore[import-untyped]

    interp = litert_interp.Interpreter(model_path=str(tflite_path))
    signatures: dict[str, dict[str, Any]] = {}
    for signature in interp.get_signature_list():
        runner = interp.get_signature_runner(signature)
        ins = {n: tuple(int(v) for v in d["shape"]) for n, d in runner.get_input_details().items()}
        outs = {
            n: tuple(int(v) for v in d["shape"]) for n, d in runner.get_output_details().items()
        }
        inputs = {"input": _by_shape(ins, lambda s: len(s) == 4)}
        if context_dim:
            inputs["context"] = _by_shape(ins, lambda s: len(s) == 2 and s[-1] == context_dim)
        if any(len(s) == 2 and s[0] == classes for s in ins.values()):
            inputs["head_w"] = _by_shape(ins, lambda s: len(s) == 2 and s[0] == classes)
            inputs["head_b"] = _by_shape(ins, lambda s: len(s) == 1)
        outputs = {"logits": _by_shape(outs, lambda s: s[-1] == classes)}
        if any(s[-1] != classes for s in outs.values()):
            outputs["features"] = _by_shape(outs, lambda s: s[-1] != classes)
        signatures[signature] = {"inputs": inputs, "outputs": outputs}
    main = "serving_default" if "serving_default" in signatures else next(iter(signatures))
    return {
        "signature": main,
        "inputs": signatures[main]["inputs"],
        "output": signatures[main]["outputs"]["logits"],
        "signatures": signatures,
    }


def _verify_coreml(
    mlpackage: Path,
    model: torch.nn.Module,
    x: torch.Tensor,
    ctx: torch.Tensor | None,
    personal: bool,
) -> str:
    """Max |Δ| against PyTorch where Core ML can run (macOS): the default function's logits
    and, with ``personal``, that function's logits and features; elsewhere a note."""
    import platform

    if platform.system() != "Darwin":
        return "n/a (macOS runtime)"
    import coremltools as ct  # type: ignore[import-untyped]

    feeds: dict[str, np.ndarray] = {"input": x.numpy()}
    if ctx is not None:
        feeds["context"] = ctx.numpy()
    with torch.no_grad():
        logits = (model(x, ctx) if ctx is not None else model(x)).numpy()
    try:
        got = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY).predict(
            feeds
        )
        worst = float(np.max(np.abs(got["logits"] - logits)))
        if personal:
            w, b = head_weights(model)
            with torch.no_grad():
                feats = model.features(x, ctx).numpy()
            p = ct.models.MLModel(
                str(mlpackage),
                compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name=PERSONAL_FUNCTION,
            ).predict({**feeds, "head_w": w, "head_b": b})
            worst = max(
                worst,
                float(np.max(np.abs(p["logits"] - logits))),
                float(np.max(np.abs(p["features"] - feats))),
            )
        return f"{worst:.2e}"
    except Exception as e:  # a report, not a gate
        return f"not run ({type(e).__name__})"


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
    # granitanet with a context: the `personal` function / signature too (the last layer as inputs).
    personal = hasattr(model, "features") and context_dim > 0

    rows: list[list[str]] = []
    last_out: Path | None = None
    # A sidecar already there keeps its litert names when litert is not exported this time
    # (the tflite is made on Linux, the mlpackage wherever coremltools runs).
    previous = out_dir / f"{stem}.json"
    litert_names: dict[str, object] | None = (
        json.loads(previous.read_text(encoding="utf-8")).get("litert")
        if previous.is_file()
        else None
    )

    for fmt in args.formats:
        if fmt == "coreml":
            out = export_coreml(
                model, out_dir / f"{stem}.mlpackage", dynamic_batch=dynamic, personal=personal
            )
            parity = _verify_coreml(out, model, x, ctx, personal)
        elif fmt == "litert":
            out = export_litert(model, out_dir / f"{stem}.tflite", personal=personal)
            diff = _verify_litert(out, model, x, ctx)
            parity = f"{diff:.2e}"
            litert_names = _litert_names(out, context_dim, len(classes))
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
        if personal and fmt in ("coreml", "litert"):
            inputs_desc += (
                f"; {PERSONAL_FUNCTION}: + head_w ({len(classes)},128) + head_b ({len(classes)})"
                " -> logits, features"
            )
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
            head=head_weights(model) if personal else None,
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
