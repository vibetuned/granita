"""What the conditioning vector buys: a context model scored with and without it.

Scores one checkpoint on a split twice, with the archive's context vectors and
with every vector zeroed, and reports top-1 per source, macro F1, and the
confusion counts of the shape-identical pairs the context is meant to settle.

    uv run python scripts/context_effect.py \\
        --checkpoint checkpoints/granitanet_v5_ctx/best_model.pt \\
        --data ../musicset/data/symbols/dataset.npz --out results/context_effect.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tabulate import tabulate

from granita.config import SOURCE_NAMES
from granita.data.dataset import load_split_dataset, make_loader
from granita.eval.evaluate import EvalMetrics, evaluate_model
from granita.models.backbone import create_model
from granita.train.trainer import resolve_device

PAIRS = [
    ("augmentationDot", "articStaccatoAbove"),
    ("fing1", "barlineSingle"),
    ("articStaccatissimoAbove", "barlineSingle"),
    ("noteWhole", "augmentationDot"),
    ("restQuarter", "accidentalNatural"),
    ("noteQuarter", "noteHalf"),
    ("fing1", "articStaccatissimoAbove"),
    ("note8th", "note16th"),
]


def pair_errors(m: EvalMetrics, classes: list[str], a: str, b: str) -> int:
    i, j = classes.index(a), classes.index(b)
    return int(m.confusion_matrix[i, j] + m.confusion_matrix[j, i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--baseline", default=None, help="a context-free checkpoint to list alongside"
    )
    parser.add_argument("--out", default=None, help="markdown report path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dataset = load_split_dataset(args.data, args.split)
    if dataset is None:
        raise SystemExit(f"{args.data} has no {args.split!r} split")
    loader = make_loader(dataset, 512, shuffle=False, num_workers=2)
    classes = dataset.classes
    valid = None if dataset.contexts is None else dataset.contexts[:, 0] > 0

    def load(path: str) -> tuple[torch.nn.Module, int]:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        dim = int(ckpt.get("context_dim", 0))
        model = create_model(
            ckpt.get("model_name", "granitanet"), num_classes=len(classes), context_dim=dim
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device), dim

    model, dim = load(args.checkpoint)
    columns: dict[str, EvalMetrics] = {
        "with context": evaluate_model(model, loader, classes, device),
        "context masked": evaluate_model(model, loader, classes, device, mask_context=True),
    }
    if args.baseline:
        base, _ = load(args.baseline)
        columns["context-free model"] = evaluate_model(
            base, loader, classes, device, mask_context=True
        )

    names = list(columns)
    rows: list[list[object]] = [
        ["top-1", *[f"{m.top1_accuracy * 100:.2f}%" for m in columns.values()]]
    ]
    rows.append(["macro F1", *[f"{m.macro_f1 * 100:.2f}%" for m in columns.values()]])
    for src in SOURCE_NAMES:
        if all(src in m.source_accuracy for m in columns.values()):
            rows.append(
                [
                    f"{src} top-1",
                    *[f"{m.source_accuracy[src] * 100:.2f}%" for m in columns.values()],
                ]
            )
    if valid is not None:
        known = f"{int(valid.sum())} / {len(valid)}"
        rows.append(["samples with a known context", known, *[""] * (len(names) - 1)])
    for a, b in PAIRS:
        if a in classes and b in classes:
            rows.append(
                [f"{a} <-> {b} errors", *[pair_errors(m, classes, a, b) for m in columns.values()]]
            )
    table = tabulate(rows, headers=["", *names], tablefmt="github")
    print(table)
    report = (
        f"# Context effect: {args.checkpoint} on the {args.split} split\n\n"
        f"Model context dim {dim}; `context masked` feeds the same model all-zero vectors.\n\n"
        f"{table}\n"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        summary = {
            name: {
                "top1": m.top1_accuracy,
                "macro_f1": m.macro_f1,
                "source_accuracy": m.source_accuracy,
                "pair_errors": {
                    f"{a}|{b}": pair_errors(m, classes, a, b)
                    for a, b in PAIRS
                    if a in classes and b in classes
                },
            }
            for name, m in columns.items()
        }
        json_path = Path(args.out).with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[*] Saved {args.out}")


if __name__ == "__main__":
    main()
