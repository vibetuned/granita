"""Score a checkpoint on held-out HOMUS writers under several stroke rendering styles.

Answers "is a brush-drawn image a better input than a thin pen?" with the raw
strokes musicset keeps in ``strokes_test.json`` (writers never seen in training).

    uv run python scripts/stroke_style_sweep.py \\
        --checkpoint checkpoints/granitanet_v3/best_model.pt \\
        --strokes ../musicset/data/symbols/strokes_test.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

from granita.eval.evaluate import metrics_from_predictions
from granita.infer.predictor import GranitaClassifier, render_strokes

# name: (pen width, brush, fixed fraction or None for the drawing's natural size)
STYLES: dict[str, tuple[float, bool, float | None]] = {
    "pen 0.05": (0.05, False, None),
    "pen 0.085 (default)": (0.085, False, None),
    "pen 0.12": (0.12, False, None),
    "brush 0.085": (0.085, True, None),
    "brush 0.12": (0.12, True, None),
    "pen 0.085, size ignored (fraction 0.7)": (0.085, False, 0.7),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strokes", required=True, help="strokes_test.json from musicset symbols")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--out", default=None, help="JSON results (default: next to the checkpoint)"
    )
    args = parser.parse_args()

    records = json.loads(Path(args.strokes).read_text(encoding="utf-8"))
    clf = GranitaClassifier.from_checkpoint(args.checkpoint, device=args.device)
    class_to_idx = clf.class_to_idx
    records = [r for r in records if r["label"] in class_to_idx]
    y_true = np.array([class_to_idx[r["label"]] for r in records], dtype=np.int64)
    writers = np.array([r["writer"] for r in records])
    print(f"{len(records)} strokes from {len(set(writers.tolist()))} held-out writers")

    rows = []
    results: dict[str, dict[str, float]] = {}
    for name, (pen_width, brush, fixed_fraction) in STYLES.items():
        images = [
            render_strokes(
                r["strokes"],
                32,
                fraction=float(r["fraction"]) if fixed_fraction is None else fixed_fraction,
                pen_width=pen_width,
                brush=brush,
            )
            for r in records
        ]
        preds: list[int] = []
        for i in range(0, len(images), args.batch_size):
            preds.extend(p.class_idx for p in clf.predict_batch(images[i : i + args.batch_size]))
        y_pred = np.array(preds, dtype=np.int64)
        m = metrics_from_predictions(y_true, y_pred, clf.classes, top3_correct=0)
        # macro F1 over the classes the strokes actually cover (HOMUS has 32 of granita's 61)
        present = sorted({int(c) for c in y_true})
        macro_f1 = float(np.mean([m.per_class_f1[clf.classes[c]] for c in present]))
        if fixed_fraction is None and not brush and abs(pen_width - 0.085) < 1e-9:
            per_class = {clf.classes[c]: float((y_pred[y_true == c] == c).mean()) for c in present}
            worst = sorted(per_class.items(), key=lambda kv: kv[1])[:8]
            print("weakest classes with the default pen:")
            for cls_name, acc in worst:
                print(f"  {cls_name:<24} {acc * 100:5.1f}%")
        per_writer: dict[int, list[bool]] = defaultdict(list)
        for w, ok in zip(writers.tolist(), (y_true == y_pred).tolist(), strict=True):
            per_writer[int(w)].append(bool(ok))
        writer_acc = np.array([float(np.mean(v)) for v in per_writer.values()])
        results[name] = {
            "top1": m.top1_accuracy,
            "macro_f1": macro_f1,
            "writer_mean": float(writer_acc.mean()),
            "writer_min": float(writer_acc.min()),
        }
        rows.append(
            [
                name,
                f"{m.top1_accuracy * 100:.2f}%",
                f"{macro_f1 * 100:.2f}%",
                f"{writer_acc.mean() * 100:.2f}%",
                f"{writer_acc.min() * 100:.2f}%",
            ]
        )
    print(
        tabulate(
            rows,
            headers=["Style", "Top-1", "Macro F1", "Per-writer mean", "Worst writer"],
            tablefmt="github",
        )
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "stroke_style_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[*] Saved {out}")


if __name__ == "__main__":
    main()
