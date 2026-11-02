"""Pillow's pixels and PyTorch's answer for every reply in a shell's replies file: a
diagnostic asset for the shells' rasteriser on real strokes (the fixtures are polylines
of a few points; a pen gives a thousand). Rendered at each reply's recorded fraction, with
its recorded context::

    uv run python scripts/replies_fixtures.py models/battuta/replies/android_replies.jsonl \\
        --output apps/android/app/src/androidTest/assets/replies_fixtures.json

The Android shell's ``FixtureTest.repliesRasteriser`` reads that asset from the test apk
(skipped without it) and logs, per drawing, its own pixels against Pillow's and its answer
against PyTorch's. The asset is a few megabytes and not tracked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from granita.context import context_from_mapping, context_vector
from granita.infer.predictor import GranitaClassifier, render_strokes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("replies", type=Path, help="a shell's replies.jsonl")
    parser.add_argument("--checkpoint", default="models/battuta/best_model.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clf = GranitaClassifier.from_checkpoint(args.checkpoint, device="cpu")
    rows = [
        json.loads(line)
        for line in args.replies.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = []
    for r in rows:
        strokes = [[(p[0], p[1]) for p in s] for s in r["strokes"]]
        px = render_strokes(strokes, 32, fraction=r["fraction"])
        x = ((torch.from_numpy(px).float() / 255 - 0.5) / 0.5).view(1, 1, 32, 32)
        vec = context_vector(context_from_mapping(r["context"]))
        with torch.no_grad():
            p = torch.softmax(clf.model(x, torch.from_numpy(vec)[None])[0], -1)
        out.append(
            {
                "label": r["label"],
                "shell": r.get("predicted"),
                "strokes": [[[q[0], q[1]] for q in s] for s in r["strokes"]],
                "fraction": r["fraction"],
                "context_vector": [float(v) for v in vec],
                "pixels": px.flatten().tolist(),
                "top": {"label": clf.classes[int(p.argmax())], "confidence": float(p.max())},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"replies": out}), encoding="utf-8")
    print(f"wrote {args.output}: {len(out)} replies with Pillow's pixels and PyTorch's answer")


if __name__ == "__main__":
    main()
