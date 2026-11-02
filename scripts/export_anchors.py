"""Anchors for the shells' on-device training: feature vectors the released model read,
with its own logits — what a last layer retrained on a user's replies is held to (the
anchor term of the shells' trainer; the user's design, 2026-11-02).

From a ``musicset symbols`` dataset with contexts (``--data``: K per class from the val
split, among the samples the model gets right, at random with a fixed seed), or, as an
interim set, from the shell fixtures (``--fixtures``, 13 drawings)::

    uv run python scripts/export_anchors.py --data /path/to/dataset.npz --per-class 4
    uv run python scripts/export_anchors.py --fixtures models/battuta/fixtures.json

From a dataset the anchors lean toward handwriting (``--prefer``, the user's choice,
2026-11-02): a class's K come from real handwriting (HOMUS, EMNIST, Fornés) as far as it
has them, then from stylus templates, then score crops, then font renders. The note values
get none (``--skip``, the user's choice too): only the trainer's pull toward the original
weights holds them.

Writes ``models/battuta/anchors.json``: ``{"source", "classes", "anchors": [{"label",
"features", "logits", "source"}]}``, ``label`` an index into ``classes``, ``source`` the
sample's (``granita.config.SOURCE_NAMES``) when it comes from a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from granita.config import CONTEXT_DIM, SOURCE_NAMES
from granita.infer.predictor import GranitaClassifier

PREFER = "hand+emnist+fornes,stroke,crop,font"
"""Source tiers, best first: a class's anchors fill from the first tier that has samples."""
NOTE_VALUES = ("noteWhole", "noteHalf", "noteQuarter", "note8th", "note16th")


def source_tiers(spec: str) -> list[set[int]]:
    """``"hand+emnist,stroke"`` -> ``[{3, 4}, {1}]``, indices into ``SOURCE_NAMES``."""
    tiers = []
    for tier in spec.split(","):
        names = [n.strip() for n in tier.split("+")]
        unknown = [n for n in names if n not in SOURCE_NAMES]
        if unknown:
            raise SystemExit(f"--prefer: unknown sources {unknown}, not in {SOURCE_NAMES}")
        tiers.append({SOURCE_NAMES.index(n) for n in names})
    return tiers


def from_dataset(
    clf: GranitaClassifier,
    data: Path,
    per_class: int,
    seed: int,
    prefer: list[set[int]],
    skip: set[str],
) -> list[dict[str, object]]:
    d = np.load(data)
    x = d["x_val"]
    y = d["y_val"]
    s = d["s_val"] if "s_val" in d.files else np.zeros(len(x), dtype=np.int64)
    classes = [str(c) for c in d["classes"]] if "classes" in d.files else clf.classes
    if list(classes) != clf.classes:
        raise SystemExit(
            f"{data}: its classes are not the model's ({len(classes)} vs {len(clf.classes)})"
        )
    c = d["c_val"] if "c_val" in d.files else np.zeros((len(x), CONTEXT_DIM), dtype=np.float32)
    feats = np.empty((len(x), 128), dtype=np.float32)
    logits = np.empty((len(x), len(clf.classes)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = ((torch.from_numpy(x[i : i + 512]).float() / 255.0 - 0.5) / 0.5).unsqueeze(1)
            cb = torch.from_numpy(c[i : i + 512]).float()
            feats[i : i + 512] = clf.model.features(xb, cb).numpy()
            logits[i : i + 512] = clf.model(xb, cb).numpy()
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []
    for k, name in enumerate(clf.classes):
        if name in skip:
            continue
        right = (y == k) & (logits.argmax(1) == k)
        picked: list[int] = []
        for tier in prefer:
            pool = np.flatnonzero(right & np.isin(s, list(tier)))
            picked += rng.permutation(pool)[: per_class - len(picked)].tolist()
        for i in picked:
            out.append(
                {
                    "label": int(k),
                    "features": feats[i].round(5).tolist(),
                    "logits": logits[i].round(5).tolist(),
                    "source": SOURCE_NAMES[int(s[i])],
                }
            )
    return out


def from_fixtures(path: Path) -> list[dict[str, object]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"label": int(f["top"]["index"]), "features": f["features"], "logits": f["logits"]}
        for f in doc["fixtures"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", default="models/battuta/best_model.pt")
    parser.add_argument(
        "--data", type=Path, help="a musicset symbols dataset.npz with the model's classes"
    )
    parser.add_argument("--fixtures", type=Path, help="the shell fixtures instead: an interim set")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument(
        "--prefer",
        default=PREFER,
        help=f"source tiers, best first, '+' within a tier (default {PREFER})",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=list(NOTE_VALUES),
        help="classes that get no anchors (default the note values; none with an empty list)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/battuta/anchors.json"))
    args = parser.parse_args()
    clf = GranitaClassifier.from_checkpoint(args.checkpoint, device="cpu")
    if args.data:
        skip = set(args.skip)
        if skip - set(clf.classes):
            raise SystemExit(f"--skip: not the model's classes: {sorted(skip - set(clf.classes))}")
        anchors = from_dataset(
            clf, args.data, args.per_class, args.seed, source_tiers(args.prefer), skip
        )
        source = (
            f"{args.data.name}: val split, {args.per_class} per class among the ones the model "
            f"gets right, sources by preference {args.prefer}, seed {args.seed}"
            + (f", none for {', '.join(sorted(skip))}" if skip else "")
        )
    elif args.fixtures:
        anchors = from_fixtures(args.fixtures)
        source = f"{args.fixtures.name} (interim: the shell fixtures, not a dataset)"
    else:
        raise SystemExit("one of --data or --fixtures")
    args.output.write_text(
        json.dumps(
            {
                "source": source,
                "checkpoint": str(args.checkpoint),
                "classes": clf.classes,
                "anchors": anchors,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}: {len(anchors)} anchors ({source})")


if __name__ == "__main__":
    main()
