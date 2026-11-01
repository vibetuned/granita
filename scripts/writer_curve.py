"""Held-out-writer accuracy as a function of how many HOMUS writers the training set holds.

Sizes the handwriting collection effort: the archive must carry ``w_<split>``
writer ids (``musicset symbols --homus``). Each point trains granitanet on the
full synthetic + crop data plus the first ``n`` training writers (seeded
shuffle, nested subsets) and scores the untouched held-out test writers.

    uv run python scripts/writer_curve.py --data ../musicset/data/symbols/dataset.npz \\
        --writers 0 5 10 20 40 80 --out results/writer_curve
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from granita.config import TrainConfig
from granita.data.dataset import SymbolNpzDataset, make_loader
from granita.data.transforms import get_eval_transforms
from granita.eval.evaluate import metrics_from_predictions
from granita.models.backbone import create_model
from granita.train.trainer import resolve_device, run_training

HAND = 3


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    ds = SymbolNpzDataset(x, np.zeros(len(x), dtype=np.int64), [], transform=get_eval_transforms())
    preds: list[int] = []
    model.eval()
    for images, _, _ in make_loader(ds, 512, shuffle=False, num_workers=0):
        preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
    return np.array(preds, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="dataset.npz with w_<split> writer ids")
    parser.add_argument("--writers", type=int, nargs="+", default=[0, 5, 10, 20, 40, 80])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", default="checkpoints/writer_curve")
    parser.add_argument("--out", default="results/writer_curve", help="prefix for .md and .json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    data = np.load(args.data)
    classes = [str(c) for c in data["classes"]]
    w_train = data["w_train"]
    train_writers = np.array(sorted({int(w) for w in w_train if w >= 0}))
    np.random.default_rng(args.seed).shuffle(train_writers)
    hand_test = data["s_test"] == HAND
    x_hand, y_hand, w_hand = (
        data["x_test"][hand_test],
        data["y_test"][hand_test],
        data["w_test"][hand_test],
    )
    present = sorted({int(c) for c in y_hand})
    print(
        f"{len(train_writers)} training writers available, {len(set(w_hand.tolist()))} held-out "
        f"writers with {len(y_hand)} strokes over {len(present)} classes"
    )

    ckpt_root = Path(args.checkpoints)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[object]] = []
    results: list[dict[str, object]] = []
    last_per_class: dict[str, float] = {}
    for n in args.writers:
        chosen = set(train_writers[:n].tolist())
        keep = (w_train < 0) | np.isin(w_train, list(chosen))
        subset_dir = ckpt_root / f"writers_{n:02d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset = subset_dir / "dataset.npz"
        arrays = {k: data[k] for k in data.files}
        for k in ("x_train", "y_train", "s_train", "w_train", "c_train"):
            if k in arrays:
                arrays[k] = data[k][keep]
        np.savez_compressed(subset, **arrays)
        n_hand = int((arrays["s_train"] == HAND).sum())
        print(f"\n=== {n} writers: {int(keep.sum())} training samples, {n_hand} hand")

        cfg = TrainConfig(
            data_path=subset,
            output_dir=subset_dir,
            model_name="granitanet",
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=4,
            seed=args.seed,
            device=device,
        )
        t0 = time.time()
        best = run_training(cfg)
        seconds = time.time() - t0
        subset.unlink()  # the filtered archive is large and reproducible

        ckpt = torch.load(best, map_location=device, weights_only=True)
        model = create_model("granitanet", num_classes=len(classes)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_pred = predict(model, x_hand, device)
        m = metrics_from_predictions(y_hand, y_pred, classes, top3_correct=0)
        macro_f1 = float(np.mean([m.per_class_f1[classes[c]] for c in present]))
        per_writer = np.array(
            [
                float((y_pred[w_hand == w] == y_hand[w_hand == w]).mean())
                for w in sorted(set(w_hand.tolist()))
            ]
        )
        per_class = {classes[c]: float((y_pred[y_hand == c] == c).mean()) for c in present}
        last_per_class = per_class
        results.append(
            {
                "writers": n,
                "hand_train_samples": n_hand,
                "hand_top1": m.top1_accuracy,
                "hand_macro_f1": macro_f1,
                "writer_mean": float(per_writer.mean()),
                "writer_min": float(per_writer.min()),
                "per_class": per_class,
                "best_epoch": int(ckpt.get("epoch", 0)),
                "train_seconds": seconds,
                "checkpoint": str(best),
            }
        )
        rows.append(
            [
                n,
                n_hand,
                f"{m.top1_accuracy * 100:.1f}%",
                f"{macro_f1 * 100:.1f}%",
                f"{per_writer.mean() * 100:.1f}%",
                f"{per_writer.min() * 100:.1f}%",
            ]
        )
        print(tabulate(rows[-1:], tablefmt="plain"))

    headers = [
        "Training writers",
        "Hand train samples",
        "Held-out top-1",
        "Macro F1",
        "Per-writer mean",
        "Worst writer",
    ]
    table = tabulate(rows, headers=headers, tablefmt="github")
    print("\n" + table)
    worst = sorted(last_per_class.items(), key=lambda kv: kv[1])[:8]
    md = (
        "# Held-out-writer accuracy vs. number of HOMUS training writers\n\n"
        f"granitanet, {args.epochs} epochs, seed {args.seed}; "
        "same synthetic + crop data in every row; "
        f"{len(set(w_hand.tolist()))} held-out writers, {len(y_hand)} strokes, "
        "pen 0.085 at natural size.\n\n"
        f"{table}\n\n"
        f"Weakest classes at {args.writers[-1]} writers: "
        + ", ".join(f"{c} {a * 100:.0f}%" for c, a in worst)
        + "\n"
    )
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[*] Saved {out.with_suffix('.md')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
