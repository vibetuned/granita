"""End-to-end training smoke test on a tiny synthetic archive (CPU)."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from granita.config import TrainConfig
from granita.infer.predictor import GranitaClassifier
from granita.train.ablation import format_ablation_report, run_ablation
from granita.train.trainer import build_scheduler, run_training


def _tiny_archive(folder: Path) -> Path:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True)

    def make(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.full((n, 32, 32), 255, dtype=np.uint8)
        y = np.arange(n, dtype=np.int64) % 2
        for i in range(n):
            if y[i] == 0:
                x[i, 14:18, 4:28] = 0  # horizontal bar
            else:
                x[i, 4:28, 14:18] = 0  # vertical bar
            x[i] = np.clip(x[i].astype(int) + rng.integers(-20, 20, (32, 32)), 0, 255)
        return x, y, (np.arange(n) % 3).astype(np.uint8)

    xt, yt, st = make(64)
    xv, yv, sv = make(16)
    xs, ys, ss = make(16)

    def ctx(y: np.ndarray) -> np.ndarray:  # class 1 sits above the staff, class 0 on it
        c = np.zeros((len(y), 9), dtype=np.float32)
        c[:, 0] = 1.0
        c[:, 1] = np.where(y == 1, -0.8, 0.0)
        return c

    np.savez(
        folder / "dataset.npz",
        x_train=xt,
        y_train=yt,
        s_train=st,
        c_train=ctx(yt),
        x_val=xv,
        y_val=yv,
        s_val=sv,
        c_val=ctx(yv),
        x_test=xs,
        y_test=ys,
        s_test=ss,
        c_test=ctx(ys),
        classes=np.array(["articTenutoAbove", "fing1"]),
    )
    return folder / "dataset.npz"


def test_build_scheduler_warmup_then_cosine() -> None:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, epochs=4, steps_per_epoch=5, warmup_epochs=1)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.2 and lrs[4] == 1.0  # linear warmup over 5 steps
    assert all(a >= b for a, b in pairwise(lrs[4:]))  # then decays
    assert lrs[-1] < 0.05


def test_run_training_and_ablation(tmp_path: Path) -> None:
    npz = _tiny_archive(tmp_path / "data")
    cfg = TrainConfig(
        data_path=npz,
        output_dir=tmp_path / "run",
        model_name="granitanet",
        batch_size=16,
        epochs=2,
        warmup_epochs=1,
        context_dim=9,
        num_workers=0,
        device="cpu",
    )
    best = run_training(cfg)
    assert best == tmp_path / "run" / "best_model.pt"

    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]
    assert set(history[0]["val_source_accuracy"]) == {"font", "stroke", "crop"}

    best_ckpt = torch.load(best, map_location="cpu", weights_only=True)
    last_ckpt = torch.load(
        tmp_path / "run" / "last_model.pt", map_location="cpu", weights_only=True
    )
    assert "optimizer_state_dict" not in best_ckpt and "optimizer_state_dict" in last_ckpt
    assert best_ckpt["classes"] == ["articTenutoAbove", "fing1"]
    assert best_ckpt["config"]["warmup_epochs"] == 1
    assert best_ckpt["context_dim"] == 9 and best_ckpt["context_features"][0] == "valid"
    assert history[0]["val_top1_noctx"] is not None

    from safetensors import safe_open

    with safe_open(str(tmp_path / "run" / "best_model.safetensors"), "pt") as f:
        meta = f.metadata()
    assert meta["model_name"] == "granitanet"
    assert json.loads(meta["classes"]) == best_ckpt["classes"]

    clf = GranitaClassifier.from_checkpoint(best, device="cpu")
    assert clf.context_dim == 9
    assert clf.predict(np.full((32, 32), 255, dtype=np.uint8)).label in clf.classes

    abl_cfg = TrainConfig(
        data_path=npz,
        output_dir=tmp_path / "ablation",
        batch_size=16,
        epochs=1,
        warmup_epochs=0,
        num_workers=0,
        device="cpu",
    )
    results = run_ablation(["tiny"], abl_cfg)
    assert len(results) == 1 and results[0].eval_split == "test"
    assert set(results[0].source_accuracy) == {"font", "stroke", "crop"}
    assert results[0].epochs == 1
    report = format_ablation_report(results)
    assert "Test Top-1 by Sample Source" in report and "Stroke Top-1" in report
    # A second run reuses the trained arm instead of retraining
    again = run_ablation(["tiny"], abl_cfg)
    assert again[0].checkpoint_path == results[0].checkpoint_path
    assert (tmp_path / "ablation" / "ablation_report.md").is_file()
