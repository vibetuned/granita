# Granita

A lightweight, high-performance music symbol and glyph classifier for score editors and Optical Music Recognition (OMR).

Granita's first job is **handwriting recognition in the [Battuta](https://battuta.vibetuned.com/) editor**: the user draws a symbol with a stylus, the strokes are rasterised exactly like the training data (`GranitaClassifier.predict_strokes`), and the predicted class is pushed back to the editor, which inserts it into the score. A **conditioning vector** describes *where* in the score the stroke was drawn (staff position, place in the measure, nearest notehead, size, all in staff spaces; `granita.context` mirrors `musicset.context`), so the same pen mark can mean different things in different places: a dot beside a notehead is an augmentation dot, above it a staccato. Train with `--context`; the model still works when the editor sends none.

It also complements **[Kakigori](../kakigori)** (CenterNet detector + GNN Humdrum `**kern` pipeline) and **[Gelato](../gelato)** (Gemma-based vision-language OMR): kakigori detects a single `fing` box, and `GranitaClassifier.predict_box` resolves it to `fing1`..`fing5` with the same crop protocol musicset uses to build the training crops. It classifies 61 musical symbols: fingerings, ornaments, articulations, accidentals, dynamics and pedal marks, plus notes by duration, rests, clefs, time signatures, barlines and augmentation dots (the classes of the HOMUS handwritten dataset).

---

## Architecture & Ablation Backbones

Granita includes five distinct model families tailored for $32 \times 32$ single-channel music glyphs so you can run controlled architectural ablations:

| Model (`--model`) | Variants | Architecture Highlights |
|---|---|---|
| **`regla`** | `regla_t`, `regla_s` | **ReGLA** (from `kakigori` / arXiv 2602.05262): Cascaded `DW3x3`+`DW5x5` `ELRF` blocks in stages 1–2 and **ReLU Gated Modulated Attention (`RGMA`)** linear attention in stages 3–4. |
| **`iformer`** | `iformer_t`, `iformer_s` | **iFormer** (ICLR 2025): Mobile hybrid combining `7x7` ConvNeXt-style inverted bottleneck blocks in stages 1–2 with **Single-Head Modulation Attention (`SHMA`)** in stages 3–4. |
| **`lsnet`** | `lsnet_t`, `lsnet_s` | **LSNet** (CVPR 2025, *"See Large, Focus Small"*): Combines `7x7` **Large-Kernel Perception (`LKP`)** weight generation with `3x3` dynamic **Small-Kernel Aggregation (`SKA`)**. |
| **`granitanet`** | `granitanet` | Modern residual ConvNet with Squeeze-and-Excitation (SE) channel attention and GELU activations. |
| **`tiny`** | `tiny` | Ultra-compact 3-stage CNN (~80k parameters) for microsecond latency in interactive editing tools. |

- **Astral Toolchain**: Built with the classic `uv`, `ruff`, and `ty` toolchain.
- **OMR Semantic Grouping**: Tracks and reports metrics across musical categories:
  - **Fingering**: `fing1`, `fing2`, `fing3`, `fing4`, `fing5`
  - **Ornament**: `trill`, `mordent`, `turn`, `fermata`, `arpeg`
  - **Articulation**: `articStaccatoAbove`, `articTenutoAbove`, `articAccentAbove`, `articStaccatissimoAbove`, `articMarcatoAbove`
  - **Accidental**: `accidentalFlat`, `accidentalNatural`, `accidentalSharp`, `accidentalDoubleSharp`, `accidentalDoubleFlat`
  - **Dynamic**: `dynamicPiano`, `dynamicMezzo`, `dynamicForte`, `dynamicSforzando`, `dynamicZ`, `dynamicPP`, `dynamicMP`, `dynamicMF`, `dynamicFF`, `dynamicFFF`, `dynamicForzando`, `dynamicSforzato`
  - **Pedal**: `pedal`
- **Data Integrations**: Natively loads both compressed `.npz` datasets (e.g. from `musicset/data/symbols/dataset.npz`) and standard `ImageFolder` directory hierarchies.
- **Per-source metrics**: `musicset symbols` tags every sample as `font` (SMuFL glyph render), `stroke` (stylus stroke template), `crop` (real score crop) or `hand` (real pen strokes from HOMUS, whose 100 writers never straddle splits); training logs, `eval` and the ablation report break accuracy down by source, so the **hand** column on the test split is a writer-held-out handwriting number and **stroke** the synthetic one.
- **Export Ready**: Export any model family to ONNX (ONNX Runtime, browsers), `torch.export` (`.pt2`) or TorchScript (deprecated upstream).

---

## Installation & Setup

Set up the project with `uv`:

```bash
cd /home/flux/projects/granita
uv sync
```

Install the ONNX toolchain with `uv sync --extra export`. Run checks using `uv run`:

```bash
# Linting & Formatting
uv run ruff check .
uv run ruff format .

# Type checking with ty
uv run ty check

# Unit tests
uv run pytest
```

---

## CLI Usage

Granita provides a unified CLI:

### 1. System, Model & Dataset Info
```bash
uv run granita info
```

### 2. Building the dataset
The training data comes from [musicset](../musicset): synthetic SMuFL glyphs and stylus stroke templates, plus real crops from built score datasets. Crops are split **by score** and byte-identical images are kept in one split only, so the test numbers are not inflated by engravings repeated across pages (an earlier build leaked ~20% of each fingering test class that way).
```bash
cd ../musicset
uv run musicset symbols data/symbols -n 2000 \
    --from-dataset ../kakigori/data/train-lite2 ../kakigori/data/validation-lite2 ../kakigori/data/synthetic-lite2 \
    --max-crops-per-class 500 --preset mixed --workers 8
# add the HOMUS pen strokes (writer-disjoint split, train samples rendered twice with tablet augmentation)
uv run musicset symbols data/symbols -n 2000 --homus ../music_hand/HOMUS.zip --homus-copies 2 --workers 8
# stylus-only evaluation set (different seed, so no overlap with the main set)
uv run musicset symbols data/symbols-tablet -n 300 --preset tablet --seed 1 --split 0 0 1 --no-png
```

### 3. Training a Specific Backbone
Train any architecture (`regla`, `iformer`, `lsnet`, `granitanet`, `tiny`, or their `_t` / `_s` variants):
```bash
uv run granita train \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --output-dir checkpoints/regla_v1 \
    --model regla \
    --epochs 25 \
    --batch-size 128
```
Linear warmup (`--warmup-epochs`, default 2) precedes the cosine decay; `history.json` in the output directory records every epoch. `--context` feeds the archive's `c_<split>` vectors to the model, with `--context-dropout` (default 0.25) of the training samples zeroed so the model also works without a context; the log and `eval` then also report accuracy with the context masked. `best_model.pt` holds weights only, `last_model.pt` also the optimizer state.

### 4. Multi-Model Ablation Study
Train and benchmark multiple architectures back-to-back under identical seeds, hyperparameters, and splits (measuring parameter counts, training time, batch-128 latency, test Top-1/Top-3/Macro F1, and per-group F1):
```bash
uv run granita ablate \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --output-dir checkpoints/ablation \
    --models tiny granitanet regla iformer lsnet \
    --epochs 15 \
    --batch-size 256
```
Writes `checkpoints/ablation/ablation_report.md` and `ablation_results.json`, including a per-source table (font / stroke / crop top-1). Arms whose `best_model.pt` already exists are reused unless `--force` is given.

### 5. Evaluation
Evaluate a checkpoint on the validation or test split:
```bash
uv run granita eval \
    --checkpoint checkpoints/regla_v1/best_model.pt \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --split test
```
Any split works (`train` is evaluated without augmentation); the report lists per-source accuracy and the most frequent confusions. Point `--data` at `../musicset/data/symbols-tablet/dataset.npz` for the stylus-only benchmark.

### 6. Single Glyph or Stroke Inference
Classify an isolated glyph image, or the pen strokes an editor captured (JSON `[[[x, y], ...], ...]`, any units):
```bash
uv run granita infer --checkpoint checkpoints/regla_v1/best_model.pt --image path/to/glyph.png --top-k 5
uv run granita infer --checkpoint checkpoints/regla_v1/best_model.pt --strokes strokes.json --fraction 0.7
```
`--fraction` is the glyph's size relative to the canvas (musicset draws templates at 0.55-0.88); pass a small value for a dot-sized mark and a large one for a dynamic when the editor knows the staff scale. `--brush` renders a pressure-like brush (width follows pen speed, soft edge) instead of a uniform pen.

`scripts/stroke_style_sweep.py` scores a checkpoint on the held-out HOMUS writers in `strokes_test.json` under several pen and brush styles, to pick the rendering the editor should use.

### 7. Model Export
Export weights for deployment (dynamic batch axis unless `--static-batch`). `--sidecar` writes `<output>.json` with the class list, the input tensor spec, the stroke rasterisation rules (pen width, canvas, size from the staff space) and the context features, everything the editor needs to preprocess exactly like granita. A model trained with `--context` exports a second input, `context`.
```bash
# ONNX (.onnx) for ONNX Runtime / onnxruntime-web in the editor
uv run granita export --checkpoint checkpoints/regla_v1/best_model.pt --output checkpoints/regla.onnx --format onnx --sidecar

# torch.export program (.pt2), the successor of TorchScript
uv run granita export --checkpoint checkpoints/regla_v1/best_model.pt --output checkpoints/regla.pt2 --format pt2

# TorchScript (.pt), deprecated upstream since torch 2.14
uv run granita export --checkpoint checkpoints/regla_v1/best_model.pt --output checkpoints/regla.pt --format torchscript
```

---

## Results

Every run writes its tables next to its weights under `checkpoints/<run>/`; `uv run python scripts/collect_results.py --exclude ablation_smoke smoke_run` copies them into [results/](results/) (tracked in git, weights excluded) and regenerates `results/INDEX.md`. [results/README.md](results/README.md) holds the conclusions: backbone ablation, per-source accuracy with held-out HOMUS writers, the pen-versus-brush sweep, the synthetic-stroke transfer test, and the writer learning curve (`scripts/writer_curve.py`).

## Python API

You can import and use `GranitaClassifier` or any backbone (`GranitaReGLA`, `GranitaIFormer`, `GranitaLSNet`, `GranitaNet`, `GranitaTiny`) directly in `kakigori`, `gelato`, or standalone scripts:

```python
from PIL import Image
from granita import GranitaClassifier, create_model
from granita.context import context_from_strokes

# Instantiate any model architecture for custom pipelines
model = create_model("regla", num_classes=33)

# Or load a trained classifier from checkpoint
classifier = GranitaClassifier.from_checkpoint("checkpoints/regla_v1/best_model.pt")

image = Image.open("glyph.png")
pred = classifier.predict(image, top_k=3)            # isolated glyph (padded to square, never squashed)
pred = classifier.predict_strokes(strokes, top_k=3, staff_space=12.0)  # editor pen strokes, natural size
ctx = context_from_strokes(strokes, 12.0, y_staff=-2.5, note_dx=0.1, note_dy=-1.6)  # where it was drawn
pred = classifier.predict_strokes(strokes, staff_space=12.0, context=ctx)
pred = classifier.predict_box(page, (x1, y1, x2, y2))  # kakigori detection box on a page image

print(f"Symbol:     {pred.label}")
print(f"Category:   {pred.group}")
print(f"Confidence: {pred.confidence:.2%}")
print(f"Top 3:      {pred.top_k}")
```

---

## Project Structure

```
granita/
├── .gitignore
├── .python-version
├── LICENSE                     # AGPL-3.0-only
├── results/                    # result tables and conclusions (tracked)
├── scripts/
│   ├── collect_results.py      # checkpoints/*/ tables -> results/
│   ├── context_effect.py       # a context model scored with and without its context
│   ├── stroke_style_sweep.py   # pen vs brush rendering sweep on held-out HOMUS writers
│   └── writer_curve.py         # held-out accuracy vs number of HOMUS training writers
├── pyproject.toml              # Build config, dependencies, ruff, ty, and pytest settings
├── README.md
├── src/
│   └── granita/
│       ├── __init__.py         # Package exports
│       ├── config.py           # classes, semantic groups, sample sources, context features, training defaults
│       ├── context.py          # conditioning vector (mirror of musicset.context)
│       ├── cli.py              # CLI entry point (info, train, ablate, eval, infer, export)
│       ├── data/
│       │   ├── dataset.py      # SymbolNpzDataset (with sources) & SymbolFolderDataset, split loaders
│       │   └── transforms.py   # Training augmentations & evaluation normalization
│       ├── models/
│       │   ├── backbone.py     # GranitaNet, GranitaTiny & model factory
│       │   ├── regla.py        # GranitaReGLA (ELRF + RGMA linear attention)
│       │   ├── iformer.py      # GranitaIFormer (ConvNeXt + SHMA modulation attention)
│       │   ├── lsnet.py        # GranitaLSNet (LKP + dynamic SKA convolution)
│       │   └── export.py       # ONNX, torch.export (.pt2) and TorchScript exporters
│       ├── train/
│       │   ├── trainer.py      # AMP training loop, warmup + cosine schedule, history, checkpointing
│       │   └── ablation.py     # Multi-model ablation study & latency benchmark runner
│       ├── eval/
│       │   └── evaluate.py     # Metrics (Top-1, Top-3, Macro F1, group, per-source, confusions)
│       └── infer/
│           └── predictor.py    # GranitaClassifier: images, editor strokes, page boxes
└── tests/
    ├── test_dataset.py
    ├── test_evaluate.py
    ├── test_export.py
    ├── test_models.py
    ├── test_predictor.py
    └── test_trainer.py
```
