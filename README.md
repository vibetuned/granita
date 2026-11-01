# Granita

A lightweight, high-performance music symbol and glyph classifier for score editors and Optical Music Recognition (OMR).

Granita's first job is **handwriting recognition in the [Battuta](https://battuta.vibetuned.com/) editor**: the user draws a symbol with a stylus, the strokes are rasterised exactly like the training data (`GranitaClassifier.predict_strokes`), and the predicted class is pushed back to the editor, which inserts it into the score. A **conditioning vector** describes *where* in the score the stroke was drawn (staff position, place in the measure, nearest notehead, size, all in staff spaces; `granita.context` mirrors `musicset.context`), so the same pen mark can mean different things in different places: a dot beside a notehead is an augmentation dot, above it a staccato. Train with `--context`; the model still works when the editor sends none.

It also serves as a second stage for an OMR detector: `GranitaClassifier.predict_box` classifies the glyph inside a detection box with the same crop protocol [musicset](https://github.com/vibetuned/musicset) uses to build the training crops, for example to resolve a single `fing` box into `fing1`..`fing5`. It classifies 61 musical symbols: fingerings, ornaments, articulations, accidentals, dynamics and pedal marks, plus notes by duration, rests, clefs, time signatures, barlines and augmentation dots (the classes of the HOMUS handwritten dataset).

---

## Architecture & Ablation Backbones

Granita includes five distinct model families tailored for $32 \times 32$ single-channel music glyphs so you can run controlled architectural ablations:

| Model (`--model`) | Variants | Architecture Highlights |
|---|---|---|
| **`regla`** | `regla_t`, `regla_s` | **ReGLA** (arXiv 2602.05262): Cascaded `DW3x3`+`DW5x5` `ELRF` blocks in stages 1–2 and **ReLU Gated Modulated Attention (`RGMA`)** linear attention in stages 3–4. |
| **`iformer`** | `iformer_t`, `iformer_s` | **iFormer** (ICLR 2025): Mobile hybrid combining `7x7` ConvNeXt-style inverted bottleneck blocks in stages 1–2 with **Single-Head Modulation Attention (`SHMA`)** in stages 3–4. |
| **`lsnet`** | `lsnet_t`, `lsnet_s` | **LSNet** (CVPR 2025, *"See Large, Focus Small"*): Combines `7x7` **Large-Kernel Perception (`LKP`)** weight generation with `3x3` dynamic **Small-Kernel Aggregation (`SKA`)**. |
| **`granitanet`** | `granitanet` | Modern residual ConvNet with Squeeze-and-Excitation (SE) channel attention and GELU activations; the released model. |
| **`tiny`** | `tiny` | Ultra-compact 3-stage CNN (~80k parameters) for microsecond latency in interactive editing tools. |

- **Astral Toolchain**: Built with the classic `uv`, `ruff`, and `ty` toolchain.
- **OMR Semantic Grouping**: Tracks and reports metrics across 11 musical categories (61 classes):
  - **Fingering**: `fing1`, `fing2`, `fing3`, `fing4`, `fing5`
  - **Ornament**: `trill`, `mordent`, `turn`, `fermata`, `arpeg`
  - **Articulation**: `articStaccatoAbove`, `articTenutoAbove`, `articAccentAbove`, `articStaccatissimoAbove`, `articMarcatoAbove`
  - **Accidental**: `accidentalFlat`, `accidentalNatural`, `accidentalSharp`, `accidentalDoubleSharp`, `accidentalDoubleFlat`
  - **Dynamic**: `dynamicPiano`, `dynamicMezzo`, `dynamicForte`, `dynamicSforzando`, `dynamicZ`, `dynamicPP`, `dynamicMP`, `dynamicMF`, `dynamicFF`, `dynamicFFF`, `dynamicForzando`, `dynamicSforzato`
  - **Pedal**: `pedal`
  - **Note**: `noteWhole`, `noteHalf`, `noteQuarter`, `note8th`, `note16th`, `note32nd`, `note64th`, `augmentationDot`
  - **Rest**: `restWholeHalf`, `restQuarter`, `rest8th`, `rest16th`, `rest32nd`, `rest64th`
  - **Clef**: `gClef`, `fClef`, `cClef`
  - **Time Signature**: `timeSigCommon`, `timeSigCutCommon`, `timeSig2over2`, `timeSig2over4`, `timeSig3over4`, `timeSig3over8`, `timeSig4over4`, `timeSig6over8`, `timeSig9over8`, `timeSig12over8`
  - **Barline**: `barlineSingle`
- **Data Integrations**: Natively loads the compressed `.npz` archives that `musicset symbols` writes (`dataset.npz` next to `labels.json`) and standard `ImageFolder` directory hierarchies.
- **Per-source metrics**: `musicset symbols` tags every sample as `font` (SMuFL glyph render), `stroke` (stylus stroke template), `crop` (real score crop) or `hand` (real pen strokes from HOMUS, whose 100 writers never straddle splits); training logs, `eval` and the ablation report break accuracy down by source, so the **hand** column on the test split is a writer-held-out handwriting number and **stroke** the synthetic one.
- **Export Ready**: Export any model family to ONNX (ONNX Runtime, browsers), Apple Core ML (`.mlpackage`), Google LiteRT (`.tflite`), `torch.export` (`.pt2`) or TorchScript (deprecated upstream).

---

## Installation & Setup

Granita is a standalone project; the training data comes from [musicset](https://github.com/vibetuned/musicset), which is an optional dependency.

```bash
uv sync                      # the classifier, training, evaluation and inference
uv sync --extra data         # + musicset, to build datasets with `uv run musicset symbols`
uv sync --extra export       # + ONNX, Core ML and LiteRT toolchains
```

On Linux `torch` comes from the CUDA 13 index, elsewhere from PyPI; `litert-torch` has
Linux wheels only, so the LiteRT export needs Linux (Core ML and ONNX export anywhere).

Run checks using `uv run`:

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

Granita provides a unified CLI. The examples below assume the dataset was built into `data/symbols/`; `data/` and `checkpoints/` are ignored by git.

### 1. System, Model & Dataset Info
```bash
uv run granita info
```

### 2. Building the dataset
`musicset symbols` (from the `data` extra) synthesises SMuFL glyphs and stylus stroke templates, crops real symbols from built musicset score datasets, and mixes in real pen strokes from the HOMUS dataset (Calvo-Zaragoza & Oncina, 2014; 15,200 symbols drawn on a tablet by 100 writers, which you download separately). Crops are split **by score** and HOMUS by writer, byte-identical images are kept in one split only, and every sample carries its source and its conditioning vector.
```bash
# synthetic symbols + real crops from one or more built musicset datasets
uv run musicset symbols data/symbols -n 2000 \
    --from-dataset path/to/built-dataset-a path/to/built-dataset-b \
    --max-crops-per-class 500 --preset mixed --workers 8

# add the HOMUS pen strokes (writer-disjoint split, training samples rendered twice with tablet augmentation)
uv run musicset symbols data/symbols -n 2000 --homus path/to/HOMUS.zip --homus-copies 2 --workers 8

# contributions from your own editor (JSON Lines, see the musicset README), split by contributor
uv run musicset symbols data/symbols -n 2000 --homus path/to/HOMUS.zip --strokes path/to/contributions.jsonl

# stylus-only evaluation set (different seed, so no overlap with the main set)
uv run musicset symbols data/symbols-tablet -n 300 --preset tablet --seed 1 --split 0 0 1 --no-png
```

### 3. Training a Specific Backbone
Train any architecture (`regla`, `iformer`, `lsnet`, `granitanet`, `tiny`, or their `_t` / `_s` variants):
```bash
uv run granita train \
    --data data/symbols/dataset.npz \
    --output-dir checkpoints/granitanet_ctx \
    --model granitanet \
    --context \
    --epochs 15 \
    --batch-size 256
```
Linear warmup (`--warmup-epochs`, default 2) precedes the cosine decay; `history.json` in the output directory records every epoch. `--context` feeds the archive's `c_<split>` vectors to the model, with `--context-dropout` (default 0.25) of the training samples zeroed so the model also works without a context; the log and `eval` then also report accuracy with the context masked. `best_model.pt` holds weights only, `last_model.pt` also the optimizer state.

### 4. Multi-Model Ablation Study
Train and benchmark multiple architectures back-to-back under identical seeds, hyperparameters, and splits (measuring parameter counts, training time, batch-128 latency, test Top-1/Top-3/Macro F1, and per-group F1):
```bash
uv run granita ablate \
    --data data/symbols/dataset.npz \
    --output-dir checkpoints/ablation \
    --models tiny granitanet regla iformer lsnet \
    --epochs 15 \
    --batch-size 256
```
Writes `checkpoints/ablation/ablation_report.md` and `ablation_results.json`, including a per-source table (font / stroke / crop / hand top-1). Arms whose `best_model.pt` already exists are reused unless `--force` is given.

### 5. Evaluation
Evaluate a checkpoint on any split (`train` is evaluated without augmentation):
```bash
uv run granita eval \
    --checkpoint checkpoints/granitanet_ctx/best_model.pt \
    --data data/symbols/dataset.npz \
    --split test
```
The report lists per-source accuracy and the most frequent confusions; a model trained with `--context` is scored twice, with and without its context. Point `--data` at `data/symbols-tablet/dataset.npz` for the stylus-only benchmark.

### 6. Single Glyph or Stroke Inference
Classify an isolated glyph image, or the pen strokes an editor captured (JSON `[[[x, y], ...], ...]`, any units):
```bash
uv run granita infer --checkpoint checkpoints/granitanet_ctx/best_model.pt --image path/to/glyph.png --top-k 5
uv run granita infer --checkpoint checkpoints/granitanet_ctx/best_model.pt --strokes strokes.json --staff-space 12
uv run granita infer --checkpoint checkpoints/granitanet_ctx/best_model.pt --strokes strokes.json --staff-space 12 \
    --context '{"y_staff": -2.5, "x_measure": 0.4, "barline": 3.0, "note_dx": 0.1, "note_dy": -1.6}'
```
`--staff-space` (pixels between two staff lines in the strokes' units) keeps the drawing at its natural size; without it `--fraction` sets the glyph's span as a fraction of the canvas. `--brush` renders a pressure-like brush instead of a uniform pen (the uniform pen scored better, see the results). `scripts/stroke_style_sweep.py` scores a checkpoint on the held-out HOMUS writers in `strokes_test.json` under several pen and brush styles.

### 7. Model Export
Export weights for deployment (dynamic batch axis unless `--static-batch`). `--sidecar` writes `<output>.json` with the class list, the input tensor spec, the stroke rasterisation rules (pen width, canvas, size from the staff space) and the context features, everything an editor needs to preprocess exactly like granita. A model trained with `--context` exports a second input, `context`.
```bash
# ONNX (.onnx) for ONNX Runtime / onnxruntime-web in the editor
uv run granita export --checkpoint checkpoints/granitanet_ctx/best_model.pt --output exports/granitanet.onnx --format onnx --sidecar

# Apple Core ML (.mlpackage) for iOS / iPadOS / macOS
uv run granita export --checkpoint checkpoints/granitanet_ctx/best_model.pt --output exports/granitanet.mlpackage --format coreml --sidecar

# Google LiteRT (.tflite) for Android / edge runtimes
uv run granita export --checkpoint checkpoints/granitanet_ctx/best_model.pt --output exports/granitanet.tflite --format litert --sidecar

# torch.export program (.pt2), the successor of TorchScript
uv run granita export --checkpoint checkpoints/granitanet_ctx/best_model.pt --output exports/granitanet.pt2 --format pt2

# TorchScript (.pt), deprecated upstream since torch 2.14
uv run granita export --checkpoint checkpoints/granitanet_ctx/best_model.pt --output exports/granitanet.pt --format torchscript

# Every format at once, with the sidecar, checking LiteRT, ONNX and pt2 against PyTorch
uv run python scripts/export_mobile.py --checkpoint checkpoints/granitanet_ctx/best_model.pt --output-dir exports
```

---

## Released models

[models/](models/) holds the released weights and their exports, tracked in git on purpose so a checkout has everything an app needs: the checkpoint, Core ML, LiteRT, ONNX and pt2 files, the sidecar and the test tables. [models/README.md](models/README.md) documents the inputs and the stroke rasterisation rules for the editor.

## Mobile test shells

[apps/ios](apps/ios/) (SwiftUI, Core ML) and [apps/android](apps/android/) (Kotlin, LiteRT's
CompiledModel API) are small apps that run the released exports on a tablet: draw a symbol
on a staff with the pen and see the top five classes, the 32 × 32 image the model saw, its
time and the context derived from the strokes. Each carries the three steps an editor needs
(the stroke rasteriser, the context vector, the model fed and read) and a test that checks
them against [`models/granitanet/fixtures.json`](models/granitanet/fixtures.json), made by
`scripts/shell_fixtures.py` from the PyTorch checkpoint. Each app's README says how to build
and run it.

## Results

Every run writes its tables next to its weights under `checkpoints/<run>/`; `uv run python scripts/collect_results.py` copies them into [results/](results/) (tracked, weights excluded) and regenerates `results/INDEX.md`. [results/README.md](results/README.md) holds the conclusions: backbone ablation, per-source accuracy with held-out HOMUS writers, the pen-versus-brush sweep, the synthetic-stroke transfer test, the writer learning curve (`scripts/writer_curve.py`) and the effect of the conditioning vector (`scripts/context_effect.py`).

## Python API

You can import and use `GranitaClassifier` or any backbone (`GranitaReGLA`, `GranitaIFormer`, `GranitaLSNet`, `GranitaNet`, `GranitaTiny`) in your own pipeline:

```python
from PIL import Image
from granita import GranitaClassifier, create_model
from granita.context import context_from_strokes

# Instantiate any model architecture for custom pipelines
model = create_model("granitanet", num_classes=61, context_dim=9)

# Or load a trained classifier from a checkpoint
classifier = GranitaClassifier.from_checkpoint("models/granitanet/best_model.pt")

image = Image.open("glyph.png")
pred = classifier.predict(image, top_k=3)  # isolated glyph (padded to square, never squashed)
pred = classifier.predict_strokes(strokes, top_k=3, staff_space=12.0)  # editor pen strokes, natural size
ctx = context_from_strokes(strokes, 12.0, y_staff=-2.5, note_dx=0.1, note_dy=-1.6)  # where it was drawn
pred = classifier.predict_strokes(strokes, staff_space=12.0, context=ctx)
pred = classifier.predict_box(page, (x1, y1, x2, y2))  # a detection box on a page image

print(f"Symbol:     {pred.label}")
print(f"Category:   {pred.group}")
print(f"Confidence: {pred.confidence:.2%}")
print(f"Top 3:      {pred.top_k}")
```

---

## Project Structure

```
granita/
├── .gitattributes              # model artifacts are binary
├── .gitignore
├── .python-version
├── LICENSE                     # AGPL-3.0-only
├── apps/
│   ├── ios/                    # test shell: SwiftUI + Core ML (xcodegen project.yml)
│   └── android/                # test shell: Kotlin + LiteRT (Gradle)
├── models/                     # released weights and exports (tracked)
├── results/                    # result tables and conclusions (tracked)
├── scripts/
│   ├── collect_results.py      # checkpoints/*/ tables -> results/
│   ├── context_effect.py       # a context model scored with and without its context
│   ├── export_mobile.py        # every export format + sidecar, with parity checks
│   ├── shell_fixtures.py       # strokes, images and logits for the test shells' checks
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
│       │   ├── dataset.py      # SymbolNpzDataset (sources, contexts) & SymbolFolderDataset, split loaders
│       │   └── transforms.py   # Training augmentations & evaluation normalization
│       ├── models/
│       │   ├── backbone.py     # GranitaNet (with context input), GranitaTiny & model factory
│       │   ├── regla.py        # GranitaReGLA (ELRF + RGMA linear attention)
│       │   ├── iformer.py      # GranitaIFormer (ConvNeXt + SHMA modulation attention)
│       │   ├── lsnet.py        # GranitaLSNet (LKP + dynamic SKA convolution)
│       │   └── export.py       # ONNX, Core ML, LiteRT, torch.export (.pt2) and TorchScript exporters
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
