# Granita

A lightweight, high-performance music symbol and glyph classifier for Optical Music Recognition (OMR).

Granita complements **[Kakigori](../kakigori)** (CenterNet detector + GNN Humdrum `**kern` pipeline) and **[Gelato](../gelato)** (Gemma-based vision-language OMR) by providing dedicated, sub-millisecond isolated music glyph recognition. It classifies 33 core musical symbols across fingerings, ornaments, articulations, accidentals, dynamics, and pedal markings from rendered scores, scanned manuscripts, and stylus/tablet inputs.

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
- **Export Ready**: Export any model family to TorchScript and ONNX for embedding in C++, ONNX Runtime, or desktop scoring applications.

---

## Installation & Setup

Set up the project with `uv`:

```bash
cd /home/flux/projects/granita
uv sync
```

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

Granita provides a unified CLI:

### 1. System, Model & Dataset Info
```bash
uv run granita info
```

### 2. Training a Specific Backbone
Train any architecture (`regla`, `iformer`, `lsnet`, `granitanet`, `tiny`, or their `_t` / `_s` variants):
```bash
uv run granita train \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --output-dir checkpoints/regla_v1 \
    --model regla \
    --epochs 25 \
    --batch-size 128
```

### 3. Multi-Model Ablation Study
Train and benchmark multiple architectures back-to-back under identical seeds, hyperparameters, and splits (measuring parameter counts, training time, batch-128 latency, test Top-1/Top-3/Macro F1, and per-group F1):
```bash
uv run granita ablate \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --output-dir checkpoints/ablation \
    --models tiny granitanet regla iformer lsnet \
    --epochs 15 \
    --batch-size 256
```
Writes `checkpoints/ablation/ablation_report.md` and `ablation_results.json`.

### 4. Evaluation
Evaluate a checkpoint on the validation or test split:
```bash
uv run granita eval \
    --checkpoint checkpoints/regla_v1/best_model.pt \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --split test
```

### 5. Single Glyph Inference
Classify any isolated glyph image:
```bash
uv run granita infer \
    --checkpoint checkpoints/regla_v1/best_model.pt \
    --image path/to/glyph.png \
    --top-k 5
```

### 6. Model Export
Export weights for deployment:
```bash
# Export to TorchScript (.pt)
uv run granita export \
    --checkpoint checkpoints/regla_v1/best_model.pt \
    --output checkpoints/regla.pt \
    --format torchscript

# Export to ONNX (.onnx)
uv run granita export \
    --checkpoint checkpoints/regla_v1/best_model.pt \
    --output checkpoints/regla.onnx \
    --format onnx
```

---

## Python API

You can import and use `GranitaClassifier` or any backbone (`GranitaReGLA`, `GranitaIFormer`, `GranitaLSNet`, `GranitaNet`, `GranitaTiny`) directly in `kakigori`, `gelato`, or standalone scripts:

```python
from PIL import Image
from granita import GranitaClassifier, create_model

# Instantiate any model architecture for custom pipelines
model = create_model("regla", num_classes=33)

# Or load a trained classifier from checkpoint
classifier = GranitaClassifier.from_checkpoint("checkpoints/regla_v1/best_model.pt")

image = Image.open("glyph.png")
pred = classifier.predict(image, top_k=3)

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
├── pyproject.toml              # Build config, dependencies, ruff, ty, and pytest settings
├── README.md
├── src/
│   └── granita/
│       ├── __init__.py         # Package exports
│       ├── config.py           # 33 classes, semantic groups, training defaults
│       ├── cli.py              # CLI entry point (info, train, ablate, eval, infer, export)
│       ├── data/
│       │   ├── dataset.py      # SymbolNpzDataset & SymbolFolderDataset
│       │   └── transforms.py   # Training augmentations & evaluation normalization
│       ├── models/
│       │   ├── backbone.py     # GranitaNet, GranitaTiny & model factory
│       │   ├── regla.py        # GranitaReGLA (ELRF + RGMA linear attention)
│       │   ├── iformer.py      # GranitaIFormer (ConvNeXt + SHMA modulation attention)
│       │   ├── lsnet.py        # GranitaLSNet (LKP + dynamic SKA convolution)
│       │   └── export.py       # TorchScript and ONNX exporters
│       ├── train/
│       │   ├── trainer.py      # AMP training loop, scheduler, checkpointing
│       │   └── ablation.py     # Multi-model ablation study & latency benchmark runner
│       ├── eval/
│       │   └── evaluate.py     # Metrics (Top-1, Top-3, Macro F1, Group Breakdown)
│       └── infer/
│           └── predictor.py    # GranitaClassifier Python API
└── tests/
    ├── test_dataset.py
    ├── test_export.py
    ├── test_models.py
    └── test_predictor.py
```
