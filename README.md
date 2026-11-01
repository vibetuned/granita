# Granita

A lightweight, high-performance music symbol and glyph classifier for Optical Music Recognition (OMR).

Granita complements **[Kakigori](../kakigori)** (CenterNet detector + GNN Humdrum `**kern` pipeline) and **[Gelato](../gelato)** (Gemma-based vision-language OMR) by providing dedicated, sub-millisecond isolated music glyph recognition. It classifies 33 core musical symbols across fingerings, ornaments, articulations, accidentals, dynamics, and pedal markings from rendered scores, scanned manuscripts, and stylus/tablet inputs.

---

## Architecture & Features

- **Specialized Glyph Backbones**:
  - `GranitaNet` (default): Modern residual ConvNet with channel attention (Squeeze-and-Excitation) and GELU activations, tailored for 32×32 single-channel glyph patches.
  - `GranitaTiny`: Ultra-compact 3-stage CNN (~40k parameters) for microsecond latency in interactive editing tools.
- **Astral Toolchain**: Built with the classic `uv`, `ruff`, and `ty` toolchain.
- **OMR Semantic Grouping**: Tracks and reports metrics across musical categories:
  - **Fingering**: `fing1`, `fing2`, `fing3`, `fing4`, `fing5`
  - **Ornament**: `trill`, `mordent`, `turn`, `fermata`, `arpeg`
  - **Articulation**: `articStaccatoAbove`, `articTenutoAbove`, `articAccentAbove`, `articStaccatissimoAbove`, `articMarcatoAbove`
  - **Accidental**: `accidentalFlat`, `accidentalNatural`, `accidentalSharp`, `accidentalDoubleSharp`, `accidentalDoubleFlat`
  - **Dynamic**: `dynamicPiano`, `dynamicMezzo`, `dynamicForte`, `dynamicSforzando`, `dynamicZ`, `dynamicPP`, `dynamicMP`, `dynamicMF`, `dynamicFF`, `dynamicFFF`, `dynamicForzando`, `dynamicSforzato`
  - **Pedal**: `pedal`
- **Data Integrations**: Natively loads both compressed `.npz` datasets (e.g. from `musicset/data/symbols/dataset.npz`) and standard `ImageFolder` directory hierarchies.
- **Export Ready**: Export to TorchScript and ONNX for embedding in C++, ONNX Runtime, or desktop scoring applications (e.g. PySide6 / Qt).

---

## Installation & Setup

Set up the project with `uv`:

```bash
cd /home/flux/projects/granita
uv sync
```

Activate the environment or run commands using `uv run`:

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

### 1. System & Dataset Info
```bash
uv run granita info
```

### 2. Training
Train a model on an `.npz` archive or image folder:
```bash
uv run granita train \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --output-dir checkpoints/granitanet_v1 \
    --model granitanet \
    --epochs 25 \
    --batch-size 128
```

### 3. Evaluation
Evaluate a checkpoint on the validation or test split:
```bash
uv run granita eval \
    --checkpoint checkpoints/granitanet_v1/best_model.pt \
    --data /home/flux/projects/musicset/data/symbols/dataset.npz \
    --split test
```

### 4. Single Glyph Inference
Classify any isolated glyph image:
```bash
uv run granita infer \
    --checkpoint checkpoints/granitanet_v1/best_model.pt \
    --image path/to/glyph.png \
    --top-k 5
```

### 5. Model Export
Export weights for deployment:
```bash
# Export to TorchScript (.pt)
uv run granita export \
    --checkpoint checkpoints/granitanet_v1/best_model.pt \
    --output checkpoints/granitanet.pt \
    --format torchscript

# Export to ONNX (.onnx)
uv run granita export \
    --checkpoint checkpoints/granitanet_v1/best_model.pt \
    --output checkpoints/granitanet.onnx \
    --format onnx
```

---

## Python API

You can import and use `GranitaClassifier` directly in `kakigori`, `gelato`, or standalone scripts:

```python
from PIL import Image
from granita import GranitaClassifier

# Load classifier from checkpoint
classifier = GranitaClassifier.from_checkpoint("checkpoints/granitanet_v1/best_model.pt")

# Predict single image (accepts PIL Image, numpy array, or torch Tensor)
image = Image.open("glyph.png")
pred = classifier.predict(image, top_k=3)

print(f"Symbol:     {pred.label}")
print(f"Category:   {pred.group}")
print(f"Confidence: {pred.confidence:.2%}")
print(f"Top 3:      {pred.top_k}")

# Batch prediction
predictions = classifier.predict_batch([img1, img2, img3])
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
│       ├── cli.py              # CLI entry point (granita)
│       ├── data/
│       │   ├── dataset.py      # SymbolNpzDataset & SymbolFolderDataset
│       │   └── transforms.py   # Training augmentations & evaluation normalization
│       ├── models/
│       │   ├── backbone.py     # GranitaNet, GranitaTiny architectures
│       │   └── export.py       # TorchScript and ONNX exporters
│       ├── train/
│       │   └── trainer.py      # AMP training loop, scheduler, checkpointing
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
