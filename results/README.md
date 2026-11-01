# Granita results

Every training run writes its tables next to its weights under `checkpoints/<run>/`
(gitignored); `uv run python scripts/collect_results.py --exclude ablation_smoke smoke_run`
copies the tables here and regenerates [INDEX.md](INDEX.md). Weights are never copied.
All runs: granitanet unless stated, 15 epochs, batch 256, 2 warmup epochs, seed 42.

Dataset builds live in `../musicset/data/`:

| build | classes | train / val / test | notes |
|---|---|---|---|
| `symbols.v1-leaky` | 33 | 57,679 / 7,212 / 7,212 | per-image split; 208 test images were byte-copies of train images, almost all fingerings |
| `symbols.v2-33cls` | 33 | 65,648 / 8,210 / 8,218 | score-disjoint crops, cross-split dedup, source codes |
| `symbols` | 61 | 139,087 / 15,871 / 15,873 | + HOMUS pen strokes (80 / 10 / 10 writers, `w_<split>` writer ids), notes, rests, clefs, time signatures, barline, dot |
| `symbols-brush` | 61 | same splits | HOMUS training strokes rendered as a pressure brush half the time, pen 0.05-0.13 |
| `symbols-tablet` | 61 | 0 / 0 / 18,276 | synthetic stylus-only benchmark, `--preset tablet --seed 1` |

## 1. Backbone ablation on the 33-class set (`ablation_v2/`)

| model | params | test macro F1 | stroke top-1 | stylus set stroke top-1 | batch-128 latency |
|---|---|---|---|---|---|
| tiny | 77,793 | 99.14% | 98.64% | 98.17% | 0.15 ms |
| **granitanet** | 1,997,561 | 99.78% | 99.73% | 99.55% | 0.76 ms |
| regla | 1,454,529 | 99.79% | 99.53% | 99.32% | 2.48 ms |
| iformer | 2,668,225 | 99.66% | 98.92% | 98.93% | 2.11 ms |
| lsnet | 2,517,465 | 99.76% | 99.66% | 99.38% | 8.02 ms |

granitanet is the pick: best on strokes and second fastest. The attention backbones add nothing.
These "stroke" numbers are synthetic templates and overstate real handwriting by ~13 points (section 3).

## 2. granitanet on the 61-class set with HOMUS (`granitanet_v3/`)

| source (test split) | samples | top-1 |
|---|---|---|
| font | 10,713 | 99.45% |
| stroke (synthetic templates) | 1,476 | 97.76% |
| crop (real engravings) | 2,164 | 99.54% |
| **hand (10 held-out writers)** | 1,520 | **84.93%** |

Top confusions: augmentation dot vs. staccato (61), fing1 vs. barline (21), 16th / 32nd / 64th
notes and rests among themselves (~80, a 32-pixel limit), quarter vs. half note (18), quarter rest vs. natural (13).

## 3. Rendering sweep on the same 1,520 held-out strokes (`granitanet_v3/`, `granitanet_v4/`)

| rendering at inference | v3, pen-only training | v4, half of HOMUS rendered as brush |
|---|---|---|
| pen 0.05 | **86.25%** | 84.80% |
| pen 0.085 | 84.93% | 82.50% |
| pen 0.12 | 81.05% | 79.14% |
| brush 0.085 | 70.00% | 80.46% |
| brush 0.12 | 51.64% | 74.41% |
| pen 0.085, size ignored (fraction 0.7) | 79.54% | 79.08% |

Thin uniform pen at the drawing's natural size wins; natural size is worth 5 points. Training with
the brush makes brush input usable but never better, and costs ~2 points on every pen style.
musicset and granita brush renders were verified byte-identical, so the test is fair.

## 4. Do synthetic strokes transfer to real handwriting? No (`granitanet_v3_nohand/`)

Same data minus the HOMUS training rows, scored on the same held-out writers, pen 0.05:

| held-out writers | with HOMUS | without |
|---|---|---|
| all 32 HOMUS classes | 86.2% | 28.7% |
| 4 accidentals, which have hand-coded stroke templates | 93.1% | 29.4% |
| 28 classes with fonts only | 85.4% | 28.6% |

Templates only teach the model to recognise themselves. Every class the editor must read by pen
needs real handwriting; the 29 classes without any (fingerings, ornaments, articulations, double
flat, dynamics, pedal) are currently unusable for stylus input. Per class: `granitanet_v3_nohand/per_class_with_vs_without_homus.md`.

## 5. How many writers are needed? (`writer_curve.md`, `writer_curve.json`, `writer_curve/`)

Same synthetic + crop data in every row, plus the first n of the 80 HOMUS training writers
(nested seeded subsets, every writer ~152 strokes, rendered twice); scored on the same 10 held-out
writers, pen 0.085 at natural size.

| training writers | hand train samples | held-out top-1 | macro F1 | worst writer |
|---|---|---|---|---|
| 0 | 0 | 22.6% | 27.7% | 9.2% |
| 5 | 1,520 | 62.4% | 65.6% | 39.5% |
| 10 | 3,040 | 69.1% | 71.7% | 53.9% |
| 20 | 6,080 | 75.8% | 78.2% | 63.2% |
| 40 | 12,158 | 81.1% | 83.2% | 71.7% |
| 80 | 24,311 | 84.5% | 86.2% | 79.6% |

About +6 points per doubling of writers and no plateau by 80. The first five writers do most of
the work (+40 points); the worst writer improves fastest of all, so writer variety matters more
than samples per writer (HOMUS has only 4-8 per class per writer). Simple distinctive shapes
(8th rest, cut time, 12/8, barline) saturate at 20-40 writers; the flag-count classes (16th /
32nd / 64th notes and rests), quarter rest and whole note are still climbing at 80 and sit at
55-72%, so they need both more writers and more pixels.

Rule of thumb for collecting the 29 classes with no real strokes: 10 writers gives a usable
first model (~70%), 40 a good one (~80%), and every doubling after that is worth ~5 points.

## 6. The conditioning vector (`granitanet_v5_ctx/`, `granitanet_v5_noctx/`, `context_effect.md`)

Nine features in staff spaces (`musicset.context`): valid flag, height above or below the
staff's middle line, position in the measure, distance to the nearest barline, whether a
notehead is near and its offset, width and height. Crops carry their real page context; synthetic
and HOMUS samples get one drawn from the training crops of their class with probability 0.7
(16 classes have no crops and never get one); 25% of training contexts are zeroed so the model
also works without one. Same data, same recipe; the twin has no context input.

| test split, 15,873 samples (8,853 with a known context) | with context | context masked | context-free twin |
|---|---|---|---|
| top-1 | **98.54%** | 97.81% | 97.88% |
| hand (held-out writers) | **87.63%** | 84.61% | 84.67% |
| stroke | 98.64% | 97.02% | 97.70% |
| crop | 99.77% | 99.31% | 99.54% |
| augmentation dot vs. staccato errors | **23** | 67 | 67 |
| fing1 vs. barline errors | **7** | 21 | 21 |
| staccatissimo vs. barline errors | 2 | 10 | 7 |
| whole note vs. augmentation dot errors | 7 | 18 | 17 |
| quarter rest vs. natural errors | 7 | 13 | 13 |
| quarter vs. half note errors | 27 | 27 | 26 |

The context removes two thirds of the shape-identical errors and lifts held-out handwriting by
3 points, while the masked column shows it costs nothing when absent. Quarter versus half note
does not move: that is a visual distinction (filled head), not a positional one. Only 56% of test
samples, and 28% of the handwritten ones, had a context at all, so these gains are a lower bound
for an editor that always sends one.
