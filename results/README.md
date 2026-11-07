# Granita results

Every training run writes its tables next to its weights under `checkpoints/<run>/`
(gitignored); `uv run python scripts/collect_results.py --exclude ablation_smoke smoke_run`
copies the tables here and regenerates [INDEX.md](INDEX.md). Weights are never copied.
All runs: granitanet unless stated, 15 epochs, batch 256, 2 warmup epochs, seed 42.

Dataset builds are produced by `musicset symbols` (names below are the output directories of those builds):

| build | classes | train / val / test | notes |
|---|---|---|---|
| `symbols.v1-leaky` | 33 | 57,679 / 7,212 / 7,212 | per-image split; 208 test images were byte-copies of train images, almost all fingerings |
| `symbols.v2-33cls` | 33 | 65,648 / 8,210 / 8,218 | score-disjoint crops, cross-split dedup, source codes |
| `symbols.v6-61cls` | 61 | 145,083 / 16,625 / 16,624 | + HOMUS pen strokes (80 / 10 / 10 writers, `w_<split>` writer ids), notes, rests, clefs, time signatures, barline, dot; sections 2 to 6 used the first build of this recipe (139,087 / 15,871 / 15,873), section 7 the one with stemmed-note and time-signature crops |
| `symbols-brush` | 61 | same splits | HOMUS training strokes rendered as a pressure brush half the time, pen 0.05-0.13 |
| `symbols-tablet.v6-61cls` | 61 | 0 / 0 / 18,276 | synthetic stylus-only benchmark, `--preset tablet --seed 1` |
| `symbols.v7-51cls` | 51 | 133,742 / 15,609 / 15,546 | time signatures dropped; context v2 (16 features); + EMNIST handwriting for the fingerings and dynamics (1,000 words per class for train + val, 1,822 test words from EMNIST's test writers, source `emnist`) |
| `symbols-tablet.v7-51cls` | 51 | 0 / 0 / 15,280 | synthetic stylus-only benchmark, `--preset tablet --seed 1` |
| `symbols-battuta` | 43 | 115,310 / 13,680 / 13,655 | `--profile battuta`: the editor's stylus set; + D.S., D.C., Fine, segno, coda; EMNIST words and signs for 22 classes; HOMUS codas; context v3 (note bodies) |
| `symbols-battuta-tablet` | 43 | 0 / 0 / 12,862 | stylus-only benchmark for battuta, `--preset tablet --seed 1` |
| `symbols-battuta-fornes` | 43 | 120,417 / 14,176 / 14,198 | `symbols-battuta` + the Fornés scans (`--fornes Music_Symbols.zip --fornes-copies 2`, source `fornes`, writer-disjoint: 17 / 3 / 3 writers); every other sample is `symbols-battuta`'s but for 22 repeated images whose surviving copy carries another context |
| `symbols-battuta-replies` | 43 | 115,590 / 13,680 / 13,655 | `symbols-battuta` + the 140 cleaned shell replies (`--strokes replies/clean.jsonl --strokes-train`: all in train, rendered twice at the size the shells showed them, source `hand`, writers 1000 / 1001); val and test are `symbols-battuta`'s but for 16 repeated images whose surviving copy carries another context |
| `symbols-battuta-replies-android`, `-ipad` | 43 | 115,450 / 13,680 / 13,655 | the same with one device's 70 replies, for the cross-device test |
| `symbols-kakigori` | 56 | 109,892 / 13,742 / 13,742 | `--profile kakigori`: every class, fonts and score crops only, no stroke templates |

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

## 7. Page crops for stemmed notes and time signatures (`granitanet_v6_ctx/`, `granitanet_v6_noctx/`, `context_effect_v6.md`)

musicset now also crops standalone notes (notehead + stem + flag, unbeamed, not in chords;
duration from the `dur` attribute, present on 52% of the notes in the built datasets, else from the flag glyph and
stem) and stacked time signatures (digit rows under `meterSig`). Every class has a context pool
and the handwritten test samples with a known context rose from 28% to 69%. The HOMUS val and
test images are now rendered with the editor's pen width 0.05 (they were 0.085; the sweep in
section 3 predicted the gain). The test split gained 751 crops, so absolute numbers are not
comparable with section 6; the twins are.

| test split, 16,624 samples (12,543 with a known context) | with context | context masked | context-free twin |
|---|---|---|---|
| top-1 | **98.68%** | 97.94% | 98.06% |
| hand (held-out writers) | **88.95%** | 86.05% | 86.25% |
| stroke | 99.05% | 96.61% | 97.29% |
| crop | 99.49% | 99.07% | 99.28% |
| augmentation dot vs. staccato errors | **23** | 69 | 59 |
| fing1 vs. barline errors | **4** | 21 | 21 |
| staccatissimo vs. barline errors | 5 | 11 | 8 |
| whole note vs. augmentation dot errors | 1 | 5 | 7 |
| quarter rest vs. natural errors | 5 | 8 | 8 |
| quarter vs. half note errors | 22 | 27 | 29 |

Same picture as section 6 with more of the data covered: +0.6 top-1 and +2.7 points on held-out
handwriting from the context, the shape-identical pairs down by two thirds to four fifths.
`granitanet_v6_ctx` was the released model until section 9.

## 8. Export pipeline (`granitanet_v5_ctx`)

`scripts/export_mobile.py` exports one checkpoint to Core ML, LiteRT, ONNX and pt2 with the
editor sidecar and checks each against PyTorch. For `granitanet_v5_ctx` (two inputs, `input`
and `context`):

| format | size | max abs. difference vs. PyTorch |
|---|---|---|
| Core ML `.mlpackage` | 3.90 MB | not checkable on Linux (fp16 weights by default) |
| LiteRT `.tflite` | 7.71 MB | 3.3e-06 |
| ONNX | 7.84 MB | 2.9e-06 |
| pt2 | 7.89 MB | 0 |

## 9. Time signatures out, EMNIST in, context v2 (`granitanet_v7_ctx/`, `granitanet_v7_noctx/`, `context_effect_v7.md`)

Three data changes after testing the released model in the editor, then the same recipe as
section 7 (15 epochs, batch 256, context dropout 0.25, context-free twin):

- The ten time-signature classes are gone: stylus marks kept landing in those digit stacks.
  51 classes; indices up to `cClef` are unchanged, `barlineSingle` is now 50.
- EMNIST (`byclass`) digits and letters composed into handwritten fingerings `1`–`5` and
  dynamics `p`, `mp`, `sfz`, ... (source `emnist`): 1,000 words per class for train + val from
  EMNIST's train file, 1,822 test words from its test file, i.e. other writers. Until now the
  17 fingering and dynamic classes had no real handwriting at all (section 4).
- Context v2, 16 features: the v1 continuous ones plus explicit flags for the staff zone (in,
  above or below the staff) and, when a notehead is within 8 spaces, whether the symbol sits in
  its column above or below it, or beside it to the left or right. On the real crops: rests are
  inside the staff 100% of the time, fingerings above it 95–98%, dynamics and pedal below it
  90–94%, the augmentation dot right of its note 97%, the sharp left of it 97%, staccato above
  or below the note 55 / 45.

| test split, 15,546 samples (11,677 with a known context) | with context | context masked | context-free twin |
|---|---|---|---|
| top-1 | **98.59%** | 97.76% | 97.88% |
| macro F1 | **98.57%** | 97.67% | 97.81% |
| hand (held-out HOMUS writers, 1,119) | **86.95%** | 83.91% | 82.84% |
| emnist (held-out EMNIST writers, 1,822) | **98.79%** | 98.19% | 98.30% |
| stroke | 98.85% | 97.15% | 97.97% |
| crop | 99.71% | 98.84% | 99.21% |
| font | 99.69% | 99.25% | 99.33% |
| augmentation dot vs. staccato errors | **25** | 77 | 61 |
| fing1 vs. barline errors | **6** | 25 | 20 |
| staccatissimo vs. barline errors | 10 | 16 | 7 |
| whole note vs. augmentation dot errors | 2 | 9 | 10 |
| quarter rest vs. natural errors | 1 | 2 | 2 |
| quarter vs. half note errors | 19 | 19 | 23 |

The context is worth +0.7 top-1 and **+4.1 points on held-out handwriting** (v6: +2.7), and
again removes two thirds of the dot / staccato and fing1 / barline confusions. On the
stylus-only benchmark `symbols-tablet`, which has no context, the context model runs masked
and loses a quarter point on strokes (96.96% vs. 97.21% for the twin; 98.74% vs. 98.82% top-1).

The headline handwriting number fell from 88.95% (section 7) to 86.95%, but that is the test
set, not the model: the 400 time-signature strokes were the easiest part of the HOMUS test
split. Scored on the 51 shared classes only, each model on its own test split:

| 51 shared classes, test | v6 ctx (61 classes) | v7 ctx (51 classes) |
|---|---|---|
| top-1 | 98.55% | **98.59%** |
| hand (1,120 / 1,119 strokes) | 86.70% | **86.95%** |
| stroke | **99.05%** | 98.85% |
| crop | 99.39% | **99.71%** |
| font | **99.76%** | 99.69% |

So on held-out data the time signatures were not pulling the other classes down (v6 sent only
2 shared-class test samples to a time-signature class); what the editor showed was stylus marks
landing in classes that no longer exist. EMNIST words score 98.8% on unseen writers (98.2%
masked: the context barely matters for them), which makes the fingerings and dynamics usable by
pen for the first time. Weakest classes by pen are still the flag counts (64th 66%, 32nd 68%,
16th 73%, the 32 px resolution limit) and the 32nd rest (85%).

`granitanet_v7_ctx` was the released model until section 10; the exports in section 8
were regenerated from it (LiteRT 1.3e-06, ONNX 2.3e-06, pt2 exact vs. PyTorch).

## 10. Two variants, navigation marks, context from the note body (`battuta_v8_ctx/`, `battuta_v8_noctx/`, `kakigori_v8/`, `context_effect_v8.md`)

Granita is now two models, one per consumer, built from musicset's `--profile` class sets:

- **battuta** (43 classes), the editor's stylus model. Every rest but the quarter is gone
  (the editor infers a rest's duration from the measure), as are the double accidentals,
  `dynamicMezzo` and `dynamicZ` (only ever parts of mp, mf, sfz), `pedal`, `note32nd`,
  `note64th` and `barlineSingle`.
- **kakigori** (56 classes), the whole catalog for the OMR pipeline, trained on fonts and
  score crops only, without context (the OMR pipeline computes none yet).

Both gained five navigation marks: D.S., D.C. and Fine (SMuFL glyphs or Verovio's text
font, `dir` crops whose whole text is the mark, EMNIST letters with drawn periods), segno
and coda (SMuFL glyphs, `repeatMark` crops). Handwriting for the signs is built from
EMNIST letters: a coda is an `O` crossed by an `X` turned upright, a segno a capital `S`
slashed by a `1` with two dots, a turn a capital `S` mirrored and laid on its side (a
plain `S` on its side matches the inverted turn, IoU 0.58 against 0.39). Every HOMUS writer
also draws codas from their own whole-note ovals and barlines, so the pen-stroke test
writers cover the coda too. Lowercase EMNIST letters now keep their x-height and
descenders inside words, so `mp` no longer reads as `MP`.

The context's note features now measure from the nearest note's **body**, the box of its
notehead, stem and flag, or of all the noteheads and the stem of its chord, without
accidentals, dots or articulations. Before, they measured from the nearest notehead, so
the same staccato read differently on a stem-up and a stem-down note. The above / below /
left / right flags now test the body's edges. On real crops: augmentation dot right of the
body 94%, sharp left 93%, arpeggio left 83%, fingering above 89%, staccato above or below
54 / 45.

| battuta test split, 13,655 samples (10,220 with a known context) | with context | context masked | context-free twin |
|---|---|---|---|
| top-1 | **99.30%** | 98.61% | 98.85% |
| macro F1 | **99.27%** | 98.43% | 98.73% |
| hand (10 held-out HOMUS writers, 740) | **92.43%** | 90.27% | 90.68% |
| emnist (EMNIST's test writers, 2,288) | 99.65% | 99.65% | 99.78% |
| stroke | 99.21% | 96.99% | 99.05% |
| crop | 99.76% | 98.53% | 99.17% |
| font | 99.78% | 99.43% | 99.26% |
| augmentation dot vs. staccato errors | **20** | 87 | 64 |
| whole note vs. augmentation dot errors | 4 | 16 | 13 |
| fing1 vs. staccatissimo errors | 4 | 10 | 8 |
| quarter vs. half note errors | 19 | 18 | 20 |
| 8th vs. 16th note errors | 20 | 20 | 25 |

| kakigori test split, 13,742 samples (no context) | |
|---|---|
| top-1 | 99.57% |
| macro F1 | 99.59% |
| font | 99.76% |
| crop | 98.75% |
| top confusions | staccato to dot 25, staccatissimo to barline 10, dot to staccato 6 |

The five navigation marks score 99.9-100% F1 in both variants, on every source: fonts,
crops, EMNIST test writers and the held-out HOMUS writers' codas (60 / 60). D.S. has the
fewest crops (5 in test, 59 in all).

Read the headline handwriting number with care. Battuta's 92.43% against v7's 86.95%
(section 9) is mostly the class set. Scored on the 38 classes both share, answering only
among battuta's classes, the two context models tie:

| 38 shared classes, test | v7 ctx (51 classes) | v8 battuta ctx |
|---|---|---|
| top-1 | 99.10% | 99.21% |
| hand | 92.19% (679) | 91.76% (680) |
| crop | 99.69% | 99.73% |
| emnist | 99.26% | 99.60% |

Even that tie is only indicative. **The HOMUS test writers differ between the two builds**
(v7: 7, 12, 14, 29, 41, 53, 54, 82, 86, 97; v8: 19, 29, 46, 65, 69, 76, 78, 80, 86, 90):
musicset draws the writer split from a random stream that the steps before it also use,
so every change upstream reshuffles the writers. Per-writer accuracy spans 80-95%
(section 5), so hand numbers compare only within one build. Within this build the context
adds 1.75 points of held-out handwriting (92.43 vs. 90.68) and cuts the dot / staccato
confusion by two thirds, as before.

On the stylus-only benchmark `symbols-battuta-tablet`, which has no context, the context
model runs masked and trails its twin: 99.00% vs. 99.27% top-1, 97.09% vs. 98.50% on stroke
templates. The editor always sends a context, so this only matters for callers that send
none.

`battuta_v8_ctx` was the released battuta model (`models/battuta/`) until 2026-11-07
(section 12), `kakigori_v8` is the kakigori one (`models/kakigori/`).

## 11. Fornés scanned clefs and accidentals (`battuta_v9_fornes_ctx/`)

musicset gained a sixth source, `fornes`: the CVC *Music Symbols* dataset (Fornés,
Lladós & Sánchez, 2008), 4,094 scans of handwritten flats, naturals, sharps, double
sharps and treble, bass and alto clefs. Battuta keeps six of the seven classes (no
double sharp): 6,146 samples, the train scans rendered twice, their strokes thickened
to a pen (the scans' fine lines faded at 32 px), split by writer with the accidental
writers and the clef-only writers dealt separately so val and test hold both. The scans
draw from their own random stream, so this build is v8's plus the scans: same HOMUS test
writers, same everything else. Same recipe as `battuta_v8_ctx`, the data the only change.

| test split of `symbols-battuta-fornes` | samples | v8 ctx | v9 + Fornés |
|---|---|---|---|
| all | 14,198 | 99.11% | 99.30% |
| fornes (3 held-out writers) | 543 | 93.92% | 99.45% |
| hand (10 held-out HOMUS writers) | 740 | 92.43% | 92.43% |
| hand, the six Fornés classes | 240 | 100.00% | 99.58% |
| font / stroke / crop / emnist | | 99.78 / 99.29 / 99.76 / 99.65% | 99.73 / 99.45 / 99.80 / 99.61% |
| every class but the six | 11,905 | 99.21% | 99.20% |
| `symbols-battuta-tablet` (stylus-only, context masked) | 12,862 | 99.00% | 99.01% |

Fornés teaches the model to read old-score handwriting of those six classes (+5.5 points
on unseen Fornés writers) and changes nothing for the editor: the held-out HOMUS writers'
clefs and accidentals were already all read right, their errors are notes, rests and
dots, which Fornés does not have. One run each; the other differences are noise. Not
released: `models/battuta/` stays `battuta_v8_ctx`.

## 12. The shells' replies (`battuta_v10_replies_ctx/`, `battuta_v10_replies_noctx/`, `battuta_v10_android_ctx/`, `battuta_v10_ipad_ctx/`)

140 replies recorded with the test shells on 2026-11-02, one writer per device (70 with an S
Pen on a Galaxy Tab, 70 with an Apple Pencil on an iPad), cleaned with
`scripts/replies_viewer.py` (two relabelled, none dropped), mostly note values (58 8th, 26
quarter, 23 16th, 22 half). They go to train only (`--strokes-train`, off the shared random
stream, so the HOMUS writers and every other sample stay `symbols-battuta`'s) and train at the
fraction the shells recorded: musicset's staff-space rule (HOMUS's 8.8 spaces to the canvas)
would have drawn them at a median 0.33-0.36 of the canvas, where the shells show the model
0.9 (the 3.0-space editor rule, mostly at its cap). Scored the way the shells feed the model,
each reply at its recorded fraction with its recorded context:

| model | Galaxy Tab replies (70) | iPad replies (70) |
|---|---|---|
| v8 (`battuta_v8_ctx`) | 28.6% | 27.1% |
| trained with the Galaxy Tab replies | *92.9%* | **90.0%** |
| trained with the iPad replies | **87.1%** | *95.7%* |
| trained with both (`battuta_v10_replies_ctx`, released) | *95.7%* | *98.6%* |
| the same without context | *94.3%* | *98.6%* |

Bold: the other device's replies, never seen in training; italics: replies the model trained
on. v8's errors were almost all notes (8th read as half or quarter, quarter as half); what is
left is mostly quarter notes read as half. It is one hand on two devices, so the bold numbers
measure the editor's input from a known writer, not new writers.

On everything else the release holds (one run each):

| | v8 | v10 |
|---|---|---|
| test top-1 (13,655 samples, the same but for 11 repeated images with another context) | 99.30% | 99.27% |
| hand: 10 held-out HOMUS writers (740) | 92.43% | 91.76% (12 lost, 7 gained; 6 of the lost 16th notes read as 8th) |
| emnist | 99.65% | 99.74% |
| context masked / context-free twin | 98.61% / 98.85% | 98.70% / 98.76% |
| `symbols-battuta-tablet` (stylus-only) | 99.00% | 98.98% |

`battuta_v10_replies_ctx` is the released battuta model (`models/battuta/`) since 2026-11-07.

