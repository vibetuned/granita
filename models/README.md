# Released models

Weights and exports tracked in git on purpose (`!models/**` in `.gitignore`), so a checkout on
another machine has everything the Battuta extension and the OMR pipeline need. Training runs
stay under `checkpoints/` (ignored); `scripts/export_mobile.py` regenerates every file here and
`scripts/shell_fixtures.py` the battuta fixtures. Conclusions and the full tables are in
[`../results/README.md`](../results/README.md), section 10.

| folder | for | classes | context | training run |
|---|---|---|---|---|
| [`battuta/`](battuta/) | the Battuta editor's stylus input | 43 | 16 features | `battuta_v8_ctx` |
| [`kakigori/`](kakigori/) | the OMR pipeline's second stage (`GranitaClassifier.predict_box`) | 56 | none | `kakigori_v8` |

Both are granitanet, 2.0 M parameters, trained 2026-11-02 for 15 epochs, batch 256, on
`musicset symbols --profile <variant> -n 2000` with crops from built musicset score datasets
capped at 500 per class (`--context-p 0.7`, seed 0).

## battuta (training run `battuta_v8_ctx`)

The editor's stylus set: fingerings, ornaments, articulations, flat / natural / sharp, ten
dynamics, whole to 16th notes, the augmentation dot, the quarter rest, the three clefs and
the navigation marks D.S., D.C., Fine, segno and coda. Trained with HOMUS pen strokes (80
training writers rendered twice, codas composed from each writer's own strokes), EMNIST
handwriting at 1,000 words or signs per class, stylus stroke templates and context dropout
0.25.

| test split (13,655 samples) | with context | context masked |
|---|---|---|
| top-1 | 99.30% | 98.61% |
| held-out handwriting (10 HOMUS writers never seen in training) | 92.43% | 90.27% |
| EMNIST handwriting (EMNIST's test writers) | 99.65% | 99.65% |
| augmentation dot vs. staccato errors | 20 | 87 |
| fing1 vs. staccatissimo errors | 4 | 10 |

The handwriting number is not comparable with the previous 51-class release: on the classes
both share they tie, and the test writers differ between the two builds (results, section 10).

| file | what it is | checked against PyTorch |
|---|---|---|
| `granitanet.mlpackage/` | Core ML program, fp16 weights (3.9 MB), **two functions** since 2026-11-02: `infer` (the default: `input`, `context` → `logits`, as before) and `personal` (`input`, `context`, `head_w`, `head_b` → `logits`, `features`), which makes it iOS 18+ | on this Mac: max abs. diff 1.3e-02 on both functions (CPU); the `apps/ios` tests on a device: the previous release 3.6e-02 to 4.5e-02 on the simulator, the same top-1 on every fixture |
| `granitanet.tflite` | LiteRT / TFLite, static batch 1 (7.7 MB); **to be re-exported on Linux** with the `personal` signature next to `serving_default` (`scripts/export_mobile.py --formats litert`, which also writes the signatures' tensor names into the sidecar) — the file here is the single-signature one still | max abs. diff 1.6e-06; re-run the `apps/android` tests on a device |
| `granitanet.onnx` | ONNX opset 18, dynamic batch (7.8 MB) | max abs. diff 1.4e-06 |
| `granitanet.pt2` | `torch.export` program, dynamic batch (7.9 MB) | exact |
| `granitanet.json` | sidecar: class list, groups, input and output spec, stroke rasterisation rules, context features and the note-body definition, the tflite's tensor names (per signature, once re-exported), and `head`: the last layer's weights `(43 × 128)` and bias, what an app's `personal` copy starts from | |
| `anchors.json` | feature vectors the model read with its own logits, `label` an index into `classes` (`scripts/export_anchors.py`): what the shells' on-device trainer holds the retrained layer to. **Interim: the 13 fixtures**; the real set comes from the dataset's val split, K per class (`--data`), on the machine that has it | |
| `fixtures.json` | 13 stroke sets on a staff, six beside a note, with the note's body, their context, rendered image and logits (`scripts/shell_fixtures.py`), for the mobile test shells in `apps/`. The `barline` and `half rest` fixtures now read as `fing1` and `articTenutoAbove`: battuta has neither class, and the fixtures pin numbers, not accuracy | |
| `best_model.pt` | PyTorch checkpoint (weights only) for `GranitaClassifier.from_checkpoint` and re-export | |
| `history.json`, `eval_test.md`, `eval_tablet.md`, `context_effect.md` | training curve, test tables, the stylus-only benchmark | |

### Feeding the model from the editor

Two inputs, `input` `(N, 1, 32, 32)` float32 and `context` `(N, 16)` float32; one output,
`logits` `(N, 43)`, softmax for confidences. Core ML and ONNX keep those names; the tflite
has litert-torch's instead, signature `serving_default` with inputs `args_0` (the image)
and `args_1` (the context) and output `output_0` (the sidecar's `litert` block). The
sidecar spells out every rule; in short:

1. Rasterise the pen strokes on a 96 px canvas: centre the strokes' bounding box, keep the
   aspect ratio, make the longer side span `fraction = clamp(span / (3.0 * staff_space), 0.08, 0.9)`
   of the canvas, draw a uniform pen `0.05 * 96` px wide with round caps, no blur, then
   downsample to 32 x 32 bilinearly. Black ink on white paper (255). The 3.0 was
   calibrated by hand on an iPad and a Galaxy Tab with the test shells (2026-11-01): at
   musicset's HOMUS value, 8.8, symbols written at their natural size came out a few
   pixels tall, far below the sizes fonts, templates and EMNIST words are trained at.
2. Normalise: `x = (pixel / 255 - 0.5) / 0.5`.
3. Context, all in staff spaces with `y` positive downwards from the staff's middle line.
   The note features describe the nearest note's **body**: the bounding box of its
   notehead, stem and flag, or of every notehead plus the stem and flag of its chord,
   without accidentals, dots or articulations (the note before the mark is added). `dx`,
   `dy` are the symbol's centre minus the body's centre, `w`, `h` the body's size, and a
   note counts when its body is within 8 spaces of the symbol's centre (see
   `granita.context.context_vector`). In this order:
   `valid` (1), `in_staff` (|y| ≤ 2), `above_staff` (y < -2), `below_staff` (y > 2),
   `y / 6` clipped to ±2, measure position mapped to [-1, 1], barline distance / 8 clipped to
   [0, 1], `has_note`, `note_above` and `note_below` (dy < -h/2 or dy > h/2, with
   |dx| ≤ w/2 + 0.5), `note_left` and `note_right` (dx < -w/2 or dx > w/2, with
   |dy| ≤ h/2 + 0.5), `dx / 8` and `dy / 8` clipped to ±1, width / 8 and height / 8 of the
   symbol clipped to [0, 1.5]. Send all zeros when unknown; the model was trained for that
   too.

All three steps are implemented in Swift and in Kotlin in the test shells (`apps/ios`,
`apps/android`), whose placed notes are noteheads with a 3.5-space stem (the nearest
counts), and are checked against `fixtures.json`. Regenerating `fixtures.json` on another
machine reproduces the pixels and contexts exactly and the logits to within 1e-05.

### The `personal` function: the last layer in the app's hands

Since 2026-11-02 the exports carry a second entry point for an app that trains the last
layer on its users' replies without touching the network: the Core ML package's function
`personal` (`MultiFunctionDescriptor`, the default function `infer` unchanged) and the
tflite's signature of that name (`litert_torch.signature`, next to `serving_default`).
It takes `head_w` `(43, 128)` and `head_b` `(43)` with the image and the context, and
gives `logits` `(1, 43)` and `features` `(1, 128)` — what that layer read. Fed the
sidecar's `head` it computes exactly what `infer` does; fed a trained copy, the app's
own classifier, the whole network still on the accelerator (`HeadInputs` in
`granita.models.export`, `GranitaNet.features`). The shells train that copy with plain
gradient steps on a small replay buffer of the replies, the anchors of `anchors.json`
holding the layer to the released model's answers and an L2 term to its weights
(`apps/ios/Sources/Head.swift`, `apps/android/.../Head.kt`). `fixtures.json` carries each
fixture's `features` for the shells' tests.

## kakigori (training run `kakigori_v8`)

Every catalog class (the battuta set plus the double accidentals, `dynamicMezzo`,
`dynamicZ`, `pedal`, 32nd and 64th notes, every rest and `barlineSingle`), trained on SMuFL
font renders and score crops only: no stroke templates, no handwriting, no context input.

| test split (13,742 samples) | |
|---|---|
| top-1 | 99.57% |
| macro F1 | 99.59% |
| score crops | 98.75% |
| top confusions | staccato to dot 25, staccatissimo to barline 10 |

Both top confusions are the ones the battuta context resolves; the OMR pipeline computes no
context yet, so this model goes without.

| file | what it is | checked against PyTorch |
|---|---|---|
| `granitanet.onnx` | ONNX opset 18, dynamic batch (7.8 MB), one input `input` `(N, 1, 32, 32)`, output `logits` `(N, 56)` | max abs. diff 1.8e-06 |
| `granitanet.pt2` | `torch.export` program, dynamic batch (7.8 MB) | exact |
| `granitanet.json` | sidecar: class list, groups, input and output spec | |
| `best_model.pt` | PyTorch checkpoint for `GranitaClassifier.from_checkpoint(...).predict_box` | |
| `history.json`, `eval_test.md` | training curve and test tables | |

`predict_box(page, box)` crops a detection box with the same protocol musicset used for the
training crops (`crop_glyph`, a port of musicset's `_crop_box_to_patch`).
