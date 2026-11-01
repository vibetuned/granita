# Released models

Weights and exports tracked in git on purpose (`!models/**` in `.gitignore`), so a checkout on
another machine has everything the Battuta extension needs. Training runs stay under
`checkpoints/` (ignored); `scripts/export_mobile.py` regenerates every file here and
`scripts/shell_fixtures.py` the fixtures.

## granitanet (training run `granitanet_v7_ctx`)

granitanet with the 16-feature conditioning vector, 51 classes, 2.0 M parameters, trained
2026-11-01 on `musicset symbols` (`-n 2000`, crops from built musicset score datasets capped at
500 per class, HOMUS with 80 training writers rendered twice, EMNIST handwriting for the
fingerings and dynamics at 1,000 words per class, `--context-p 0.7`, seed 0), 15 epochs,
batch 256, context dropout 0.25. The time-signature classes of the previous release are gone.
Conclusions and the full tables are in [`../results/README.md`](../results/README.md),
sections 6 to 9.

| test split (15,546 samples) | with context | context masked |
|---|---|---|
| top-1 | 98.59% | 97.76% |
| held-out handwriting (10 HOMUS writers never seen in training) | 86.95% | 83.91% |
| handwritten fingerings and dynamics (EMNIST's test writers) | 98.79% | 98.19% |
| augmentation dot vs. staccato errors | 25 | 77 |
| fing1 vs. barline errors | 6 | 25 |

| file | what it is | checked against PyTorch |
|---|---|---|
| `granitanet.mlpackage/` | Core ML program, iOS 16+, fp16 weights (3.9 MB) | fp16: max abs. diff 3.6e-02 (CPU) and 4.5e-02 (every unit) on the iPad simulator over the 13 fixtures, with and without context, the same top-1 on each (`apps/ios` tests, 2026-11-01) |
| `granitanet.tflite` | LiteRT / TFLite, static batch 1 (7.7 MB) | max abs. diff 1.3e-06; 7.6e-06 on a Galaxy Tab S8 Ultra (LiteRT 2.3.2, CPU) over the fixtures (`apps/android` tests, 2026-11-01) |
| `granitanet.onnx` | ONNX opset 18, dynamic batch (7.8 MB) | max abs. diff 2.3e-06 |
| `granitanet.pt2` | `torch.export` program, dynamic batch (7.9 MB) | exact |
| `granitanet.json` | sidecar: class list, groups, input and output spec, stroke rasterisation rules, context features, the tflite's tensor names | |
| `fixtures.json` | 13 stroke sets on a staff, six beside a notehead, with their context, rendered image and logits (`scripts/shell_fixtures.py`), for the mobile test shells in `apps/` | |
| `best_model.pt` | PyTorch checkpoint (weights only) for `GranitaClassifier.from_checkpoint` and re-export | |
| `history.json`, `eval_test.md`, `context_effect.md` | training curve and test tables | |

### Feeding the model from the editor

Two inputs, `input` `(N, 1, 32, 32)` float32 and `context` `(N, 16)` float32; one output,
`logits` `(N, 51)`, softmax for confidences. Core ML and ONNX keep those names; the tflite
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
3. Context, all in staff spaces with `y` positive downwards from the staff's middle line and
   `dx`, `dy` the symbol centre minus the nearest notehead's centre within 8 spaces (see
   `granita.context.context_vector`), in this order:
   `valid` (1), `in_staff` (|y| ≤ 2), `above_staff` (y < -2), `below_staff` (y > 2),
   `y / 6` clipped to ±2, measure position mapped to [-1, 1], barline distance / 8 clipped to
   [0, 1], `has_note`, `note_above` and `note_below` (|dx| < 1 and dy ≤ -0.5 / ≥ 0.5),
   `note_left` and `note_right` (dx ≤ -0.5 / ≥ 0.5), `dx / 8` and `dy / 8` clipped to ±1,
   width / 8 and height / 8 clipped to [0, 1.5]. Send all zeros when unknown; the model was
   trained for that too.

All three steps are implemented in Swift and in Kotlin in the test shells (`apps/ios`,
`apps/android`), checked against `fixtures.json` with this release on an iPad and a Galaxy
Tab: the 16-feature context vectors match exactly, the models match PyTorch as the table
says, and the drawing comes out close to Pillow's — Android's Canvas within a grey level
on every fixture, Core Graphics less closely (each app's README and test output say how
much). Regenerating `fixtures.json` on another machine reproduces the pixels and contexts
exactly and the logits to within 1e-05 (float noise in the last rounded digit).
