# Released models

Weights and exports tracked in git on purpose (`!models/**` in `.gitignore`), so a checkout on
another machine has everything the Battuta extension needs. Training runs stay under
`checkpoints/` (ignored); `scripts/export_mobile.py` regenerates every file here.

## granitanet (training run `granitanet_v6_ctx`)

granitanet with the 9-feature conditioning vector, 61 classes, 2.0 M parameters, trained
2026-11-01 on `musicset symbols` (`-n 2000`, crops from built musicset score datasets capped at 500
per class, HOMUS with 80 training writers rendered twice, `--context-p 0.7`, seed 0), 15 epochs,
batch 256, context dropout 0.25. Conclusions and the full tables are in
[`../results/README.md`](../results/README.md), sections 6 to 8.

| test split (16,624 samples) | with context | context masked |
|---|---|---|
| top-1 | 98.68% | 97.94% |
| held-out handwriting (10 HOMUS writers never seen in training) | 88.95% | 86.05% |
| augmentation dot vs. staccato errors | 23 | 69 |
| fing1 vs. barline errors | 4 | 21 |

| file | what it is | checked against PyTorch |
|---|---|---|
| `granitanet.mlpackage/` | Core ML program, iOS 16+, fp16 weights (3.9 MB) | fp16: max abs. diff 1.5e-02 on the Mac (CPU), 2.4e-02 (CPU) and 3.3e-02 (every unit) on the iPad simulator over the 13 fixtures, the same top-1 on each |
| `granitanet.tflite` | LiteRT / TFLite, static batch 1 (7.7 MB) | max abs. diff 2.4e-06; 1.1e-05 on a Galaxy Tab S8 Ultra (LiteRT 2.3.2, CPU) over the fixtures |
| `granitanet.onnx` | ONNX opset 18, dynamic batch (7.8 MB) | max abs. diff 1.9e-06 |
| `granitanet.pt2` | `torch.export` program, dynamic batch (7.9 MB) | exact |
| `granitanet.json` | sidecar: class list, groups, input and output spec, stroke rasterisation rules, context features, the tflite's tensor names | |
| `fixtures.json` | 13 stroke sets on a staff with their context, rendered image and logits (`scripts/shell_fixtures.py`), for the mobile test shells in `apps/` | |
| `best_model.pt` | PyTorch checkpoint (weights only) for `GranitaClassifier.from_checkpoint` and re-export | |
| `history.json`, `eval_test.md`, `context_effect.md` | training curve and test tables | |

### Feeding the model from the editor

Two inputs, `input` `(N, 1, 32, 32)` float32 and `context` `(N, 9)` float32; one output,
`logits` `(N, 61)`, softmax for confidences. Core ML and ONNX keep those names; the tflite
has litert-torch's instead, signature `serving_default` with inputs `args_0` (the image)
and `args_1` (the context) and output `output_0` (the sidecar's `litert` block). The
sidecar spells out every rule; in short:

1. Rasterise the pen strokes on a 96 px canvas: centre the strokes' bounding box, keep the
   aspect ratio, make the longer side span `fraction = clamp(span / (8.8 * staff_space), 0.08, 0.9)`
   of the canvas, draw a uniform pen `0.05 * 96` px wide with round caps, no blur, then
   downsample to 32 x 32 bilinearly. Black ink on white paper (255).
2. Normalise: `x = (pixel / 255 - 0.5) / 0.5`.
3. Context (all in staff spaces, see `granita.context.context_vector` for the scaling): valid
   flag, centre height from the staff's middle line (down positive) / 6, position in the
   measure mapped to [-1, 1], distance to the nearest barline / 8, whether a notehead is within
   8 spaces, its offset / 8, width / 8, height / 8. Send all zeros when unknown; the model was
   trained for that too.

All three steps are implemented in Swift and in Kotlin in the test shells (`apps/ios`,
`apps/android`), checked against `fixtures.json` on an iPad and a Galaxy Tab.
