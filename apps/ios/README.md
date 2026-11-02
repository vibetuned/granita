# granita's iOS test shell

A small SwiftUI app that runs the released Core ML export
(`models/battuta/granitanet.mlpackage`) on an iPad or iPhone: draw a symbol on a
staff with the Pencil or a finger and the app shows the top five classes with their
confidences, the 32 × 32 image the model saw, the model's time, and the context it
derived from the strokes (height on the staff, place in the measure, size). *Use the
context* off sends the all-zero vector instead, to compare.

Every stroke's end classifies all the strokes since the last *Clear*, so a sharp can
take its four strokes.

*Add notes* then taps on the staff place noteheads (snapped to the lines and spaces; a
tap on a note takes it away, *Clear notes* all of them) with their stems, so the
context's note features are real: the symbol's offset from the centre of the nearest
note's body (notehead and stem, as granita measures every note) and whether it sits
above, below, left or right of that body. Two steppers change the size rule, to
calibrate it against natural-size writing:
how many staff spaces fill the canvas (granita's 3.0, which these steppers calibrated:
a symbol's longer side over that many spaces is its fraction of the canvas) and the
smallest fraction a symbol is drawn at (granita's 0.08: a floor lifts the small symbols
and leaves the large ones at their natural size). The status line shows the fraction
used.

## The reply, the dataset and training on the device

Under the five answers, tap the class the drawing **was** — one of the five, or
*Another…* for the whole list — and it is saved to `replies.jsonl` in the app's
Documents folder (the Files app shows it; *Share the replies* sends it) as one line in
musicset's contribution format (`musicset symbols --strokes`): `label`, `writer` (this
install's id), `staff_space`, the `strokes` with each sample's time and pressure, the
`context` facts, and the shell's own `staff`, `notes`, `fraction`, the model's
`predicted` answer, `device`, `app` and `time`. *Discard* drops the drawing; nothing is
saved. Every reply clears the staff for the next drawing; the notes stay.

*Train on N replies* retrains the model's last layer on this device: the Core ML
package's `personal` function takes that layer's weights as inputs (`head_w`, `head_b`,
the sidecar's `head` to start from) and gives back the 128 features the layer reads,
so the whole network still runs on the Neural Engine and the layer is a plain
`Linear(128 -> classes)` the app owns. The trainer (`Head.swift`) takes gradient steps
on a tiny replay buffer — the newest eight replies of each class, so one symbol
written fifty times does not drown the others — holds the layer to the **anchors**
(`anchors.json`: features the released model read with its own answers, the KL from
those answers on every anchor each epoch) and pulls the weights back toward the
original ones (an L2 anchor). Thirty epochs take well under a second. The trained
layer is saved (`Documents/head.json`), used from then on and shown in the status
line; *Reset the layer* puts the original back. The anchors shipped today are the 13
fixtures (an interim set); `scripts/export_anchors.py --data` makes the real one from
the dataset.

## What an editor would lift

- `StrokeRaster.swift`: the strokes to the model's image, the way granita's
  `render_strokes` does it (96 px canvas, natural size from the staff space, 5 px pen,
  no anti-aliasing, Pillow's bilinear reduction to 32 px re-implemented exactly).
- `SymbolContext.swift`: granita's `SymbolContext` and `context_vector`.
- `GranitaModel.swift`: the export loaded by URL through its `personal` function, fed
  `input`, `context` and the layer, read as `logits` and `features`.
- `Head.swift`: the layer, the replay buffer, the anchors and the trainer.
- `Annotations.swift`: the replies file, musicset's record.

## Building

The project is generated, not tracked:

```sh
brew install xcodegen          # once
cd apps/ios
xcodegen generate              # GranitaShell.xcodeproj
open GranitaShell.xcodeproj    # or:
xcodebuild -scheme GranitaShell -destination 'platform=iOS Simulator,name=iPad Pro 11-inch (M5)' test
```

The model, its sidecar and the anchors are read straight from `models/battuta`; Xcode
compiles the `.mlpackage` into the app (iOS 18 or later: the multi-function package's
floor — `MLModelConfiguration.functionName` picks `personal`; the fp16 program alone
needed 16). The signing team is Vibetuned's, as battuta's.

## The tests

`Tests/FixtureTests.swift` runs `models/battuta/fixtures.json` (made by
`scripts/shell_fixtures.py` with the PyTorch checkpoint) through the app:

- **the model**, fed the reference pixels and context, on the CPU and on every
  compute unit: the same top-1, logits within 0.1 (the weights are fp16);
- **the personal function**: its features are the reference's (within fp16), and the
  sidecar's layer applied to them is its logits;
- **the trainer**: the fixtures as replies stay right and near the anchors; ten
  replies calling the quarter rest an arpeggio make it one, a reset undoes that;
- **the replies file**: a record written and read back, musicset's keys in the file;
- **the context** derived from the strokes: the reference vector;
- **the rasteriser**: the size rule exactly, and the model's answer from the app's
  own drawing the same as from Pillow's. The drawing is a port of Pillow's own
  (`StrokeRaster.Paper`, from its `Draw.c`: the wide line's polygon with its
  whole-pixel offsets, the scan conversion, the joint's pie slice; the dots'
  ellipses approximated) — Core Graphics drew every stroke fatter and the quarter
  rest fixture read as an arpeggio (2026-11-02). The pixels are still only
  guarded against a gross error (mean difference under 10 grey levels, fewer than
  half the ink pixels far off for a glyph of more than 16), the exact numbers
  printed: on the Mac they agree with Pillow to a pixel or two, except where
  `fixtures.json` rounds the fraction (the sharp, fingering 3).

The numbers print in the test log (`coreml units …`, one line per fixture).
