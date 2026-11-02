# granita's iOS test shell

A small SwiftUI app that runs the released Core ML export
(`models/battuta/granitanet.mlpackage`) on an iPad or iPhone: draw a symbol on a
staff with the Pencil or a finger and the app shows the top five classes with their
confidences, the 32 × 32 image the model saw, the model's time, and the context it
derived from the strokes (height on the staff, place in the measure, size). *Use the
context* off sends the all-zero vector instead, to compare.

Every stroke's end classifies all the strokes since the last *Clear*, so a sharp can
take its four strokes.

*Add a note* then a tap on the staff places a notehead (snapped to the lines and
spaces; *Remove the note* takes it away) with its stem, so the context's note features
are real: the symbol's offset from the centre of the note's body (notehead and stem, as
granita measures every note) and whether it sits above, below, left or right of that
body. Two steppers change the size rule, to calibrate it against natural-size writing:
how many staff spaces fill the canvas (granita's 3.0, which these steppers calibrated:
a symbol's longer side over that many spaces is its fraction of the canvas) and the
smallest fraction a symbol is drawn at (granita's 0.08: a floor lifts the small symbols
and leaves the large ones at their natural size). The status line shows the fraction
used.

## What an editor would lift

- `StrokeRaster.swift`: the strokes to the model's image, the way granita's
  `render_strokes` does it (96 px canvas, natural size from the staff space, 5 px pen,
  no anti-aliasing, Pillow's bilinear reduction to 32 px re-implemented exactly).
- `SymbolContext.swift`: granita's `SymbolContext` and `context_vector`.
- `GranitaModel.swift`: the export loaded by URL, fed `input` and `context`, read as
  `logits`.

## Building

The project is generated, not tracked:

```sh
brew install xcodegen          # once
cd apps/ios
xcodegen generate              # GranitaShell.xcodeproj
open GranitaShell.xcodeproj    # or:
xcodebuild -scheme GranitaShell -destination 'platform=iOS Simulator,name=iPad Pro 11-inch (M5)' test
```

The model and its sidecar are read straight from `models/battuta`; Xcode compiles
the `.mlpackage` into the app (iOS 16 or later, the fp16 program's floor). The signing
team is Vibetuned's, as battuta's.

## The tests

`Tests/FixtureTests.swift` runs `models/battuta/fixtures.json` (made by
`scripts/shell_fixtures.py` with the PyTorch checkpoint) through the app:

- **the model**, fed the reference pixels and context, on the CPU and on every
  compute unit: the same top-1, logits within 0.1 (the weights are fp16);
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
