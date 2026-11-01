# granita's iOS test shell

A small SwiftUI app that runs the released Core ML export
(`models/granitanet/granitanet.mlpackage`) on an iPad or iPhone: draw a symbol on a
staff with the Pencil or a finger and the app shows the top five classes with their
confidences, the 32 × 32 image the model saw, the model's time, and the context it
derived from the strokes (height on the staff, place in the measure, size). *Use the
context* off sends the all-zero vector instead, to compare.

Every stroke's end classifies all the strokes since the last *Clear*, so a sharp can
take its four strokes.

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

The model and its sidecar are read straight from `models/granitanet`; Xcode compiles
the `.mlpackage` into the app (iOS 16 or later, the fp16 program's floor). The signing
team is Vibetuned's, as battuta's.

## The tests

`Tests/FixtureTests.swift` runs `models/granitanet/fixtures.json` (made by
`scripts/shell_fixtures.py` with the PyTorch checkpoint) through the app:

- **the model**, fed the reference pixels and context, on the CPU and on every
  compute unit: the same top-1, logits within 0.1 (the weights are fp16);
- **the context** derived from the strokes: the reference vector;
- **the rasteriser**: the size rule exactly; the drawing close to Pillow's (Core
  Graphics rasterises a wide line differently: within 4 grey levels on average, at
  most 31 pixels far off); and the reference answer in the model's top two from the
  app's own drawing. How many top-1s agree is printed, not asserted: a shape the
  model is unsure of can flip on a few pixels.

The numbers print in the test log (`coreml units …`, one line per fixture).
