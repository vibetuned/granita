# granita's Android test shell

A small Kotlin app (plain views, no Compose) that runs the released LiteRT export
(`models/battuta/granitanet.tflite`) on a tablet or phone: draw a symbol on a staff
with the S Pen or a finger and the app shows the top five classes with their
confidences, the 32 × 32 image the model saw, the model's time, and the context it
derived from the strokes (height on the staff, place in the measure, size). *Use the
context* off sends the all-zero vector instead, to compare.

Every stroke's end classifies all the strokes since the last *Clear*, so a sharp can
take its four strokes.

*Add a note* then a tap on the staff places a notehead (snapped to the lines and
spaces; *Remove the note* takes it away) with its stem, so the context's note features
are real: the symbol's offset from the centre of the note's body (notehead and stem, as
granita measures every note) and whether it sits above, below, left or right of that
body. Two pairs of − / + buttons change the size rule, to calibrate it against
natural-size writing: how many staff spaces fill the canvas (granita's 3.0, which these
buttons calibrated: a symbol's longer side over that many spaces is its fraction of the
canvas) and the smallest fraction a symbol is drawn at (granita's 0.08: a floor lifts
the small symbols and leaves the large ones at their natural size). The status line
shows the fraction used.

## What an editor would lift

- `StrokeRaster.kt`: the strokes to the model's image, the way granita's
  `render_strokes` does it (96 px canvas, natural size from the staff space, 5 px pen,
  no anti-aliasing, Pillow's bilinear reduction to 32 px re-implemented exactly).
- `SymbolContext.kt`: granita's `SymbolContext` and `context_vector`.
- `GranitaModel.kt`: LiteRT's CompiledModel API (2.3.2, CPU). The tflite keeps
  litert-torch's names, not the sidecar's: signature `serving_default`, inputs
  `args_0` (the image) and `args_1` (the context), output `output_0`.

## Building

Gradle 8.14.6 (the wrapper), AGP 8.13.2 and Kotlin 2.3.21; Android Studio's own
Java is too new for this Gradle, so point at a JDK 17–21:

```sh
cd apps/android
export ANDROID_HOME=~/Library/Android/sdk JAVA_HOME=$(/usr/libexec/java_home -v 21)
./gradlew :app:assembleDebug                       # app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:installDebug                        # onto the connected tablet
./gradlew :app:connectedDebugAndroidTest           # the fixture tests, on it
```

The model, its sidecar and the fixtures are copied from `models/battuta` into the
build's assets at build time (`copyGranitaAssets`); nothing is duplicated here. minSdk
24, as battuta.

## The tests

`src/androidTest/.../FixtureTest.kt` runs `models/battuta/fixtures.json` (made by
`scripts/shell_fixtures.py` with the PyTorch checkpoint) through the app, on a device
or an emulator:

- **the model**, fed the reference pixels and context: the same top-1, logits within
  1e-3 (fp32);
- **the context** derived from the strokes: the reference vector;
- **the rasteriser**: the size rule exactly, and the model's answer from the app's
  own drawing the same as from Pillow's. Android's Canvas rasterises a wide line a
  little differently from Pillow, so the pixels are only guarded against a gross
  error (mean difference under 10 grey levels, fewer than half the ink pixels far
  off for a glyph of more than 16); the exact numbers are logged.

The numbers go to logcat under the tag `granita`
(`adb logcat -s granita`).
