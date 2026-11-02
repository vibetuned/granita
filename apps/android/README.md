# granita's Android test shell

A small Kotlin app (plain views, no Compose) that runs the released LiteRT export
(`models/battuta/granitanet.tflite`) on a tablet or phone: draw a symbol on a staff
with the S Pen or a finger and the app shows the top five classes with their
confidences, the 32 × 32 image the model saw, the model's time, and the context it
derived from the strokes (height on the staff, place in the measure, size). *Use the
context* off sends the all-zero vector instead, to compare.

Every stroke's end classifies all the strokes since the last *Clear*, so a sharp can
take its four strokes.

*Add notes* then taps on the staff place noteheads (snapped to the lines and spaces; a
tap on a note takes it away, *Clear notes* all of them) with their stems, so the
context's note features are real: the symbol's offset from the centre of the nearest
note's body (notehead and stem, as granita measures every note) and whether it sits
above, below, left or right of that body. Two pairs of − / + buttons change the size
rule, to calibrate it against
natural-size writing: how many staff spaces fill the canvas (granita's 3.0, which these
buttons calibrated: a symbol's longer side over that many spaces is its fraction of the
canvas) and the smallest fraction a symbol is drawn at (granita's 0.08: a floor lifts
the small symbols and leaves the large ones at their natural size). The status line
shows the fraction used.

## The reply, the dataset and training on the device

Under the five answers, tap the class the drawing **was** — one of the five, or
*Another…* for the whole list — and it is saved to `replies.jsonl` in the app's
external files (`Android/data/com.vibetuned.granita.shell/files/`; *Share the replies*
hands it to another app through a FileProvider) as one line in musicset's contribution
format (`musicset symbols --strokes`): `label`, `writer` (this install's id),
`staff_space`, the `strokes` with each sample's time and pressure, the `context` facts,
and the shell's own `staff`, `notes`, `fraction`, the model's `predicted` answer,
`device`, `app` and `time`. *Discard* drops the drawing; nothing is saved. Every reply
clears the staff for the next drawing; the notes stay.

*Train on N replies* retrains the model's last layer on this device: the tflite's
`personal` signature takes that layer's weights as inputs (`head_w`, `head_b`, the
sidecar's `head` to start from) and gives back the 128 features the layer reads, so
the whole network still runs as compiled and the layer is a plain `Linear(128 ->
classes)` the app owns. The trainer (`Head.kt`) takes gradient steps on a tiny replay
buffer — the newest eight replies of each class, so one symbol written fifty times
does not drown the others — holds the layer to the **anchors** (`anchors.json`:
features the released model read with its own answers, the KL from those answers on
every anchor each epoch) and pulls the weights back toward the original ones (an L2
anchor). The trained layer is saved (`files/head.json`), used from then on and shown
in the status line; *Reset the layer* puts the original back. The anchors are 152
drawings of the dataset's val split that the model gets right, four per class,
handwritten where the class has handwriting, none for the note values
(`scripts/export_anchors.py --data`). A tflite exported without the `personal`
signature still runs: the status line says so and Train stays off.

## What an editor would lift

- `StrokeRaster.kt`: the strokes to the model's image, the way granita's
  `render_strokes` does it (96 px canvas, natural size from the staff space, 5 px pen,
  no anti-aliasing, Pillow's bilinear reduction to 32 px re-implemented exactly).
- `SymbolContext.kt`: granita's `SymbolContext` and `context_vector`.
- `GranitaModel.kt`: LiteRT's CompiledModel API (2.3.2, CPU). The tflite keeps
  litert-torch's names, not the sidecar's — the sidecar's `litert` block lists them per
  signature: `serving_default` with `args_0` (the image), `args_1` (the context) and
  `output_0`; `personal` with the layer's `head_w` and `head_b` after them and the
  `features` as a second output.
- `Head.kt`: the layer, the replay buffer, the anchors and the trainer.
- `Annotations.kt`: the replies file, musicset's record, and its sharing.

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

The model, its sidecar, the anchors and the fixtures are copied from `models/battuta`
into the build's assets at build time (`copyGranitaAssets`); nothing is duplicated
here. minSdk 24, as battuta.

## The tests

`src/androidTest/.../FixtureTest.kt` runs `models/battuta/fixtures.json` (made by
`scripts/shell_fixtures.py` with the PyTorch checkpoint) through the app, on a device
or an emulator:

- **the model**, fed the reference pixels and context: the same top-1, logits within
  1e-3 (fp32);
- **the personal signature** (skipped on a tflite without it): its features are the
  reference's, and the sidecar's layer applied to them is its logits;
- **the trainer** (likewise): the fixtures as replies stay right and near the anchors;
  ten replies calling the quarter rest an arpeggio make it one, a reset undoes that;
- **the replies file**: a record written and read back, musicset's keys in the file;
- **the context** derived from the strokes: the reference vector;
- **the rasteriser**: the size rule exactly, and the model's answer from the app's
  own drawing the same as from Pillow's. The drawing is a port of Pillow's own
  (`StrokeRaster.Paper`, from its `Draw.c`: the wide line's polygon with its
  whole-pixel offsets, the scan conversion, the joint's pie slice; the dots'
  ellipses approximated) — the Canvas drew every stroke fatter (2026-11-02). The
  pixels are still only guarded against a gross error (mean difference under 10
  grey levels, fewer than half the ink pixels far off for a glyph of more than
  16), the exact numbers logged: they agree with Pillow to a pixel or two, except
  where `fixtures.json` rounds the fraction (the sharp, fingering 3).

The numbers go to logcat under the tag `granita`
(`adb logcat -s granita`).
