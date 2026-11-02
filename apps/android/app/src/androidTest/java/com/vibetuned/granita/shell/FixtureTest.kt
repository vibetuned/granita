package com.vibetuned.granita.shell

import android.graphics.PointF
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * The LiteRT export and this app's preprocessing against granita's reference
 * (models/battuta/fixtures.json, from scripts/shell_fixtures.py), on a device or an
 * emulator: the model fed the reference pixels gives the reference logits, the context
 * derived from the strokes is the reference vector, and the app's own rasteriser leads
 * the model to the same answer. The numbers go to logcat under the tag "granita".
 */
@RunWith(AndroidJUnit4::class)
class FixtureTest {
    class Fixture(json: JSONObject) {
        val drawnAs: String = json.getString("drawn_as")
        val strokes: List<List<PointF>> = json.getJSONArray("strokes").let { all ->
            List(all.length()) { i ->
                all.getJSONArray(i).let { s ->
                    List(s.length()) { j -> s.getJSONArray(j).let { PointF(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) } }
                }
            }
        }
        val note: PointF? = json.optJSONArray("note")?.let { PointF(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) }
        val fraction = json.getDouble("fraction")
        val contextVector = json.getJSONArray("context_vector").floats()
        val pixels = json.getJSONArray("pixels").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val logits = json.getJSONArray("logits").floats()
        val logitsNoContext = json.getJSONArray("logits_no_context").floats()
        /** What the last layer read, from the `personal` signature. */
        val features = json.getJSONArray("features").floats()
        val topIndex: Int = json.getJSONObject("top").getInt("index")
        val topLabel: String = json.getJSONObject("top").getString("label")
        val topConfidence = json.getJSONObject("top").getDouble("confidence")
    }

    companion object {
        const val TAG = "granita"

        private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

        val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
        val doc = JSONObject(assets.open("fixtures.json").bufferedReader().use { it.readText() })
        val staff = doc.getJSONObject("staff").let {
            Staff(it.getDouble("staff_space"), it.getDouble("middle_y"), it.getDouble("left_x"), it.getDouble("right_x"))
        }
        val fixtures = doc.getJSONArray("fixtures").let { a -> List(a.length()) { Fixture(a.getJSONObject(it)) } }

        /** The staff with the fixture's notehead, if it has one. */
        fun staffFor(f: Fixture) = Staff(staff.space, staff.middleY, staff.leftX, staff.rightX, f.note)

        fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
            assertEquals(a.size, b.size)
            var worst = 0f
            for (i in a.indices) worst = max(worst, abs(a[i] - b[i]))
            return worst
        }
    }

    /** The export, fed the reference pixels and context: the same logits, fp32. */
    @Test
    fun modelMatchesPyTorch() {
        GranitaModel(assets).use { model ->
            var worst = 0f
            for (f in fixtures) {
                val p = model.predict(StrokeRaster.normalize(f.pixels), f.contextVector)
                worst = max(worst, maxAbsDiff(p.logits, f.logits))
                assertEquals("${f.drawnAs} with context", f.topLabel, p.top[0].label)
                val q = model.predict(StrokeRaster.normalize(f.pixels), SymbolContext.UNKNOWN)
                worst = max(worst, maxAbsDiff(q.logits, f.logitsNoContext))
            }
            Log.i(TAG, "litert: max |Δ| logits vs PyTorch $worst")
            assertTrue("fp32: logits within 1e-3 of PyTorch, got $worst", worst < 1e-3f)
        }
    }

    /**
     * The `personal` signature: the features it gives are the reference's, and the sidecar's
     * last layer applied to them is the logits it gives. Skipped on a tflite without the
     * signature (one exported before it, or without the Linux export).
     */
    @Test
    fun personalSignature() {
        GranitaModel(assets).use { model ->
            assumeTrue("no personal signature in this tflite", model.personal)
            val head = model.head!!
            assertEquals(model.sidecar.classes.size, head.classes)
            assertEquals(128, head.width)
            var worstFeatures = 0f
            var worstHead = 0f
            for (f in fixtures) {
                val p = model.predict(StrokeRaster.normalize(f.pixels), f.contextVector)
                assertEquals(f.drawnAs, 128, p.features.size)
                worstFeatures = max(worstFeatures, maxAbsDiff(p.features, f.features))
                worstHead = max(worstHead, maxAbsDiff(head.logits(p.features), p.logits))
            }
            Log.i(TAG, "personal: max |Δ| features vs PyTorch $worstFeatures, head·features vs logits $worstHead")
            assertTrue("features within 1e-3 of PyTorch, got $worstFeatures", worstFeatures < 1e-3f)
            assertTrue("the head applied to the features is the logits, got $worstHead", worstHead < 1e-3f)
        }
    }

    /**
     * The trainer on the fixtures as replies: the layer keeps them right and stays near the
     * anchors; ten replies calling the quarter rest an arpeggio turn it into one — against
     * the interim anchors, which hold that very drawing — and a reset puts the original back.
     */
    @Test
    fun trainerFollowsReplies() {
        GranitaModel(assets).use { model ->
            assumeTrue("no personal signature in this tflite", model.personal)
            val origin = model.head!!.copy()
            val anchors = Anchors.load(assets)
            assertNotNull("anchors.json is an asset", anchors)
            val examples = fixtures.map { f -> Example(model.predict(StrokeRaster.normalize(f.pixels), f.contextVector).features, f.topIndex) }
            val head = origin.copy()
            val report = Trainer.train(head, origin, ReplayBuffer(examples), anchors!!.anchors)
            assertEquals(1f, report.accuracy, 1e-6f)
            assertTrue("anchor drift ${report.anchorDrift}", report.anchorDrift < 0.1f)
            val rest = fixtures.first { it.drawnAs == "quarter rest" }
            val arpeg = model.sidecar.classes.indexOf("arpeg")
            val restFeatures = model.predict(StrokeRaster.normalize(rest.pixels), rest.contextVector).features
            val wrong = ReplayBuffer(examples + List(10) { Example(restFeatures, arpeg) })
            assertEquals("the buffer keeps a class's newest few", ReplayBuffer.PER_CLASS, wrong.examples.count { it.label == arpeg })
            val corrected = origin.copy()
            Trainer.train(corrected, origin, wrong, anchors.anchors)
            model.head = corrected
            assertEquals("arpeg", model.predict(StrokeRaster.normalize(rest.pixels), rest.contextVector).top[0].label)
            model.head = origin
            assertEquals("restQuarter", model.predict(StrokeRaster.normalize(rest.pixels), rest.contextVector).top[0].label)
        }
    }

    /** A reply written and read back, musicset's keys in the file, its context whole. */
    @Test
    fun repliesRoundTrip() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = java.io.File(context.cacheDir, "replies-" + System.nanoTime()).apply { mkdirs() }
        val store = Annotations(context, dir)
        assertEquals(0, store.count)
        val ctx = SymbolContext(-0.1, 0.5, 12.4, -1.0, 1.4, 1.6, 2.6, 1.3, 4.0)
        val staff = Staff(24.0, 300.0, 100.0, 700.0, listOf(PointF(424f, 300f)))
        store.append(Reply("accidentalSharp", store.writer, 24.0, listOf(listOf(doubleArrayOf(395.0, 255.0, 0.0, 0.5), doubleArrayOf(410.0, 275.0, 16.0, 0.6))), ctx, staff, 0.889, GranitaModel.Top("accidentalSharp", "accidental", 0.9f), "s-pen", "granita-shell/android", Annotations.now()))
        assertEquals(1, store.count)
        val text = store.file.readText()
        assertTrue(text, text.contains("\"staff_space\":24") && text.contains("\"note_dx\":-1") && text.contains("\"label\":\"accidentalSharp\""))
        val back = store.all().single()
        assertEquals(listOf(395.0, 255.0, 0.0, 0.5), back.strokes[0][0].toList())
        assertEquals(ctx.vector().toList(), back.context.vector().toList())
        assertEquals(1, back.staff.notes.size)
        assertEquals(1, Annotations(context, dir).count)
    }

    /**
     * The context derived from strokes on the fixture staff, with its notehead when the
     * fixture has one (the note offsets and the above / below / left / right flags).
     */
    @Test
    fun contextVector() {
        for (f in fixtures) {
            val v = SymbolContext.fromStrokes(f.strokes, staffFor(f)).vector()
            assertEquals(f.contextVector.size, v.size)
            for (i in v.indices) assertEquals(f.drawnAs, f.contextVector[i], v[i], 1e-5f)
        }
    }

    /**
     * The app's rasteriser against Pillow's: the size rule exactly, and the model's answer
     * from the app's own drawing the same as from Pillow's. Android's Canvas and Pillow
     * drew a wide line differently before the drawing became a port of Pillow's own
     * (2026-11-02); the pixels are still only guarded against a gross error (a wrong
     * scale, axis or pen: mean |Δ| under 10 grey levels, fewer than half the ink pixels far
     * off for a glyph of more than 16) and the exact numbers are logged — a pixel or two
     * now, but for the fixtures whose fraction the file rounds.
     */
    @Test
    fun rasteriserMatchesGranita() {
        GranitaModel(assets).use { model ->
            val report = mutableListOf<String>()
            for (f in fixtures) {
                assertEquals(f.drawnAs, f.fraction, StrokeRaster.fraction(f.strokes, staff.space), 1e-6)
                val px = StrokeRaster.render(f.strokes, f.fraction)
                var sum = 0.0
                var farOff = 0
                for (i in px.indices) {
                    val d = abs(px[i] - f.pixels[i])
                    sum += d
                    if (d > 64) farOff++
                }
                val meanDiff = sum / px.size
                val ink = f.pixels.count { it < 128 }
                assertTrue("${f.drawnAs}: the drawing is not Pillow's (mean |Δ| $meanDiff)", meanDiff < 10)
                // A dot has six ink pixels; its edge alone can move half of them.
                if (ink > 16) assertTrue("${f.drawnAs}: the drawing is not Pillow's ($farOff of $ink ink pixels far off)", farOff * 2 < ink)
                val p = model.predict(StrokeRaster.normalize(px), f.contextVector)
                assertEquals(f.drawnAs, f.topLabel, p.top[0].label)
                report.add(
                    String.format(
                        Locale.ROOT, "%s: mean |Δ| %.2f, %d of %d ink pixels far off, top-1 %s (reference %s at %.0f %%)",
                        f.drawnAs, meanDiff, farOff, ink, p.top[0].label, f.topLabel, f.topConfidence * 100,
                    ),
                )
            }
            Log.i(TAG, report.joinToString("\n"))
        }
    }
}
