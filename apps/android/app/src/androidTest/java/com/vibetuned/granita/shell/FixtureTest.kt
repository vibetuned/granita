package com.vibetuned.granita.shell

import android.graphics.PointF
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * The LiteRT export and this app's preprocessing against granita's reference
 * (models/granitanet/fixtures.json, from scripts/shell_fixtures.py), on a device or an
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
        val fraction = json.getDouble("fraction")
        val contextVector = json.getJSONArray("context_vector").floats()
        val pixels = json.getJSONArray("pixels").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val logits = json.getJSONArray("logits").floats()
        val logitsNoContext = json.getJSONArray("logits_no_context").floats()
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

    /** The context derived from strokes on the fixture staff. */
    @Test
    fun contextVector() {
        for (f in fixtures) {
            val v = SymbolContext.fromStrokes(f.strokes, staff).vector()
            assertEquals(f.contextVector.size, v.size)
            for (i in v.indices) assertEquals(f.drawnAs, f.contextVector[i], v[i], 1e-5f)
        }
    }

    /**
     * The app's rasteriser against Pillow's: the size rule exactly; the drawing close
     * (Android's Canvas and Pillow rasterise a wide line differently); and the reference
     * answer in the model's top two from the app's own drawing. The top-1 agreement is
     * logged, not asserted: a shape the model is unsure of can flip on a few pixels.
     */
    @Test
    fun rasteriserMatchesGranita() {
        GranitaModel(assets).use { model ->
            val report = mutableListOf<String>()
            var agreed = 0
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
                assertTrue("${f.drawnAs}: the drawing drifted from Pillow's (mean |Δ| $meanDiff)", meanDiff < 4)
                assertTrue("${f.drawnAs}: the drawing drifted from Pillow's ($farOff pixels far off)", farOff < 32)
                val p = model.predict(StrokeRaster.normalize(px), f.contextVector)
                val labels = p.top.take(2).map { it.label }
                assertTrue("${f.drawnAs}: ${f.topLabel} not in the top two $labels", f.topLabel in labels)
                if (labels[0] == f.topLabel) agreed++
                report.add(
                    String.format(
                        Locale.ROOT, "%s: mean |Δ| %.2f, %d of %d pixels far off, top-1 %s (reference %s at %.0f %%)",
                        f.drawnAs, meanDiff, farOff, px.size, p.top[0].label, f.topLabel, f.topConfidence * 100,
                    ),
                )
            }
            Log.i(TAG, "rasteriser: top-1 agrees on $agreed of ${fixtures.size} fixtures")
            Log.i(TAG, report.joinToString("\n"))
        }
    }
}
