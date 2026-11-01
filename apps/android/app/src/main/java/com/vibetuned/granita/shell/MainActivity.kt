package com.vibetuned.granita.shell

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.Locale
import java.util.concurrent.Executors

/**
 * The Android test shell for granita's LiteRT export: draw a symbol on a staff with the
 * S Pen or a finger and see what the model makes of it: the top five classes with their
 * confidences, the 32 x 32 image the model saw, the model's time, and the context
 * derived from the strokes. Every stroke's end classifies all the strokes since the last
 * clear, so a sharp can take its four strokes. The pieces an editor would lift are
 * StrokeRaster, SymbolContext and GranitaModel.
 */
class MainActivity : Activity() {
    private val executor = Executors.newSingleThreadExecutor()
    private var model: GranitaModel? = null
    private lateinit var staff: StaffView
    private lateinit var useContext: CheckBox
    private lateinit var preview: ImageView
    private lateinit var results: TextView
    private lateinit var status: TextView

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
        staff = StaffView(this)
        root.addView(staff, LinearLayout.LayoutParams(MATCH_PARENT, dp(300)))

        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val clear = Button(this).apply { text = "Clear" }
        useContext = CheckBox(this).apply { text = "Use the context"; isChecked = true }
        controls.addView(clear)
        controls.addView(useContext, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(16) })
        root.addView(controls, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8) })

        val answer = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        preview = ImageView(this).apply { setBackgroundColor(Color.LTGRAY) }
        answer.addView(preview, LinearLayout.LayoutParams(dp(128), dp(128)))
        results = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            text = "Draw a symbol on the staff."
        }
        answer.addView(results, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { marginStart = dp(16) })
        root.addView(answer, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(12) })

        status = TextView(this).apply { textSize = 12f; setTextColor(Color.DKGRAY); text = "loading the model…" }
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8) })
        setContentView(root)

        clear.setOnClickListener {
            staff.clear()
            results.text = "Draw a symbol on the staff."
            preview.setImageDrawable(null)
        }
        useContext.setOnCheckedChangeListener { _, _ -> classify(staff.strokes, staff.staff) }
        staff.onStrokeEnd = { strokes -> classify(strokes, staff.staff) }

        executor.execute {
            try {
                val loaded = GranitaModel(assets)
                model = loaded
                runOnUiThread { status.text = "LiteRT ${GranitaModel.RUNTIME}, CPU · ${loaded.sidecar.classes.size} classes" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "the model failed to load: $e" }
            }
        }
    }

    private fun classify(strokes: List<List<PointF>>, geometry: Staff) {
        val m = model ?: return
        if (strokes.isEmpty()) return
        val withContext = useContext.isChecked
        executor.execute {
            val fraction = StrokeRaster.fraction(strokes, geometry.space)
            val pixels = StrokeRaster.render(strokes, fraction)
            val context = SymbolContext.fromStrokes(strokes, geometry)
            val prediction = try {
                m.predict(StrokeRaster.normalize(pixels), if (withContext) context.vector() else SymbolContext.UNKNOWN)
            } catch (e: Exception) {
                runOnUiThread { status.text = "the model failed: $e" }
                return@execute
            }
            val bitmap = Bitmap.createBitmap(StrokeRaster.SIZE, StrokeRaster.SIZE, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(IntArray(pixels.size) { Color.rgb(pixels[it], pixels[it], pixels[it]) }, 0, StrokeRaster.SIZE, 0, 0, StrokeRaster.SIZE, StrokeRaster.SIZE)
            val shown = Bitmap.createScaledBitmap(bitmap, dp(128), dp(128), false)
            runOnUiThread {
                preview.setImageBitmap(shown)
                results.text = prediction.top.joinToString("\n") {
                    String.format(Locale.ROOT, "%-24s %-12s %5.1f %%", it.label, it.group, it.confidence * 100)
                }
                status.text = String.format(
                    Locale.ROOT,
                    "%.2f ms · y %.2f · x %.2f · barline %.1f · %.1f × %.1f spaces",
                    prediction.nanos / 1e6, context.yStaff, context.xMeasure, context.barline, context.width, context.height,
                )
            }
        }
    }

    override fun onDestroy() {
        executor.shutdown()
        model?.close()
        super.onDestroy()
    }
}
