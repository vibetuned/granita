package com.vibetuned.granita.shell

import android.app.Activity
import android.app.AlertDialog
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
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

/**
 * The Android test shell for granita's LiteRT export: draw a symbol on a staff with the
 * S Pen or a finger and see what the model makes of it: the top five classes with their
 * confidences, the 32 x 32 image the model saw, the model's time, and the context
 * derived from the strokes. Every stroke's end classifies all the strokes since the last
 * clear, so a sharp can take its four strokes. Noteheads can be placed on the staff, which
 * give the context its note features (the nearest counts); the − / + buttons change the
 * size rule (how many staff spaces fill the canvas, granita's 3.0, and the smallest
 * fraction a symbol is drawn at, granita's 0.08) to calibrate it against natural-size
 * writing.
 *
 * Then the reply (2026-11-02): tap the class the drawing was — one of the five, or any
 * from the list — and it is saved for the dataset (`Annotations`, musicset's contribution
 * record); Discard drops it. "Train" retrains the model's last layer on the replies, on
 * this device, held to the anchors and the original weights (`Trainer`); the trained
 * layer is kept and used until "Reset". Share sends the replies file. The pieces an
 * editor would lift are StrokeRaster, SymbolContext, GranitaModel and Head.
 */
class MainActivity : Activity() {
    private class Drawing(val strokes: List<List<PointF>>, val samples: List<List<Sample>>, val staff: Staff, val pencil: Boolean)
    private class Shown(val prediction: GranitaModel.Prediction, val context: SymbolContext, val fraction: Double, val drawing: Drawing)

    private val executor = Executors.newSingleThreadExecutor()
    private var model: GranitaModel? = null
    private var spacesPerCanvas = StrokeRaster.CANVAS_STAFF_SPACES
    private var minFraction = StrokeRaster.FRACTION_RANGE.start
    private var shown: Shown? = null
    private var report: Trainer.Report? = null
    private var headTrained = false
    private var training = false
    private lateinit var replies: Annotations
    private lateinit var staff: StaffView
    private lateinit var notesButton: Button
    private lateinit var clearNotes: Button
    private lateinit var useContext: CheckBox
    private lateinit var sizeLabel: TextView
    private lateinit var floorLabel: TextView
    private lateinit var preview: ImageView
    private lateinit var answers: LinearLayout
    private lateinit var replyRow: LinearLayout
    private lateinit var contextLine: TextView
    private lateinit var trainButton: Button
    private lateinit var resetButton: Button
    private lateinit var shareButton: Button
    private lateinit var status: TextView

    private val headFile: File get() = File(filesDir, "head.json")

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        replies = Annotations(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
        staff = StaffView(this)
        root.addView(staff, LinearLayout.LayoutParams(MATCH_PARENT, dp(300)))

        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val clear = Button(this).apply { text = "Clear" }
        notesButton = Button(this)
        clearNotes = Button(this).apply { text = "Clear notes"; isEnabled = false }
        useContext = CheckBox(this).apply { text = "Use the context"; isChecked = true }
        controls.addView(clear)
        controls.addView(notesButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8) })
        controls.addView(clearNotes, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8) })
        controls.addView(useContext, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(16) })
        root.addView(controls, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8) })

        val size = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val smaller = Button(this).apply { text = "−" }
        val larger = Button(this).apply { text = "+" }
        sizeLabel = TextView(this)
        size.addView(smaller)
        size.addView(larger)
        size.addView(sizeLabel, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(12) })
        root.addView(size, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        val floor = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val lower = Button(this).apply { text = "−" }
        val higher = Button(this).apply { text = "+" }
        floorLabel = TextView(this)
        floor.addView(lower)
        floor.addView(higher)
        floor.addView(floorLabel, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(12) })
        root.addView(floor, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        val answer = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        preview = ImageView(this).apply { setBackgroundColor(Color.LTGRAY) }
        answer.addView(preview, LinearLayout.LayoutParams(dp(128), dp(128)))
        val right = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        answers = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        right.addView(answers, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        replyRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; visibility = android.view.View.GONE }
        val another = Button(this).apply { text = "Another…" }
        val discard = Button(this).apply { text = "Discard" }
        replyRow.addView(another)
        replyRow.addView(discard, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8) })
        right.addView(replyRow, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        contextLine = TextView(this).apply { textSize = 12f; setTextColor(Color.DKGRAY) }
        right.addView(contextLine, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        answer.addView(right, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { marginStart = dp(16) })
        root.addView(answer, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(12) })

        val learn = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        trainButton = Button(this)
        resetButton = Button(this).apply { text = "Reset the layer" }
        shareButton = Button(this).apply { text = "Share the replies" }
        learn.addView(trainButton)
        learn.addView(resetButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8) })
        learn.addView(shareButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8) })
        root.addView(learn, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8) })

        status = TextView(this).apply { textSize = 12f; setTextColor(Color.DKGRAY); text = "loading the model…" }
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8) })
        setContentView(ScrollView(this).apply { addView(root) })
        showNotesState()
        showSize()
        showAnswers()
        showLearning()

        clear.setOnClickListener { clearDrawing() }
        notesButton.setOnClickListener {
            staff.placingNotes = !staff.placingNotes
            showNotesState()
        }
        clearNotes.setOnClickListener {
            staff.clearNotes()
            showNotesState()
            classify()
        }
        staff.onNotesChanged = {
            showNotesState()
            classify()
        }
        smaller.setOnClickListener { spacesPerCanvas = (spacesPerCanvas - 0.5).coerceAtLeast(3.0); showSize(); classify() }
        larger.setOnClickListener { spacesPerCanvas = (spacesPerCanvas + 0.5).coerceAtMost(12.0); showSize(); classify() }
        lower.setOnClickListener { minFraction = (minFraction - 0.04).coerceAtLeast(StrokeRaster.FRACTION_RANGE.start); showSize(); classify() }
        higher.setOnClickListener { minFraction = (minFraction + 0.04).coerceAtMost(StrokeRaster.FRACTION_RANGE.endInclusive); showSize(); classify() }
        useContext.setOnCheckedChangeListener { _, _ -> classify() }
        staff.onStrokeEnd = { classify() }
        another.setOnClickListener {
            val classes = model?.sidecar?.classes ?: return@setOnClickListener
            AlertDialog.Builder(this).setTitle("It was a…").setItems(classes.toTypedArray()) { _, i -> reply(classes[i]) }.show()
        }
        discard.setOnClickListener { clearDrawing() }
        trainButton.setOnClickListener { train() }
        resetButton.setOnClickListener { resetHead() }
        shareButton.setOnClickListener { startActivity(replies.shareIntent()) }

        executor.execute {
            try {
                val loaded = GranitaModel(assets)
                if (loaded.personal) HeadWeights.load(headFile)?.let { loaded.head = it; headTrained = true }
                model = loaded
                runOnUiThread {
                    status.text = "LiteRT ${GranitaModel.RUNTIME}, CPU · ${loaded.sidecar.classes.size} classes" + if (loaded.personal) " · the personal signature" else " · no personal signature in this tflite: no training"
                    showLearning()
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "the model failed to load: $e" }
            }
        }
    }

    private fun showNotesState() {
        notesButton.text = if (staff.placingNotes) "Done with the notes" else "Add notes"
        clearNotes.isEnabled = staff.notes.isNotEmpty()
    }

    private fun showSize() {
        sizeLabel.text = String.format(Locale.ROOT, "%.1f staff spaces fill the canvas", spacesPerCanvas)
        floorLabel.text = String.format(Locale.ROOT, "smallest fraction %.2f", minFraction)
    }

    private fun showLearning() {
        val m = model
        trainButton.text = if (training) "Training…" else "Train on ${replies.count} replies"
        trainButton.isEnabled = !training && replies.count > 0 && m != null && m.personal
        resetButton.isEnabled = headTrained
        shareButton.isEnabled = replies.count > 0
        var line = "${replies.count} replies saved as ${Annotations.FILE_NAME}"
        val r = report
        if (r != null) {
            line += String.format(Locale.ROOT, " · trained on %d (the buffer) with %d anchors: loss %.3f, %.0f %% right, anchor drift %.4f", r.examples, r.anchors, r.loss, r.accuracy * 100, r.anchorDrift)
        } else if (headTrained) {
            line += " · the trained layer is in use"
        }
        if (m != null) status.text = line
    }

    /** The top five as rows to tap — the class the drawing was — with Another… and Discard under them. */
    private fun showAnswers() {
        answers.removeAllViews()
        val s = shown
        if (s == null) {
            answers.addView(TextView(this).apply { typeface = Typeface.MONOSPACE; text = "Draw a symbol on the staff, then tap what it was." })
            replyRow.visibility = android.view.View.GONE
            contextLine.text = ""
            return
        }
        answers.addView(TextView(this).apply { textSize = 12f; setTextColor(Color.DKGRAY); text = "Tap the class it was — the reply is saved:" })
        for (t in s.prediction.top) {
            answers.addView(TextView(this).apply {
                typeface = Typeface.MONOSPACE
                text = String.format(Locale.ROOT, "%-24s %-12s %5.1f %%", t.label, t.group, t.confidence * 100)
                setPadding(dp(8), dp(8), dp(8), dp(8))
                setBackgroundResource(android.R.drawable.list_selector_background)
                setOnClickListener { reply(t.label) }
            })
        }
        replyRow.visibility = android.view.View.VISIBLE
        var line = String.format(
            Locale.ROOT,
            "%.2f ms%s · fraction %.2f · y %.2f · x %.2f · barline %.1f · %.1f × %.1f spaces",
            s.prediction.nanos / 1e6, if (headTrained) " (the trained layer)" else "", s.fraction, s.context.yStaff, s.context.xMeasure, s.context.barline, s.context.width, s.context.height,
        )
        if (s.context.noteDx != null && s.context.noteDy != null) {
            line += String.format(Locale.ROOT, " · note dx %.2f dy %.2f", s.context.noteDx, s.context.noteDy)
        }
        contextLine.text = line
    }

    private fun clearDrawing() {
        staff.clear()
        shown = null
        preview.setImageDrawable(null)
        showAnswers()
    }

    private fun classify() {
        val m = model ?: return
        val drawing = Drawing(staff.strokes, staff.samples, staff.staff, staff.pencil)
        if (drawing.strokes.isEmpty()) return
        val withContext = useContext.isChecked
        val spaces = spacesPerCanvas
        val floor = minFraction
        executor.execute {
            val fraction = StrokeRaster.fraction(drawing.strokes, drawing.staff.space, spaces, floor)
            val pixels = StrokeRaster.render(drawing.strokes, fraction)
            val context = SymbolContext.fromStrokes(drawing.strokes, drawing.staff)
            val prediction = try {
                m.predict(StrokeRaster.normalize(pixels), if (withContext) context.vector() else SymbolContext.UNKNOWN)
            } catch (e: Exception) {
                runOnUiThread { status.text = "the model failed: $e" }
                return@execute
            }
            val bitmap = Bitmap.createBitmap(StrokeRaster.SIZE, StrokeRaster.SIZE, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(IntArray(pixels.size) { Color.rgb(pixels[it], pixels[it], pixels[it]) }, 0, StrokeRaster.SIZE, 0, 0, StrokeRaster.SIZE, StrokeRaster.SIZE)
            val scaled = Bitmap.createScaledBitmap(bitmap, dp(128), dp(128), false)
            runOnUiThread {
                preview.setImageBitmap(scaled)
                shown = Shown(prediction, context, fraction, drawing)
                showAnswers()
            }
        }
    }

    /** The drawing was `label`: saved for the dataset, and the staff cleared for the next. */
    private fun reply(label: String) {
        val s = shown ?: return
        val d = s.drawing
        val record = Reply(
            label = label,
            writer = replies.writer,
            staffSpace = d.staff.space,
            strokes = d.samples.map { stroke -> stroke.map { p -> val base = doubleArrayOf(p.x.toDouble(), p.y.toDouble(), Math.round(p.t * 10) / 10.0); if (p.pressure != null) base + (Math.round(p.pressure * 1000) / 1000.0) else base } },
            context = s.context,
            staff = d.staff,
            fraction = s.fraction,
            predicted = s.prediction.top.firstOrNull(),
            device = if (d.pencil) "s-pen" else "finger",
            app = "granita-shell/android",
            time = Annotations.now(),
        )
        try {
            replies.append(record)
        } catch (e: Exception) {
            status.text = "the reply was not saved: $e"
            return
        }
        clearDrawing()
        showLearning()
    }

    /** The last layer retrained on this device from every reply, on the executor. */
    private fun train() {
        val m = model ?: return
        val original = m.sidecar.head?.let { HeadWeights(it.weight, it.bias) } ?: return
        if (!m.personal || training) return
        training = true
        showLearning()
        val records = replies.all()
        val anchors = Anchors.load(assets)?.anchors ?: emptyList()
        val classes = m.sidecar.classes
        executor.execute {
            val examples = ArrayList<Example>()
            for (r in records) {
                val label = classes.indexOf(r.label)
                if (label < 0) continue
                val strokes = r.strokes.map { s -> s.map { PointF(it[0].toFloat(), it[1].toFloat()) } }.filter { it.isNotEmpty() }
                if (strokes.isEmpty()) continue
                val px = StrokeRaster.render(strokes, r.fraction)
                val p = m.predict(StrokeRaster.normalize(px), r.context.vector())
                if (p.features.isNotEmpty()) examples.add(Example(p.features, label))
            }
            val head = original.copy()
            val result = Trainer.train(head, original, ReplayBuffer(examples), anchors)
            m.head = head
            head.save(headFile)
            runOnUiThread {
                headTrained = true
                report = result
                training = false
                showLearning()
                classify()
            }
        }
    }

    private fun resetHead() {
        val m = model ?: return
        val original = m.sidecar.head?.let { HeadWeights(it.weight, it.bias) } ?: return
        m.head = original
        headFile.delete()
        headTrained = false
        report = null
        showLearning()
        classify()
    }

    override fun onDestroy() {
        executor.shutdown()
        model?.close()
        super.onDestroy()
    }
}
