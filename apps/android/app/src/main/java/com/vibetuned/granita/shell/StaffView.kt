package com.vibetuned.granita.shell

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * A staff to draw on: five lines a staff space apart, a barline at each end, and a
 * notehead where one was placed (a tap, snapped to the lines and spaces). Every other
 * touch, S Pen or finger, is one stroke; the strokes are kept in the view's own pixels,
 * the units the model then reads them in (the staff space is known).
 */
class StaffView(context: Context) : View(context) {
    private val density = resources.displayMetrics.density
    var staffSpace = 24f * density
    private val inset = 24f * density
    private val strokeList = mutableListOf<List<PointF>>()
    private var current = mutableListOf<PointF>()
    /** The notehead's centre, in the view's pixels, once placed. */
    var note: PointF? = null
        private set
    /** While set, the next touch places the notehead instead of drawing. */
    var placingNote = false
    private var placingTouch = false
    var onStrokeEnd: ((List<List<PointF>>) -> Unit)? = null
    var onNotePlaced: (() -> Unit)? = null

    val strokes: List<List<PointF>> get() = strokeList.toList()
    val staff: Staff
        get() = Staff(staffSpace.toDouble(), height / 2.0, inset.toDouble(), (width - inset).toDouble(), note)

    private val linePaint = Paint().apply {
        color = Color.rgb(140, 140, 140)
        strokeWidth = density
        isAntiAlias = true
    }
    private val notePaint = Paint().apply {
        color = Color.rgb(64, 64, 64)
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val stemPaint = Paint().apply {
        color = Color.rgb(64, 64, 64)
        strokeWidth = 1.5f * density
        isAntiAlias = true
    }
    private val inkPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }

    init {
        setBackgroundColor(Color.WHITE)
    }

    fun clear() {
        strokeList.clear()
        current = mutableListOf()
        invalidate()
    }

    fun removeNote() {
        note = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val s = staff
        for (i in -2..2) {
            val y = (s.middleY + i * s.space).toFloat()
            canvas.drawLine(s.leftX.toFloat(), y, s.rightX.toFloat(), y, linePaint)
        }
        for (x in listOf(s.leftX, s.rightX)) {
            canvas.drawLine(x.toFloat(), (s.middleY - 2 * s.space).toFloat(), x.toFloat(), (s.middleY + 2 * s.space).toFloat(), linePaint)
        }
        note?.let { n ->
            // Ledger lines from the staff out to the note.
            val steps = ((n.y - s.middleY) / s.space).roundToInt()
            if (abs(steps) >= 3) {
                for (k in 3..abs(steps)) {
                    val y = (s.middleY + k * Integer.signum(steps) * s.space).toFloat()
                    canvas.drawLine(n.x - s.space.toFloat(), y, n.x + s.space.toFloat(), y, linePaint)
                }
            }
            val sp = s.space.toFloat()
            canvas.drawOval(RectF(n.x - 0.65f * sp, n.y - 0.5f * sp, n.x + 0.65f * sp, n.y + 0.5f * sp), notePaint)
            // The stem: up on the right from the middle line down, else down on the left.
            val up = n.y >= s.middleY
            val x = n.x + (if (up) 0.6f else -0.6f) * sp
            canvas.drawLine(x, n.y, x, n.y + (if (up) -3.5f else 3.5f) * sp, stemPaint)
        }
        for (stroke in strokeList + listOf(current)) {
            if (stroke.isEmpty()) continue
            val path = Path()
            path.moveTo(stroke[0].x, stroke[0].y)
            if (stroke.size == 1) path.lineTo(stroke[0].x, stroke[0].y)
            for (p in stroke.drop(1)) path.lineTo(p.x, p.y)
            canvas.drawPath(path, inkPaint)
        }
    }

    /** A tap's point snapped to the nearest line or space, within the barlines. */
    private fun snapped(x: Float, y: Float): PointF {
        val s = staff
        val halfSteps = ((y - s.middleY) / (s.space / 2)).roundToInt()
        return PointF(x.coerceIn(s.leftX.toFloat(), s.rightX.toFloat()), (s.middleY + halfSteps * s.space / 2).toFloat())
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                if (placingNote) {
                    placingTouch = true
                    return true
                }
                current = mutableListOf(PointF(event.x, event.y))
            }
            MotionEvent.ACTION_MOVE -> {
                if (placingTouch) return true
                // The S Pen samples faster than the screen refreshes; take every sample.
                for (h in 0 until event.historySize) current.add(PointF(event.getHistoricalX(h), event.getHistoricalY(h)))
                current.add(PointF(event.x, event.y))
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (placingTouch) {
                    placingTouch = false
                    if (event.actionMasked == MotionEvent.ACTION_UP) {
                        placingNote = false
                        note = snapped(event.x, event.y)
                        invalidate()
                        performClick()
                        onNotePlaced?.invoke()
                    }
                    return true
                }
                current.add(PointF(event.x, event.y))
                strokeList.add(current.toList())
                current = mutableListOf()
                invalidate()
                if (event.actionMasked == MotionEvent.ACTION_UP) performClick()
                onStrokeEnd?.invoke(strokes)
                return true
            }
            else -> return false
        }
        invalidate()
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
