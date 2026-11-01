package com.vibetuned.granita.shell

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.view.MotionEvent
import android.view.View

/**
 * A staff to draw on: five lines a staff space apart, a barline at each end. Every
 * touch, S Pen or finger, is one stroke; the strokes are kept in the view's own pixels,
 * the units the model then reads them in (the staff space is known).
 */
class StaffView(context: Context) : View(context) {
    private val density = resources.displayMetrics.density
    var staffSpace = 24f * density
    private val inset = 24f * density
    private val strokeList = mutableListOf<List<PointF>>()
    private var current = mutableListOf<PointF>()
    var onStrokeEnd: ((List<List<PointF>>) -> Unit)? = null

    val strokes: List<List<PointF>> get() = strokeList.toList()
    val staff: Staff
        get() = Staff(staffSpace.toDouble(), height / 2.0, inset.toDouble(), (width - inset).toDouble())

    private val linePaint = Paint().apply {
        color = Color.rgb(140, 140, 140)
        strokeWidth = density
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

    override fun onDraw(canvas: Canvas) {
        val s = staff
        for (i in -2..2) {
            val y = (s.middleY + i * s.space).toFloat()
            canvas.drawLine(s.leftX.toFloat(), y, s.rightX.toFloat(), y, linePaint)
        }
        for (x in listOf(s.leftX, s.rightX)) {
            canvas.drawLine(x.toFloat(), (s.middleY - 2 * s.space).toFloat(), x.toFloat(), (s.middleY + 2 * s.space).toFloat(), linePaint)
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

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                current = mutableListOf(PointF(event.x, event.y))
            }
            MotionEvent.ACTION_MOVE -> {
                // The S Pen samples faster than the screen refreshes; take every sample.
                for (h in 0 until event.historySize) current.add(PointF(event.getHistoricalX(h), event.getHistoricalY(h)))
                current.add(PointF(event.x, event.y))
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
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
