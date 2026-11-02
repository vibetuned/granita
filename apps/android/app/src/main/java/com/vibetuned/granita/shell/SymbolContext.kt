package com.vibetuned.granita.shell

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min

/**
 * The staff a canvas draws, in the strokes' units: the distance between two lines,
 * the middle line, and the barline at each end.
 */
class Staff(
    val space: Double,
    val middleY: Double,
    val leftX: Double,
    val rightX: Double,
    /** The centres of the noteheads drawn on it. */
    val notes: List<PointF> = emptyList(),
) {
    /** The staff with one note at most, as the code and tests written then make it. */
    constructor(space: Double, middleY: Double, leftX: Double, rightX: Double, note: PointF?) :
        this(space, middleY, leftX, rightX, listOfNotNull(note))

    /** The first note's centre, for the code written when there was at most one. */
    val note: PointF? get() = notes.firstOrNull()

    /**
     * The body of a note as (left, top, right, bottom): its notehead (1.3 x 1 staff spaces)
     * and its 3.5-space stem, up on the right from the middle line down, else down on the
     * left. Granita measures the note offsets from this box's centre.
     */
    fun body(n: PointF): DoubleArray {
        val up = n.y >= middleY
        val top = n.y - (if (up) 3.5 else 0.5) * space
        return doubleArrayOf(n.x - 0.65 * space, top, n.x + 0.65 * space, top + 4.0 * space)
    }

    /** The body nearest a point — the least gap to it — as granita's context takes the nearest note; null without notes. */
    fun nearestBody(x: Double, y: Double): DoubleArray? = notes.map { body(it) }.minByOrNull { gap(it, x, y) }

    /** The first note's body, for the code and tests written when there was at most one. */
    fun noteBody(): DoubleArray? = note?.let { body(it) }

    private fun gap(b: DoubleArray, x: Double, y: Double): Double =
        hypot(maxOf(b[0] - x, x - b[2], 0.0), maxOf(b[1] - y, y - b[3], 0.0))
}

/**
 * Where a mark sits in the score, in staff spaces: granita's `SymbolContext` and
 * `context_vector` (granita/context.py), kept identical. The note is the one the
 * staff view draws, when there is one, measured by its body (notehead and stem).
 */
data class SymbolContext(
    /** Centre from the staff's middle line, positive downwards. */
    val yStaff: Double,
    /** 0 at the left barline, 1 at the right. */
    val xMeasure: Double,
    /** Distance to the nearest barline. */
    val barline: Double,
    /** Centre minus the centre of the nearest note's body; null without one. */
    val noteDx: Double?,
    val noteDy: Double?,
    val width: Double,
    val height: Double,
    /** Width and height of that body; null without one. */
    val noteW: Double? = null,
    val noteH: Double? = null,
) {
    /**
     * The model's `context` input: sixteen floats roughly in [-1, 1] (granita's
     * `CONTEXT_FEATURES` order): valid, in / above / below the staff, height, measure
     * position, barline distance, note near, symbol above / below / left / right of its body,
     * its offset, width, height.
     */
    fun vector(): FloatArray {
        val hasNote = noteDx != null && noteDy != null
        val dx = noteDx ?: 0.0
        val dy = noteDy ?: 0.0
        val hw = 0.5 * (noteW ?: 0.0)
        val hh = 0.5 * (noteH ?: 0.0)
        val over = hasNote && abs(dx) <= hw + NOTE_MARGIN
        val beside = hasNote && abs(dy) <= hh + NOTE_MARGIN
        fun clamp(v: Double, lo: Double, hi: Double) = v.coerceIn(lo, hi).toFloat()
        fun flag(on: Boolean) = if (on) 1f else 0f
        return floatArrayOf(
            1f,
            flag(abs(yStaff) <= STAFF_HALF_HEIGHT),
            flag(yStaff < -STAFF_HALF_HEIGHT),
            flag(yStaff > STAFF_HALF_HEIGHT),
            clamp(yStaff / 6, -2.0, 2.0),
            clamp(xMeasure * 2 - 1, -1.0, 1.0),
            clamp(barline / 8, 0.0, 1.0),
            flag(hasNote),
            flag(over && dy < -hh),
            flag(over && dy > hh),
            flag(beside && dx < -hw),
            flag(beside && dx > hw),
            clamp(dx / NOTE_RANGE, -1.0, 1.0),
            clamp(dy / NOTE_RANGE, -1.0, 1.0),
            clamp(width / 8, 0.0, 1.5),
            clamp(height / 8, 0.0, 1.5),
        )
    }

    companion object {
        /** Staff spaces within which a note's body counts as the symbol's neighbour. */
        const val NOTE_RANGE = 8.0
        /** Half the staff's height: a centre within this of the middle line is inside the lines. */
        const val STAFF_HALF_HEIGHT = 2.0
        /**
         * Staff spaces the symbol may stick out past the body's sides and still count as above
         * or below it (or past its top and bottom and still count as left or right of it).
         */
        const val NOTE_MARGIN = 0.5
        const val DIM = 16
        /** What to send when nothing is known: all zeros, the valid flag included. */
        val UNKNOWN = FloatArray(DIM)

        /**
         * From strokes drawn on a staff: the note counts when its body is within NOTE_RANGE
         * staff spaces of the strokes' centre.
         */
        fun fromStrokes(strokes: List<List<PointF>>, staff: Staff): SymbolContext {
            val b = StrokeRaster.bounds(strokes)
            var noteDx: Double? = null
            var noteDy: Double? = null
            var noteW: Double? = null
            var noteH: Double? = null
            staff.nearestBody(b.centreX, b.centreY)?.let { (left, top, right, bottom) ->
                val gx = maxOf(left - b.centreX, b.centreX - right, 0.0) / staff.space
                val gy = maxOf(top - b.centreY, b.centreY - bottom, 0.0) / staff.space
                if (hypot(gx, gy) <= NOTE_RANGE) {
                    noteDx = (b.centreX - 0.5 * (left + right)) / staff.space
                    noteDy = (b.centreY - 0.5 * (top + bottom)) / staff.space
                    noteW = (right - left) / staff.space
                    noteH = (bottom - top) / staff.space
                }
            }
            return SymbolContext(
                yStaff = (b.centreY - staff.middleY) / staff.space,
                xMeasure = (b.centreX - staff.leftX) / (staff.rightX - staff.leftX),
                barline = min(b.centreX - staff.leftX, staff.rightX - b.centreX) / staff.space,
                noteDx = noteDx,
                noteDy = noteDy,
                width = (b.maxX - b.minX) / staff.space,
                height = (b.maxY - b.minY) / staff.space,
                noteW = noteW,
                noteH = noteH,
            )
        }
    }
}
