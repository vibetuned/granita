package com.vibetuned.granita.shell

import android.graphics.PointF
import kotlin.math.min

/**
 * The staff a canvas draws, in the strokes' units: the distance between two lines,
 * the middle line, and the barline at each end.
 */
class Staff(val space: Double, val middleY: Double, val leftX: Double, val rightX: Double)

/**
 * Where a mark sits in the score, in staff spaces: granita's `SymbolContext` and
 * `context_vector` (granita/context.py), kept identical. The shell draws no notes,
 * so the nearest notehead is always unknown.
 */
data class SymbolContext(
    /** Centre from the staff's middle line, positive downwards. */
    val yStaff: Double,
    /** 0 at the left barline, 1 at the right. */
    val xMeasure: Double,
    /** Distance to the nearest barline. */
    val barline: Double,
    /** Centre minus the nearest notehead's centre; null without one. */
    val noteDx: Double?,
    val noteDy: Double?,
    val width: Double,
    val height: Double,
) {
    /** The model's `context` input: nine floats roughly in [-1, 1]. */
    fun vector(): FloatArray {
        val hasNote = noteDx != null && noteDy != null
        fun clamp(v: Double, lo: Double, hi: Double) = v.coerceIn(lo, hi).toFloat()
        return floatArrayOf(
            1f,
            clamp(yStaff / 6, -2.0, 2.0),
            clamp(xMeasure * 2 - 1, -1.0, 1.0),
            clamp(barline / 8, 0.0, 1.0),
            if (hasNote) 1f else 0f,
            clamp((noteDx ?: 0.0) / NOTE_RANGE, -1.0, 1.0),
            clamp((noteDy ?: 0.0) / NOTE_RANGE, -1.0, 1.0),
            clamp(width / 8, 0.0, 1.5),
            clamp(height / 8, 0.0, 1.5),
        )
    }

    companion object {
        /** Staff spaces within which a notehead counts as the symbol's neighbour. */
        const val NOTE_RANGE = 8.0
        const val DIM = 9
        /** What to send when nothing is known: all zeros, the valid flag included. */
        val UNKNOWN = FloatArray(DIM)

        /** From strokes drawn on a staff. */
        fun fromStrokes(strokes: List<List<PointF>>, staff: Staff): SymbolContext {
            val b = StrokeRaster.bounds(strokes)
            return SymbolContext(
                yStaff = (b.centreY - staff.middleY) / staff.space,
                xMeasure = (b.centreX - staff.leftX) / (staff.rightX - staff.leftX),
                barline = min(b.centreX - staff.leftX, staff.rightX - b.centreX) / staff.space,
                noteDx = null,
                noteDy = null,
                width = (b.maxX - b.minX) / staff.space,
                height = (b.maxY - b.minY) / staff.space,
            )
        }
    }
}
