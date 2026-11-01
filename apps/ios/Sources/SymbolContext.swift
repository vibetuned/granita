import CoreGraphics

/// The staff a canvas draws, in the strokes' units: the distance between two lines,
/// the middle line, and the barline at each end.
struct Staff {
    var space: Double
    var middleY: Double
    var leftX: Double
    var rightX: Double
}

/// Where a mark sits in the score, in staff spaces — granita's `SymbolContext` and
/// `context_vector` (granita/context.py), kept identical. The shell draws no notes,
/// so the nearest notehead is always unknown.
struct SymbolContext: Equatable {
    /// Centre from the staff's middle line, positive downwards.
    var yStaff: Double
    /// 0 at the left barline, 1 at the right.
    var xMeasure: Double
    /// Distance to the nearest barline.
    var barline: Double
    /// Centre minus the nearest notehead's centre; nil without one.
    var noteDx: Double?
    var noteDy: Double?
    var width: Double
    var height: Double

    /// Staff spaces within which a notehead counts as the symbol's neighbour.
    static let noteRange = 8.0
    static let dim = 9
    /// What to send when nothing is known: all zeros, the valid flag included.
    static let unknown = [Float](repeating: 0, count: dim)

    /// From strokes drawn on a staff.
    init(strokes: [[CGPoint]], staff: Staff) {
        let b = StrokeRaster.bounds(strokes)
        let (cx, cy) = b.centre
        yStaff = (cy - staff.middleY) / staff.space
        xMeasure = (cx - staff.leftX) / (staff.rightX - staff.leftX)
        barline = min(cx - staff.leftX, staff.rightX - cx) / staff.space
        noteDx = nil
        noteDy = nil
        width = (b.maxX - b.minX) / staff.space
        height = (b.maxY - b.minY) / staff.space
    }

    /// The model's `context` input: nine floats roughly in [-1, 1].
    var vector: [Float] {
        let hasNote = noteDx != nil && noteDy != nil
        func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Float { Float(max(lo, min(hi, v))) }
        return [
            1,
            clamp(yStaff / 6, -2, 2),
            clamp(xMeasure * 2 - 1, -1, 1),
            clamp(barline / 8, 0, 1),
            hasNote ? 1 : 0,
            clamp((noteDx ?? 0) / Self.noteRange, -1, 1),
            clamp((noteDy ?? 0) / Self.noteRange, -1, 1),
            clamp(width / 8, 0, 1.5),
            clamp(height / 8, 0, 1.5),
        ]
    }
}
