import CoreGraphics

/// The staff a canvas draws, in the strokes' units: the distance between two lines,
/// the middle line, and the barline at each end.
struct Staff {
    var space: Double
    var middleY: Double
    var leftX: Double
    var rightX: Double
    /// The centre of the notehead drawn on it, if any.
    var note: CGPoint? = nil

    /// The body of that note: its notehead (1.3 x 1 staff spaces) and its 3.5-space stem,
    /// up on the right from the middle line down, else down on the left. Granita measures
    /// the note offsets from this box's centre.
    var noteBody: CGRect? {
        guard let n = note else { return nil }
        let up = Double(n.y) >= middleY
        let top = Double(n.y) - (up ? 3.5 : 0.5) * space
        return CGRect(x: Double(n.x) - 0.65 * space, y: top, width: 1.3 * space, height: 4.0 * space)
    }
}

/// Where a mark sits in the score, in staff spaces — granita's `SymbolContext` and
/// `context_vector` (granita/context.py), kept identical. The note is the one the
/// staff canvas draws, when there is one, measured by its body (notehead and stem).
struct SymbolContext: Equatable {
    /// Centre from the staff's middle line, positive downwards.
    var yStaff: Double
    /// 0 at the left barline, 1 at the right.
    var xMeasure: Double
    /// Distance to the nearest barline.
    var barline: Double
    /// Centre minus the centre of the nearest note's body; nil without one.
    var noteDx: Double?
    var noteDy: Double?
    var width: Double
    var height: Double
    /// Width and height of that body; nil without one.
    var noteW: Double?
    var noteH: Double?

    /// Staff spaces within which a note's body counts as the symbol's neighbour.
    static let noteRange = 8.0
    /// Half the staff's height: a centre within this of the middle line is inside the lines.
    static let staffHalfHeight = 2.0
    /// Staff spaces the symbol may stick out past the body's sides and still count as above
    /// or below it (or past its top and bottom and still count as left or right of it).
    static let noteMargin = 0.5
    static let dim = 16
    /// What to send when nothing is known: all zeros, the valid flag included.
    static let unknown = [Float](repeating: 0, count: dim)

    /// From strokes drawn on a staff: the note counts when its body is within `noteRange`
    /// staff spaces of the strokes' centre.
    init(strokes: [[CGPoint]], staff: Staff) {
        let b = StrokeRaster.bounds(strokes)
        let (cx, cy) = b.centre
        yStaff = (cy - staff.middleY) / staff.space
        xMeasure = (cx - staff.leftX) / (staff.rightX - staff.leftX)
        barline = min(cx - staff.leftX, staff.rightX - cx) / staff.space
        if let body = staff.noteBody {
            let gx = max(Double(body.minX) - cx, cx - Double(body.maxX), 0) / staff.space
            let gy = max(Double(body.minY) - cy, cy - Double(body.maxY), 0) / staff.space
            let near = (gx * gx + gy * gy).squareRoot() <= Self.noteRange
            noteDx = near ? (cx - Double(body.midX)) / staff.space : nil
            noteDy = near ? (cy - Double(body.midY)) / staff.space : nil
            noteW = near ? Double(body.width) / staff.space : nil
            noteH = near ? Double(body.height) / staff.space : nil
        } else {
            noteDx = nil
            noteDy = nil
            noteW = nil
            noteH = nil
        }
        width = (b.maxX - b.minX) / staff.space
        height = (b.maxY - b.minY) / staff.space
    }

    /// The model's `context` input: sixteen floats roughly in [-1, 1] (granita's
    /// `CONTEXT_FEATURES` order): valid, in / above / below the staff, height, measure
    /// position, barline distance, note near, symbol above / below / left / right of its body,
    /// its offset, width, height.
    var vector: [Float] {
        let hasNote = noteDx != nil && noteDy != nil
        let dx = noteDx ?? 0
        let dy = noteDy ?? 0
        let hw = 0.5 * (noteW ?? 0)
        let hh = 0.5 * (noteH ?? 0)
        let over = hasNote && abs(dx) <= hw + Self.noteMargin
        let beside = hasNote && abs(dy) <= hh + Self.noteMargin
        func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Float { Float(max(lo, min(hi, v))) }
        func flag(_ on: Bool) -> Float { on ? 1 : 0 }
        return [
            1,
            flag(abs(yStaff) <= Self.staffHalfHeight),
            flag(yStaff < -Self.staffHalfHeight),
            flag(yStaff > Self.staffHalfHeight),
            clamp(yStaff / 6, -2, 2),
            clamp(xMeasure * 2 - 1, -1, 1),
            clamp(barline / 8, 0, 1),
            flag(hasNote),
            flag(over && dy < -hh),
            flag(over && dy > hh),
            flag(beside && dx < -hw),
            flag(beside && dx > hw),
            clamp(dx / Self.noteRange, -1, 1),
            clamp(dy / Self.noteRange, -1, 1),
            clamp(width / 8, 0, 1.5),
            clamp(height / 8, 0, 1.5),
        ]
    }
}
