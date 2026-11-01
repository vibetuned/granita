import SwiftUI
import UIKit

/// A staff to draw on: five lines a staff space apart, a barline at each end, and a
/// notehead where one was placed (a tap, snapped to the lines and spaces). Every
/// other touch, Pencil or finger, is one stroke; the strokes are kept in the view's
/// own points, the units the model then reads them in (the staff space is known).
final class StaffView: UIView {
    var staffSpace: CGFloat = 24
    private let inset: CGFloat = 24
    private(set) var strokes: [[CGPoint]] = []
    private var current: [CGPoint] = []
    /// The notehead's centre, in the view's points, once placed.
    private(set) var note: CGPoint?
    /// While set, the next touch places the notehead instead of drawing.
    var placingNote = false
    private var placingTouch = false
    var onStrokeEnd: (([[CGPoint]]) -> Void)?
    var onNotePlaced: (() -> Void)?

    var staff: Staff {
        Staff(space: staffSpace, middleY: bounds.midY, leftX: inset, rightX: bounds.maxX - inset, note: note)
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .white
        isMultipleTouchEnabled = false
        contentMode = .redraw
    }

    required init?(coder: NSCoder) { fatalError("not from a storyboard") }

    func clear() {
        strokes = []
        current = []
        setNeedsDisplay()
    }

    func removeNote() {
        note = nil
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let s = staff
        ctx.setStrokeColor(UIColor(white: 0.55, alpha: 1).cgColor)
        ctx.setLineWidth(1)
        for i in -2...2 {
            let y = s.middleY + Double(i) * s.space
            ctx.move(to: CGPoint(x: s.leftX, y: y))
            ctx.addLine(to: CGPoint(x: s.rightX, y: y))
        }
        for x in [s.leftX, s.rightX] {
            ctx.move(to: CGPoint(x: x, y: s.middleY - 2 * s.space))
            ctx.addLine(to: CGPoint(x: x, y: s.middleY + 2 * s.space))
        }
        if let n = note {
            // Ledger lines from the staff out to the note.
            let steps = Int(((Double(n.y) - s.middleY) / s.space).rounded())
            if abs(steps) >= 3 {
                for k in stride(from: 3, through: abs(steps), by: 1) {
                    let y = s.middleY + Double(k * steps.signum()) * s.space
                    ctx.move(to: CGPoint(x: Double(n.x) - s.space, y: y))
                    ctx.addLine(to: CGPoint(x: Double(n.x) + s.space, y: y))
                }
            }
        }
        ctx.strokePath()
        if let n = note {
            ctx.setFillColor(UIColor(white: 0.25, alpha: 1).cgColor)
            ctx.fillEllipse(in: CGRect(x: Double(n.x) - 0.65 * s.space, y: Double(n.y) - 0.5 * s.space, width: 1.3 * s.space, height: s.space))
            // The stem: up on the right from the middle line down, else down on the left.
            let up = Double(n.y) >= s.middleY
            let x = Double(n.x) + (up ? 0.6 : -0.6) * s.space
            ctx.setStrokeColor(UIColor(white: 0.25, alpha: 1).cgColor)
            ctx.setLineWidth(1.5)
            ctx.move(to: CGPoint(x: x, y: Double(n.y)))
            ctx.addLine(to: CGPoint(x: x, y: Double(n.y) + (up ? -3.5 : 3.5) * s.space))
            ctx.strokePath()
        }
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(2.5)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        for stroke in strokes + [current] where !stroke.isEmpty {
            ctx.beginPath()
            ctx.addLines(between: stroke.count == 1 ? [stroke[0], stroke[0]] : stroke)
            ctx.strokePath()
        }
    }

    /// A tap's point snapped to the nearest line or space, within the barlines.
    private func snapped(_ p: CGPoint) -> CGPoint {
        let s = staff
        let halfSteps = ((Double(p.y) - s.middleY) / (s.space / 2)).rounded()
        return CGPoint(x: min(max(Double(p.x), s.leftX), s.rightX), y: s.middleY + halfSteps * s.space / 2)
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        if placingNote {
            placingTouch = true
            return
        }
        current = [touch.location(in: self)]
        setNeedsDisplay()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, !placingTouch else { return }
        // The Pencil samples faster than the screen refreshes; take every sample.
        let samples = event?.coalescedTouches(for: touch) ?? [touch]
        current.append(contentsOf: samples.map { $0.location(in: self) })
        setNeedsDisplay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if placingTouch {
            placingTouch = false
            placingNote = false
            if let touch = touches.first {
                note = snapped(touch.location(in: self))
                setNeedsDisplay()
                onNotePlaced?()
            }
            return
        }
        finish(touches)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        if placingTouch {
            placingTouch = false
            return
        }
        finish(touches)
    }

    private func finish(_ touches: Set<UITouch>) {
        if let touch = touches.first { current.append(touch.location(in: self)) }
        if !current.isEmpty { strokes.append(current) }
        current = []
        setNeedsDisplay()
        onStrokeEnd?(strokes)
    }
}

/// Whether the staff has a notehead, is waiting for a tap to place one, or has none.
enum NoteState: Equatable {
    case none, placing, placed
}

/// The staff in SwiftUI. Bump `generation` to clear the strokes.
struct StaffCanvas: UIViewRepresentable {
    @Binding var generation: Int
    @Binding var noteState: NoteState
    var onStrokeEnd: ([[CGPoint]], Staff) -> Void
    var onNotePlaced: (Staff) -> Void

    func makeUIView(context: Context) -> StaffView {
        let view = StaffView()
        view.onStrokeEnd = { [weak view] strokes in
            guard let view else { return }
            onStrokeEnd(strokes, view.staff)
        }
        view.onNotePlaced = { [weak view] in
            guard let view else { return }
            onNotePlaced(view.staff)
        }
        return view
    }

    func updateUIView(_ view: StaffView, context: Context) {
        if context.coordinator.generation != generation {
            context.coordinator.generation = generation
            view.clear()
        }
        switch noteState {
        case .none:
            view.placingNote = false
            if view.note != nil { view.removeNote() }
        case .placing:
            view.placingNote = true
        case .placed:
            view.placingNote = false
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var generation = 0
    }
}
