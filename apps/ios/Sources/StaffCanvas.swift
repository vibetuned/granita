import SwiftUI
import UIKit

/// One point of a stroke as the pen gave it: where, when (ms from the drawing's first
/// sample) and how hard (0–1 for the Pencil; nil for a finger).
struct Sample {
    var point: CGPoint
    var t: Double
    var pressure: Double?
}

/// A staff to draw on: five lines a staff space apart, a barline at each end, and the
/// noteheads placed on it (a tap, snapped to the lines and spaces, while placing; a tap
/// on a note takes it away). Every other touch, Pencil or finger, is one stroke; the
/// strokes are kept in the view's own points, the units the model then reads them in
/// (the staff space is known), with their samples' time and pressure for the dataset.
final class StaffView: UIView {
    var staffSpace: CGFloat = 24
    private let inset: CGFloat = 24
    private(set) var strokes: [[CGPoint]] = []
    private(set) var samples: [[Sample]] = []
    private var current: [Sample] = []
    private var started: TimeInterval?
    /// The noteheads' centres, in the view's points.
    private(set) var notes: [CGPoint] = []
    /// While set, a touch places a notehead — or removes the one it lands on — instead of drawing.
    var placingNotes = false
    private var placingTouch = false
    /// Whether the last stroke came from the Pencil.
    private(set) var pencil = false
    var onStrokeEnd: (([[CGPoint]]) -> Void)?
    var onNotesChanged: (() -> Void)?

    var staff: Staff {
        Staff(space: staffSpace, middleY: bounds.midY, leftX: inset, rightX: bounds.maxX - inset, notes: notes)
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
        samples = []
        current = []
        started = nil
        setNeedsDisplay()
    }

    func clearNotes() {
        notes = []
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
        for n in notes {
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
        for n in notes {
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
        for stroke in strokes + [current.map { $0.point }] where !stroke.isEmpty {
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

    /// The note a tap lands on: within a space of a head's centre.
    private func noteAt(_ p: CGPoint) -> Int? {
        notes.firstIndex { hypot($0.x - p.x, $0.y - p.y) <= staffSpace }
    }

    private func sample(_ touch: UITouch) -> Sample {
        let now = touch.timestamp
        if started == nil { started = now }
        let pressure = touch.type == .pencil && touch.maximumPossibleForce > 0 ? Double(touch.force / touch.maximumPossibleForce) : nil
        return Sample(point: touch.location(in: self), t: (now - started!) * 1000, pressure: pressure)
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        if placingNotes {
            placingTouch = true
            return
        }
        pencil = touch.type == .pencil
        current = [sample(touch)]
        setNeedsDisplay()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, !placingTouch else { return }
        // The Pencil samples faster than the screen refreshes; take every sample.
        let coalesced = event?.coalescedTouches(for: touch) ?? [touch]
        current.append(contentsOf: coalesced.map { sample($0) })
        setNeedsDisplay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if placingTouch {
            placingTouch = false
            if let touch = touches.first {
                let p = touch.location(in: self)
                if let i = noteAt(p) { notes.remove(at: i) } else { notes.append(snapped(p)) }
                setNeedsDisplay()
                onNotesChanged?()
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
        if let touch = touches.first { current.append(sample(touch)) }
        if !current.isEmpty {
            strokes.append(current.map { $0.point })
            samples.append(current)
        }
        current = []
        setNeedsDisplay()
        onStrokeEnd?(strokes)
    }
}

/// The staff in SwiftUI. Bump `generation` to clear the strokes, `notesGeneration` to clear the notes.
struct StaffCanvas: UIViewRepresentable {
    @Binding var generation: Int
    @Binding var notesGeneration: Int
    @Binding var placingNotes: Bool
    var onStrokeEnd: (StaffView) -> Void
    var onNotesChanged: (StaffView) -> Void

    func makeUIView(context: Context) -> StaffView {
        let view = StaffView()
        view.onStrokeEnd = { [weak view] _ in
            guard let view else { return }
            onStrokeEnd(view)
        }
        view.onNotesChanged = { [weak view] in
            guard let view else { return }
            onNotesChanged(view)
        }
        return view
    }

    func updateUIView(_ view: StaffView, context: Context) {
        if context.coordinator.generation != generation {
            context.coordinator.generation = generation
            view.clear()
        }
        if context.coordinator.notesGeneration != notesGeneration {
            context.coordinator.notesGeneration = notesGeneration
            view.clearNotes()
        }
        view.placingNotes = placingNotes
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var generation = 0
        var notesGeneration = 0
    }
}
