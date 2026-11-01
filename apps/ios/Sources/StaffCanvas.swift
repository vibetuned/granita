import SwiftUI
import UIKit

/// A staff to draw on: five lines a staff space apart, a barline at each end. Every
/// touch, Pencil or finger, is one stroke; the strokes are kept in the view's own
/// points, the units the model then reads them in (the staff space is known).
final class StaffView: UIView {
    var staffSpace: CGFloat = 24
    private let inset: CGFloat = 24
    private(set) var strokes: [[CGPoint]] = []
    private var current: [CGPoint] = []
    var onStrokeEnd: (([[CGPoint]]) -> Void)?

    var staff: Staff {
        Staff(space: staffSpace, middleY: bounds.midY, leftX: inset, rightX: bounds.maxX - inset)
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
        ctx.strokePath()
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

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        current = [touch.location(in: self)]
        setNeedsDisplay()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        // The Pencil samples faster than the screen refreshes; take every sample.
        let samples = event?.coalescedTouches(for: touch) ?? [touch]
        current.append(contentsOf: samples.map { $0.location(in: self) })
        setNeedsDisplay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) { finish(touches) }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { finish(touches) }

    private func finish(_ touches: Set<UITouch>) {
        if let touch = touches.first { current.append(touch.location(in: self)) }
        if !current.isEmpty { strokes.append(current) }
        current = []
        setNeedsDisplay()
        onStrokeEnd?(strokes)
    }
}

/// The staff in SwiftUI. Bump `generation` to clear it.
struct StaffCanvas: UIViewRepresentable {
    @Binding var generation: Int
    var onStrokeEnd: ([[CGPoint]], Staff) -> Void

    func makeUIView(context: Context) -> StaffView {
        let view = StaffView()
        view.onStrokeEnd = { [weak view] strokes in
            guard let view else { return }
            onStrokeEnd(strokes, view.staff)
        }
        return view
    }

    func updateUIView(_ view: StaffView, context: Context) {
        if context.coordinator.generation != generation {
            context.coordinator.generation = generation
            view.clear()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var generation = 0
    }
}
