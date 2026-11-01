import SwiftUI

/// The staff, the controls and the model's answer: the top five classes with their
/// confidences, the 32 x 32 image the model saw, the model's time, and the context
/// derived from the strokes. Every stroke's end classifies all the strokes since
/// the last clear, so a sharp can take its four strokes. A notehead can be placed
/// on the staff, which gives the context its note features; the steppers change the
/// size rule (how many staff spaces fill the canvas, granita's 3.0, and the smallest
/// fraction a symbol is drawn at, granita's 0.08) to calibrate it against natural-size
/// writing.
struct ContentView: View {
    @State private var model: Result<GranitaModel, any Error>?
    @State private var generation = 0
    @State private var useContext = true
    @State private var noteState = NoteState.none
    @State private var spacesPerCanvas = StrokeRaster.canvasStaffSpaces
    @State private var minFraction = StrokeRaster.fractionRange.lowerBound
    @State private var strokes: [[CGPoint]] = []
    @State private var staff: Staff?
    @State private var prediction: GranitaModel.Prediction?
    @State private var pixels: [UInt8]?
    @State private var context: SymbolContext?
    @State private var fraction: Double?
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StaffCanvas(generation: $generation, noteState: $noteState) { strokes, staff in
                self.strokes = strokes
                self.staff = staff
                classify()
            } onNotePlaced: { staff in
                self.staff = staff
                noteState = .placed
                classify()
            }
            .frame(height: 300)
            .border(Color.gray.opacity(0.4))
            HStack(spacing: 16) {
                Button("Clear") {
                    generation += 1
                    strokes = []
                    prediction = nil
                    pixels = nil
                    context = nil
                    fraction = nil
                }
                .buttonStyle(.bordered)
                Button(noteLabel) {
                    switch noteState {
                    case .none: noteState = .placing
                    case .placing: noteState = .none
                    case .placed:
                        noteState = .none
                        classify()
                    }
                }
                .buttonStyle(.bordered)
                Toggle("Use the context", isOn: $useContext)
                    .onChange(of: useContext) { _ in classify() }
                    .frame(maxWidth: 220)
                Spacer()
            }
            Stepper(String(format: "%.1f staff spaces fill the canvas", spacesPerCanvas), value: $spacesPerCanvas, in: 3...12, step: 0.5)
                .onChange(of: spacesPerCanvas) { _ in classify() }
                .frame(maxWidth: 420)
            Stepper(String(format: "smallest fraction %.2f", minFraction), value: $minFraction, in: 0.08...0.9, step: 0.04)
                .onChange(of: minFraction) { _ in classify() }
                .frame(maxWidth: 420)
            if let failure {
                Text(failure).foregroundColor(.red)
            }
            HStack(alignment: .top, spacing: 16) {
                if let pixels {
                    PixelsView(pixels: pixels)
                        .frame(width: 128, height: 128)
                        .border(Color.gray)
                }
                VStack(alignment: .leading, spacing: 4) {
                    if let p = prediction {
                        ForEach(Array(p.top.enumerated()), id: \.offset) { _, t in
                            HStack {
                                Text(t.label).bold()
                                Text(t.group).foregroundColor(.secondary)
                                Spacer()
                                Text(String(format: "%.1f %%", t.confidence * 100)).monospacedDigit()
                            }
                        }
                        Text(String(format: "%.2f ms · Core ML, %@", p.seconds * 1000, unitsName))
                            .font(.caption).foregroundColor(.secondary)
                    } else {
                        Text("Draw a symbol on the staff.").foregroundColor(.secondary)
                    }
                    if let c = context, let fraction {
                        Text(contextLine(c, fraction: fraction))
                            .font(.caption.monospacedDigit()).foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: 420, alignment: .leading)
            }
            Spacer()
        }
        .padding()
        .task {
            if model == nil { model = Result { try GranitaModel() } }
            if case .failure(let error) = model { failure = "\(error)" }
        }
    }

    private var noteLabel: String {
        switch noteState {
        case .none: return "Add a note"
        case .placing: return "Tap the staff where the note goes"
        case .placed: return "Remove the note"
        }
    }

    private var unitsName: String {
        guard case .success(let loaded)? = model else { return "" }
        switch loaded.computeUnits {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU and GPU"
        case .cpuAndNeuralEngine: return "CPU and Neural Engine"
        default: return "all compute units"
        }
    }

    private func contextLine(_ c: SymbolContext, fraction: Double) -> String {
        var line = String(
            format: "fraction %.2f · y %.2f · x %.2f · barline %.1f · %.1f × %.1f spaces",
            fraction, c.yStaff, c.xMeasure, c.barline, c.width, c.height)
        if let dx = c.noteDx, let dy = c.noteDy {
            line += String(format: " · note dx %.2f dy %.2f", dx, dy)
        }
        return line
    }

    private func classify() {
        guard var staff, !strokes.isEmpty, case .success(let loaded)? = model else { return }
        if noteState == .none { staff.note = nil }
        let fraction = StrokeRaster.fraction(
            of: strokes, staffSpace: staff.space, spacesPerCanvas: spacesPerCanvas, minFraction: minFraction)
        let px = StrokeRaster.render(strokes, fraction: fraction)
        let ctx = SymbolContext(strokes: strokes, staff: staff)
        do {
            prediction = try loaded.predict(
                input: StrokeRaster.normalize(px), context: useContext ? ctx.vector : SymbolContext.unknown)
            pixels = px
            context = ctx
            self.fraction = fraction
            failure = nil
        } catch {
            failure = "\(error)"
        }
    }
}

/// The model's input, each pixel a square, no smoothing.
struct PixelsView: View {
    let pixels: [UInt8]

    var body: some View {
        Canvas { gc, size in
            let n = StrokeRaster.size
            let w = size.width / CGFloat(n)
            for i in 0..<(n * n) {
                let rect = CGRect(x: CGFloat(i % n) * w, y: CGFloat(i / n) * w, width: w, height: w)
                gc.fill(Path(rect), with: .color(Color(white: Double(pixels[i]) / 255)))
            }
        }
    }
}
