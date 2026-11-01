import SwiftUI

/// The staff, the controls and the model's answer: the top five classes with their
/// confidences, the 32 x 32 image the model saw, the model's time, and the context
/// derived from the strokes. Every stroke's end classifies all the strokes since
/// the last clear, so a sharp can take its four strokes.
struct ContentView: View {
    @State private var model: Result<GranitaModel, any Error>?
    @State private var generation = 0
    @State private var useContext = true
    @State private var strokes: [[CGPoint]] = []
    @State private var staff: Staff?
    @State private var prediction: GranitaModel.Prediction?
    @State private var pixels: [UInt8]?
    @State private var context: SymbolContext?
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StaffCanvas(generation: $generation) { strokes, staff in
                self.strokes = strokes
                self.staff = staff
                classify()
            }
            .frame(height: 300)
            .border(Color.gray.opacity(0.4))
            HStack {
                Button("Clear") {
                    generation += 1
                    strokes = []
                    prediction = nil
                    pixels = nil
                    context = nil
                }
                .buttonStyle(.bordered)
                Toggle("Use the context", isOn: $useContext)
                    .onChange(of: useContext) { _ in classify() }
                    .frame(maxWidth: 260)
                Spacer()
            }
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
                    if let c = context {
                        Text(String(
                            format: "y %.2f · x %.2f · barline %.1f · %.1f × %.1f spaces",
                            c.yStaff, c.xMeasure, c.barline, c.width, c.height))
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

    private var unitsName: String {
        guard case .success(let loaded)? = model else { return "" }
        switch loaded.computeUnits {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU and GPU"
        case .cpuAndNeuralEngine: return "CPU and Neural Engine"
        default: return "all compute units"
        }
    }

    private func classify() {
        guard let staff, !strokes.isEmpty, case .success(let loaded)? = model else { return }
        let fraction = StrokeRaster.fraction(of: strokes, staffSpace: staff.space)
        let px = StrokeRaster.render(strokes, fraction: fraction)
        let ctx = SymbolContext(strokes: strokes, staff: staff)
        do {
            prediction = try loaded.predict(
                input: StrokeRaster.normalize(px), context: useContext ? ctx.vector : SymbolContext.unknown)
            pixels = px
            context = ctx
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
