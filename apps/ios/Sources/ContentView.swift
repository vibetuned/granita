import SwiftUI

/// The iOS test shell for granita's Core ML export: draw a symbol on a staff with the
/// Pencil or a finger and see what the model makes of it — the top five classes with
/// their confidences, the 32 x 32 image the model saw, its time, and the context derived
/// from the strokes. Every stroke's end classifies all the strokes since the last clear,
/// so a sharp can take its four strokes. Noteheads can be placed on the staff, which give
/// the context its note features (the nearest counts); the steppers change the size rule.
///
/// Then the reply (2026-11-02): tap the class the drawing was — one of the five, or any
/// from the list — and it is saved for the dataset (`Annotations`, musicset's contribution
/// record); Discard drops it. "Train" retrains the model's last layer on the replies, on
/// this device, held to the anchors and the original weights (`Trainer`); the trained
/// layer is kept and used until "Reset". Share sends the replies file.
struct ContentView: View {
    /// What is on the staff: the strokes with their samples, the staff with its notes, the pen.
    struct Drawing {
        var strokes: [[CGPoint]]
        var samples: [[Sample]]
        var staff: Staff
        var pencil: Bool
    }

    @State private var model: Result<GranitaModel, any Error>?
    @State private var generation = 0
    @State private var notesGeneration = 0
    @State private var placingNotes = false
    @State private var useContext = true
    @State private var spacesPerCanvas = StrokeRaster.canvasStaffSpaces
    @State private var minFraction = StrokeRaster.fractionRange.lowerBound
    @State private var drawing: Drawing?
    @State private var prediction: GranitaModel.Prediction?
    @State private var pixels: [UInt8]?
    @State private var context: SymbolContext?
    @State private var fraction: Double?
    @State private var failure: String?
    @State private var choosing = false
    @State private var replies = Annotations()
    @State private var replyCount = 0
    @State private var training = false
    @State private var report: Trainer.Report?
    @State private var headTrained = false

    private var headURL: URL { replies.url.deletingLastPathComponent().appendingPathComponent("head.json") }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StaffCanvas(generation: $generation, notesGeneration: $notesGeneration, placingNotes: $placingNotes) { view in
                drawing = Drawing(strokes: view.strokes, samples: view.samples, staff: view.staff, pencil: view.pencil)
                classify()
            } onNotesChanged: { view in
                if drawing == nil { drawing = Drawing(strokes: [], samples: [], staff: view.staff, pencil: false) }
                drawing?.staff = view.staff
                classify()
            }
            .frame(height: 300)
            .border(Color.gray.opacity(0.4))
            HStack(spacing: 16) {
                Button("Clear") { clearDrawing() }
                    .buttonStyle(.bordered)
                Button(placingNotes ? "Done with the notes" : "Add notes") { placingNotes.toggle() }
                    .buttonStyle(.bordered)
                Button("Clear notes") {
                    notesGeneration += 1
                    drawing?.staff.notes = []
                    classify()
                }
                .buttonStyle(.bordered)
                .disabled(drawing?.staff.notes.isEmpty ?? true)
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
                        // The reply: the class it was, among the five or any other, or none.
                        ForEach(Array(p.top.enumerated()), id: \.offset) { _, t in
                            Button {
                                reply(t.label)
                            } label: {
                                HStack {
                                    Text(t.label).bold()
                                    Text(t.group).foregroundColor(.secondary)
                                    Spacer()
                                    Text(String(format: "%.1f %%", t.confidence * 100)).monospacedDigit()
                                }
                            }
                            .buttonStyle(.plain)
                        }
                        HStack {
                            Button("Another…") { choosing = true }.buttonStyle(.bordered)
                            Button("Discard") { clearDrawing() }.buttonStyle(.bordered)
                        }
                        Text(String(format: "%.2f ms · Core ML, %@%@", p.seconds * 1000, unitsName, headTrained ? ", the trained layer" : ""))
                            .font(.caption).foregroundColor(.secondary)
                    } else {
                        Text("Draw a symbol on the staff, then tap what it was.").foregroundColor(.secondary)
                    }
                    if let c = context, let fraction {
                        Text(contextLine(c, fraction: fraction))
                            .font(.caption.monospacedDigit()).foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: 420, alignment: .leading)
            }
            HStack(spacing: 12) {
                Button(training ? "Training…" : "Train on \(replyCount) replies") { train() }
                    .buttonStyle(.bordered)
                    .disabled(training || replyCount == 0 || loadedModel == nil)
                Button("Reset the layer") { resetHead() }
                    .buttonStyle(.bordered)
                    .disabled(!headTrained)
                if replyCount > 0 {
                    ShareLink(item: replies.url) { Label("Share the replies", systemImage: "square.and.arrow.up") }
                        .buttonStyle(.bordered)
                }
            }
            Text(statusLine).font(.caption).foregroundColor(.secondary)
            Spacer()
        }
        .padding()
        .sheet(isPresented: $choosing) {
            ClassPicker(classes: loadedModel?.sidecar.classes ?? []) { label in
                choosing = false
                reply(label)
            }
        }
        .task {
            if model == nil {
                model = Result { try GranitaModel() }
                if case .success(let m) = model, m.personal, let saved = HeadWeights.load(from: headURL) {
                    m.head = saved
                    headTrained = true
                }
                replyCount = replies.count
            }
            if case .failure(let error) = model { failure = "\(error)" }
        }
    }

    private var loadedModel: GranitaModel? {
        if case .success(let m) = model { return m }
        return nil
    }

    private var unitsName: String {
        switch loadedModel?.computeUnits {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU and GPU"
        case .cpuAndNeuralEngine: return "CPU and Neural Engine"
        default: return "all units"
        }
    }

    private var statusLine: String {
        var line = "\(replyCount) replies saved as \(Annotations.fileName)"
        if let m = loadedModel, !m.personal { line += " · the model has no personal function: no training" }
        if let r = report {
            line += String(format: " · trained on %d (the buffer) with %d anchors: loss %.3f, %.0f %% right, anchor drift %.4f", r.examples, r.anchors, r.loss, r.accuracy * 100, r.anchorDrift)
        } else if headTrained {
            line += " · the trained layer is in use"
        }
        return line
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

    private func clearDrawing() {
        generation += 1
        drawing = drawing.map { Drawing(strokes: [], samples: [], staff: $0.staff, pencil: $0.pencil) }
        prediction = nil
        pixels = nil
        context = nil
        fraction = nil
    }

    private func classify() {
        guard case .success(let m) = model, let d = drawing, !d.strokes.isEmpty else { return }
        let fraction = StrokeRaster.fraction(of: d.strokes, staffSpace: d.staff.space, spacesPerCanvas: spacesPerCanvas, minFraction: minFraction)
        let px = StrokeRaster.render(d.strokes, fraction: fraction)
        let ctx = SymbolContext(strokes: d.strokes, staff: d.staff)
        do {
            prediction = try m.predict(input: StrokeRaster.normalize(px), context: useContext ? ctx.vector : SymbolContext.unknown)
            pixels = px
            context = ctx
            self.fraction = fraction
            failure = nil
        } catch {
            failure = "\(error)"
        }
    }

    /// The drawing was `label`: saved for the dataset, and the staff cleared for the next.
    private func reply(_ label: String) {
        guard let d = drawing, let c = context, let fraction, !d.strokes.isEmpty else { return }
        let record = Reply(
            label: label,
            writer: replies.writer,
            staffSpace: d.staff.space,
            strokes: d.samples.map { stroke in stroke.map { s in [Double(s.point.x), Double(s.point.y), (s.t * 10).rounded() / 10] + (s.pressure.map { [($0 * 1000).rounded() / 1000] } ?? []) } },
            context: Reply.Context(c),
            staff: Reply.StaffRecord(space: d.staff.space, middleY: d.staff.middleY, leftX: d.staff.leftX, rightX: d.staff.rightX),
            notes: d.staff.notes.map { [Double($0.x), Double($0.y)] },
            fraction: fraction,
            predicted: prediction?.top.first.map { Reply.Predicted(label: $0.label, confidence: $0.confidence) },
            device: d.pencil ? "apple-pencil" : "finger",
            app: "granita-shell/ios",
            time: Annotations.now())
        do {
            try replies.append(record)
            replyCount = replies.count
            clearDrawing()
        } catch {
            failure = "the reply was not saved: \(error)"
        }
    }

    /// The last layer retrained on this device from every reply, on a background task.
    private func train() {
        guard let m = loadedModel, m.personal, let origin = m.sidecar.head.map(HeadWeights.init) else { return }
        training = true
        let records = replies.all()
        let anchors = Anchors.load()?.anchors ?? []
        let classes = m.sidecar.classes
        Task.detached(priority: .userInitiated) {
            var examples: [Example] = []
            for r in records {
                guard let label = classes.firstIndex(of: r.label) else { continue }
                let strokes = r.strokes.map { $0.map { CGPoint(x: $0[0], y: $0[1]) } }.filter { !$0.isEmpty }
                guard !strokes.isEmpty else { continue }
                let px = StrokeRaster.render(strokes, fraction: r.fraction)
                if let p = try? m.predict(input: StrokeRaster.normalize(px), context: SymbolContext(r.context).vector) {
                    examples.append(Example(features: p.features, label: label))
                }
            }
            var head = origin
            let result = Trainer.train(&head, origin: origin, buffer: ReplayBuffer(replies: examples), anchors: anchors)
            await MainActor.run {
                m.head = head
                try? head.save(to: headURL)
                headTrained = true
                report = result
                training = false
                classify()
            }
        }
    }

    private func resetHead() {
        guard let m = loadedModel, let original = m.sidecar.head.map(HeadWeights.init) else { return }
        m.head = original
        try? FileManager.default.removeItem(at: headURL)
        headTrained = false
        report = nil
        classify()
    }
}

/// Every class, to name the one the drawing was when it is not among the five.
struct ClassPicker: View {
    let classes: [String]
    let onPick: (String) -> Void
    @State private var query = ""

    var body: some View {
        NavigationStack {
            List(classes.filter { query.isEmpty || $0.localizedCaseInsensitiveContains(query) }, id: \.self) { c in
                Button(c) { onPick(c) }
            }
            .searchable(text: $query)
            .navigationTitle("It was a…")
        }
    }
}

/// The 32 x 32 image the model saw, enlarged without smoothing.
struct PixelsView: View {
    let pixels: [UInt8]

    var body: some View {
        Canvas { context, size in
            let cell = size.width / 32
            for y in 0..<32 {
                for x in 0..<32 {
                    let v = Double(pixels[y * 32 + x]) / 255
                    context.fill(Path(CGRect(x: Double(x) * cell, y: Double(y) * cell, width: cell, height: cell)), with: .color(Color(white: v)))
                }
            }
        }
    }
}
