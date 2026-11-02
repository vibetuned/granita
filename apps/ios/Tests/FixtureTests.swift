import CoreML
import XCTest
@testable import GranitaShell

/// The Core ML export and this app's preprocessing against granita's reference
/// (`models/battuta/fixtures.json`, from scripts/shell_fixtures.py): the model fed
/// the reference pixels gives the reference logits (within fp16), the context derived
/// from the strokes is the reference vector, and the app's own rasteriser leads the
/// model to the same answer.
final class FixtureTests: XCTestCase {
    struct Fixtures: Decodable {
        struct StaffSpec: Decodable { let staff_space, middle_y, left_x, right_x: Double }
        struct Top: Decodable { let label: String; let index: Int; let confidence: Float }
        struct Fixture: Decodable {
            let drawn_as: String
            let strokes: [[[Double]]]
            let note: [Double]?
            let fraction: Double
            let context_vector: [Float]
            let pixels: [UInt8]
            let logits: [Float]
            let logits_no_context: [Float]
            /// What the last layer read, from the `personal` function.
            let features: [Float]
            let top: Top
            var points: [[CGPoint]] { strokes.map { $0.map { CGPoint(x: $0[0], y: $0[1]) } } }
        }
        let staff: StaffSpec
        let classes: [String]
        let fixtures: [Fixture]
        var staffGeometry: Staff { Staff(space: staff.staff_space, middleY: staff.middle_y, leftX: staff.left_x, rightX: staff.right_x) }
        /// The staff with the fixture's notehead, if it has one.
        func staff(for f: Fixture) -> Staff {
            var s = staffGeometry
            s.note = f.note.map { CGPoint(x: $0[0], y: $0[1]) }
            return s
        }
    }

    static let fixtures: Fixtures = {
        let url = Bundle(for: FixtureTests.self).url(forResource: "fixtures", withExtension: "json")!
        return try! JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
    }()

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// The export, fed the reference pixels and context, on the CPU and on every unit.
    func testModelMatchesPyTorch() throws {
        for units in [MLComputeUnits.cpuOnly, .all] {
            let model = try GranitaModel(bundle: Bundle.main, computeUnits: units)
            var worst: Float = 0
            for f in Self.fixtures.fixtures {
                let p = try model.predict(input: StrokeRaster.normalize(f.pixels), context: f.context_vector)
                worst = max(worst, maxAbsDiff(p.logits, f.logits))
                XCTAssertEqual(p.top.first?.label, f.top.label, "\(f.drawn_as) with context (\(units.rawValue))")
                let q = try model.predict(input: StrokeRaster.normalize(f.pixels), context: SymbolContext.unknown)
                worst = max(worst, maxAbsDiff(q.logits, f.logits_no_context))
            }
            print("coreml units \(units.rawValue): max |Δ| logits vs PyTorch \(worst)")
            XCTAssertLessThan(worst, 0.1, "fp16 weights: logits within 0.1 of PyTorch (units \(units.rawValue))")
        }
    }

    /// The `personal` function: the features it gives are the reference's (within fp16), and
    /// the sidecar's last layer applied to them is the logits it gives.
    func testPersonalFunction() throws {
        let model = try GranitaModel(bundle: Bundle.main, computeUnits: .cpuOnly)
        XCTAssertTrue(model.personal, "the package has no personal function")
        let head = try XCTUnwrap(model.head)
        XCTAssertEqual(head.classes, model.sidecar.classes.count)
        XCTAssertEqual(head.width, 128)
        var worstFeatures: Float = 0, worstHead: Float = 0
        for f in Self.fixtures.fixtures {
            let p = try model.predict(input: StrokeRaster.normalize(f.pixels), context: f.context_vector)
            XCTAssertEqual(p.features.count, 128, f.drawn_as)
            worstFeatures = max(worstFeatures, maxAbsDiff(p.features, f.features))
            worstHead = max(worstHead, maxAbsDiff(head.logits(p.features), p.logits))
        }
        print("personal: max |Δ| features vs PyTorch \(worstFeatures), head·features vs logits \(worstHead)")
        XCTAssertLessThan(worstFeatures, 0.1)
        XCTAssertLessThan(worstHead, 0.05)
    }

    /// The trainer on the fixtures as replies: the layer keeps them right and stays near the
    /// anchors; ten replies calling the quarter rest an arpeggio turn it into one — against
    /// the interim anchors, which hold that very drawing — and a reset puts the original back.
    func testTrainerFollowsReplies() throws {
        let model = try GranitaModel(bundle: Bundle.main, computeUnits: .cpuOnly)
        let origin = try XCTUnwrap(model.head)
        let anchors = try XCTUnwrap(Anchors.load(from: Bundle.main)).anchors
        var examples: [Example] = []
        for f in Self.fixtures.fixtures {
            let p = try model.predict(input: StrokeRaster.normalize(f.pixels), context: f.context_vector)
            examples.append(Example(features: p.features, label: f.top.index))
        }
        var head = origin
        let report = Trainer.train(&head, origin: origin, buffer: ReplayBuffer(replies: examples), anchors: anchors)
        XCTAssertEqual(report.accuracy, 1)
        XCTAssertLessThan(report.anchorDrift, 0.1)
        let rest = try XCTUnwrap(Self.fixtures.fixtures.first { $0.drawn_as == "quarter rest" })
        let arpeg = try XCTUnwrap(model.sidecar.classes.firstIndex(of: "arpeg"))
        let restFeatures = try model.predict(input: StrokeRaster.normalize(rest.pixels), context: rest.context_vector).features
        var corrected = origin
        let wrong = ReplayBuffer(replies: examples + Array(repeating: Example(features: restFeatures, label: arpeg), count: 10))
        XCTAssertEqual(wrong.examples.filter { $0.label == arpeg }.count, ReplayBuffer.perClass, "the buffer keeps a class's newest few")
        _ = Trainer.train(&corrected, origin: origin, buffer: wrong, anchors: anchors)
        model.head = corrected
        XCTAssertEqual(try model.predict(input: StrokeRaster.normalize(rest.pixels), context: rest.context_vector).top.first?.label, "arpeg")
        model.head = origin
        XCTAssertEqual(try model.predict(input: StrokeRaster.normalize(rest.pixels), context: rest.context_vector).top.first?.label, "restQuarter")
    }

    /// A reply written and read back, musicset's keys in the file, its context whole.
    func testRepliesRoundTrip() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("granita-replies-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let store = Annotations(directory: dir, defaults: try XCTUnwrap(UserDefaults(suiteName: "granita.tests")))
        XCTAssertEqual(store.count, 0)
        let ctx = SymbolContext(yStaff: -0.1, xMeasure: 0.5, barline: 12.4, noteDx: -1, noteDy: 1.4, width: 1.6, height: 2.6, noteW: 1.3, noteH: 4)
        try store.append(Reply(label: "accidentalSharp", writer: store.writer, staffSpace: 24, strokes: [[[395, 255, 0, 0.5], [410, 275, 16, 0.6]]], context: Reply.Context(ctx), staff: Reply.StaffRecord(space: 24, middleY: 300, leftX: 100, rightX: 700), notes: [[424, 300]], fraction: 0.889, predicted: Reply.Predicted(label: "accidentalSharp", confidence: 0.9), device: "apple-pencil", app: "granita-shell/ios", time: Annotations.now()))
        XCTAssertEqual(store.count, 1)
        let text = try String(contentsOf: store.url, encoding: .utf8)
        XCTAssertTrue(text.contains("\"staff_space\":24") && text.contains("\"note_dx\":-1") && text.contains("\"label\":\"accidentalSharp\""), text)
        let back = try XCTUnwrap(store.all().first)
        XCTAssertEqual(back.strokes, [[[395, 255, 0, 0.5], [410, 275, 16, 0.6]]])
        XCTAssertEqual(SymbolContext(back.context).vector, ctx.vector)
        XCTAssertEqual(Annotations(directory: dir, defaults: try XCTUnwrap(UserDefaults(suiteName: "granita.tests"))).count, 1)
    }

    /// The context derived from strokes on the fixture staff, with its notehead when
    /// the fixture has one (the note offsets and the above / below / left / right flags).
    func testContextVector() {
        for f in Self.fixtures.fixtures {
            let v = SymbolContext(strokes: f.points, staff: Self.fixtures.staff(for: f)).vector
            XCTAssertEqual(v.count, f.context_vector.count)
            for (a, b) in zip(v, f.context_vector) { XCTAssertEqual(a, b, accuracy: 1e-5, f.drawn_as) }
        }
    }

    /// The app's rasteriser against Pillow's: the size rule exactly, and the model's answer
    /// from the app's own drawing the same as from Pillow's. Core Graphics and Pillow
    /// drew a wide line differently before the drawing became a port of Pillow's own
    /// (2026-11-02); the pixels are still only guarded against a gross error (a wrong
    /// scale, axis or pen: mean |Δ| under 10 grey levels, fewer than half the ink pixels
    /// far off for a glyph of more than 16) and the exact numbers are printed — a pixel or
    /// two now, but for the fixtures whose fraction the file rounds.
    func testRasteriserMatchesGranita() throws {
        let model = try GranitaModel(bundle: Bundle.main, computeUnits: .cpuOnly)
        let staff = Self.fixtures.staffGeometry
        var report: [String] = []
        for f in Self.fixtures.fixtures {
            let strokes = f.points
            XCTAssertEqual(StrokeRaster.fraction(of: strokes, staffSpace: staff.space), f.fraction, accuracy: 1e-6, f.drawn_as)
            let px = StrokeRaster.render(strokes, fraction: f.fraction)
            let meanDiff = zip(px, f.pixels).map { abs(Double($0) - Double($1)) }.reduce(0, +) / Double(px.count)
            let farOff = zip(px, f.pixels).filter { abs(Int($0) - Int($1)) > 64 }.count
            let ink = f.pixels.filter { $0 < 128 }.count
            XCTAssertLessThan(meanDiff, 10, "\(f.drawn_as): the drawing is not Pillow's")
            // A dot has six ink pixels; its edge alone can move half of them.
            if ink > 16 { XCTAssertLessThan(farOff * 2, ink, "\(f.drawn_as): the drawing is not Pillow's") }
            let p = try model.predict(input: StrokeRaster.normalize(px), context: f.context_vector)
            XCTAssertEqual(p.top.first?.label, f.top.label, f.drawn_as)
            report.append(String(
                format: "%@: mean |Δ| %.2f, %d of %d ink pixels far off, top-1 %@ (reference %@ at %.0f %%)",
                f.drawn_as, meanDiff, farOff, ink, p.top.first!.label, f.top.label, f.top.confidence * 100))
        }
        print(report.joined(separator: "\n"))
    }
}
