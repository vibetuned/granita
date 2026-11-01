import CoreML
import XCTest
@testable import GranitaShell

/// The Core ML export and this app's preprocessing against granita's reference
/// (`models/granitanet/fixtures.json`, from scripts/shell_fixtures.py): the model fed
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
            let fraction: Double
            let context_vector: [Float]
            let pixels: [UInt8]
            let logits: [Float]
            let logits_no_context: [Float]
            let top: Top
            var points: [[CGPoint]] { strokes.map { $0.map { CGPoint(x: $0[0], y: $0[1]) } } }
        }
        let staff: StaffSpec
        let classes: [String]
        let fixtures: [Fixture]
        var staffGeometry: Staff { Staff(space: staff.staff_space, middleY: staff.middle_y, leftX: staff.left_x, rightX: staff.right_x) }
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

    /// The context derived from strokes on the fixture staff.
    func testContextVector() {
        let staff = Self.fixtures.staffGeometry
        for f in Self.fixtures.fixtures {
            let v = SymbolContext(strokes: f.points, staff: staff).vector
            XCTAssertEqual(v.count, f.context_vector.count)
            for (a, b) in zip(v, f.context_vector) { XCTAssertEqual(a, b, accuracy: 1e-5, f.drawn_as) }
        }
    }

    /// The app's rasteriser against Pillow's: the size rule exactly; the drawing close
    /// (Core Graphics and Pillow rasterise a wide line differently: on the 13 fixtures
    /// the mean difference is 0.4–2.6 grey levels and at most 14 pixels are far off);
    /// and the reference answer in the model's top two from the app's own drawing. The
    /// top-1 agreement is printed, not asserted: a shape the model is unsure of (the
    /// hollow "whole note", 79 %) can flip on a few pixels.
    func testRasteriserMatchesGranita() throws {
        let model = try GranitaModel(bundle: Bundle.main, computeUnits: .cpuOnly)
        let staff = Self.fixtures.staffGeometry
        var report: [String] = []
        var agreed = 0
        for f in Self.fixtures.fixtures {
            let strokes = f.points
            XCTAssertEqual(StrokeRaster.fraction(of: strokes, staffSpace: staff.space), f.fraction, accuracy: 1e-6, f.drawn_as)
            let px = StrokeRaster.render(strokes, fraction: f.fraction)
            let meanDiff = zip(px, f.pixels).map { abs(Double($0) - Double($1)) }.reduce(0, +) / Double(px.count)
            let farOff = zip(px, f.pixels).filter { abs(Int($0) - Int($1)) > 64 }.count
            XCTAssertLessThan(meanDiff, 4, "\(f.drawn_as): the drawing drifted from Pillow's")
            XCTAssertLessThan(farOff, 32, "\(f.drawn_as): the drawing drifted from Pillow's")
            let p = try model.predict(input: StrokeRaster.normalize(px), context: f.context_vector)
            let labels = p.top.prefix(2).map(\.label)
            XCTAssertTrue(labels.contains(f.top.label), "\(f.drawn_as): \(f.top.label) not in the top two \(labels)")
            if labels.first == f.top.label { agreed += 1 }
            report.append(String(
                format: "%@: mean |Δ| %.2f, %d of %d pixels far off, top-1 %@ (reference %@ at %.0f %%)",
                f.drawn_as, meanDiff, farOff, px.count, p.top.first!.label, f.top.label, f.top.confidence * 100))
        }
        print("rasteriser: top-1 agrees on \(agreed) of \(Self.fixtures.fixtures.count) fixtures")
        print(report.joined(separator: "\n"))
    }
}
