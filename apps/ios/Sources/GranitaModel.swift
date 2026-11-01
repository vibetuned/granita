import CoreML
import Foundation

/// The Core ML export (`granitanet.mlpackage`, compiled into the app), fed the way
/// granita feeds PyTorch: `input` (1, 1, 32, 32) float32, the normalised pixels, and
/// `context` (1, 16); `logits` (1, 51) out, softmax for the confidences.
final class GranitaModel {
    struct Prediction {
        let logits: [Float]
        let probabilities: [Float]
        let top: [(label: String, group: String, confidence: Float)]
        /// The model call alone.
        let seconds: Double
    }

    let sidecar: Sidecar
    let computeUnits: MLComputeUnits
    private let model: MLModel

    init(bundle: Bundle = .main, computeUnits: MLComputeUnits = .all) throws {
        sidecar = try Sidecar.load(from: bundle)
        guard let url = bundle.url(forResource: "granitanet", withExtension: "mlmodelc") else {
            throw ShellError.missing("granitanet.mlmodelc")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: configuration)
        self.computeUnits = computeUnits
    }

    func predict(input: [Float], context: [Float], topK: Int = 5) throws -> Prediction {
        precondition(input.count == StrokeRaster.size * StrokeRaster.size && context.count == SymbolContext.dim)
        let image = try MLMultiArray(shape: [1, 1, 32, 32], dataType: .float32)
        image.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            for (i, v) in input.enumerated() { buffer[i] = v }
        }
        let ctx = try MLMultiArray(shape: [1, NSNumber(value: SymbolContext.dim)], dataType: .float32)
        ctx.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            for (i, v) in context.enumerated() { buffer[i] = v }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: ["input": image, "context": ctx])
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try model.prediction(from: features)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        guard let array = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ShellError.model("returned no logits")
        }
        let logits = (0..<array.count).map { Float(truncating: array[$0]) }
        let probabilities = softmax(logits)
        let order = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }.prefix(topK)
        let top = order.map { i -> (label: String, group: String, confidence: Float) in
            let label = sidecar.classes[i]
            return (label, sidecar.groups[label] ?? "other", probabilities[i])
        }
        return Prediction(logits: logits, probabilities: probabilities, top: Array(top), seconds: seconds)
    }
}

func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    let e = x.map { exp($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}
