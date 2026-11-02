import CoreML
import Foundation

/// The Core ML export (`granitanet.mlpackage`, compiled into the app), fed the way
/// granita feeds PyTorch: `input` (1, 1, 32, 32) float32, the normalised pixels, and
/// `context` (1, 16); `logits` (1, 43) out, softmax for the confidences.
///
/// A package with a `personal` function (the sidecar has a `head`) is run through it
/// (iOS 18, `MLModelConfiguration.functionName`): the last layer's weights go in as
/// `head_w` / `head_b` and the `features` that layer reads come out, so the app can
/// train that layer on the device (`Head.swift`) and still run the whole network on
/// the accelerator. The layer fed is `head`, the app's copy: the sidecar's at first.
final class GranitaModel {
    struct Prediction {
        let logits: [Float]
        let probabilities: [Float]
        let top: [(label: String, group: String, confidence: Float)]
        /// What the last layer read, 128 values — empty without the `personal` function.
        let features: [Float]
        /// The model call alone.
        let seconds: Double
    }

    let sidecar: Sidecar
    let computeUnits: MLComputeUnits
    /// The last layer as fed to the `personal` function; set to a trained copy to use it.
    var head: HeadWeights? {
        didSet { headArrays = nil }
    }
    private let model: MLModel
    private var headArrays: (w: MLMultiArray, b: MLMultiArray)?

    init(bundle: Bundle = .main, computeUnits: MLComputeUnits = .all) throws {
        sidecar = try Sidecar.load(from: bundle)
        guard let url = bundle.url(forResource: "granitanet", withExtension: "mlmodelc") else {
            throw ShellError.missing("granitanet.mlmodelc")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        if let h = sidecar.head {
            configuration.functionName = h.function
            head = HeadWeights(h)
        }
        model = try MLModel(contentsOf: url, configuration: configuration)
        self.computeUnits = computeUnits
    }

    /// Whether the package is run through its `personal` function, the last layer in hand.
    var personal: Bool { head != nil }

    private static func array(_ shape: [Int], _ values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            for (i, v) in values.enumerated() { buffer[i] = v }
        }
        return array
    }

    func predict(input: [Float], context: [Float], topK: Int = 5) throws -> Prediction {
        precondition(input.count == StrokeRaster.size * StrokeRaster.size && context.count == SymbolContext.dim)
        var inputs: [String: Any] = [
            "input": try Self.array([1, 1, 32, 32], input),
            "context": try Self.array([1, SymbolContext.dim], context),
        ]
        if let head {
            if headArrays == nil {
                headArrays = (try Self.array([head.classes, head.width], head.weight.flatMap { $0 }), try Self.array([head.classes], head.bias))
            }
            inputs["head_w"] = headArrays!.w
            inputs["head_b"] = headArrays!.b
        }
        let features = try MLDictionaryFeatureProvider(dictionary: inputs)
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try model.prediction(from: features)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        guard let array = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ShellError.model("returned no logits")
        }
        let logits = (0..<array.count).map { Float(truncating: array[$0]) }
        let read = output.featureValue(for: "features")?.multiArrayValue
        let feats = read.map { a in (0..<a.count).map { Float(truncating: a[$0]) } } ?? []
        let probabilities = softmax(logits)
        let order = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }.prefix(topK)
        let top = order.map { i -> (label: String, group: String, confidence: Float) in
            let label = sidecar.classes[i]
            return (label, sidecar.groups[label] ?? "other", probabilities[i])
        }
        return Prediction(logits: logits, probabilities: probabilities, top: Array(top), features: feats, seconds: seconds)
    }
}

func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    let e = x.map { exp($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}
