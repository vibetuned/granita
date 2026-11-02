import Foundation

/// The model's sidecar (`granitanet.json`, written by `granita export --sidecar`):
/// the class list, each class's group and, since the `personal` function, the last
/// layer's weights the app starts its own copy from. The input spec and the stroke
/// rules it also states are mirrored in code, in StrokeRaster and SymbolContext.
struct Sidecar: Decodable {
    /// The last layer, `Linear(128 -> classes)`, as trained: the `personal` Core ML
    /// function takes a copy as inputs `head_w` / `head_b` and gives the features back.
    struct Head: Decodable {
        let function: String
        let shape: [Int]
        let weight: [[Float]]
        let bias: [Float]
    }

    let model: String
    let classes: [String]
    let groups: [String: String]
    let head: Head?

    static func load(from bundle: Bundle = .main) throws -> Sidecar {
        guard let url = bundle.url(forResource: "granitanet", withExtension: "json") else {
            throw ShellError.missing("granitanet.json")
        }
        return try JSONDecoder().decode(Sidecar.self, from: Data(contentsOf: url))
    }
}

enum ShellError: Error, CustomStringConvertible {
    case missing(String)
    case model(String)

    var description: String {
        switch self {
        case .missing(let what): return "\(what) is not in the app bundle"
        case .model(let what): return "the model \(what)"
        }
    }
}
