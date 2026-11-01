import Foundation

/// The model's sidecar (`granitanet.json`, written by `granita export --sidecar`):
/// the class list and each class's group. The input spec and the stroke rules it
/// also states are mirrored in code, in StrokeRaster and SymbolContext.
struct Sidecar: Decodable {
    let model: String
    let classes: [String]
    let groups: [String: String]

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
