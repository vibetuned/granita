import Foundation

/// A reply saved for the dataset: musicset's contribution record (`musicset.strokes`:
/// `label`, `writer`, `staff_space`, `strokes` with time and pressure, `context`,
/// `device`, `app`, `time`) and what else the shell knows — the staff the drawing was
/// made on, the notes placed on it, the fraction it was drawn at, and the model's answer.
/// Keys are snake_case in the file.
struct Reply: Codable {
    struct Context: Codable {
        var yStaff, xMeasure, barline: Double
        var noteDx, noteDy, noteW, noteH: Double?
        var width, height: Double
    }

    struct StaffRecord: Codable {
        var space, middleY, leftX, rightX: Double
    }

    struct Predicted: Codable {
        var label: String
        var confidence: Float
    }

    var label: String
    var writer: String
    var staffSpace: Double
    /// One list per pen-down: `[x, y, t_ms, pressure]`, or `[x, y, t_ms]` from a finger.
    var strokes: [[[Double]]]
    var context: Context
    var staff: StaffRecord
    /// The noteheads on the staff, `[x, y]` each, in the strokes' coordinates.
    var notes: [[Double]]
    var fraction: Double
    var predicted: Predicted?
    var device: String
    var app: String
    var time: String
}

extension Reply.Context {
    init(_ c: SymbolContext) {
        self.init(yStaff: c.yStaff, xMeasure: c.xMeasure, barline: c.barline, noteDx: c.noteDx, noteDy: c.noteDy, noteW: c.noteW, noteH: c.noteH, width: c.width, height: c.height)
    }
}

extension SymbolContext {
    /// The context a reply recorded, for its vector again.
    init(_ r: Reply.Context) {
        self.init(yStaff: r.yStaff, xMeasure: r.xMeasure, barline: r.barline, noteDx: r.noteDx, noteDy: r.noteDy, width: r.width, height: r.height, noteW: r.noteW, noteH: r.noteH)
    }
}

/// The replies on this device: `replies.jsonl` in the app's Documents folder (the Files
/// app shows it; a Share button sends it), one record a line, appended as they come.
/// The writer id is this install's, a UUID kept in the defaults.
final class Annotations {
    static let fileName = "replies.jsonl"
    let url: URL
    let writer: String
    private(set) var count: Int

    init(directory: URL? = nil, defaults: UserDefaults = .standard) {
        let dir = directory ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        url = dir.appendingPathComponent(Self.fileName)
        if let id = defaults.string(forKey: "granita.writer") {
            writer = id
        } else {
            writer = UUID().uuidString.lowercased()
            defaults.set(writer, forKey: "granita.writer")
        }
        count = ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n", omittingEmptySubsequences: true).count
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    static func now() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: Date())
    }

    func append(_ reply: Reply) throws {
        var line = try Self.encoder.encode(reply)
        line.append(contentsOf: [0x0A])
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: url, options: .atomic)
        }
        count += 1
    }

    /// Every reply in the file, in order; a line that does not parse is skipped.
    func all() -> [Reply] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n", omittingEmptySubsequences: true).compactMap { try? Self.decoder.decode(Reply.self, from: Data($0.utf8)) }
    }
}
