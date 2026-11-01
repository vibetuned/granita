/**
 * The iOS test shell for granita's Core ML export: draw a symbol on a staff with
 * the Pencil or a finger and see what the model makes of it. The pieces an editor
 * would lift are StrokeRaster (the strokes to the model's image), SymbolContext
 * (where the mark sits) and GranitaModel (the export, fed and read).
 */
import SwiftUI

@main
struct GranitaShellApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
