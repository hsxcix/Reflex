import AppKit

/// `--smoke-test <report>`: types one line through the normal key path (one wrong press
/// first), exports CSV and JSON next to the report, writes `passed=true|false`, quits.
/// Runs against a throwaway store (see `AppModel.init`). Driven by tools/smoke.sh.
@MainActor
enum SmokeTest {
    static func run(model: AppModel, report: URL) async {
        var notes: [String] = []
        let passed = await check(model: model, report: report, notes: &notes)
        let text = (["passed=\(passed)"] + notes).joined(separator: "\n") + "\n"
        try? text.write(to: report, atomically: true, encoding: .utf8)
        NSApp.terminate(nil)
    }

    private static func check(model: AppModel, report: URL, notes: inout [String]) async -> Bool {
        guard let engine = model.engine else {
            notes.append("error=engine did not open: \(model.startupError ?? "?")")
            return false
        }
        model.screen = .typing
        var t = Int64(ProcessInfo.processInfo.systemUptime * 1_000_000)
        func press(_ chars: String, code: String, modifiers: UInt32) {
            t += 120_000
            model.handle(CapturedKey(code: code, chars: chars, modifiers: modifiers, timestampUs: t, isDead: false))
        }

        let line = Array(model.view.lines.first ?? "")
        guard !line.isEmpty, model.view.cursor == 0 else {
            notes.append("error=no line to type")
            return false
        }
        notes.append("line=\(String(line))")
        let lineToken = model.lineToken

        // A wrong press keeps the cursor and marks the position.
        let wrong = ["#", "q", "x"].first { $0 != String(line[0]) }!
        press(wrong, code: "", modifiers: 0)
        guard model.view.cursor == 0, model.view.errorPositions.contains(0) else {
            notes.append("error=wrong press moved the cursor or was not marked")
            return false
        }

        // The right presses move it on until the line is finished.
        for (i, c) in line.enumerated() {
            guard model.lineToken == lineToken else { break }
            guard Int(model.view.cursor) == i else {
                notes.append("error=cursor \(model.view.cursor) expected \(i)")
                return false
            }
            let hint = engine.hint()
            let modifiers: UInt32 = (hint?.shift == true ? 1 : 0) | (hint?.option == true ? 2 : 0)
            press(String(c), code: hint?.codes.last ?? "", modifiers: modifiers)
        }
        guard model.lineToken == lineToken + 1 else {
            notes.append("error=line did not finish")
            return false
        }
        notes.append("keys=\(line.count + 1)")

        let dir = report.deletingLastPathComponent().appendingPathComponent("export", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try engine.exportCsv(path: dir.appendingPathComponent("keystrokes.csv").path)
            try engine.exportJson(path: dir.appendingPathComponent("export.json").path)
        } catch {
            notes.append("error=export failed: \(error.localizedDescription)")
            return false
        }
        return true
    }
}
