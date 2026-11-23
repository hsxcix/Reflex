import AppKit

/// `--smoke-test <report>`: types one line through the real key path (synthetic NSEvents
/// into `KeyCaptureNSView`: key-code mapping, repeat filter), one wrong press first, checks
/// that an auto-repeat is ignored and that stats load, exports CSV and JSON next to the
/// report, writes `passed=true|false`, quits.
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
        let capture = KeyCaptureNSView()
        capture.onKey = model.handle
        var t = ProcessInfo.processInfo.systemUptime
        var unmapped: [String] = []
        func press(_ chars: String, code: String, modifiers: UInt32, repeating: Bool = false) {
            t += 0.12
            guard let keyCode = PhysicalKey.keyCode(for: code) else { unmapped.append(code); return }
            var flags: NSEvent.ModifierFlags = []
            if modifiers & 1 != 0 { flags.insert(.shift) }
            if modifiers & 2 != 0 { flags.insert(.option) }
            guard let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: flags, timestamp: t,
                                               windowNumber: 0, context: nil, characters: chars,
                                               charactersIgnoringModifiers: chars, isARepeat: repeating,
                                               keyCode: keyCode) else { unmapped.append(code); return }
            capture.keyDown(with: event)
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
        press(wrong, code: "Backslash", modifiers: 0)
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
            notes.append("error=line did not finish (unmapped codes: \(unmapped))")
            return false
        }
        notes.append("keys=\(line.count + 1)")

        // A held key's auto-repeat is not a keystroke.
        let cursor = model.view.cursor
        if let c = model.view.lines.first?.unicodeScalars.first, let hint = engine.hint() {
            press(String(c), code: hint.codes.last ?? "", modifiers: hint.shift ? 1 : 0, repeating: true)
        }
        guard model.view.cursor == cursor else {
            notes.append("error=auto-repeat was counted as a keystroke")
            return false
        }

        // Stats load (separate read connection) and show the wrong press as a mix-up.
        do {
            let stats = try engine.stats()
            guard !stats.confusions.isEmpty else {
                notes.append("error=stats show no mix-up after a wrong press")
                return false
            }
            notes.append("confusions=\(stats.confusions.count)")
        } catch {
            notes.append("error=stats failed: \(error.localizedDescription)")
            return false
        }

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
