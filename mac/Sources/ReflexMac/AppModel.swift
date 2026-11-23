import AppKit
import Observation
import SwiftUI

enum Screen { case typing, stats }

enum ExportFormat { case csv, json }

/// Bridges the Rust core to SwiftUI. All decisions are made by the core; this only
/// forwards events and mirrors state.
@MainActor
@Observable
final class AppModel {
    var screen: Screen = .typing
    private(set) var view: TypingView
    private(set) var pool: PoolSettings
    private(set) var keyboard: [KeyCap] = []
    private(set) var hint: KeyHint?
    private(set) var lastBatch: BatchSummary?
    private(set) var batchToken = 0
    private(set) var errorToken = 0
    private(set) var lineToken = 0
    private(set) var startupError: String?
    private(set) var stats: Stats?
    /// Shown as an alert; set when something the user did could not be completed.
    var errorMessage: String?
    private(set) var exporting = false

    @ObservationIgnored let engine: ReflexEngine?
    @ObservationIgnored private var hesitationTask: Task<Void, Never>?
    @ObservationIgnored private var idleTask: Task<Void, Never>?
    @ObservationIgnored private var mouseTask: Task<Void, Never>?
    @ObservationIgnored private var layoutObserver: NSObjectProtocol?
    @ObservationIgnored private var resignObserver: NSObjectProtocol?
    @ObservationIgnored private var clockObservers: [NSObjectProtocol] = []
    /// `--smoke-test <report>`: run the automated check against a throwaway store, then quit.
    @ObservationIgnored let smokeReport: URL?

    static let hesitationDelay: Duration = .milliseconds(1200)

    init() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--smoke-test"), args.indices.contains(i + 1) {
            smokeReport = URL(fileURLWithPath: args[i + 1])
        } else {
            smokeReport = nil
        }
        let dir: URL
        if let smokeReport {
            dir = smokeReport.deletingLastPathComponent().appendingPathComponent("store", isDirectory: true)
        } else if let custom = ProcessInfo.processInfo.environment["REFLEX_DATA"], !custom.isEmpty {
            dir = URL(fileURLWithPath: custom, isDirectory: true)
        } else {
            dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Reflex", isDirectory: true)
        }
        do {
            engine = try ReflexEngine.open(dataDir: dir.path, resourcesDir: Bundle.main.resourcePath ?? "",
                                           utcOffsetMin: AppModel.utcOffsetMinutes())
            startupError = nil
        } catch {
            engine = nil
            startupError = error.localizedDescription
        }
        view = engine?.view() ?? TypingView(lines: [], cursor: 0, errorPositions: [], liveWpm: 0,
                                             liveAccuracy: 1, todayMinutes: 0, layoutName: "", realWpm: nil)
        pool = engine?.pool() ?? PoolSettings(languages: ["de", "en"], vocabulary: "broad", code: false)
        applyInputSource()
        layoutObserver = InputSource.observe { [weak self] in
            MainActor.assumeIsolated { self?.applyInputSource() }
        }
        // Switching to another app is a pause, not a slow keystroke.
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.engine?.pause() }
        }
        // Days end at local midnight: follow time zone and daylight saving changes.
        for name in [NSNotification.Name.NSSystemTimeZoneDidChange, .NSCalendarDayChanged,
                     NSApplication.didBecomeActiveNotification] {
            clockObservers.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.syncClock() }
            })
        }
    }

    static func utcOffsetMinutes() -> Int32 {
        Int32(TimeZone.current.secondsFromGMT() / 60)
    }

    private func syncClock() {
        guard let engine else { return }
        do { try engine.setUtcOffset(utcOffsetMin: AppModel.utcOffsetMinutes()) } catch { report(error) }
    }

    /// Show an error once; while an alert is open, further errors don't stack up.
    func report(_ error: Error) {
        guard errorMessage == nil else { return }
        errorMessage = error.localizedDescription
    }

    /// True while keys are coming in; turns false 2.5 s after the last key.
    private(set) var isTyping = false
    /// True for a moment after the mouse moved (reveals the top bar while typing).
    private(set) var mouseActive = false

    func handle(_ key: CapturedKey) {
        guard let engine, screen == .typing else { return }
        let event = KeyEvent(code: key.code, chars: key.chars, modifiers: key.modifiers,
                             timestampUs: key.timestampUs, isDead: key.isDead)
        let result: KeyResult
        do {
            result = try engine.keyEvent(event: event)
        } catch {
            report(error)
            return
        }
        hint = nil
        markTyping()
        if !result.correct && !key.isDead { errorToken += 1 }
        if result.lineFinished { lineToken += 1 }
        if let batch = result.batch {
            lastBatch = batch
            batchToken += 1
        }
        view = engine.view()
        scheduleHesitationHint()
    }

    func togglePool(language: String) {
        var langs = pool.languages
        if langs.contains(language) {
            guard langs.count > 1 else { return }
            langs.removeAll { $0 == language }
        } else {
            langs.append(language)
        }
        updatePool(PoolSettings(languages: langs, vocabulary: pool.vocabulary, code: pool.code))
    }

    func setVocabulary(_ vocabulary: String) {
        updatePool(PoolSettings(languages: pool.languages, vocabulary: vocabulary, code: pool.code))
    }

    func toggleCode() {
        updatePool(PoolSettings(languages: pool.languages, vocabulary: pool.vocabulary, code: !pool.code))
    }

    func showStats() {
        engine?.pause()
        do { stats = try engine?.stats() } catch { report(error) }
        screen = .stats
    }

    /// While the stats screen is open, refresh a few times so the forecast (computed in
    /// the background) shows up.
    func refreshStatsLoop() async {
        for _ in 0..<6 {
            try? await Task.sleep(for: .seconds(1.5))
            guard screen == .stats else { return }
            stats = try? engine?.stats()
        }
    }

    func showTyping() {
        engine?.pause()
        screen = .typing
    }

    func exportData(_ format: ExportFormat) {
        guard let engine, !exporting else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = format == .csv ? "reflex-keystrokes.csv" : "reflex-export.json"
        panel.allowedContentTypes = [format == .csv ? .commaSeparatedText : .json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        exporting = true
        // Large histories take a moment: export off the main thread, report failures.
        Task.detached {
            let failure: String?
            do {
                switch format {
                case .csv: try engine.exportCsv(path: url.path)
                case .json: try engine.exportJson(path: url.path)
                }
                failure = nil
            } catch {
                failure = "Export failed: \(error.localizedDescription)"
            }
            await MainActor.run { [weak self] in
                self?.exporting = false
                if let failure, self?.errorMessage == nil { self?.errorMessage = failure }
            }
        }
    }

    func refresh() {
        guard let engine else { return }
        view = engine.view()
    }

    private func updatePool(_ settings: PoolSettings) {
        guard let engine else { return }
        do { try engine.setPool(settings: settings) } catch { report(error) }
        pool = engine.pool()
        view = engine.view()
    }

    private func applyInputSource() {
        guard let engine else { return }
        let source = InputSource.current()
        do { try engine.setLayout(layout: source.layout, geometry: source.geometry) } catch { report(error) }
        keyboard = engine.keyboard()
        view = engine.view()
    }

    func mouseMoved() {
        if !mouseActive { mouseActive = true }
        mouseTask?.cancel()
        mouseTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            self?.mouseActive = false
        }
    }

    private func markTyping() {
        if !isTyping { isTyping = true }
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2.5))
            guard !Task.isCancelled else { return }
            self?.isTyping = false
        }
    }

    private func scheduleHesitationHint() {
        hesitationTask?.cancel()
        hesitationTask = Task { [weak self] in
            try? await Task.sleep(for: AppModel.hesitationDelay)
            guard !Task.isCancelled, let self, self.screen == .typing else { return }
            self.hint = self.engine?.hint()
        }
    }
}
