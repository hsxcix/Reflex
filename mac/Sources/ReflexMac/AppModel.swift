import AppKit
import Observation
import SwiftUI

enum Screen { case typing, stats }

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
    private(set) var lastKeyAt = Date.distantPast
    private(set) var startupError: String?
    private(set) var stats: Stats?

    @ObservationIgnored let engine: ReflexEngine?
    @ObservationIgnored private var hesitationTask: Task<Void, Never>?
    @ObservationIgnored private var layoutObserver: NSObjectProtocol?

    static let hesitationDelay: Duration = .milliseconds(1200)

    init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Reflex", isDirectory: true)
        do {
            engine = try ReflexEngine.open(dataDir: dir.path, resourcesDir: Bundle.main.resourcePath ?? "")
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
    }

    var isTyping: Bool { Date().timeIntervalSince(lastKeyAt) < 2.5 }

    func handle(_ key: CapturedKey) {
        guard let engine, screen == .typing else { return }
        let event = KeyEvent(code: key.code, chars: key.chars, modifiers: key.modifiers,
                             timestampUs: key.timestampUs, isDead: key.isDead)
        guard let result = try? engine.keyEvent(event: event) else { return }
        lastKeyAt = Date()
        hint = nil
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
        stats = try? engine?.stats()
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

    func exportData() {
        guard let engine else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "reflex-export.json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? engine.export(path: url.path)
    }

    func refresh() {
        guard let engine else { return }
        view = engine.view()
    }

    private func updatePool(_ settings: PoolSettings) {
        guard let engine else { return }
        try? engine.setPool(settings: settings)
        pool = engine.pool()
        view = engine.view()
    }

    private func applyInputSource() {
        guard let engine else { return }
        let source = InputSource.current()
        engine.setLayout(layout: source.layout, geometry: source.geometry)
        keyboard = engine.keyboard()
        view = engine.view()
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
