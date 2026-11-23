import SwiftUI

@main
struct ReflexApp: App {
    private let model: AppModel

    init() {
        model = AppModel()
    }

    var body: some Scene {
        Window("Reflex", id: "main") {
            RootView()
                .environment(model)
                .frame(minWidth: 760, minHeight: 460)
        }
        .defaultSize(width: 980, height: 560)
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(after: .importExport) {
                Button("Export Keystrokes as CSV…") { model.exportData(.csv) }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                Button("Export All Data as JSON…") { model.exportData(.json) }
                    .keyboardShortcut("e", modifiers: [.command, .shift, .option])
            }
            CommandMenu("Practice") {
                Button("Type") { model.screen = .typing }
                    .keyboardShortcut("1", modifiers: .command)
                Button("Progress") { model.showStats() }
                    .keyboardShortcut("2", modifiers: .command)
            }
        }
    }
}

struct RootView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            Backdrop()
            switch model.screen {
            case .typing: TypingScreen().transition(.opacity)
            case .stats: StatsScreen().transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: reduceMotion ? 0.1 : 0.2), value: model.screen)
        .ignoresSafeArea()
        .alert("Reflex", isPresented: Binding(get: { model.errorMessage != nil },
                                              set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
        .task {
            if let report = model.smokeReport { await SmokeTest.run(model: model, report: report) }
        }
    }
}
