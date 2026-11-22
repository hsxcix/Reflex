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
                Button("Export Data…") { model.exportData() }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
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

    var body: some View {
        ZStack {
            Backdrop()
            switch model.screen {
            case .typing: TypingScreen().transition(.opacity)
            case .stats: StatsScreen().transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: model.screen)
        .ignoresSafeArea()
    }
}
