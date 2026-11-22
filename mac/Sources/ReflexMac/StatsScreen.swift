import SwiftUI

struct StatsScreen: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack {
            Text("Progress").font(Theme.mono(22, weight: .semibold)).foregroundStyle(Theme.text)
            Text("Press Tab to type").font(Theme.mono(13)).foregroundStyle(Theme.muted)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(StatsKeys(onBack: model.showTyping))
    }
}

/// Tab or Esc returns to typing from the stats screen.
struct StatsKeys: NSViewRepresentable {
    var onBack: () -> Void

    func makeNSView(context: Context) -> KeyCaptureNSView {
        let view = KeyCaptureNSView()
        view.onTab = onBack
        view.onEscape = onBack
        return view
    }

    func updateNSView(_ view: KeyCaptureNSView, context: Context) {
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
    }
}
