import SwiftUI

/// The only controls: languages, vocabulary depth, code toggle. Plus live numbers.
struct TopBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 22) {
            Text("reflex")
                .font(Theme.mono(15, weight: .semibold))
                .foregroundStyle(Theme.accent)
            Divider().frame(height: 14)
            HStack(spacing: 6) {
                Chip(label: "DE", on: model.pool.languages.contains("de")) { model.togglePool(language: "de") }
                Chip(label: "EN", on: model.pool.languages.contains("en")) { model.togglePool(language: "en") }
            }
            HStack(spacing: 6) {
                ForEach(["common", "broad", "full"], id: \.self) { v in
                    Chip(label: v, on: model.pool.vocabulary == v) { model.setVocabulary(v) }
                }
            }
            Chip(label: "{ code }", on: model.pool.code) { model.toggleCode() }
            Spacer()
            LiveNumbers()
            Button { model.showStats() } label: {
                Image(systemName: "chart.line.uptrend.xyaxis")
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.muted)
            .help("Progress (Esc)")
        }
        .padding(.horizontal, 30)
        .frame(height: 30)
    }
}

struct LiveNumbers: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 14) {
            Text(model.view.layoutName).foregroundStyle(Theme.muted)
            if model.view.liveWpm > 0 {
                Text("\(Int(model.view.liveWpm.rounded())) wpm")
                Text(String(format: "%.0f%%", model.view.liveAccuracy * 100))
            }
            if let real = model.view.realWpm {
                Text("real \(Int(real.rounded()))").foregroundStyle(Theme.accent)
                    .help("Estimated speed on normal text, from your typist model")
            }
            DailyRing(minutes: model.view.todayMinutes)
        }
        .font(Theme.mono(12))
        .foregroundStyle(Theme.text.opacity(0.8))
    }
}

struct Chip: View {
    let label: String
    let on: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(Theme.mono(12, weight: on ? .semibold : .regular))
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .foregroundStyle(on ? Theme.accent : Theme.muted)
                .background(Capsule().fill(on ? Theme.accent.opacity(0.13) : Theme.chip.opacity(0.6)))
        }
        .buttonStyle(.plain)
        .focusable(false)
    }
}

/// Small ring that fills up over 15 minutes of daily practice (information, not a rule).
struct DailyRing: View {
    let minutes: Double

    var body: some View {
        ZStack {
            Circle().stroke(Theme.chip, lineWidth: 2.5)
            Circle()
                .trim(from: 0, to: min(1, minutes / 15))
                .stroke(Theme.accent, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 14, height: 14)
        .help("\(Int(minutes)) of 15 minutes today")
    }
}
