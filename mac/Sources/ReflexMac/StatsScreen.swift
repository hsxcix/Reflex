import Charts
import SwiftUI

/// Progress: real speed over time with forecast, keyboard heatmap, slowest pairs,
/// typical mistakes. Tab or Esc returns to typing.
struct StatsScreen: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let stats = model.stats
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                header(stats)
                if let stats {
                    HStack(alignment: .top, spacing: 26) {
                        Panel(title: "speed over time") { SpeedChart(stats: stats) }
                            .frame(minWidth: 420)
                        VStack(spacing: 26) {
                            Panel(title: "slowest pairs") { PairList(pairs: stats.slowPairs) }
                            Panel(title: "typical mix-ups") { ConfusionList(items: stats.confusions) }
                        }
                        .frame(width: 250)
                    }
                    Panel(title: "keyboard · where time goes") {
                        KeyboardDrawing(caps: model.keyboard, highlight: nil,
                                        heat: Dictionary(uniqueKeysWithValues: stats.keyHeat.map { ($0.code, $0.value) }))
                            .frame(height: 210)
                    }
                } else {
                    Text("Type a little first; your progress appears here.")
                        .font(Theme.mono(14)).foregroundStyle(Theme.muted)
                }
                Text("tab · back to typing").font(Theme.mono(12)).foregroundStyle(Theme.muted)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 40)
            .padding(.top, 46)
            .padding(.bottom, 30)
        }
        .scrollIndicators(.never)
        .background(StatsKeys(onBack: model.showTyping))
        .task { await model.refreshStatsLoop() }
    }

    @ViewBuilder
    private func header(_ stats: Stats?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 34) {
            Text("reflex").font(Theme.mono(15, weight: .semibold)).foregroundStyle(Theme.accent)
            Metric(value: stats?.realWpm.map { "\(Int($0.rounded()))" } ?? "–", label: "real wpm",
                   help: "Predicted speed on normal text, errors included")
            Metric(value: stats.map { "\($0.streakDays)" } ?? "0", label: "day streak")
            Metric(value: stats.map { "\(Int($0.todayMinutes.rounded()))" } ?? "0", label: "min today")
            Metric(value: stats.map { formatHours($0.totalMinutes) } ?? "0", label: "practised")
            Metric(value: stats.map { "\(Int(($0.confidence * 100).rounded()))%" } ?? "–", label: "model confidence",
                   help: "How well your typist model knows you; grows as you type")
            Spacer()
        }
    }

    private func formatHours(_ minutes: Double) -> String {
        minutes < 60 ? "\(Int(minutes))m" : String(format: "%.1fh", minutes / 60)
    }
}

struct Metric: View {
    let value: String
    let label: String
    var help: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(Theme.mono(26, weight: .semibold)).foregroundStyle(Theme.text)
            Text(label).font(Theme.mono(11)).foregroundStyle(Theme.muted)
        }
        .help(help ?? label)
    }
}

struct Panel<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(Theme.mono(12)).foregroundStyle(Theme.muted)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.chip.opacity(0.35)))
    }
}

struct SpeedChart: View {
    let stats: Stats

    var body: some View {
        let days = stats.days.map { Date(timeIntervalSince1970: Double($0.dayMs) / 1000) }
        let last = days.last ?? Date()
        Chart {
            ForEach(Array(stats.days.enumerated()), id: \.offset) { i, d in
                LineMark(x: .value("day", days[i]), y: .value("wpm", d.wpm), series: .value("s", "measured"))
                    .foregroundStyle(Theme.muted)
                    .interpolationMethod(.monotone)
                if let real = d.realWpm {
                    LineMark(x: .value("day", days[i]), y: .value("wpm", real), series: .value("s", "real"))
                        .foregroundStyle(Theme.accent)
                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                        .interpolationMethod(.monotone)
                }
            }
            ForEach(stats.forecast, id: \.days) { f in
                let date = last.addingTimeInterval(Double(f.days) * 86_400)
                AreaMark(x: .value("day", date), yStart: .value("low", f.wpm - f.plusMinus),
                         yEnd: .value("high", f.wpm + f.plusMinus))
                    .foregroundStyle(Theme.accent.opacity(0.12))
                LineMark(x: .value("day", date), y: .value("wpm", f.wpm), series: .value("s", "forecast"))
                    .foregroundStyle(Theme.accent.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
            }
        }
        .chartYAxis { AxisMarks(position: .leading) }
        .chartYScale(domain: .automatic(includesZero: false))
        .frame(height: 230)
        .overlay(alignment: .topTrailing) {
            if let f = stats.forecast.last {
                Text("~\(Int(f.wpm.rounded())) wpm in \(f.days / 7) weeks at 15 min/day")
                    .font(Theme.mono(11)).foregroundStyle(Theme.accent)
            }
        }
    }
}

struct PairList: View {
    let pairs: [PairStat]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if pairs.isEmpty {
                Text("learning…").font(Theme.mono(12)).foregroundStyle(Theme.muted)
            }
            ForEach(Array(pairs.prefix(6).enumerated()), id: \.offset) { _, p in
                HStack {
                    Text("“\(p.text)”").font(Theme.mono(14, weight: .medium)).foregroundStyle(Theme.text)
                    Spacer()
                    Text("\(Int(p.ms)) ms").font(Theme.mono(12)).foregroundStyle(Theme.muted)
                }
            }
        }
    }
}

struct ConfusionList: View {
    let items: [Confusion]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if items.isEmpty {
                Text("no mix-ups yet").font(Theme.mono(12)).foregroundStyle(Theme.muted)
            }
            ForEach(Array(items.prefix(6).enumerated()), id: \.offset) { _, c in
                HStack {
                    Text(c.expected).foregroundStyle(Theme.text)
                    Image(systemName: "arrow.right").font(.system(size: 9)).foregroundStyle(Theme.muted)
                    Text(c.typed.uppercased()).foregroundStyle(Theme.error)
                    Spacer()
                    Text("\(c.count)×").foregroundStyle(Theme.muted)
                }
                .font(Theme.mono(13))
            }
        }
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
        view.onTab = onBack
        view.onEscape = onBack
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
    }
}
