import Charts
import SwiftUI

/// Progress: real speed over time with forecast, keyboard heatmap, slowest pairs,
/// typical mistakes. Tab or Esc returns to typing.
enum HeatMode: String, CaseIterable { case errors, speed, predicted }

struct StatsScreen: View {
    @Environment(AppModel.self) private var model
    @StateObject private var heatMode = Local(HeatMode.errors)

    var body: some View {
        let stats = model.stats
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                header(stats)
                if let stats, stats.learningStopped {
                    Text("Learning stopped because of an internal error. Typing still works; restart Reflex to resume learning.")
                        .font(Theme.mono(12)).foregroundStyle(Theme.error)
                }
                if let stats, stats.totalKeystrokes > 0 {
                    HStack(alignment: .top, spacing: 26) {
                        Panel(title: "speed over time") { SpeedChart(stats: stats) }
                            .frame(minWidth: 420)
                        VStack(spacing: 26) {
                            Panel(title: "slowest pairs") { PairList(pairs: stats.slowPairs, empty: "learning…") }
                            Panel(title: "fading soon · scheduled before you forget") {
                                PairList(pairs: stats.fadingPairs, empty: "nothing fading yet", unit: "ms slower in a week")
                            }
                            Panel(title: "typical mix-ups") { ConfusionList(items: stats.confusions) }
                        }
                        .frame(width: 270)
                    }
                    if let from = stats.transitionFrom {
                        Panel(title: "switching from \(from)") { TransitionView(stats: stats) }
                    }
                    let heat = KeyHeatMap(stats: stats, mode: heatMode.value)
                    Panel(title: heat.title) {
                        Picker("Heatmap", selection: $heatMode.value) {
                            ForEach(HeatMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(width: 300)
                        KeyboardDrawing(caps: model.keyboard, highlight: nil, heat: heat.values, details: heat.details)
                            .frame(height: 210)
                        if heat.values.isEmpty {
                            Text(heatMode.value == .predicted ? "learning…" : "no data for this layout yet")
                                .font(Theme.mono(12)).foregroundStyle(Theme.muted)
                        }
                    }
                } else if stats != nil {
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
            Metric(value: stats?.measuredWpm.map { "\(Int($0.rounded()))" } ?? "–", label: "real wpm (measured)",
                   help: "Measured on recent measurement lines: sentences never used for practice, errors included, adjusted for how hard each sentence is")
            Metric(value: stats?.realWpm.map { "\(Int($0.rounded()))" } ?? "–", label: "real wpm (model)",
                   help: "Your typist model's prediction for normal text it never trained on")
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

/// Heat values (0…1) and tooltip text per key for one heatmap mode. Keys without data
/// are left out and drawn neutral.
struct KeyHeatMap {
    let values: [String: Double]
    let details: [String: String]
    let title: String

    init(stats: Stats, mode: HeatMode) {
        switch mode {
        case .predicted:
            title = "keyboard · predicted time per key (model)"
            values = Dictionary(uniqueKeysWithValues: stats.keyHeat.map { ($0.code, $0.value) })
            details = Dictionary(uniqueKeysWithValues: stats.keyHeat.map { ($0.code, "~\(Int($0.ms)) ms predicted") })
        case .speed:
            title = "keyboard · measured time per key, last 30 days"
            let timed = stats.keyMeasured.compactMap { k in k.meanMs.map { (k, $0) } }
            let lo = timed.map(\.1).min() ?? 0, hi = timed.map(\.1).max() ?? 0
            values = Dictionary(uniqueKeysWithValues: timed.map { k, ms in (k.code, hi > lo ? (ms - lo) / (hi - lo) : 0) })
            details = Dictionary(uniqueKeysWithValues: timed.map { k, ms in (k.code, "\(Int(ms.rounded())) ms · \(k.attempts) tries") })
        case .errors:
            title = "keyboard · measured error rate, last 30 days (keys with 5+ tries)"
            // Too few tries say nothing ("1 of 1 = 100 %"): such keys stay neutral.
            let enough = stats.keyMeasured.filter { $0.attempts >= 5 }
            // Scale to the worst key, but at least 10 %, so a few slips don't glow red.
            let top = max(enough.map(\.errorRate).max() ?? 0, 0.1)
            values = Dictionary(uniqueKeysWithValues: enough.map { ($0.code, min($0.errorRate / top, 1)) })
            details = Dictionary(uniqueKeysWithValues: enough.map {
                ($0.code, "\(Int(($0.errorRate * 100).rounded()))% errors · \($0.errors) of \($0.attempts) tries")
            })
        }
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
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(value)
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
        // Forecast points are "days from now" (core), so they start today, not on the last
        // practice day.
        let today = Calendar.current.startOfDay(for: Date())
        Chart {
            ForEach(Array(stats.days.enumerated()), id: \.offset) { i, d in
                LineMark(x: .value("day", days[i]), y: .value("wpm", d.wpm), series: .value("s", "measured"))
                    .foregroundStyle(Theme.muted)
                    .interpolationMethod(.monotone)
                if let real = d.realWpm {
                    LineMark(x: .value("day", days[i]), y: .value("wpm", real), series: .value("s", "model"))
                        .foregroundStyle(Theme.accent.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                        .interpolationMethod(.monotone)
                }
                if let measured = d.measuredWpm {
                    LineMark(x: .value("day", days[i]), y: .value("wpm", measured), series: .value("s", "measured real"))
                        .foregroundStyle(Theme.accent)
                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                        .interpolationMethod(.monotone)
                    PointMark(x: .value("day", days[i]), y: .value("wpm", measured))
                        .foregroundStyle(Theme.accent).symbolSize(18)
                }
            }
            ForEach(stats.forecast, id: \.days) { f in
                let date = Calendar.current.date(byAdding: .day, value: Int(f.days), to: today) ?? today
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
        .accessibilityLabel("Speed over time")
        .overlay(alignment: .topTrailing) {
            if let f = stats.forecast.last {
                Text("~\(Int(f.wpm.rounded())) wpm in \(Self.weeks(f.days)) at 15 min/day")
                    .font(Theme.mono(11)).foregroundStyle(Theme.accent)
            }
        }
    }
}

extension SpeedChart {
    static func weeks(_ days: UInt32) -> String {
        days % 7 == 0 ? "\(days / 7) weeks" : String(format: "%.1f weeks", Double(days) / 7)
    }
}

struct PairList: View {
    let pairs: [PairStat]
    var empty = "learning…"
    var unit = "ms"

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if pairs.isEmpty {
                Text(empty).font(Theme.mono(12)).foregroundStyle(Theme.muted)
            }
            ForEach(Array(pairs.prefix(6).enumerated()), id: \.offset) { _, p in
                HStack {
                    Text("“\(p.text)”").font(Theme.mono(14, weight: .medium)).foregroundStyle(Theme.text)
                    Spacer()
                    Text(unit == "ms" ? "\(Int(p.ms)) ms" : "+\(Int(p.ms.rounded())) ms").font(Theme.mono(12)).foregroundStyle(Theme.muted)
                        .help(unit)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}

/// Moving to another layout: speed on both, and how the moved keys are coming along.
struct TransitionView: View {
    let stats: Stats

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 30) {
                ForEach(Array(stats.layouts.sorted { $0.keystrokes > $1.keystrokes }.enumerated()), id: \.offset) { _, l in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(l.measuredWpm.map { "\(Int($0.rounded()))" } ?? "~\(Int(l.estimatedWpm.rounded()))")
                            .font(Theme.mono(22, weight: .semibold))
                            .foregroundStyle(l.active ? Theme.accent : Theme.text)
                        Text("\(l.name)\(l.active ? " · now" : "") · \(Int(l.keystrokes / 1000))k keys")
                            .font(Theme.mono(11)).foregroundStyle(Theme.muted)
                    }
                    .accessibilityElement(children: .combine)
                }
                if let ratio = ratio {
                    Text("\(Int((ratio * 100).rounded()))% of your old speed")
                        .font(Theme.mono(13)).foregroundStyle(Theme.text)
                }
                Spacer()
            }
            if !stats.movedKeys.isEmpty {
                Text("moved keys · time per press now vs. before").font(Theme.mono(11)).foregroundStyle(Theme.muted)
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 4), alignment: .leading, spacing: 6) {
                    ForEach(Array(stats.movedKeys.enumerated()), id: \.offset) { _, k in
                        HStack(spacing: 6) {
                            Text(k.ch).font(Theme.mono(14, weight: .semibold)).foregroundStyle(Theme.text).frame(width: 18)
                            Text(k.nowMs.map { "\(Int($0))" } ?? "–").foregroundStyle(Theme.accent)
                            Text("/ \(k.beforeMs.map { "\(Int($0))" } ?? "–") ms").foregroundStyle(Theme.muted)
                        }
                        .font(Theme.mono(12))
                        .help("\(k.nowAttempts) tries on the new layout, \(Int((k.nowErrorRate * 100).rounded()))% errors")
                        .accessibilityElement(children: .combine)
                    }
                }
            }
        }
    }

    private var ratio: Double? {
        guard let now = stats.layouts.first(where: \.active),
              let old = stats.layouts.filter({ !$0.active }).max(by: { $0.keystrokes < $1.keystrokes }) else { return nil }
        let a = now.measuredWpm ?? now.estimatedWpm, b = old.measuredWpm ?? old.estimatedWpm
        return b > 0 ? a / b : nil
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
                    Text(c.typed).foregroundStyle(Theme.error)
                    Spacer()
                    Text("\(c.count)×").foregroundStyle(Theme.muted)
                }
                .font(Theme.mono(13))
                .accessibilityElement(children: .combine)
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
