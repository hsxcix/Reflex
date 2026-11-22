import SwiftUI

struct TypingScreen: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            TopBar()
                .opacity(model.isTyping && !model.mouseActive ? 0.12 : 1)
                .animation(.easeInOut(duration: 0.35), value: model.isTyping)
                .padding(.top, 34)
            Spacer(minLength: 20)
            TextStage()
                .padding(.horizontal, 56)
            Spacer(minLength: 20)
            ZStack {
                KeyboardHintView()
                BatchLine()
            }
            .frame(height: 190)
            .padding(.bottom, 26)
        }
        .background(KeyCaptureView(onKey: model.handle, onEscape: model.showStats))
        .onContinuousHover { phase in
            if case .active = phase { model.mouseMoved() }
        }
        .overlay {
            if let error = model.startupError {
                Text("Reflex couldn't open its data: \(error)")
                    .font(.callout).foregroundStyle(Theme.error).padding()
            }
        }
    }
}

/// The three visible lines with caret, typed/untyped colouring and error marks.
struct TextStage: View {
    @Environment(AppModel.self) private var model
    @StateObject private var shake = Local<CGFloat>(0)

    var body: some View {
        GeometryReader { geo in
            let lines = model.view.lines
            let longest = max(lines.map(\.count).max() ?? 40, 40)
            let size = min(30, max(17, geo.size.width / (CGFloat(longest) * 0.62)))
            let charWidth = size * 0.6021
            let lineHeight = size * 1.75
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                    LineText(line: line, index: index, size: size, view: model.view)
                        .frame(height: lineHeight, alignment: .leading)
                        .overlay(alignment: .topLeading) {
                            if index == 0 {
                                Caret(size: size)
                                    .offset(x: CGFloat(model.view.cursor) * charWidth + shake.value,
                                            y: lineHeight * 0.5 + size * 0.62)
                                    .animation(.easeOut(duration: 0.07), value: model.view.cursor)
                            }
                        }
                }
            }
            .id(model.lineToken)
            .transition(.asymmetric(insertion: .move(edge: .bottom).combined(with: .opacity), removal: .opacity))
            .animation(.easeOut(duration: 0.18), value: model.lineToken)
            .frame(width: geo.size.width, height: geo.size.height, alignment: .center)
        }
        .frame(height: 170)
        .onChange(of: model.errorToken) {
            withAnimation(.spring(response: 0.12, dampingFraction: 0.25)) { shake.value = 3 }
            withAnimation(.spring(response: 0.2, dampingFraction: 0.5).delay(0.08)) { shake.value = 0 }
        }
    }
}

struct LineText: View {
    let line: String
    let index: Int
    let size: CGFloat
    let view: TypingView

    var body: some View {
        Text(attributed).font(Theme.mono(size)).lineLimit(1).fixedSize()
    }

    private var attributed: AttributedString {
        var result = AttributedString()
        let errors = Set(view.errorPositions.map(Int.init))
        let cursor = Int(view.cursor)
        for (i, ch) in line.enumerated() {
            var piece = AttributedString(String(ch == " " && index == 0 && errors.contains(i) ? "·" : ch))
            if index == 0 && i < cursor {
                piece.foregroundColor = errors.contains(i) ? Theme.error.opacity(0.75) : Theme.typed
            } else if index == 0 && i == cursor && errors.contains(i) {
                piece.foregroundColor = Theme.error
            } else {
                piece.foregroundColor = index == 0 ? Theme.text : Theme.text.opacity(index == 1 ? 0.55 : 0.3)
            }
            result += piece
        }
        return result
    }
}

struct Caret: View {
    let size: CGFloat

    var body: some View {
        RoundedRectangle(cornerRadius: 1.5)
            .fill(Theme.accent)
            .frame(width: size * 0.6, height: 2.5)
            .shadow(color: Theme.accent.opacity(0.6), radius: 4)
    }
}

/// Quiet one-line summary after each ~45 s batch.
struct BatchLine: View {
    @Environment(AppModel.self) private var model
    @StateObject private var visible = Local(false)

    var body: some View {
        Group {
            if let b = model.lastBatch, visible.value {
                HStack(spacing: 14) {
                    Text("\(Int(b.wpm.rounded())) wpm").foregroundStyle(Theme.text)
                    Text(String(format: "%.1f%%", b.accuracy * 100))
                    Text(b.trend >= 0 ? "↑ \(String(format: "%.1f", b.trend))" : "↓ \(String(format: "%.1f", -b.trend))")
                        .foregroundStyle(b.trend >= 0 ? Theme.accent : Theme.muted)
                    if b.fatigueHint {
                        Text("· good moment for a short break").foregroundStyle(Theme.accent)
                    }
                }
                .font(Theme.mono(14))
                .foregroundStyle(Theme.muted)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .frame(maxHeight: .infinity, alignment: .bottom)
        .animation(.easeInOut(duration: 0.4), value: visible.value)
        .onChange(of: model.batchToken) {
            visible.value = true
            Task {
                try? await Task.sleep(for: .seconds(5))
                visible.value = false
            }
        }
    }
}
