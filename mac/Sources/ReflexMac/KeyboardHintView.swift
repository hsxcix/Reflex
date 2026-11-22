import SwiftUI

/// Faint keyboard that fades in only when you hesitate, showing the next key
/// in its finger colour.
struct KeyboardHintView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let hint = model.hint
        KeyboardDrawing(caps: model.keyboard, highlight: hint)
            .opacity(hint == nil ? 0 : 1)
            .animation(.easeInOut(duration: hint == nil ? 0.15 : 0.35), value: hint == nil)
            .allowsHitTesting(false)
    }
}

struct KeyboardDrawing: View {
    let caps: [KeyCap]
    let highlight: KeyHint?
    var heat: [String: Double] = [:]

    var body: some View {
        GeometryReader { geo in
            let unit = min(geo.size.width / 15.2, geo.size.height / 5.4)
            let width = unit * 15
            ZStack(alignment: .topLeading) {
                ForEach(caps, id: \.code) { cap in
                    capView(cap, unit: unit)
                        .offset(x: CGFloat(cap.x) * unit, y: CGFloat(cap.row) * unit * 1.05)
                }
                if highlight?.shift == true { modifierCap("⇧", x: 0, unit: unit) }
                if highlight?.option == true { modifierCap("⌥", x: 1.5, unit: unit) }
            }
            .frame(width: width, height: unit * 5.3, alignment: .topLeading)
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }

    private func capView(_ cap: KeyCap, unit: CGFloat) -> some View {
        let isTarget = highlight?.codes.contains(cap.code) == true
        let finger = Theme.fingers[Int(cap.finger) % Theme.fingers.count]
        let heatValue = heat[cap.code]
        return RoundedRectangle(cornerRadius: unit * 0.16)
            .fill(fill(isTarget: isTarget, finger: finger, heat: heatValue))
            .overlay(
                RoundedRectangle(cornerRadius: unit * 0.16)
                    .stroke(isTarget ? finger : Theme.muted.opacity(0.18), lineWidth: isTarget ? 2 : 1)
            )
            .overlay(
                Text(cap.label.uppercased())
                    .font(Theme.mono(unit * 0.3, weight: isTarget ? .bold : .regular))
                    .foregroundStyle(isTarget ? Theme.text : Theme.muted.opacity(0.7))
            )
            .frame(width: CGFloat(cap.width) * unit - unit * 0.1, height: unit * 0.92)
    }

    private func fill(isTarget: Bool, finger: Color, heat: Double?) -> Color {
        if isTarget { return finger.opacity(0.35) }
        if let heat { return Theme.error.opacity(0.08 + 0.55 * heat) }
        return Theme.chip.opacity(0.35)
    }

    private func modifierCap(_ label: String, x: CGFloat, unit: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: unit * 0.16)
            .fill(Theme.accent.opacity(0.3))
            .overlay(Text(label).font(Theme.mono(unit * 0.32, weight: .bold)).foregroundStyle(Theme.text))
            .frame(width: unit * 1.15, height: unit * 0.92)
            .offset(x: x * unit, y: 3 * unit * 1.05)
    }
}
