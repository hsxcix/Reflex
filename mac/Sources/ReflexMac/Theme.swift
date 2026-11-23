import SwiftUI

/// Reflex's palette: deep graphite with a warm amber accent (and a paper-light variant).
enum Theme {
    static let accent = Color("accent", light: 0xC9781A, dark: 0xF2A541)
    static let error = Color("error", light: 0xD9473B, dark: 0xFF6B5E)
    static let text = Color("text", light: 0x2A2925, dark: 0xE9E4DA)
    static let typed = Color("typed", light: 0xB2ACA0, dark: 0x5C5F66)
    static let muted = Color("muted", light: 0x8E887C, dark: 0x7D818A)
    static let surface = Color("surface", light: 0xF3EFE7, dark: 0x15171B)
    static let chip = Color("chip", light: 0xE6E0D4, dark: 0x22252B)

    /// Finger colours for the on-screen keyboard (left pinky … right pinky).
    static func fingerName(_ index: Int) -> String {
        let names = ["left pinky", "left ring", "left middle", "left index", "left thumb",
                     "right thumb", "right index", "right middle", "right ring", "right pinky"]
        return names.indices.contains(index) ? names[index] : "finger"
    }

    static let fingers: [Color] = [
        Color(hex: 0xE07A5F), Color(hex: 0xE9A23B), Color(hex: 0x8AB17D), Color(hex: 0x4FA3A5), Color(hex: 0x9C9C9C),
        Color(hex: 0x9C9C9C), Color(hex: 0x5B8DEF), Color(hex: 0x9D7BD8), Color(hex: 0xD86FA8), Color(hex: 0xC9675B),
    ]

    static func mono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }

    /// A colour that follows light/dark mode.
    init(_ name: String, light: UInt32, dark: UInt32) {
        self.init(nsColor: NSColor(name: NSColor.Name(name)) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor(Color(hex: isDark ? dark : light))
        })
    }
}

/// Native translucent window background with Reflex's tint on top.
struct Backdrop: View {
    var body: some View {
        ZStack {
            VisualEffect(material: .underWindowBackground)
            Theme.surface.opacity(0.86)
        }
    }
}

struct VisualEffect: NSViewRepresentable {
    var material: NSVisualEffectView.Material

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = .behindWindow
        view.state = .followsWindowActiveState
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {
        view.material = material
    }
}
