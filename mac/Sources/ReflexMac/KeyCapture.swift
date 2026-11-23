import AppKit
import Carbon.HIToolbox
import SwiftUI

/// A key press as Reflex needs it: physical position, produced characters, precise time.
struct CapturedKey {
    let code: String
    let chars: String
    let modifiers: UInt32
    let timestampUs: Int64
    let isDead: Bool
    /// The keyboard that sent the key is ISO (key left of "Z"); can change when an
    /// external keyboard is used.
    var iso: Bool = PhysicalKey.isISO
}

/// Invisible first-responder view that receives every key while the typing screen is shown.
struct KeyCaptureView: NSViewRepresentable {
    var onKey: (CapturedKey) -> Void
    var onEscape: () -> Void
    var onFlags: (NSEvent.ModifierFlags) -> Void = { _ in }

    func makeNSView(context: Context) -> KeyCaptureNSView {
        let view = KeyCaptureNSView()
        update(view)
        return view
    }

    func updateNSView(_ view: KeyCaptureNSView, context: Context) {
        update(view)
        // Take focus back only if nothing else (a menu, a text field) holds it.
        DispatchQueue.main.async {
            if let window = view.window, window.firstResponder == nil || window.firstResponder === window {
                window.makeFirstResponder(view)
            }
        }
    }

    private func update(_ view: KeyCaptureNSView) {
        view.onKey = onKey
        view.onEscape = onEscape
        view.onFlags = onFlags
    }
}

final class KeyCaptureNSView: NSView {
    var onKey: ((CapturedKey) -> Void)?
    var onEscape: (() -> Void)?
    var onTab: (() -> Void)?
    var onFlags: ((NSEvent.ModifierFlags) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    override func keyDown(with event: NSEvent) {
        let flags = event.modifierFlags
        if flags.contains(.command) || flags.contains(.control) {
            super.keyDown(with: event)
            return
        }
        if event.keyCode == UInt16(kVK_Escape) {
            onEscape?()
            return
        }
        if event.keyCode == UInt16(kVK_Tab) {
            onTab?()
            return
        }
        // A held key auto-repeats every ~30 ms: not a keystroke, and it would flood the
        // model with fake presses.
        if event.isARepeat { return }
        let iso = PhysicalKey.isISO(event)
        guard let code = PhysicalKey.code(for: event.keyCode, iso: iso) else { return }
        let chars = event.characters ?? ""
        let isDead = chars.isEmpty
        var modifiers: UInt32 = 0
        if flags.contains(.shift) { modifiers |= 1 }
        if flags.contains(.option) { modifiers |= 2 }
        onKey?(CapturedKey(
            code: code,
            chars: code == "Enter" ? "\n" : chars,
            modifiers: modifiers,
            timestampUs: Int64(event.timestamp * 1_000_000),
            isDead: isDead,
            iso: iso
        ))
    }

    // Swallow key-up so the system doesn't beep; modifier changes report Caps Lock.
    override func keyUp(with event: NSEvent) {}
    override func flagsChanged(with event: NSEvent) { onFlags?(event.modifierFlags) }
}

/// macOS virtual key codes → W3C codes *by position* (what the core expects).
enum PhysicalKey {
    /// Form factor of the keyboard used last (asked fresh each time: external keyboards).
    static var isISO: Bool { KBGetLayoutType(Int16(LMGetKbdType())) == kKeyboardISO }

    /// Form factor of the keyboard that sent `event`.
    static func isISO(_ event: NSEvent) -> Bool {
        guard let type = event.cgEvent?.getIntegerValueField(.keyboardEventKeyboardType), type > 0 else { return isISO }
        return KBGetLayoutType(Int16(truncatingIfNeeded: type)) == kKeyboardISO
    }

    static func code(for keyCode: UInt16, iso: Bool = PhysicalKey.isISO) -> String? {
        let k = Int(keyCode)
        let isISO = iso
        // On ISO keyboards macOS reports the key left of "1" as Section and the key left
        // of "Z" as Grave; by position they are Backquote and IntlBackslash.
        if k == kVK_ISO_Section { return isISO ? "Backquote" : "IntlBackslash" }
        if k == kVK_ANSI_Grave { return isISO ? "IntlBackslash" : "Backquote" }
        return table[k]
    }

    /// Virtual key code for a W3C code (inverse of `code(for:)`; used by the app check).
    static func keyCode(for code: String, iso: Bool = PhysicalKey.isISO) -> UInt16? {
        if code == "Backquote" { return UInt16(iso ? kVK_ISO_Section : kVK_ANSI_Grave) }
        if code == "IntlBackslash" { return iso ? UInt16(kVK_ANSI_Grave) : nil }
        return table.first { $0.value == code }.map { UInt16($0.key) }
    }

    private static let table: [Int: String] = [
        kVK_ANSI_1: "Digit1", kVK_ANSI_2: "Digit2", kVK_ANSI_3: "Digit3", kVK_ANSI_4: "Digit4",
        kVK_ANSI_5: "Digit5", kVK_ANSI_6: "Digit6", kVK_ANSI_7: "Digit7", kVK_ANSI_8: "Digit8",
        kVK_ANSI_9: "Digit9", kVK_ANSI_0: "Digit0", kVK_ANSI_Minus: "Minus", kVK_ANSI_Equal: "Equal",
        kVK_ANSI_Q: "KeyQ", kVK_ANSI_W: "KeyW", kVK_ANSI_E: "KeyE", kVK_ANSI_R: "KeyR", kVK_ANSI_T: "KeyT",
        kVK_ANSI_Y: "KeyY", kVK_ANSI_U: "KeyU", kVK_ANSI_I: "KeyI", kVK_ANSI_O: "KeyO", kVK_ANSI_P: "KeyP",
        kVK_ANSI_LeftBracket: "BracketLeft", kVK_ANSI_RightBracket: "BracketRight",
        kVK_ANSI_A: "KeyA", kVK_ANSI_S: "KeyS", kVK_ANSI_D: "KeyD", kVK_ANSI_F: "KeyF", kVK_ANSI_G: "KeyG",
        kVK_ANSI_H: "KeyH", kVK_ANSI_J: "KeyJ", kVK_ANSI_K: "KeyK", kVK_ANSI_L: "KeyL",
        kVK_ANSI_Semicolon: "Semicolon", kVK_ANSI_Quote: "Quote", kVK_ANSI_Backslash: "Backslash",
        kVK_ANSI_Z: "KeyZ", kVK_ANSI_X: "KeyX", kVK_ANSI_C: "KeyC", kVK_ANSI_V: "KeyV", kVK_ANSI_B: "KeyB",
        kVK_ANSI_N: "KeyN", kVK_ANSI_M: "KeyM", kVK_ANSI_Comma: "Comma", kVK_ANSI_Period: "Period",
        kVK_ANSI_Slash: "Slash", kVK_Space: "Space", kVK_Return: "Enter",
    ]
}

/// Which keyboard layout is active, and notifications when it changes.
enum InputSource {
    struct Current {
        let layout: String
        let geometry: String
        /// Reflex has a table for this layout; otherwise it judges by the closest one.
        let supported: Bool
        /// The layout's name as macOS shows it (e.g. "Französisch").
        let name: String
    }

    /// Exact macOS layout ids Reflex knows (the part after "com.apple.keylayout.").
    private static let known: [String: String] = [
        "german": "de-iso",
        "austrian": "de-iso",
        "us": "us-ansi",
        "abc": "us-ansi",
        "usextended": "us-ansi",
        "usinternational-pc": "us-ansi",
    ]

    static func current(iso: Bool = PhysicalKey.isISO) -> Current {
        let geometry = iso ? "iso" : "ansi"
        let fallback = iso ? "de-iso" : "us-ansi"
        guard let source = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
              let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else {
            return Current(layout: fallback, geometry: geometry, supported: false, name: "?")
        }
        let id = (Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String).lowercased()
        var name = id
        if let n = TISGetInputSourceProperty(source, kTISPropertyLocalizedName) {
            name = Unmanaged<CFString>.fromOpaque(n).takeUnretainedValue() as String
        }
        if id.contains("eurkey") { return Current(layout: "eurkey", geometry: geometry, supported: true, name: name) }
        let short = id.components(separatedBy: "keylayout.").last ?? id
        if let layout = known[short] { return Current(layout: layout, geometry: geometry, supported: true, name: name) }
        return Current(layout: fallback, geometry: geometry, supported: false, name: name)
    }

    static func observe(_ handler: @escaping () -> Void) -> NSObjectProtocol {
        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String),
            object: nil, queue: .main
        ) { _ in handler() }
    }
}
