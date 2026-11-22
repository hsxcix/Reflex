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
}

/// Invisible first-responder view that receives every key while the typing screen is shown.
struct KeyCaptureView: NSViewRepresentable {
    var onKey: (CapturedKey) -> Void
    var onEscape: () -> Void

    func makeNSView(context: Context) -> KeyCaptureNSView {
        let view = KeyCaptureNSView()
        view.onKey = onKey
        view.onEscape = onEscape
        return view
    }

    func updateNSView(_ view: KeyCaptureNSView, context: Context) {
        view.onKey = onKey
        view.onEscape = onEscape
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
    }
}

final class KeyCaptureNSView: NSView {
    var onKey: ((CapturedKey) -> Void)?
    var onEscape: (() -> Void)?
    var onTab: (() -> Void)?

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
        guard let code = PhysicalKey.code(for: event.keyCode) else { return }
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
            isDead: isDead
        ))
    }

    // Swallow key-up and modifier changes so the system doesn't beep.
    override func keyUp(with event: NSEvent) {}
    override func flagsChanged(with event: NSEvent) {}
}

/// macOS virtual key codes → W3C codes *by position* (what the core expects).
enum PhysicalKey {
    static let isISO: Bool = KBGetLayoutType(Int16(LMGetKbdType())) == kKeyboardISO

    static func code(for keyCode: UInt16) -> String? {
        let k = Int(keyCode)
        // On ISO keyboards macOS reports the key left of "1" as Section and the key left
        // of "Z" as Grave; by position they are Backquote and IntlBackslash.
        if k == kVK_ISO_Section { return isISO ? "Backquote" : "IntlBackslash" }
        if k == kVK_ANSI_Grave { return isISO ? "IntlBackslash" : "Backquote" }
        return table[k]
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
    static func current() -> (layout: String, geometry: String) {
        let geometry = PhysicalKey.isISO ? "iso" : "ansi"
        guard let source = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
              let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else {
            return ("de-iso", geometry)
        }
        let id = (Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String).lowercased()
        if id.contains("eurkey") { return ("eurkey", geometry) }
        if id.contains("german") { return ("de-iso", geometry) }
        if id.hasSuffix(".us") || id.contains("abc") || id.contains("usinternational") || id.contains(".us.") {
            return ("us-ansi", geometry)
        }
        return (geometry == "iso" ? "de-iso" : "us-ansi", geometry)
    }

    static func observe(_ handler: @escaping () -> Void) -> NSObjectProtocol {
        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String),
            object: nil, queue: .main
        ) { _ in handler() }
    }
}
