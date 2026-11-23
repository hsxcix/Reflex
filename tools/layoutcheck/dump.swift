import Carbon
let codes: [(String, Int)] = [("KeyA",0),("KeyS",1),("KeyD",2),("KeyF",3),("KeyH",4),("KeyG",5),("KeyZ",6),("KeyX",7),("KeyC",8),("KeyV",9),("Sect",10),("KeyB",11),("KeyQ",12),("KeyW",13),("KeyE",14),("KeyR",15),("KeyY",16),("KeyT",17),("Digit1",18),("Digit2",19),("Digit3",20),("Digit4",21),("Digit6",22),("Digit5",23),("Equal",24),("Digit9",25),("Digit7",26),("Minus",27),("Digit8",28),("Digit0",29),("BracketRight",30),("KeyO",31),("KeyU",32),("BracketLeft",33),("KeyI",34),("KeyP",35),("KeyL",37),("KeyJ",38),("Quote",39),("KeyK",40),("Semicolon",41),("Backslash",42),("Comma",43),("Slash",44),("KeyN",45),("KeyM",46),("Period",47),("Space",49),("Grave",50)]
// Dumps a macOS keyboard layout as: code, base, shift, option, shift+option.
// "Sect" is keycode 10 (left of "1" on ISO), "Grave" keycode 50 (left of "Z" on ISO).
let id = CommandLine.arguments[1]
if id == "--kbd-type" {
    print(KBGetLayoutType(Int16(LMGetKbdType())) == kKeyboardISO ? "iso" : "ansi")
    exit(0)
}
let list = TISCreateInputSourceList([kTISPropertyInputSourceID: id] as CFDictionary, true).takeRetainedValue() as! [TISInputSource]
guard let src = list.first, let ptr = TISGetInputSourceProperty(src, kTISPropertyUnicodeKeyLayoutData) else { print("missing \(id)"); exit(1) }
let data = Unmanaged<CFData>.fromOpaque(ptr).takeUnretainedValue()
let layout = unsafeBitCast(CFDataGetBytePtr(data), to: UnsafePointer<UCKeyboardLayout>.self)
func tr(_ code: Int, _ mods: UInt32) -> String {
    var dead: UInt32 = 0; var len = 0; var buf = [UniChar](repeating: 0, count: 8)
    UCKeyTranslate(layout, UInt16(code), UInt16(kUCKeyActionDown), (mods >> 8) & 0xFF, UInt32(LMGetKbdType()), 0, &dead, 8, &len, &buf)
    if len == 0 && dead != 0 {
        UCKeyTranslate(layout, 49, UInt16(kUCKeyActionDown), 0, UInt32(LMGetKbdType()), 0, &dead, 8, &len, &buf)
        return "dead:" + String(utf16CodeUnits: buf, count: len)
    }
    return String(utf16CodeUnits: buf, count: len)
}
for (name, c) in codes {
    let cols = [0, UInt32(shiftKey), UInt32(optionKey), UInt32(shiftKey | optionKey)].map { tr(c, $0) }
    print(([name] + cols).joined(separator: "\t"))
}
