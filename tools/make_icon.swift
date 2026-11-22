// Renders Reflex's app icon (1024 px PNG). Usage: swift tools/make_icon.swift out.png
import AppKit

let size: CGFloat = 1024
let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()
let ctx = NSGraphicsContext.current!.cgContext

// macOS icon grid: 824 px rounded square centred in 1024.
let inset: CGFloat = 100
let rect = CGRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
let shape = CGPath(roundedRect: rect, cornerWidth: 185, cornerHeight: 185, transform: nil)
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 30, color: NSColor.black.withAlphaComponent(0.35).cgColor)
ctx.addPath(shape)
ctx.setFillColor(NSColor(red: 0.08, green: 0.09, blue: 0.105, alpha: 1).cgColor)
ctx.fillPath()
ctx.restoreGState()

ctx.saveGState()
ctx.addPath(shape)
ctx.clip()
let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [
    NSColor(red: 0.16, green: 0.17, blue: 0.2, alpha: 1).cgColor,
    NSColor(red: 0.07, green: 0.08, blue: 0.095, alpha: 1).cgColor,
] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(gradient, start: CGPoint(x: 0, y: size - inset), end: CGPoint(x: 0, y: inset), options: [])
ctx.restoreGState()

// A monospace "r" with the amber caret underneath: the moment before the next key.
let amber = NSColor(red: 0.95, green: 0.65, blue: 0.25, alpha: 1)
let font = NSFont.monospacedSystemFont(ofSize: 520, weight: .medium)
let text = NSAttributedString(string: "r", attributes: [
    .font: font,
    .foregroundColor: NSColor(red: 0.91, green: 0.89, blue: 0.85, alpha: 1),
])
let textSize = text.size()
text.draw(at: NSPoint(x: (size - textSize.width) / 2 - 10, y: 330))
let caret = CGRect(x: 512 - 150, y: 300, width: 300, height: 34)
ctx.saveGState()
ctx.setShadow(offset: .zero, blur: 40, color: amber.withAlphaComponent(0.8).cgColor)
ctx.addPath(CGPath(roundedRect: caret, cornerWidth: 17, cornerHeight: 17, transform: nil))
ctx.setFillColor(amber.cgColor)
ctx.fillPath()
ctx.restoreGState()

image.unlockFocus()
let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
