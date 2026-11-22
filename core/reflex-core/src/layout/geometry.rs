//! Physical key positions and the standard touch-typing finger for each key.

use super::{KeyId, KEY_CODES};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Geometry {
    /// ISO: L-shaped Return, extra key left of Z (German MacBooks).
    Iso,
    /// ANSI: wide left Shift, Backslash above Return.
    Ansi,
}

impl Geometry {
    pub fn as_str(self) -> &'static str {
        match self {
            Geometry::Iso => "iso",
            Geometry::Ansi => "ansi",
        }
    }

    pub fn parse(s: &str) -> Option<Geometry> {
        match s {
            "iso" => Some(Geometry::Iso),
            "ansi" => Some(Geometry::Ansi),
            _ => None,
        }
    }

    pub fn has_key(self, key: KeyId) -> bool {
        !(self == Geometry::Ansi && KEY_CODES[key as usize] == "IntlBackslash")
    }

    pub fn key(self, key: KeyId) -> KeyGeometry {
        let code = KEY_CODES[key as usize];
        let (row, x) = position(self, code);
        let finger = finger_for(code);
        KeyGeometry { row, x, finger }
    }

    /// All keys present on this geometry, for drawing the keyboard.
    pub fn keys(self) -> impl Iterator<Item = KeyId> {
        (0..KEY_CODES.len() as KeyId).filter(move |k| self.has_key(*k))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Finger {
    LeftPinky = 0,
    LeftRing = 1,
    LeftMiddle = 2,
    LeftIndex = 3,
    LeftThumb = 4,
    RightThumb = 5,
    RightIndex = 6,
    RightMiddle = 7,
    RightRing = 8,
    RightPinky = 9,
}

impl Finger {
    pub fn hand(self) -> Hand {
        if (self as u8) <= 4 { Hand::Left } else { Hand::Right }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hand {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyGeometry {
    /// 0 = number row, 1 = top letter row, 2 = home row, 3 = bottom row, 4 = space row.
    pub row: u8,
    /// Horizontal position in key widths from the left edge.
    pub x: f32,
    pub finger: Finger,
}

const NUMBER_ROW: [&str; 13] = [
    "Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6", "Digit7", "Digit8",
    "Digit9", "Digit0", "Minus", "Equal",
];
const TOP_ROW: [&str; 12] = [
    "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO", "KeyP", "BracketLeft",
    "BracketRight",
];
const HOME_ROW: [&str; 11] =
    ["KeyA", "KeyS", "KeyD", "KeyF", "KeyG", "KeyH", "KeyJ", "KeyK", "KeyL", "Semicolon", "Quote"];
const BOTTOM_ROW: [&str; 10] =
    ["KeyZ", "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period", "Slash"];

fn position(geometry: Geometry, code: &str) -> (u8, f32) {
    let find = |row: &[&str]| row.iter().position(|c| *c == code).map(|i| i as f32);
    if let Some(i) = find(&NUMBER_ROW) {
        return (0, i);
    }
    if let Some(i) = find(&TOP_ROW) {
        return (1, 1.5 + i);
    }
    if let Some(i) = find(&HOME_ROW) {
        return (2, 1.75 + i);
    }
    if let Some(i) = find(&BOTTOM_ROW) {
        return (3, 2.25 + i);
    }
    match (code, geometry) {
        ("Backslash", Geometry::Iso) => (2, 12.75),
        ("Backslash", Geometry::Ansi) => (1, 13.5),
        ("IntlBackslash", _) => (3, 1.25),
        ("Space", _) => (4, 7.0),
        ("Enter", Geometry::Iso) => (1, 13.75),
        ("Enter", Geometry::Ansi) => (2, 13.0),
        _ => unreachable!("no position for {code}"),
    }
}

fn finger_for(code: &str) -> Finger {
    use Finger::*;
    match code {
        "Backquote" | "Digit1" | "KeyQ" | "KeyA" | "KeyZ" | "IntlBackslash" => LeftPinky,
        "Digit2" | "KeyW" | "KeyS" | "KeyX" => LeftRing,
        "Digit3" | "KeyE" | "KeyD" | "KeyC" => LeftMiddle,
        "Digit4" | "Digit5" | "KeyR" | "KeyT" | "KeyF" | "KeyG" | "KeyV" | "KeyB" => LeftIndex,
        "Space" => RightThumb,
        "Digit6" | "Digit7" | "KeyY" | "KeyU" | "KeyH" | "KeyJ" | "KeyN" | "KeyM" => RightIndex,
        "Digit8" | "KeyI" | "KeyK" | "Comma" => RightMiddle,
        "Digit9" | "KeyO" | "KeyL" | "Period" => RightRing,
        _ => RightPinky,
    }
}
