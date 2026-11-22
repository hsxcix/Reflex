//! Keyboard layouts: which physical key (and modifiers) produces which character,
//! plus key geometry (row, horizontal position, finger) for the model's features.
//!
//! Physical keys use W3C `KeyboardEvent.code` names by *position*: `Backquote` is
//! always the key left of `Digit1`, `IntlBackslash` the ISO key left of `KeyZ`.

mod geometry;

pub use geometry::{Finger, Geometry, Hand, KeyGeometry};

use std::collections::HashMap;

/// Modifier layers a character can sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    Base = 0,
    Shift = 1,
    Option = 2,
    ShiftOption = 3,
}

impl Layer {
    pub const ALL: [Layer; 4] = [Layer::Base, Layer::Shift, Layer::Option, Layer::ShiftOption];

    pub fn shift(self) -> bool {
        matches!(self, Layer::Shift | Layer::ShiftOption)
    }

    pub fn option(self) -> bool {
        matches!(self, Layer::Option | Layer::ShiftOption)
    }
}

/// One key press: a physical key on a modifier layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stroke {
    pub key: KeyId,
    pub layer: Layer,
}

/// Index into [`KEY_CODES`].
pub type KeyId = u8;

/// A movement: key × layer × dead-key flag (see [`Layout::motor`]).
pub type Motor = u16;

/// Motor id used for "start of text" in pairs.
pub const MOTOR_START: Motor = u16::MAX;

/// All physical keys Reflex knows, in a fixed order (the order is part of the model's
/// vocabulary, so only ever append).
pub const KEY_CODES: [&str; 50] = [
    "Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6", "Digit7", "Digit8",
    "Digit9", "Digit0", "Minus", "Equal", "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU",
    "KeyI", "KeyO", "KeyP", "BracketLeft", "BracketRight", "KeyA", "KeyS", "KeyD", "KeyF", "KeyG",
    "KeyH", "KeyJ", "KeyK", "KeyL", "Semicolon", "Quote", "Backslash", "IntlBackslash", "KeyZ",
    "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period", "Slash", "Space", "Enter",
];

pub fn key_id(code: &str) -> Option<KeyId> {
    KEY_CODES.iter().position(|c| *c == code).map(|i| i as KeyId)
}

/// Layouts shipped with Reflex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutId {
    DeIso,
    UsAnsi,
    EurKey,
}

impl LayoutId {
    pub const ALL: [LayoutId; 3] = [LayoutId::DeIso, LayoutId::UsAnsi, LayoutId::EurKey];

    pub fn as_str(self) -> &'static str {
        match self {
            LayoutId::DeIso => "de-iso",
            LayoutId::UsAnsi => "us-ansi",
            LayoutId::EurKey => "eurkey",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            LayoutId::DeIso => "German",
            LayoutId::UsAnsi => "U.S.",
            LayoutId::EurKey => "EurKEY",
        }
    }

    pub fn parse(s: &str) -> Option<LayoutId> {
        LayoutId::ALL.into_iter().find(|l| l.as_str() == s)
    }

    /// Index used as the model's layout embedding id.
    pub fn index(self) -> usize {
        self as usize
    }

    fn table(self) -> &'static str {
        match self {
            LayoutId::DeIso => include_str!("../../../data/layouts/de-iso.tsv"),
            LayoutId::UsAnsi => include_str!("../../../data/layouts/us-ansi.tsv"),
            LayoutId::EurKey => include_str!("../../../data/layouts/eurkey.tsv"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Output {
    Char(char),
    Dead(char),
}

/// A layout on a given physical keyboard geometry.
#[derive(Debug, Clone)]
pub struct Layout {
    pub id: LayoutId,
    pub geometry: Geometry,
    /// Shortest stroke sequence for every typable character.
    sequences: HashMap<char, Vec<Stroke>>,
}

impl Layout {
    pub fn new(id: LayoutId, geometry: Geometry) -> Layout {
        let outputs = parse_table(id.table());
        let mut direct: HashMap<char, Stroke> = HashMap::new();
        let mut dead: Vec<(char, Stroke)> = Vec::new();
        // Iterate cheapest layer first so the easiest stroke wins.
        for layer in Layer::ALL {
            for (key, layers) in &outputs {
                if !geometry.has_key(*key) {
                    continue;
                }
                let stroke = Stroke { key: *key, layer };
                match layers[layer as usize] {
                    Some(Output::Char(c)) => {
                        direct.entry(c).or_insert(stroke);
                    }
                    Some(Output::Dead(mark)) => dead.push((mark, stroke)),
                    None => {}
                }
            }
        }
        let mut sequences: HashMap<char, Vec<Stroke>> =
            direct.iter().map(|(c, s)| (*c, vec![*s])).collect();
        for (mark, dead_stroke) in &dead {
            for (base, composed) in compositions(*mark) {
                if sequences.contains_key(&composed) {
                    continue;
                }
                if let Some(base_stroke) = direct.get(&base) {
                    sequences.insert(composed, vec![*dead_stroke, *base_stroke]);
                }
            }
        }
        sequences.insert('\n', vec![Stroke { key: key_id("Enter").unwrap(), layer: Layer::Base }]);
        Layout { id, geometry, sequences }
    }

    /// Strokes needed to type `c`, or `None` if this layout can't produce it.
    pub fn strokes(&self, c: char) -> Option<&[Stroke]> {
        self.sequences.get(&c).map(Vec::as_slice)
    }

    pub fn can_type(&self, text: &str) -> bool {
        text.chars().all(|c| self.sequences.contains_key(&c))
    }

    /// The final (character-producing) stroke for `c`.
    pub fn main_stroke(&self, c: char) -> Option<Stroke> {
        self.strokes(c).and_then(|s| s.last().copied())
    }

    /// Motor id of the movement that produces `c`: physical key, modifier layer and
    /// whether a dead key precedes it. Practice is tracked per movement, so skills
    /// carry over between layouts exactly where the movements are the same.
    pub fn motor(&self, c: char) -> Option<Motor> {
        let strokes = self.strokes(c)?;
        let main = strokes.last()?;
        Some(main.key as Motor * 8 + main.layer as Motor + if strokes.len() > 1 { 4 } else { 0 })
    }

    pub fn key(&self, key: KeyId) -> KeyGeometry {
        self.geometry.key(key)
    }

    /// Which character a key produces on the base layer (for the on-screen keyboard).
    pub fn label(&self, key: KeyId, layer: Layer) -> Option<String> {
        let outputs = parse_table(self.id.table());
        outputs.iter().find(|(k, _)| *k == key).and_then(|(_, layers)| match layers[layer as usize] {
            Some(Output::Char(' ')) => Some(String::new()),
            Some(Output::Char(c)) => Some(c.to_string()),
            Some(Output::Dead(m)) => Some(m.to_string()),
            None => None,
        })
    }
}

fn parse_table(table: &str) -> Vec<(KeyId, [Option<Output>; 4])> {
    table
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            let key = key_id(cols[0]).unwrap_or_else(|| panic!("unknown key code {}", cols[0]));
            let mut layers = [None; 4];
            for (i, cell) in cols[1..].iter().take(4).enumerate() {
                layers[i] = parse_cell(cell);
            }
            (key, layers)
        })
        .collect()
}

fn parse_cell(cell: &str) -> Option<Output> {
    match cell {
        "-" | "" => None,
        "space" => Some(Output::Char(' ')),
        "minus" => Some(Output::Char('-')),
        _ => match cell.strip_prefix("dead:") {
            Some(mark) => mark.chars().next().map(Output::Dead),
            None => cell.chars().next().map(Output::Char),
        },
    }
}

/// Characters a dead key composes with the following base character. A dead key followed
/// by space yields the mark itself.
fn compositions(mark: char) -> Vec<(char, char)> {
    let pairs: &[(char, char)] = match mark {
        '¨' => &[('a', 'ä'), ('o', 'ö'), ('u', 'ü'), ('A', 'Ä'), ('O', 'Ö'), ('U', 'Ü'), ('e', 'ë'), ('i', 'ï')],
        '^' => &[('a', 'â'), ('e', 'ê'), ('i', 'î'), ('o', 'ô'), ('u', 'û')],
        '´' => &[('a', 'á'), ('e', 'é'), ('i', 'í'), ('o', 'ó'), ('u', 'ú')],
        '`' => &[('a', 'à'), ('e', 'è'), ('i', 'ì'), ('o', 'ò'), ('u', 'ù')],
        '~' => &[('n', 'ñ'), ('a', 'ã'), ('o', 'õ')],
        _ => &[],
    };
    let mut all = pairs.to_vec();
    all.push((' ', mark));
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn de() -> Layout {
        Layout::new(LayoutId::DeIso, Geometry::Iso)
    }

    #[test]
    fn german_basics() {
        let l = de();
        let z = l.main_stroke('z').unwrap();
        assert_eq!(KEY_CODES[z.key as usize], "KeyY");
        let at = l.main_stroke('@').unwrap();
        assert_eq!((KEY_CODES[at.key as usize], at.layer), ("KeyL", Layer::Option));
        assert_eq!(l.strokes('ä').unwrap().len(), 1);
        // ^ is a dead key: ^ then space.
        let caret = l.strokes('^').unwrap();
        assert_eq!(caret.len(), 2);
        assert_eq!(KEY_CODES[caret[1].key as usize], "Space");
    }

    #[test]
    fn us_umlauts_via_dead_key() {
        let l = Layout::new(LayoutId::UsAnsi, Geometry::Ansi);
        let ae = l.strokes('Ä').unwrap();
        assert_eq!(ae.len(), 2);
        assert_eq!((KEY_CODES[ae[0].key as usize], ae[0].layer), ("KeyU", Layer::Option));
        assert_eq!(ae[1].layer, Layer::Shift);
        assert!(l.can_type("Größe {x} = 5 € @home"));
    }

    #[test]
    fn eurkey_umlauts_direct() {
        let l = Layout::new(LayoutId::EurKey, Geometry::Ansi);
        assert_eq!(l.strokes('ö').unwrap().len(), 1);
        assert_eq!(l.main_stroke('ß').unwrap().layer, Layer::Option);
    }

    #[test]
    fn every_content_char_typable_on_all_layouts() {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß0123456789 .,;:!?'\"()-/%&€$@#+=*[]{}<>|\\_~^`";
        // German is only defined on ISO hardware (`<>` live on the ISO-only key).
        let combos = [
            (LayoutId::DeIso, Geometry::Iso),
            (LayoutId::UsAnsi, Geometry::Iso),
            (LayoutId::UsAnsi, Geometry::Ansi),
            (LayoutId::EurKey, Geometry::Iso),
            (LayoutId::EurKey, Geometry::Ansi),
        ];
        for (id, geometry) in combos {
            let l = Layout::new(id, geometry);
            for c in chars.chars() {
                assert!(l.strokes(c).is_some(), "{c} on {id:?}/{geometry:?}");
            }
        }
    }
}
