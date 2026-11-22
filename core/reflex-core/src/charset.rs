//! Fixed character vocabulary for the model (only ever append, ids are baked into weights).

/// Id 0 = unknown, 1 = start of text; real characters start at 2.
const CHARS: &str = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß0123456789.,;:!?'\"()-/%&€$@#+=*[]{}<>|\\_~^`§°\n";

pub const UNKNOWN: usize = 0;
pub const START: usize = 1;

pub fn size() -> usize {
    CHARS.chars().count() + 2
}

pub fn id(c: char) -> usize {
    CHARS.chars().position(|x| x == c).map_or(UNKNOWN, |i| i + 2)
}

/// Id of an optional character (`None` = start of text).
pub fn id_or_start(c: Option<char>) -> usize {
    c.map_or(START, id)
}

#[cfg(test)]
mod tests {
    #[test]
    fn ids_are_stable() {
        assert_eq!(super::id(' '), 2);
        assert_eq!(super::id('a'), 3);
        assert_eq!(super::id('✓'), super::UNKNOWN);
        assert!(super::size() > 100);
    }
}
