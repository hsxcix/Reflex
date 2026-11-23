//! Live typing state: the visible lines, the cursor, stop-on-error judging, timing,
//! and batch bookkeeping. Pure logic; persistence and learning live elsewhere.

use crate::content::{ContentKind, Lang};
use crate::layout::{Geometry, KeyId, LayoutId};
use crate::record::{CharRecord, TypedLine};
use crate::store::KeyEventRecord;
use std::collections::VecDeque;

/// Gaps longer than this are pauses: not timed, not counted as active time.
pub const PAUSE_US: i64 = 2_000_000;
/// A batch ends at the first line boundary after this much active typing.
pub const BATCH_ACTIVE_SEC: f64 = 45.0;

/// A line chosen by the trainer, before it is typed.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedLine {
    /// Text to type, including the trailing space that leads into the next line.
    pub text: String,
    pub kind: ContentKind,
    pub lang: Lang,
}

impl PlannedLine {
    pub fn new(text: &str, kind: ContentKind, lang: Lang) -> PlannedLine {
        PlannedLine { text: format!("{} ", text.trim_end()), kind, lang }
    }
}

#[derive(Debug, Clone)]
pub struct KeyInput {
    pub key: Option<KeyId>,
    pub chars: String,
    pub modifiers: u32,
    pub timestamp_us: i64,
    pub wall_ms: i64,
    /// A dead key started a composition; the composed character follows.
    pub is_dead: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyOutcome {
    pub correct: bool,
    pub advanced: bool,
    pub line_finished: bool,
    pub batch_finished: bool,
}

/// What a finished line hands to the engine for storage and learning.
#[derive(Debug, Clone)]
pub struct FinishedLine {
    pub line: TypedLine,
    pub keys: Vec<KeyEventRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BatchTotals {
    pub active_sec: f64,
    pub chars: u32,
    pub errors: u32,
}

impl BatchTotals {
    pub fn wpm(&self) -> f64 {
        if self.active_sec <= 0.0 { 0.0 } else { self.chars as f64 / 5.0 / (self.active_sec / 60.0) }
    }

    pub fn accuracy(&self) -> f64 {
        if self.chars == 0 { 1.0 } else { 1.0 - self.errors as f64 / self.chars as f64 }
    }
}

#[derive(Debug, Default)]
struct Attempt {
    pressed: bool,
    error: bool,
    wrong_key: Option<KeyId>,
    latency_ms: Option<f32>,
    wall_ms: i64,
    session_sec: f32,
}

pub struct TypingSession {
    layout: LayoutId,
    geometry: Geometry,
    lines: VecDeque<PlannedLine>,
    pos: usize,
    attempt: Attempt,
    records: Vec<CharRecord>,
    keys: Vec<KeyEventRecord>,
    error_positions: Vec<usize>,
    last_us: Option<i64>,
    last_correct_us: Option<i64>,
    prev_char: Option<char>,
    /// Character typed right before the current line started.
    line_prev_char: Option<char>,
    session_sec: f64,
    batch: BatchTotals,
}

impl TypingSession {
    pub fn new(layout: LayoutId, geometry: Geometry) -> TypingSession {
        TypingSession {
            layout,
            geometry,
            lines: VecDeque::new(),
            pos: 0,
            attempt: Attempt::default(),
            records: Vec::new(),
            keys: Vec::new(),
            error_positions: Vec::new(),
            last_us: None,
            last_correct_us: None,
            prev_char: None,
            line_prev_char: None,
            session_sec: 0.0,
            batch: BatchTotals::default(),
        }
    }

    pub fn set_layout(&mut self, layout: LayoutId, geometry: Geometry) {
        self.layout = layout;
        self.geometry = geometry;
    }

    pub fn push_line(&mut self, line: PlannedLine) {
        self.lines.push_back(line);
    }

    /// Drop every line, including progress on the current one (layout switched mid-line:
    /// keys typed so far belong to the old layout). Returns true if progress was dropped.
    pub fn drop_all_lines(&mut self) -> bool {
        let had_progress = self.pos > 0 || !self.keys.is_empty();
        if self.pos > 0 {
            self.prev_char = self.line_prev_char;
        }
        self.lines.clear();
        self.pos = 0;
        self.attempt = Attempt::default();
        self.records.clear();
        self.keys.clear();
        self.error_positions.clear();
        had_progress
    }

    /// Drop queued (not yet started) lines, e.g. after the pool changed.
    pub fn clear_upcoming(&mut self) {
        // Keep the current line once anything was typed on it (keys already belong to it).
        let keep = if self.pos > 0 || !self.keys.is_empty() { 1 } else { 0 };
        self.lines.truncate(keep);
    }

    pub fn queued(&self) -> usize {
        self.lines.len()
    }

    pub fn lines(&self) -> impl Iterator<Item = &PlannedLine> {
        self.lines.iter()
    }

    pub fn cursor(&self) -> usize {
        self.pos
    }

    pub fn error_positions(&self) -> &[usize] {
        &self.error_positions
    }

    pub fn session_sec(&self) -> f64 {
        self.session_sec
    }

    pub fn batch(&self) -> BatchTotals {
        self.batch
    }

    pub fn expected(&self) -> Option<char> {
        self.lines.front().and_then(|l| l.text.chars().nth(self.pos))
    }

    /// Raw keys of the line being typed (moved into [`FinishedLine`] when it completes).
    pub fn open_keys(&self) -> &[KeyEventRecord] {
        &self.keys
    }

    /// Character before the cursor (for hints and context).
    pub fn prev_char(&self) -> Option<char> {
        self.prev_char
    }

    /// Process one key event. Returns the outcome and, when a line was completed, the line.
    pub fn key(&mut self, input: &KeyInput) -> (KeyOutcome, Option<FinishedLine>) {
        let Some(expected) = self.expected() else { return (KeyOutcome::default(), None) };
        let gap = self.last_us.map(|t| input.timestamp_us - t);
        if let Some(g) = gap {
            let active = g.clamp(0, PAUSE_US) as f64 / 1e6;
            self.session_sec += active;
            self.batch.active_sec += active;
        }
        self.last_us = Some(input.timestamp_us);
        let correct = !input.is_dead && input.chars == expected.to_string();
        self.keys.push(KeyEventRecord {
            pos: self.pos as u16,
            t_us: input.timestamp_us,
            wall_ms: input.wall_ms,
            typed: input.chars.clone(),
            key: input.key,
            modifiers: input.modifiers,
            correct,
        });
        if input.is_dead {
            return (KeyOutcome::default(), None);
        }
        if !self.attempt.pressed {
            self.attempt = Attempt {
                pressed: true,
                error: !correct,
                wrong_key: if correct { None } else { input.key },
                latency_ms: self.last_correct_us.and_then(|t| {
                    let dt = input.timestamp_us - t;
                    (dt > 0 && dt < PAUSE_US).then_some(dt as f32 / 1000.0)
                }),
                wall_ms: input.wall_ms,
                session_sec: self.session_sec as f32,
            };
            if !correct {
                self.error_positions.push(self.pos);
            }
        }
        if !correct {
            return (KeyOutcome { correct: false, ..Default::default() }, None);
        }
        if self.pos == 0 {
            self.line_prev_char = self.prev_char;
        }
        self.records.push(CharRecord {
            pos: self.pos as u16,
            latency_ms: self.attempt.latency_ms,
            error: self.attempt.error,
            wrong_key: self.attempt.wrong_key,
            wall_ms: self.attempt.wall_ms,
            session_sec: self.attempt.session_sec,
        });
        self.batch.chars += 1;
        self.batch.errors += self.attempt.error as u32;
        self.attempt = Attempt::default();
        self.last_correct_us = Some(input.timestamp_us);
        self.prev_char = Some(expected);
        self.pos += 1;
        let line_len = self.lines.front().map_or(0, |l| l.text.chars().count());
        let mut outcome = KeyOutcome { correct: true, advanced: true, ..Default::default() };
        if self.pos < line_len {
            return (outcome, None);
        }
        let planned = self.lines.pop_front().expect("current line exists");
        let finished = FinishedLine {
            line: TypedLine {
                text: planned.text,
                kind: planned.kind,
                lang: planned.lang,
                layout: self.layout,
                geometry: self.geometry,
                prev_char: self.line_prev_char,
                chars: std::mem::take(&mut self.records),
            },
            keys: std::mem::take(&mut self.keys),
        };
        self.pos = 0;
        self.error_positions.clear();
        outcome.line_finished = true;
        outcome.batch_finished = self.batch.active_sec >= BATCH_ACTIVE_SEC;
        (outcome, Some(finished))
    }

    /// Close the current batch and start a new one.
    pub fn take_batch(&mut self) -> BatchTotals {
        std::mem::take(&mut self.batch)
    }

    /// A long pause (window hidden etc.): the next key starts fresh timing.
    pub fn pause(&mut self) {
        self.last_us = None;
        self.last_correct_us = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(c: &str, t_ms: i64) -> KeyInput {
        KeyInput { key: None, chars: c.into(), modifiers: 0, timestamp_us: t_ms * 1000, wall_ms: t_ms, is_dead: false }
    }

    #[test]
    fn stop_on_error_and_records() {
        let mut s = TypingSession::new(LayoutId::DeIso, Geometry::Iso);
        s.push_line(PlannedLine::new("ab", ContentKind::Words, Lang::De));
        assert!(s.key(&input("a", 0)).0.advanced);
        let (o, _) = s.key(&input("x", 200));
        assert!(!o.correct && !o.advanced);
        assert_eq!(s.error_positions(), &[1]);
        assert!(s.key(&input("b", 400)).0.advanced);
        let (o, finished) = s.key(&input(" ", 550));
        assert!(o.line_finished);
        let line = finished.unwrap().line;
        assert_eq!(line.text, "ab ");
        assert_eq!(line.chars.len(), 3);
        assert!(line.chars[1].error);
        // Latency of 'b' is measured to the first press (the wrong one).
        assert_eq!(line.chars[1].latency_ms, Some(200.0));
        assert_eq!(line.chars[2].latency_ms, Some(150.0));
        assert_eq!(s.batch().chars, 3);
        assert_eq!(s.batch().errors, 1);
    }

    #[test]
    fn pauses_are_not_timed() {
        let mut s = TypingSession::new(LayoutId::DeIso, Geometry::Iso);
        s.push_line(PlannedLine::new("ab", ContentKind::Words, Lang::De));
        s.key(&input("a", 0));
        s.key(&input("b", 5000));
        s.key(&input(" ", 5100));
        assert!((s.batch().active_sec - 2.1).abs() < 1e-9);
    }
}
