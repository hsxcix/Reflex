//! Practice history: how often and how recently each character pair has been typed.
//! These are inputs to the typist model, which learns from data how practice and
//! time away change speed (learning and forgetting are not hand-coded).

use std::collections::HashMap;

const DAY_MS: f64 = 86_400_000.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Entry {
    count: f32,
    last_ms: i64,
}

#[derive(Debug, Clone, Default)]
pub struct PracticeHistory {
    bigrams: HashMap<(char, char), Entry>,
    chars: HashMap<char, f32>,
    total: f64,
}

/// History-derived model inputs for one keystroke.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryFeatures {
    pub log_bigram_count: f32,
    pub log_days_since: f32,
    pub never_seen: f32,
    pub log_char_count: f32,
    pub log_total: f32,
}

impl PracticeHistory {
    pub fn features(&self, prev: Option<char>, c: char, now_ms: i64) -> HistoryFeatures {
        let entry = prev.and_then(|p| self.bigrams.get(&(p, c)));
        let (count, days, never) = match entry {
            Some(e) => (e.count, ((now_ms - e.last_ms).max(0) as f64 / DAY_MS) as f32, 0.0),
            None => (0.0, 365.0, 1.0),
        };
        HistoryFeatures {
            log_bigram_count: count.ln_1p(),
            log_days_since: days.ln_1p(),
            never_seen: never,
            log_char_count: self.chars.get(&c).copied().unwrap_or(0.0).ln_1p(),
            log_total: (self.total as f32).ln_1p(),
        }
    }

    /// Record that `c` was typed after `prev` at `now_ms`.
    pub fn record(&mut self, prev: Option<char>, c: char, now_ms: i64) {
        if let Some(p) = prev {
            let e = self.bigrams.entry((p, c)).or_default();
            e.count += 1.0;
            e.last_ms = now_ms;
        }
        *self.chars.entry(c).or_default() += 1.0;
        self.total += 1.0;
    }

    pub fn record_text(&mut self, prev: Option<char>, text: &str, now_ms: i64) {
        let mut p = prev;
        for c in text.chars() {
            self.record(p, c, now_ms);
            p = Some(c);
        }
    }

    pub fn bigram_count(&self, prev: char, c: char) -> f32 {
        self.bigrams.get(&(prev, c)).map_or(0.0, |e| e.count)
    }

    pub fn total(&self) -> f64 {
        self.total
    }

    /// Seed with prior experience (used by the simulator for "already knows how to type").
    pub fn seed(&mut self, prev: char, c: char, count: f32, last_ms: i64) {
        self.bigrams.insert((prev, c), Entry { count, last_ms });
        *self.chars.entry(c).or_default() += count;
        self.total += count as f64;
    }
}
