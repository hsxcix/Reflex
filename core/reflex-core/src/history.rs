//! Practice history: how often, how recently and how well each *movement pair* (previous
//! key press → this key press, with modifiers) has been typed. Tracking movements instead
//! of characters means skills transfer between layouts exactly where movements match.
//!
//! These are the typist model's personal inputs. Running averages of speed and errors are
//! frozen at the start of each session while practice counts keep counting, so the model
//! can learn from data how much *more practice* helps a pair given how weak it was: the
//! exact question the planner asks. Learning and forgetting are not hand-coded.

use crate::layout::{Layout, MOTOR_START, Motor};
use std::collections::HashMap;

const DAY_MS: f64 = 86_400_000.0;
/// A pause longer than this starts a new session.
const SESSION_GAP_MS: i64 = 30 * 60 * 1000;
/// Smoothing for running averages (recent keystrokes count more: skills change).
const EMA: f32 = 0.15;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Skill {
    /// Running average of log latency over correct, timed presses.
    log_latency: f32,
    timed: f32,
    /// Running average error rate.
    error: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Entry {
    count: f32,
    last_ms: i64,
    live: Skill,
    frozen: Skill,
    frozen_epoch: u32,
}

#[derive(Debug, Clone, Default)]
pub struct PracticeHistory {
    pairs: HashMap<(Motor, Motor), Entry>,
    singles: HashMap<Motor, f32>,
    total: f64,
    epoch: u32,
    last_ms: i64,
    global: Entry,
}

/// History-derived model inputs for one keystroke.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryFeatures {
    pub log_bigram_count: f32,
    pub log_days_since: f32,
    pub never_seen: f32,
    pub log_char_count: f32,
    pub log_total: f32,
    /// How much slower this pair was than the person's average (log ratio, session start).
    pub relative_latency: f32,
    /// Error rate on this pair (session start).
    pub error_rate: f32,
    /// 0…1: how much timing data backs `relative_latency`.
    pub evidence: f32,
}

/// The movement-pair key for typing `c` after `prev` on `layout`.
pub fn pair_key(layout: &Layout, prev: Option<char>, c: char) -> Option<(Motor, Motor)> {
    let cur = layout.motor(c)?;
    let prev = prev.and_then(|p| layout.motor(p)).unwrap_or(MOTOR_START);
    Some((prev, cur))
}

impl PracticeHistory {
    /// Skill as seen from `now_ms`: frozen within the running session, live once it ended.
    fn skill(&self, e: &Entry, now_ms: i64) -> Skill {
        let same_session = now_ms - self.last_ms <= SESSION_GAP_MS;
        if same_session && e.frozen_epoch == self.epoch { e.frozen } else { e.live }
    }

    pub fn features(&self, layout: &Layout, prev: Option<char>, c: char, now_ms: i64) -> HistoryFeatures {
        let key = pair_key(layout, prev, c);
        let entry = key.and_then(|k| self.pairs.get(&k));
        let global = self.skill(&self.global, now_ms);
        let (count, days, never) = match entry {
            Some(e) => (e.count, ((now_ms - e.last_ms).max(0) as f64 / DAY_MS) as f32, 0.0),
            None => (0.0, 365.0, 1.0),
        };
        let single = key.and_then(|(_, cur)| self.singles.get(&cur)).copied().unwrap_or(0.0);
        let skill = entry.map(|e| self.skill(e, now_ms));
        let (relative_latency, evidence) = match skill {
            Some(s) if s.timed > 0.0 && global.timed > 0.0 => (s.log_latency - global.log_latency, (s.timed / 12.0).min(1.0)),
            _ => (0.0, 0.0),
        };
        HistoryFeatures {
            log_bigram_count: count.ln_1p(),
            log_days_since: days.ln_1p(),
            never_seen: never,
            log_char_count: single.ln_1p(),
            log_total: (self.total as f32).ln_1p(),
            relative_latency,
            error_rate: skill.filter(|s| s.timed > 0.0).map_or(global.error, |s| s.error),
            evidence,
        }
    }

    /// Record that `c` was typed after `prev` at `now_ms`, with its outcome.
    pub fn record(&mut self, layout: &Layout, prev: Option<char>, c: char, now_ms: i64, latency_ms: Option<f32>, error: bool) {
        let Some(key) = pair_key(layout, prev, c) else { return };
        if now_ms - self.last_ms > SESSION_GAP_MS {
            self.epoch += 1;
        }
        self.last_ms = self.last_ms.max(now_ms);
        let epoch = self.epoch;
        let global_error = self.global.live.error;
        let e = self.pairs.entry(key).or_insert(Entry {
            live: Skill { error: global_error, ..Skill::default() },
            ..Entry::default()
        });
        update(e, epoch, now_ms, latency_ms, error, EMA);
        update(&mut self.global, epoch, now_ms, latency_ms, error, 0.01);
        *self.singles.entry(key.1).or_default() += 1.0;
        self.total += 1.0;
    }

    pub fn pair_count(&self, key: (Motor, Motor)) -> f32 {
        self.pairs.get(&key).map_or(0.0, |e| e.count)
    }

    pub fn total(&self) -> f64 {
        self.total
    }
}

fn update(e: &mut Entry, epoch: u32, now_ms: i64, latency_ms: Option<f32>, error: bool, rate: f32) {
    if e.frozen_epoch != epoch {
        e.frozen = e.live;
        e.frozen_epoch = epoch;
    }
    e.count += 1.0;
    e.last_ms = now_ms;
    let s = &mut e.live;
    s.error += (error as u8 as f32 - s.error) * rate.max(1.0 / e.count);
    if let (Some(ms), false) = (latency_ms, error) {
        s.timed += 1.0;
        s.log_latency += (ms.max(20.0).ln() - s.log_latency) * rate.max(1.0 / s.timed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Geometry, LayoutId};

    #[test]
    fn movements_transfer_between_layouts() {
        let de = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let us = Layout::new(LayoutId::UsAnsi, Geometry::Iso);
        let mut h = PracticeHistory::default();
        h.record(&de, Some('a'), 's', 0, Some(150.0), false);
        // Same movement (KeyA → KeyS) on the US layout: known.
        assert_eq!(h.features(&us, Some('a'), 's', 0).never_seen, 0.0);
        // 'z' moved (KeyY on German, KeyZ on US): a different movement.
        h.record(&de, Some('a'), 'z', 0, Some(300.0), false);
        assert_eq!(h.features(&us, Some('a'), 'z', 0).never_seen, 1.0);
    }

    #[test]
    fn skill_is_frozen_within_a_session() {
        let de = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let mut h = PracticeHistory::default();
        h.record(&de, Some('a'), 's', 0, Some(100.0), false);
        h.record(&de, Some('q'), 'p', 10, Some(400.0), false);
        let hour = 60 * 60 * 1000;
        // New session: the slow pair is visible as slow.
        h.record(&de, Some('a'), 's', hour, Some(100.0), false);
        let slow = h.features(&de, Some('q'), 'p', hour + 10);
        assert!(slow.relative_latency > 0.0);
        // Practising within the session changes the count but not the frozen skill.
        h.record(&de, Some('q'), 'p', hour + 20, Some(120.0), false);
        let after = h.features(&de, Some('q'), 'p', hour + 30);
        assert_eq!(after.relative_latency, slow.relative_latency);
        assert!(after.log_bigram_count > slow.log_bigram_count);
    }
}
