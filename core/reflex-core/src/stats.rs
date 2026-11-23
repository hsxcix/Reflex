//! Derived numbers for the UI: fatigue, streaks, daily practice.

use crate::layout::{KeyId, Layout};
use crate::store::{BatchRecord, CharStat};
use std::collections::HashMap;
use std::collections::{BTreeSet, VecDeque};

const DAY_MS: i64 = 86_400_000;

/// Local calendar day of a UTC timestamp, given the UTC offset in minutes (Berlin: +60/+120).
pub fn day_index(ms: i64, utc_offset_min: i32) -> i64 {
    (ms + utc_offset_min.clamp(-840, 840) as i64 * 60_000).div_euclid(DAY_MS)
}

/// UTC timestamp of local midnight that starts `day`.
pub fn day_start_ms(day: i64, utc_offset_min: i32) -> i64 {
    day * DAY_MS - utc_offset_min.clamp(-840, 840) as i64 * 60_000
}

/// True when the last batches show a clear drop against this session's best after
/// at least 10 minutes of practice: a good moment for a break.
pub fn fatigue(recent_wpms: &VecDeque<f64>, current: f64, session_sec: f64) -> bool {
    if session_sec < 600.0 || recent_wpms.len() < 3 {
        return false;
    }
    let best = recent_wpms.iter().copied().fold(current, f64::max);
    let last3: f64 = (current + recent_wpms[0] + recent_wpms[1]) / 3.0;
    last3 < best * 0.88
}

/// Consecutive local days (ending today or yesterday) with practice. Each batch keeps the
/// calendar day it had when typed; `utc_offset_min` (now) decides what "today" is.
pub fn streak_days(batches: &[BatchRecord], now_ms: i64, utc_offset_min: i32) -> u32 {
    let days: BTreeSet<i64> = batches.iter().map(|b| b.local_day(utc_offset_min)).collect();
    let today = day_index(now_ms, utc_offset_min);
    let mut day = if days.contains(&today) { today } else { today - 1 };
    let mut streak = 0;
    while days.contains(&day) {
        streak += 1;
        day -= 1;
    }
    streak
}

/// Measured first-attempt numbers for one physical key.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredKey {
    pub key: KeyId,
    pub attempts: u32,
    pub errors: u32,
    pub error_rate: f64,
    pub mean_ms: Option<f64>,
}

/// Per-character numbers of one layout summed onto the key that types each character
/// (e.g. `a` and `A` both count for the A key). Characters the layout can't type are skipped.
pub fn key_measured(chars: &[CharStat], layout: &Layout) -> Vec<MeasuredKey> {
    let mut sums: HashMap<KeyId, (u32, u32, f64, u32)> = HashMap::new();
    for c in chars {
        let Some(stroke) = layout.main_stroke(c.expected) else { continue };
        let e = sums.entry(stroke.key).or_default();
        e.0 += c.attempts;
        e.1 += c.errors;
        e.2 += c.latency_sum_ms;
        e.3 += c.latency_n;
    }
    let mut out: Vec<MeasuredKey> = sums
        .into_iter()
        .filter(|(_, (attempts, ..))| *attempts > 0)
        .map(|(key, (attempts, errors, sum, n))| MeasuredKey {
            key,
            attempts,
            errors,
            error_rate: errors as f64 / attempts as f64,
            mean_ms: (n > 0).then(|| sum / n as f64),
        })
        .collect();
    out.sort_by_key(|m| m.key);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Geometry, LayoutId};

    fn stat(c: char, attempts: u32, errors: u32) -> CharStat {
        CharStat { expected: c, attempts, errors, latency_sum_ms: 150.0 * (attempts - errors) as f64, latency_n: attempts - errors }
    }

    #[test]
    fn measured_error_rate_per_key() {
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        // 'a' 8× with 1 miss and 'A' 2× with 1 miss: the A key has 2 errors in 10 tries.
        let keys = key_measured(&[stat('a', 8, 1), stat('A', 2, 1), stat('s', 4, 0)], &layout);
        let a = keys.iter().find(|k| k.key == layout.main_stroke('a').unwrap().key).unwrap();
        assert_eq!((a.attempts, a.errors), (10, 2));
        assert!((a.error_rate - 0.2).abs() < 1e-9);
        assert_eq!(a.mean_ms, Some(150.0));
        assert_eq!(keys.len(), 2);
    }

    fn batch(day: i64) -> BatchRecord {
        BatchRecord {
            session_id: 1,
            ended_ms: day * DAY_MS + 1000,
            seconds: 45.0,
            chars: 300,
            errors: 5,
            wpm: 80.0,
            accuracy: 0.98,
            real_wpm: None,
            arm: None,
            utc_offset_min: None,
        }
    }

    #[test]
    fn streak_counts_back_from_today() {
        let b = vec![batch(10), batch(11), batch(12), batch(14)];
        assert_eq!(streak_days(&b, 14 * DAY_MS + 5, 0), 1);
        assert_eq!(streak_days(&b, 13 * DAY_MS + 5, 0), 3);
        assert_eq!(streak_days(&b, 20 * DAY_MS, 0), 0);
    }

    #[test]
    fn days_follow_local_time() {
        // 23:30 UTC is already the next day in Berlin summer time (UTC+2).
        let late = 10 * DAY_MS + 23 * 3_600_000 + 30 * 60_000;
        assert_eq!(day_index(late, 0), 10);
        assert_eq!(day_index(late, 120), 11);
        assert_eq!(day_start_ms(11, 120), 11 * DAY_MS - 2 * 3_600_000);
        // Before the epoch, days still round down.
        assert_eq!(day_index(-1, 0), -1);
    }

    #[test]
    fn batches_keep_their_day_across_dst() {
        // Berlin, 11 July 00:30 summer time (UTC+2) = 10 July 22:30 UTC.
        let day = 200;
        let mut b = batch(day);
        b.ended_ms = day * DAY_MS - 90 * 60_000;
        b.utc_offset_min = Some(120);
        // Now it is winter time (UTC+1): the batch still counts for 11 July (day 200)...
        assert_eq!(b.local_day(60), day);
        // ...in the streak too, seen from 11 July.
        assert_eq!(streak_days(&[b.clone()], day * DAY_MS + 12 * 3_600_000, 60), 1);
        // Without a stored offset, the current one is used (old rows).
        b.utc_offset_min = None;
        assert_eq!(b.local_day(60), day - 1);
    }

    #[test]
    fn fatigue_needs_time_and_drop() {
        let w: VecDeque<f64> = [80.0, 82.0, 90.0, 91.0].into();
        assert!(!fatigue(&w, 70.0, 100.0));
        assert!(fatigue(&w, 70.0, 900.0));
        assert!(!fatigue(&w, 88.0, 900.0));
    }
}
