//! Derived numbers for the UI: fatigue, streaks, daily practice.

use crate::store::BatchRecord;
use std::collections::{BTreeSet, VecDeque};

const DAY_MS: i64 = 86_400_000;

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

/// Consecutive days (ending today or yesterday) with practice.
pub fn streak_days(batches: &[BatchRecord], now_ms: i64) -> u32 {
    let days: BTreeSet<i64> = batches.iter().map(|b| b.ended_ms / DAY_MS).collect();
    let today = now_ms / DAY_MS;
    let mut day = if days.contains(&today) { today } else { today - 1 };
    let mut streak = 0;
    while days.contains(&day) {
        streak += 1;
        day -= 1;
    }
    streak
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn streak_counts_back_from_today() {
        let b = vec![batch(10), batch(11), batch(12), batch(14)];
        assert_eq!(streak_days(&b, 14 * DAY_MS + 5), 1);
        assert_eq!(streak_days(&b, 13 * DAY_MS + 5), 3);
        assert_eq!(streak_days(&b, 20 * DAY_MS), 0);
    }

    #[test]
    fn fatigue_needs_time_and_drop() {
        let w: VecDeque<f64> = [80.0, 82.0, 90.0, 91.0].into();
        assert!(!fatigue(&w, 70.0, 100.0));
        assert!(fatigue(&w, 70.0, 900.0));
        assert!(!fatigue(&w, 88.0, 900.0));
    }
}
