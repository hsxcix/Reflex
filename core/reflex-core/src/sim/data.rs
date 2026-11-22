//! Generating typing histories of simulated people (pretraining and replay data).

use super::{Family, SimTypist};
use crate::content::{Corpus, Pool};
use crate::layout::{Geometry, Layout, LayoutId};
use crate::record::TypedLine;
use crate::session::PlannedLine;
use crate::trainer::lines::{self, RecentWords};
use crate::trainer::random::RandomSource;
use crate::trainer::{LineSource, PlanContext};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};

pub const SIM_START_MS: i64 = 1_767_225_600_000; // 2026-01-01

/// A line source that drills a few random characters hard, so pretraining sees
/// targeted practice (and its effect) as well as natural text.
struct FocusedSource {
    focus: Vec<char>,
}

impl LineSource for FocusedSource {
    fn next_line(&mut self, ctx: &PlanContext, rng: &mut StdRng) -> PlannedLine {
        let lang = ctx.pool.pick_language(rng);
        let words = Corpus::get().words(lang, ctx.pool.vocabulary);
        let focus = &self.focus;
        lines::weighted_word_line(
            words,
            |_, w| {
                let hits = w.text.chars().filter(|c| focus.contains(&c.to_ascii_lowercase())).count();
                w.freq.sqrt() * (1.0 + 25.0 * hits as f64)
            },
            ctx.layout,
            ctx.recent,
            lang,
            rng,
        )
    }
}

fn random_layout(rng: &mut StdRng) -> Layout {
    let r = rng.random::<f64>();
    if r < 0.6 {
        Layout::new(LayoutId::DeIso, Geometry::Iso)
    } else if r < 0.85 {
        Layout::new(LayoutId::UsAnsi, if rng.random() { Geometry::Iso } else { Geometry::Ansi })
    } else {
        Layout::new(LayoutId::EurKey, Geometry::Ansi)
    }
}

/// Simulate one person over several practice sessions spread across days.
pub fn simulate_history(family: Family, seed: u64, sessions: usize) -> Vec<TypedLine> {
    let mut rng = StdRng::seed_from_u64(seed);
    let home = random_layout(&mut rng);
    let params = family.sample(&home, &mut rng);
    let mut typist = SimTypist::new(params, home, SIM_START_MS);
    let switch_at = rng.random_bool(0.12).then(|| rng.random_range(1..sessions.max(2)));
    let pool = Pool { code: rng.random_bool(0.3), ..Pool::default() };
    let mut recent = RecentWords::default();
    let mut random = RandomSource::default();
    let mut now = SIM_START_MS;
    let mut out = Vec::new();
    let all_chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzäöüß.,;:-?!0123456789".chars().collect();
    for s in 0..sessions {
        if switch_at == Some(s) {
            let to = if typist.layout.id == LayoutId::DeIso { LayoutId::UsAnsi } else { LayoutId::DeIso };
            typist.set_layout(Layout::new(to, Geometry::Iso));
        }
        let mut session_sec = 0.0f32;
        let mut prev: Option<char> = None;
        let n_lines = rng.random_range(15..60);
        let mut focused = FocusedSource { focus: Vec::new() };
        for i in 0..n_lines {
            if i % 8 == 0 {
                let n = rng.random_range(2..6);
                focused.focus = all_chars.choose_multiple(&mut rng, n).copied().collect();
            }
            let ctx = PlanContext { pool: &pool, layout: &typist.layout, recent: &recent };
            let line = if rng.random_bool(0.45) {
                focused.next_line(&ctx, &mut rng)
            } else {
                random.next_line(&ctx, &mut rng)
            };
            recent.note_line(&line.text);
            let typed = typist.type_line(&line, prev, now, &mut session_sec, &mut rng);
            now = typed.chars.last().map_or(now, |c| c.wall_ms + 300);
            prev = Some(' ');
            out.push(typed);
        }
        now += (rng.random_range(0.4..3.5) * 86_400_000.0) as i64;
    }
    out
}
