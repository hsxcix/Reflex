//! Hand-written baseline: "drill your slowest, most error-prone characters". The ML planner
//! must beat this in the benchmarks, otherwise the ML isn't earning its keep.

use super::lines::{self, Mix};
use super::{LineSource, PlanContext};
use crate::content::{ContentKind, Corpus};
use crate::record::TypedLine;
use crate::session::PlannedLine;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[derive(Default)]
pub struct WeakestKeys {
    stats: HashMap<char, (f64, f64, f64)>,
}

impl WeakestKeys {
    /// Update per-character averages (latency, error count, samples) from typed lines.
    pub fn observe(&mut self, lines: &[TypedLine]) {
        for line in lines {
            let chars: Vec<char> = line.text.chars().collect();
            for rec in &line.chars {
                let Some(&c) = chars.get(rec.pos as usize) else { continue };
                let e = self.stats.entry(c.to_ascii_lowercase()).or_default();
                e.0 += rec.latency_ms.unwrap_or(0.0) as f64;
                e.1 += rec.error as u8 as f64;
                e.2 += 1.0;
            }
        }
    }

    fn slowness(&self, c: char) -> f64 {
        match self.stats.get(&c.to_ascii_lowercase()) {
            Some((lat, err, n)) if *n >= 5.0 => (lat / n) * (1.0 + 2.0 * err / n),
            _ => 0.0,
        }
    }
}

impl LineSource for WeakestKeys {
    fn next_line(&mut self, ctx: &PlanContext, rng: &mut StdRng) -> PlannedLine {
        let lang = ctx.pool.pick_language(rng);
        let mix = Mix { sentence: 0.3, real_world: 0.06, code: 0.15 };
        match mix.pick(ctx.pool, rng) {
            ContentKind::Sentence => lines::sentence_line(lang, ctx.layout, |_| 0.0, 1, rng),
            ContentKind::RealWorld => lines::real_world(lang, ctx.layout, rng),
            ContentKind::Code => lines::code_line(lang, ctx.layout, |_| 0.0, rng),
            ContentKind::Words => {
                let mut ranked: Vec<(char, f64)> = self.stats.keys().map(|c| (*c, self.slowness(*c))).collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                let worst: Vec<char> = ranked.iter().take(5).map(|(c, _)| *c).collect();
                let words = Corpus::get().words(lang, ctx.pool.vocabulary);
                lines::weighted_word_line(
                    words,
                    |_, w| {
                        let hits = w.text.chars().filter(|c| worst.contains(&c.to_ascii_lowercase())).count();
                        w.freq.sqrt() * (1.0 + 8.0 * hits as f64)
                    },
                    ctx.layout,
                    ctx.recent,
                    lang,
                    rng,
                )
            }
        }
    }
}
