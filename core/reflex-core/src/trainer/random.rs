//! Baseline trainer: frequency-weighted random words and random sentences. This is what
//! "just typing" looks like; the ML planner has to beat it.

use super::lines::{self, Mix};
use super::{LineSource, PlanContext};
use crate::content::{ContentKind, Corpus};
use crate::session::PlannedLine;
use rand::rngs::StdRng;

pub struct RandomSource {
    mix: Mix,
}

impl Default for RandomSource {
    fn default() -> RandomSource {
        RandomSource { mix: Mix { sentence: 0.4, real_world: 0.08, code: 0.15 } }
    }
}

impl LineSource for RandomSource {
    fn next_line(&mut self, ctx: &PlanContext, rng: &mut StdRng) -> PlannedLine {
        let lang = ctx.pool.pick_language(rng);
        match self.mix.pick(ctx.pool, rng) {
            ContentKind::Sentence => lines::sentence_line(lang, ctx.layout, |_| 0.0, 1, rng),
            ContentKind::RealWorld => lines::real_world(lang, ctx.layout, rng),
            ContentKind::Code => lines::code_line(lang, ctx.layout, |_| 0.0, rng),
            ContentKind::Words => {
                let words = Corpus::get().words(lang, ctx.pool.vocabulary);
                lines::weighted_word_line(words, |_, w| w.freq.sqrt(), ctx.layout, ctx.recent, lang, rng)
            }
        }
    }
}
