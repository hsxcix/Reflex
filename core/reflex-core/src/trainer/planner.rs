//! The ML planner: picks words, sentences and real-world lines whose movements the typist
//! model predicts will speed up normal typing the most, with exploration where the model
//! is unsure. Its knobs are tuned per person by the bandit.

use super::gains::GainTable;
use super::lines::{self, Mix};
use super::{LineSource, PlanContext};
use crate::content::{ContentKind, Corpus, Lang, Vocabulary};
use crate::history::pair_key;
use crate::layout::{LayoutId, Motor};
use crate::session::PlannedLine;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Arc;

/// Planner settings the bandit chooses between.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knobs {
    /// Softmax temperature over word value: low = sharp focus on the top weaknesses.
    pub temperature: f64,
    /// Share of natural sentences (also keeps "real speed" measurable).
    pub natural_share: f64,
    /// Weight of model uncertainty (exploration bonus, relative to predicted gain).
    pub explore: f64,
}

/// The arms the bandit can pick (focus × natural share).
pub const ARMS: [Knobs; 6] = [
    Knobs { temperature: 0.35, natural_share: 0.2, explore: 0.12 },
    Knobs { temperature: 0.7, natural_share: 0.2, explore: 0.12 },
    Knobs { temperature: 1.4, natural_share: 0.2, explore: 0.12 },
    Knobs { temperature: 0.35, natural_share: 0.45, explore: 0.12 },
    Knobs { temperature: 0.7, natural_share: 0.45, explore: 0.12 },
    Knobs { temperature: 1.4, natural_share: 0.45, explore: 0.12 },
];

pub struct Planner {
    gains: Arc<GainTable>,
    knobs: Knobs,
    /// Cached word weights per (language, vocabulary, layout).
    cache: HashMap<(Lang, Vocabulary, LayoutId), Vec<f64>>,
}

impl Planner {
    pub fn new(gains: Arc<GainTable>, knobs: Knobs) -> Planner {
        Planner { gains, knobs, cache: HashMap::new() }
    }

    /// Value of practising `text` per keystroke: predicted gain plus exploration bonus.
    fn value(&self, text: &str, layout: &crate::layout::Layout) -> f64 {
        let mut prev = Some(' ');
        let mut total = 0.0f64;
        let mut n = 0usize;
        let mut seen: Vec<(Motor, Motor)> = Vec::new();
        for c in text.chars().chain(std::iter::once(' ')) {
            if let Some(key) = pair_key(layout, prev, c) {
                // Diminishing returns for the same pair within one item.
                let repeat = seen.iter().filter(|k| **k == key).count() as f64;
                let g = *self.gains.gain.get(&key).unwrap_or(&0.0) as f64;
                let u = *self.gains.uncertainty.get(&key).unwrap_or(&0.0) as f64;
                total += (g + self.knobs.explore * u) / (1.0 + repeat);
                seen.push(key);
            }
            n += 1;
            prev = Some(c);
        }
        if n == 0 { 0.0 } else { total / n as f64 }
    }

    fn word_weights(&mut self, lang: Lang, vocabulary: Vocabulary, layout: &crate::layout::Layout) -> &[f64] {
        let key = (lang, vocabulary, layout.id);
        if !self.cache.contains_key(&key) {
            let words = Corpus::get().words(lang, vocabulary);
            let values: Vec<f64> = words.iter().map(|w| self.value(&w.text, layout)).collect();
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64).sqrt().max(1e-9);
            let t = self.knobs.temperature;
            let weights = values.iter().map(|v| (((v - mean) / sd) / t).clamp(-20.0, 20.0).exp()).collect();
            self.cache.insert(key, weights);
        }
        &self.cache[&key]
    }

    /// Precompute word weights for every language of `pool` (done on the learning thread,
    /// so the first line after a planner swap doesn't score 20k words on the UI thread).
    pub fn prepare(&mut self, pool: &crate::content::Pool, layout: &crate::layout::Layout) {
        for lang in &pool.languages {
            self.word_weights(*lang, pool.vocabulary, layout);
        }
    }

    fn mix(&self) -> Mix {
        // More real-world text when digits and symbols carry much of the potential gain.
        let real_world = (0.04 + 0.5 * self.gains.symbol_share as f64).min(0.3);
        Mix { sentence: self.knobs.natural_share, real_world, code: 0.18 }
    }
}

impl LineSource for Planner {
    fn next_line(&mut self, ctx: &PlanContext, rng: &mut StdRng) -> PlannedLine {
        let lang = ctx.pool.pick_language(rng);
        match self.mix().pick(ctx.pool, rng) {
            ContentKind::Sentence | ContentKind::Probe => lines::sentence_line(lang, ctx.layout, |s| self.value(s, ctx.layout), 24, rng),
            ContentKind::RealWorld => {
                let a = lines::real_world(lang, ctx.layout, rng);
                let b = lines::real_world(lang, ctx.layout, rng);
                if self.value(&a.text, ctx.layout) >= self.value(&b.text, ctx.layout) { a } else { b }
            }
            ContentKind::Code => lines::code_line(lang, ctx.layout, |s| self.value(s, ctx.layout), rng),
            ContentKind::Words => {
                let vocabulary = ctx.pool.vocabulary;
                let words = Corpus::get().words(lang, vocabulary);
                let weights = self.word_weights(lang, vocabulary, ctx.layout).to_vec();
                lines::weighted_word_line(
                    words,
                    |i, w| weights[i] * w.freq.powf(0.15),
                    ctx.layout,
                    ctx.recent,
                    lang,
                    rng,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Pool;
    use crate::layout::{Geometry, Layout};
    use crate::trainer::lines::RecentWords;
    use rand::SeedableRng;

    #[test]
    fn prefers_words_with_valuable_pairs() {
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let mut table = GainTable::default();
        // Make every pair containing 'q' hugely valuable.
        for prev in "abcdefghijklmnopqrstuvwxyz ".chars() {
            for (a, b) in [(prev, 'q'), ('q', prev)] {
                if let Some(key) = pair_key(&layout, Some(a), b) {
                    table.gain.insert(key, 10.0);
                }
            }
        }
        let mut planner = Planner::new(Arc::new(table), ARMS[1]);
        let pool = Pool { languages: vec![Lang::En], vocabulary: Vocabulary::Broad, code: false };
        let mut recent = RecentWords::default();
        let mut rng = StdRng::seed_from_u64(3);
        let (mut q_words, mut words) = (0, 0);
        for _ in 0..40 {
            let ctx = PlanContext { pool: &pool, layout: &layout, recent: &recent };
            let line = planner.next_line(&ctx, &mut rng);
            recent.note_line(&line.text);
            if line.kind == ContentKind::Words {
                for w in line.text.split_whitespace() {
                    words += 1;
                    q_words += w.contains('q') as usize;
                }
            }
        }
        // 'q' words are well under 1% of English text.
        assert!(q_words * 4 > words, "{q_words} of {words} words contain q");
    }
}
