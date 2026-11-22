//! Families of simulated typists. Pretraining only ever sees the training families;
//! benchmarks run on held-out families so a trainer can't win by memorising the simulator.

use super::{TypistParams, normal};
use crate::content::Corpus;
use crate::history::pair_key;
use crate::layout::{Layout, Motor};
use rand::Rng;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    // Training families.
    Steady,
    FastLearner,
    SlowLearner,
    ErrorProne,
    SymbolWeak,
    // Held-out benchmark families.
    Forgetful,
    WeakFingers,
    Returning,
}

impl Family {
    pub const TRAIN: [Family; 5] =
        [Family::Steady, Family::FastLearner, Family::SlowLearner, Family::ErrorProne, Family::SymbolWeak];
    pub const HELD_OUT: [Family; 3] = [Family::Forgetful, Family::WeakFingers, Family::Returning];

    pub fn sample(self, layout: &Layout, rng: &mut StdRng) -> TypistParams {
        let mut p = TypistParams {
            base_ms: normal(rng, 150.0, 25.0).clamp(95.0, 240.0),
            finger_mult: [1.0; 10],
            weak_pairs: HashMap::new(),
            base_error: normal(rng, 0.035, 0.01).clamp(0.01, 0.08),
            key_error_mult: [1.0; 64],
            learn_tau: normal(rng, 60.0, 15.0).clamp(20.0, 120.0),
            learn_alpha: normal(rng, 0.35, 0.06).clamp(0.15, 0.6),
            asym: normal(rng, 0.55, 0.06).clamp(0.4, 0.75),
            forget_share: normal(rng, 0.25, 0.07).clamp(0.05, 0.5),
            forget_days: normal(rng, 10.0, 3.0).clamp(3.0, 30.0),
            fatigue_rate: normal(rng, 0.08, 0.03).clamp(0.0, 0.2),
            noise_sigma: normal(rng, 0.3, 0.05).clamp(0.18, 0.45),
            shift_cost: normal(rng, 1.25, 0.08).clamp(1.05, 1.6),
            option_cost: normal(rng, 1.6, 0.15).clamp(1.2, 2.3),
            experience: normal(rng, 150_000.0, 50_000.0).clamp(30_000.0, 400_000.0),
        };
        for f in p.finger_mult.iter_mut() {
            *f = normal(rng, 1.0, 0.08).clamp(0.8, 1.3);
        }
        p.finger_mult[0] *= 1.12;
        p.finger_mult[9] *= 1.12;
        for e in p.key_error_mult.iter_mut() {
            *e = (normal(rng, 0.0, 0.35)).exp();
        }
        let weak_count = rng.random_range(25..70);
        let mut weak_strength = 1.9;
        match self {
            Family::Steady => {}
            Family::FastLearner => {
                p.learn_tau *= 0.5;
                p.learn_alpha *= 1.3;
            }
            Family::SlowLearner => {
                p.learn_tau *= 1.8;
                p.learn_alpha *= 0.75;
            }
            Family::ErrorProne => {
                p.base_error *= 2.0;
                p.noise_sigma *= 1.2;
            }
            Family::SymbolWeak => {
                p.shift_cost *= 1.15;
                p.option_cost *= 1.3;
                weak_strength = 2.3;
            }
            Family::Forgetful => {
                p.forget_share = 0.55;
                p.forget_days = 2.5;
            }
            Family::WeakFingers => {
                p.finger_mult[1] *= 1.25;
                p.finger_mult[8] *= 1.25;
                weak_strength = 2.4;
            }
            Family::Returning => {
                p.experience *= 0.35;
                p.base_ms *= 1.2;
                p.learn_tau *= 0.8;
            }
        }
        p.weak_pairs = sample_weak_pairs(layout, weak_count, weak_strength, rng);
        p
    }
}

/// Weak movement pairs are drawn from pairs that actually occur in text.
fn sample_weak_pairs(layout: &Layout, count: usize, strength: f32, rng: &mut StdRng) -> HashMap<(Motor, Motor), f32> {
    let corpus = Corpus::get();
    let mut pairs: Vec<(Motor, Motor)> = Vec::new();
    for lang in crate::content::Lang::ALL {
        for w in corpus.words(lang, crate::content::Vocabulary::Broad).iter().take(3000) {
            let mut prev = ' ';
            for c in w.text.chars().chain(std::iter::once(' ')) {
                if let Some(key) = pair_key(layout, Some(prev), c) {
                    pairs.push(key);
                }
                prev = c;
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let mut out = HashMap::new();
    while out.len() < count.min(pairs.len()) {
        let pair = pairs[rng.random_range(0..pairs.len())];
        let w = (strength + normal(rng, 0.0, 0.35)).clamp(1.3, 3.5);
        out.insert(pair, w);
    }
    out
}
