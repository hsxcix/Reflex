//! Simulated typists: the training ground for pretraining and the test bench for every
//! trainer. Each typist has hidden "true" skills (per finger, per character pair) that
//! improve with practice, fade without it, and suffer from fatigue. The model never sees
//! these; it only sees the keystrokes, just like with a real person.

mod family;
pub mod data;

pub use family::Family;

use crate::content::{Corpus, Lang};
use crate::history::pair_key;
use crate::layout::{Finger, KeyId, Layer, Layout, Motor, Stroke};
use crate::record::{CharRecord, TypedLine};
use crate::session::PlannedLine;
use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, LogNormal, Normal};
use std::collections::HashMap;

const DAY_MS: f64 = 86_400_000.0;

/// Hidden parameters of one simulated person.
#[derive(Debug, Clone)]
pub struct TypistParams {
    pub base_ms: f32,
    pub finger_mult: [f32; 10],
    /// Extra slowness for specific movement pairs (multiplier ≥ 1 before practice).
    pub weak_pairs: HashMap<(Motor, Motor), f32>,
    pub base_error: f32,
    pub key_error_mult: [f32; 64],
    /// Learning curve: skill = asym + (1 - asym) · (1 + n/tau)^-alpha.
    pub learn_tau: f32,
    pub learn_alpha: f32,
    pub asym: f32,
    /// Share of practice that fades, and how fast (days).
    pub forget_share: f32,
    pub forget_days: f32,
    /// Slowdown per 30 minutes of active typing beyond the first 8.
    pub fatigue_rate: f32,
    pub noise_sigma: f32,
    pub shift_cost: f32,
    pub option_cost: f32,
    pub experience: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct Practice {
    count: f32,
    last_ms: i64,
}

pub struct SimTypist {
    pub params: TypistParams,
    pub layout: Layout,
    practice: HashMap<(Motor, Motor), Practice>,
}

/// One simulated keystroke outcome.
#[derive(Debug, Clone, Copy)]
pub struct SimKey {
    pub latency_ms: f32,
    pub error: bool,
    pub wrong_key: Option<KeyId>,
}

impl SimTypist {
    pub fn new(params: TypistParams, layout: Layout, start_ms: i64) -> SimTypist {
        let mut typist = SimTypist { params, layout, practice: HashMap::new() };
        typist.seed_experience(start_ms);
        typist
    }

    /// Prior typing experience on the typist's home layout: movements that are common in
    /// real text have been typed a lot (weak ones much less, part of why they are weak).
    fn seed_experience(&mut self, start_ms: i64) {
        let corpus = Corpus::get();
        let mut freq: HashMap<(Motor, Motor), f64> = HashMap::new();
        for lang in Lang::ALL {
            for w in corpus.words(lang, crate::content::Vocabulary::Full).iter().take(5000) {
                let mut prev = ' ';
                for c in w.text.chars().chain(std::iter::once(' ')) {
                    if let Some(key) = pair_key(&self.layout, Some(prev), c) {
                        *freq.entry(key).or_default() += w.freq;
                    }
                    prev = c;
                }
            }
        }
        let total: f64 = freq.values().sum();
        for (pair, f) in freq {
            let weak = self.params.weak_pairs.contains_key(&pair);
            let count = (f / total) as f32 * self.params.experience * if weak { 0.04 } else { 1.0 };
            self.practice.insert(pair, Practice { count, last_ms: start_ms - 3 * DAY_MS as i64 });
        }
    }

    /// Switch to another layout (muscle memory stays with the movements).
    pub fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
    }

    fn effective_practice(&self, pair: (Motor, Motor), now_ms: i64) -> f32 {
        let p = self.practice.get(&pair).copied().unwrap_or_default();
        let days = ((now_ms - p.last_ms).max(0) as f64 / DAY_MS) as f32;
        let keep = 1.0 - self.params.forget_share * (1.0 - (-days / self.params.forget_days).exp());
        p.count * keep
    }

    /// Learning-curve factor in (asym, 1]; 1 = untrained.
    fn skill(&self, n: f32) -> f32 {
        let p = &self.params;
        p.asym + (1.0 - p.asym) * (1.0 + n / p.learn_tau).powf(-p.learn_alpha)
    }

    /// Expected cost (ms, errors included) of typing `c` after `prev`: the ground truth
    /// that "real speed" is measured with in benchmarks.
    pub fn expected_ms(&self, prev: Option<char>, c: char, now_ms: i64, session_sec: f32) -> Option<f32> {
        let (mean, err) = self.mean_and_error(prev, c, now_ms, session_sec)?;
        Some(mean * (1.0 + 2.0 * err))
    }

    /// Ground truth: expected cost if this pair had `extra` more repetitions.
    pub fn expected_ms_with_extra(&self, prev: Option<char>, c: char, now_ms: i64, extra: f32) -> Option<f32> {
        let (mean, err) = self.mean_and_error_n(prev, c, now_ms, 0.0, extra)?;
        Some(mean * (1.0 + 2.0 * err))
    }

    fn mean_and_error(&self, prev: Option<char>, c: char, now_ms: i64, session_sec: f32) -> Option<(f32, f32)> {
        self.mean_and_error_n(prev, c, now_ms, session_sec, 0.0)
    }

    fn mean_and_error_n(&self, prev: Option<char>, c: char, now_ms: i64, session_sec: f32, extra: f32) -> Option<(f32, f32)> {
        let p = &self.params;
        let strokes = self.layout.strokes(c)?;
        let main = *strokes.last()?;
        let prev_main = prev.and_then(|pc| self.layout.main_stroke(pc));
        let pair = pair_key(&self.layout, prev, c)?;
        let n = self.effective_practice(pair, now_ms) + extra;
        let skill = self.skill(n);
        let untrained = (skill - p.asym) / (1.0 - p.asym);
        let weak = p.weak_pairs.get(&pair).map_or(1.0, |w| 1.0 + (w - 1.0) * untrained.sqrt());
        let finger = self.layout.key(main.key).finger;
        let mut mean = p.base_ms * p.finger_mult[finger.index()] * transition(&self.layout, prev_main, main, strokes.len());
        if main.layer.shift() {
            mean *= p.shift_cost;
        }
        if main.layer.option() {
            mean *= p.option_cost;
        }
        let fatigue = 1.0 + p.fatigue_rate * ((session_sec - 480.0).max(0.0) / 1800.0);
        mean *= skill * weak * fatigue;
        let err = (p.base_error
            * p.key_error_mult[main.key as usize % 64]
            * weak.powf(1.5)
            * (0.35 + 0.65 * untrained)
            * fatigue.powi(2))
        .clamp(0.0, 0.45);
        Some((mean, err))
    }

    /// Type one character: sample a latency and whether the first press was wrong.
    pub fn press(&mut self, prev: Option<char>, c: char, now_ms: i64, session_sec: f32, rng: &mut StdRng) -> Option<SimKey> {
        let (mean, err) = self.mean_and_error(prev, c, now_ms, session_sec)?;
        let sigma = self.params.noise_sigma;
        let noise = LogNormal::new(-sigma * sigma / 2.0, sigma).expect("valid sigma").sample(rng);
        let error = rng.random::<f32>() < err;
        let wrong_key = error.then(|| self.wrong_key_for(c, rng)).flatten();
        let pair = pair_key(&self.layout, prev, c)?;
        let entry = self.practice.entry(pair).or_default();
        entry.count += 1.0;
        entry.last_ms = now_ms;
        Some(SimKey { latency_ms: (mean * noise).max(25.0), error, wrong_key })
    }

    fn wrong_key_for(&self, c: char, rng: &mut StdRng) -> Option<KeyId> {
        let target = self.layout.main_stroke(c)?;
        let tg = self.layout.key(target.key);
        let neighbours: Vec<KeyId> = self
            .layout
            .geometry
            .keys()
            .filter(|k| *k != target.key)
            .filter(|k| {
                let g = self.layout.key(*k);
                (g.row == tg.row && (g.x - tg.x).abs() < 1.3) || (g.finger == tg.finger && g.row.abs_diff(tg.row) == 1)
            })
            .collect();
        if neighbours.is_empty() { None } else { Some(neighbours[rng.random_range(0..neighbours.len())]) }
    }

    /// Type a whole planned line, producing the same record a real session would.
    pub fn type_line(&mut self, line: &PlannedLine, prev_char: Option<char>, start_ms: i64, session_sec: &mut f32, rng: &mut StdRng) -> TypedLine {
        let mut now = start_ms as f64;
        let mut prev = prev_char;
        let mut chars = Vec::with_capacity(line.text.len());
        for (pos, c) in line.text.chars().enumerate() {
            let Some(key) = self.press(prev, c, now as i64, *session_sec, rng) else {
                prev = Some(c);
                continue;
            };
            // A wrong press costs a correction (find the error, press the right key).
            let correction = if key.error { key.latency_ms * 1.6 } else { 0.0 };
            chars.push(CharRecord {
                pos: pos as u16,
                latency_ms: (prev.is_some()).then_some(key.latency_ms),
                error: key.error,
                wrong_key: key.wrong_key,
                wall_ms: now as i64,
                session_sec: *session_sec,
            });
            let spent = key.latency_ms + correction;
            now += spent as f64;
            *session_sec += spent / 1000.0;
            prev = Some(c);
        }
        TypedLine {
            text: line.text.clone(),
            kind: line.kind,
            lang: line.lang,
            layout: self.layout.id,
            geometry: self.layout.geometry,
            prev_char,
            chars,
        }
    }

    /// Ground-truth real speed (WPM) on `text` samples at time `now_ms`, rested.
    pub fn true_wpm(&self, samples: &[String], now_ms: i64) -> f64 {
        let mut total = 0.0f64;
        let mut n = 0usize;
        for s in samples {
            let mut prev = Some(' ');
            for c in s.chars() {
                if let Some(ms) = self.expected_ms(prev, c, now_ms, 0.0) {
                    total += ms as f64;
                    n += 1;
                }
                prev = Some(c);
            }
        }
        if n == 0 { 0.0 } else { 12_000.0 / (total / n as f64) }
    }
}

/// Movement cost multiplier between two strokes.
fn transition(layout: &Layout, prev: Option<Stroke>, cur: Stroke, n_strokes: usize) -> f32 {
    let g = layout.key(cur.key);
    let mut m = if g.finger == Finger::RightThumb || g.finger == Finger::LeftThumb { 0.8 } else { 1.0 };
    if let Some(p) = prev {
        let pg = layout.key(p.key);
        if p.key == cur.key {
            m *= 0.95;
        } else if pg.finger == g.finger {
            m *= 1.55 + 0.12 * pg.row.abs_diff(g.row) as f32;
        } else if pg.finger.hand() == g.finger.hand() {
            m *= 1.08 + 0.05 * pg.row.abs_diff(g.row) as f32;
        } else {
            m *= 0.86;
        }
    }
    if g.row == 0 {
        m *= 1.15;
    }
    if n_strokes > 1 {
        m *= 1.85;
    }
    if cur.layer == Layer::ShiftOption {
        m *= 1.2;
    }
    m
}

/// Draw a standard normal (helper for families).
pub(crate) fn normal(rng: &mut StdRng, mean: f32, sd: f32) -> f32 {
    Normal::new(mean, sd).expect("valid sd").sample(rng)
}
