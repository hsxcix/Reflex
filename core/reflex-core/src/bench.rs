//! Benchmarks on held-out simulated typists: does the ML trainer make people faster with
//! fewer keystrokes than random text and a hand-written heuristic, and does the typist
//! model predict better than simple averages?

use crate::brain::Brain;
use crate::content::Pool;
use crate::history::{PracticeHistory, pair_key};
use crate::layout::{Geometry, Layout, LayoutId, Motor};
use crate::model::TypistModel;
use crate::model::encode::{encode_typed_line, record_line};
use crate::record::TypedLine;
use crate::sim::data::{SIM_START_MS, simulate_history};
use crate::sim::{Family, SimTypist};
use crate::trainer::gains::eval_texts;
use crate::trainer::heuristic::WeakestKeys;
use crate::trainer::lines::RecentWords;
use crate::trainer::random::RandomSource;
use crate::trainer::{LineSource, PlanContext};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainerKind {
    Random,
    Heuristic,
    Ml,
}

impl TrainerKind {
    pub fn name(self) -> &'static str {
        match self {
            TrainerKind::Random => "random text",
            TrainerKind::Heuristic => "weakest keys",
            TrainerKind::Ml => "reflex ML",
        }
    }
}

pub struct BenchConfig {
    pub typists_per_family: usize,
    pub batches: usize,
    pub lines_per_batch: usize,
    /// A new day (session) starts after this many batches.
    pub batches_per_day: usize,
}

/// Ground-truth real speed after each batch, with cumulative keystrokes.
pub struct Curve {
    pub start_wpm: f64,
    pub points: Vec<(usize, f64)>,
}

/// Run one trainer on one simulated person from `family` (same seed → same person).
pub fn run_trainer(kind: TrainerKind, family: Family, seed: u64, cfg: &BenchConfig, model: &TypistModel) -> Curve {
    let mut rng = StdRng::seed_from_u64(seed);
    let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
    let params = family.sample(&layout, &mut rng);
    let mut typist = SimTypist::new(params, layout.clone(), SIM_START_MS);
    let pool = Pool::default();
    let truth_texts = eval_texts(&pool, &layout);
    let mut now = SIM_START_MS;
    let start_wpm = typist.true_wpm(&truth_texts, now);
    let mut recent = RecentWords::default();
    let mut random = RandomSource::default();
    let mut heuristic = WeakestKeys::default();
    let mut brain = (kind == TrainerKind::Ml).then(|| Brain::new(model.clone(), seed));
    if let Some(b) = brain.as_mut() {
        b.refresh(&layout, &pool, now);
    }
    let mut keystrokes = 0usize;
    let mut points = Vec::new();
    let mut session_sec = 0.0f32;
    let mut real_before = brain.as_ref().map_or(0.0, |b| b.gains().real_wpm);
    let mut rng_plan = StdRng::seed_from_u64(seed ^ 0xABCD);
    for batch in 0..cfg.batches {
        if batch > 0 && batch % cfg.batches_per_day == 0 {
            now += 86_400_000;
            session_sec = 0.0;
        }
        let mut planner = brain.as_ref().map(|b| b.planner());
        let mut typed: Vec<TypedLine> = Vec::new();
        let batch_start_sec = session_sec;
        for _ in 0..cfg.lines_per_batch {
            let ctx = PlanContext { pool: &pool, layout: &layout, recent: &recent };
            let line = match (kind, planner.as_mut()) {
                (TrainerKind::Ml, Some(p)) => p.next_line(&ctx, &mut rng_plan),
                (TrainerKind::Heuristic, _) => heuristic.next_line(&ctx, &mut rng_plan),
                _ => random.next_line(&ctx, &mut rng_plan),
            };
            recent.note_line(&line.text);
            let t = typist.type_line(&line, Some(' '), now, &mut session_sec, &mut rng);
            now = t.chars.last().map_or(now, |c| c.wall_ms + 200);
            keystrokes += t.chars.len();
            typed.push(t);
        }
        let minutes = (session_sec - batch_start_sec) as f64 / 60.0;
        match kind {
            TrainerKind::Ml => {
                let b = brain.as_mut().expect("ml brain");
                b.observe(&typed);
                b.learn();
                let real_after = b.refresh(&layout, &pool, now).real_wpm;
                b.end_batch(real_before, real_after, minutes);
                real_before = real_after;
            }
            TrainerKind::Heuristic => heuristic.observe(&typed),
            TrainerKind::Random => {}
        }
        points.push((keystrokes, typist.true_wpm(&truth_texts, now)));
    }
    Curve { start_wpm, points }
}

/// Keystrokes `curve` needed to first reach `level` (None if never).
pub fn keystrokes_to(curve: &Curve, level: f64) -> Option<usize> {
    curve.points.iter().find(|(_, w)| *w >= level).map(|(k, _)| *k)
}

/// Prediction quality on held-out sessions: RMSE of log latency and Brier score of errors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Quality {
    pub rmse_log_latency: f64,
    pub brier: f64,
}

/// Compare per-pair averages, the pretrained model and the fine-tuned model on the last
/// sessions of simulated people from held-out families.
pub fn prediction_quality(model: &TypistModel, typists: usize) -> (Quality, Quality, Quality) {
    let (mut base, mut pre, mut fine) = (Quality::default(), Quality::default(), Quality::default());
    let device = crate::model::cpu();
    for i in 0..typists {
        let family = Family::HELD_OUT[i % Family::HELD_OUT.len()];
        let lines = simulate_history(family, 500_000 + i as u64, 6);
        let split = lines.len() * 2 / 3;
        let (train, test) = lines.split_at(split);
        let averages = PairAverages::fit(train);
        let mut brain = Brain::new(model.clone(), i as u64);
        brain.fine_tune_steps = 40;
        for chunk in train.chunks(25) {
            brain.observe(chunk);
            brain.learn();
        }
        let mut history = PracticeHistory::default();
        for l in train {
            record_line(&mut history, &Layout::new(l.layout, l.geometry), l);
        }
        let pre_model = model.clone();
        let tuned = brain.model();
        let (mut b, mut p, mut f) = (Acc::default(), Acc::default(), Acc::default());
        for line in test {
            let layout = Layout::new(line.layout, line.geometry);
            let encoded = encode_typed_line(line, &layout, &history);
            let pp = &pre_model.predict(std::slice::from_ref(&encoded), &device)[0];
            let fp = &tuned.predict(std::slice::from_ref(&encoded), &device)[0];
            let chars: Vec<char> = line.text.chars().collect();
            for rec in &line.chars {
                let pos = rec.pos as usize;
                if pos >= pp.len() {
                    continue;
                }
                let prev = if pos == 0 { line.prev_char } else { Some(chars[pos - 1]) };
                let key = pair_key(&layout, prev, chars[pos]);
                let (bl, be) = averages.predict(key);
                let y = rec.latency_ms.filter(|_| !rec.error).map(|v| (v as f64).ln());
                b.add(y, bl, rec.error, be);
                p.add(y, pp[pos].log_latency as f64, rec.error, pp[pos].error_p as f64);
                f.add(y, fp[pos].log_latency as f64, rec.error, fp[pos].error_p as f64);
            }
            record_line(&mut history, &layout, line);
        }
        base = base.plus(b.quality(), typists);
        pre = pre.plus(p.quality(), typists);
        fine = fine.plus(f.quality(), typists);
    }
    (base, pre, fine)
}

impl Quality {
    fn plus(self, q: Quality, n: usize) -> Quality {
        Quality { rmse_log_latency: self.rmse_log_latency + q.rmse_log_latency / n as f64, brier: self.brier + q.brier / n as f64 }
    }
}

#[derive(Default)]
struct Acc {
    se: f64,
    n_lat: f64,
    brier: f64,
    n: f64,
}

impl Acc {
    fn add(&mut self, y: Option<f64>, pred_log: f64, error: bool, p_err: f64) {
        if let Some(y) = y {
            self.se += (y - pred_log).powi(2);
            self.n_lat += 1.0;
        }
        self.brier += (error as u8 as f64 - p_err).powi(2);
        self.n += 1.0;
    }

    fn quality(&self) -> Quality {
        Quality { rmse_log_latency: (self.se / self.n_lat.max(1.0)).sqrt(), brier: self.brier / self.n.max(1.0) }
    }
}

/// Baseline: average log latency and error rate per movement pair (shrunk to the global mean).
struct PairAverages {
    pairs: HashMap<(Motor, Motor), (f64, f64, f64)>,
    global: (f64, f64),
}

impl PairAverages {
    fn fit(lines: &[TypedLine]) -> PairAverages {
        let mut pairs: HashMap<(Motor, Motor), (f64, f64, f64)> = HashMap::new();
        let (mut gl, mut ge, mut gn, mut gnl) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for line in lines {
            let layout = Layout::new(line.layout, line.geometry);
            let chars: Vec<char> = line.text.chars().collect();
            for rec in &line.chars {
                let pos = rec.pos as usize;
                let prev = if pos == 0 { line.prev_char } else { Some(chars[pos - 1]) };
                let Some(key) = pair_key(&layout, prev, chars[pos]) else { continue };
                let e = pairs.entry(key).or_default();
                if let (Some(l), false) = (rec.latency_ms, rec.error) {
                    e.0 += (l as f64).ln();
                    gl += (l as f64).ln();
                    gnl += 1.0;
                }
                e.1 += rec.error as u8 as f64;
                e.2 += 1.0;
                ge += rec.error as u8 as f64;
                gn += 1.0;
            }
        }
        PairAverages { pairs, global: (gl / gnl.max(1.0), ge / gn.max(1.0)) }
    }

    fn predict(&self, key: Option<(Motor, Motor)>) -> (f64, f64) {
        let k = 5.0;
        match key.and_then(|k| self.pairs.get(&k)) {
            Some((l, e, n)) => ((l + k * self.global.0) / (n + k), (e + k * self.global.1) / (n + k)),
            None => self.global,
        }
    }
}

/// Diagnostic: how well does the model's gain table match the simulator's true gains?
/// Returns (Spearman rank correlation over pairs, share of true top-30 pairs in model top-30).
pub fn gain_accuracy(model: &TypistModel, family: Family, seed: u64, warmup_batches: usize) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
    let params = family.sample(&layout, &mut rng);
    let mut typist = SimTypist::new(params, layout.clone(), SIM_START_MS);
    let pool = Pool::default();
    let mut brain = Brain::new(model.clone(), seed);
    let mut recent = RecentWords::default();
    let mut random = RandomSource::default();
    let mut now = SIM_START_MS;
    let mut session_sec = 0.0f32;
    for _ in 0..warmup_batches {
        let mut typed = Vec::new();
        for _ in 0..8 {
            let ctx = PlanContext { pool: &pool, layout: &layout, recent: &recent };
            let line = random.next_line(&ctx, &mut rng);
            recent.note_line(&line.text);
            let t = typist.type_line(&line, Some(' '), now, &mut session_sec, &mut rng);
            now = t.chars.last().map_or(now, |c| c.wall_ms + 200);
            typed.push(t);
        }
        brain.observe(&typed);
        brain.learn();
    }
    let table = brain.refresh(&layout, &pool, now);
    let texts = eval_texts(&pool, &layout);
    let mut truth: HashMap<(Motor, Motor), f64> = HashMap::new();
    let mut n = 0usize;
    for t in &texts {
        let chars: Vec<char> = t.chars().collect();
        for i in 0..chars.len() {
            let prev = if i == 0 { Some(' ') } else { Some(chars[i - 1]) };
            let (Some(key), Some(a), Some(b)) = (
                pair_key(&layout, prev, chars[i]),
                typist.expected_ms_with_extra(prev, chars[i], now, 0.0),
                typist.expected_ms_with_extra(prev, chars[i], now, 4.0),
            ) else { continue };
            *truth.entry(key).or_default() += (a - b) as f64;
            n += 1;
        }
    }
    let keys: Vec<(Motor, Motor)> = truth.keys().copied().collect();
    let t: Vec<f64> = keys.iter().map(|k| truth[k] / n as f64).collect();
    let m: Vec<f64> = keys.iter().map(|k| *table.gain.get(k).unwrap_or(&0.0) as f64).collect();
    let top = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
        idx.truncate(30);
        idx
    };
    let (tt, mt) = (top(&t), top(&m));
    let overlap = tt.iter().filter(|i| mt.contains(i)).count() as f64 / tt.len().max(1) as f64;
    (spearman(&t, &m), overlap)
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|x, y| v[*x].total_cmp(&v[*y]));
        let mut r = vec![0.0; v.len()];
        for (pos, i) in idx.into_iter().enumerate() {
            r[i] = pos as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let ma = ra.iter().sum::<f64>() / n;
    let mb = rb.iter().sum::<f64>() / n;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}
