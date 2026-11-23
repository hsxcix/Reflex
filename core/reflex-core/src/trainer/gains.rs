//! What practice is worth: the typist model predicts how much faster normal text would
//! become if a movement pair were practised a little more, now and 1/7/30 days ahead (so a
//! skill about to fade is worth refreshing before it is gone). This "gain table" drives the
//! planner; its total is also the basis of the real-speed estimate and the forecast.

use crate::content::{Corpus, Pool, code_lines, real_world_line};
use crate::history::{PracticeHistory, pair_key};
use crate::layout::{KeyId, Layout, Motor};
use crate::model::encode::{CONT_LOG_COUNT, CONT_LOG_DAYS, CONT_NEVER, EncodeContext, EncodedLine, Token, encode_text};
use crate::model::{TokenPrediction, TypistModel};
use burn::tensor::Device;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;

/// Extra repetitions assumed when asking "what if this pair were practised more?".
const PRACTICE_REPS: f32 = 12.0;
/// Days ahead at which practice is valued, with weights. Practice now pays off now *and*
/// keeps a skill from fading: a pair the model expects you to forget soon is worth more
/// than one that stays fresh anyway. Learning and forgetting both come from the model.
pub const HORIZONS: [(f32, f32); 3] = [(1.0, 0.4), (7.0, 0.35), (30.0, 0.25)];
/// Horizon for the "fading soon" list (days).
const FADE_DAYS: f32 = 7.0;

#[derive(Debug, Clone, Default)]
pub struct GainTable {
    /// Expected ms saved per keystroke of normal text, per practised movement pair,
    /// averaged over [`HORIZONS`] (includes avoided forgetting).
    pub gain: HashMap<(Motor, Motor), f32>,
    /// Model uncertainty per pair, weighted by frequency like `gain` (ms per keystroke).
    pub uncertainty: HashMap<(Motor, Motor), f32>,
    /// Predicted real speed on normal text (WPM).
    pub real_wpm: f64,
    /// Mean predicted cost per key (ms), for the heatmap.
    pub key_cost: HashMap<KeyId, f32>,
    /// Pairs with the highest predicted cost (text form, ms), slowest first.
    pub slow_pairs: Vec<(String, f32)>,
    /// Practised pairs the model expects to slow down most within [`FADE_DAYS`] without
    /// practice (text form, ms slower per press), most fading first.
    pub fading_pairs: Vec<(String, f32)>,
    /// Mean ensemble disagreement relative to cost (0 = certain).
    pub uncertainty_ratio: f32,
    /// Share of the total gain on digits and symbols.
    pub symbol_share: f32,
}

/// Which half of the held-out reference sentences to use. The model's real-speed estimate
/// and the measured probe lines use different sentences, so neither can be tuned on the
/// other's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSet {
    /// Fixed sample the typist model's real-speed estimate and gain table are built on.
    Model,
    /// Sentences typed as measurement (probe) lines, and the simulator's ground truth.
    Probe,
}

/// Reference sentences of one half (`Model` = even positions, `Probe` = odd).
pub fn reference_half(lang: crate::content::Lang, set: ReferenceSet) -> impl Iterator<Item = &'static String> {
    let skip = if set == ReferenceSet::Model { 0 } else { 1 };
    Corpus::get().reference_sentences(lang).iter().skip(skip).step_by(2)
}

/// A fixed sample of normal text for the active pool: what "real speed" means for the model.
/// Sentences come from the held-out reference set, so the trainer never practises the
/// exact text it is measured on.
pub fn eval_texts(pool: &Pool, layout: &Layout) -> Vec<String> {
    reference_texts(pool, layout, ReferenceSet::Model, 42)
}

/// Normal-text sample from one half of the reference set (`seed` picks the sample).
pub fn reference_texts(pool: &Pool, layout: &Layout, set: ReferenceSet, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut texts = Vec::new();
    for lang in &pool.languages {
        let sentences: Vec<&String> = reference_half(*lang, set).filter(|s| layout.can_type(s) && s.chars().count() < 90).collect();
        let want = 90 / pool.languages.len().max(1);
        texts.extend(sentences.choose_multiple(&mut rng, want).map(|s| format!("{s} ")));
        for _ in 0..6 {
            let line = real_world_line(*lang, &mut rng);
            if layout.can_type(&line) {
                texts.push(format!("{line} "));
            }
        }
    }
    if pool.code {
        for line in code_lines().choose_multiple(&mut rng, 10) {
            if layout.can_type(line) {
                texts.push(format!("{line} "));
            }
        }
    }
    texts
}

fn encode_base(texts: &[String], layout: &Layout, history: &PracticeHistory, now_ms: i64) -> Vec<EncodedLine> {
    let ctx = EncodeContext { layout, history, now_ms, session_sec: 0.0 };
    texts.iter().map(|t| EncodedLine { tokens: encode_text(t, Some(' '), &ctx), labels: None }).collect()
}

fn days_of(tok: &Token) -> f32 {
    (tok.cont[CONT_LOG_DAYS] * 6.0).exp_m1()
}

fn count_of(tok: &Token) -> f32 {
    (tok.cont[CONT_LOG_COUNT] * 8.0).exp_m1()
}

fn set_days(tok: &mut Token, days: f32) {
    tok.cont[CONT_LOG_DAYS] = days.max(0.0).ln_1p() / 6.0;
}

/// The pair `days` from now if it is left alone: older, nothing else changes.
fn left_alone(tok: &Token, days: f32) -> Token {
    let mut t = *tok;
    set_days(&mut t, days_of(tok) + days);
    t
}

/// The pair `days` from now if it gets `reps` more repetitions today.
fn practised(tok: &Token, reps: f32, days: f32) -> Token {
    let mut t = *tok;
    t.cont[CONT_LOG_COUNT] = (count_of(tok) + reps).ln_1p() / 8.0;
    set_days(&mut t, days);
    t.cont[CONT_NEVER] = 0.0;
    t
}

fn map_tokens(lines: &[EncodedLine], f: impl Fn(&Token) -> Token) -> Vec<EncodedLine> {
    lines.iter().map(|l| EncodedLine { tokens: l.tokens.iter().map(&f).collect(), labels: None }).collect()
}

type Predictions = Vec<Vec<TokenPrediction>>;

fn predict_all(model: &TypistModel, lines: &[EncodedLine], device: &Device) -> Vec<Vec<TokenPrediction>> {
    lines.chunks(48).flat_map(|chunk| model.predict(chunk, device)).collect()
}

/// Predicted real speed (WPM) on normal text with `history`, rested.
pub fn predicted_real_wpm(model: &TypistModel, history: &PracticeHistory, layout: &Layout, pool: &Pool, now_ms: i64, device: &Device) -> f64 {
    let texts = eval_texts(pool, layout);
    let preds = predict_all(model, &encode_base(&texts, layout, history, now_ms), device);
    let (cost, n) = preds.iter().flatten().fold((0.0f64, 0usize), |(c, n), p| (c + p.cost_ms as f64, n + 1));
    if n == 0 { 0.0 } else { 12_000.0 / (cost / n as f64) }
}

pub fn compute_gains(model: &TypistModel, history: &PracticeHistory, layout: &Layout, pool: &Pool, now_ms: i64, device: &Device) -> GainTable {
    let texts = eval_texts(pool, layout);
    let base = encode_base(&texts, layout, history, now_ms);
    let p0 = predict_all(model, &base, device);
    // Per horizon: cost if left alone vs. cost if practised now.
    let horizons: Vec<(f32, Predictions, Predictions)> = HORIZONS
        .iter()
        .map(|&(days, w)| {
            let alone = predict_all(model, &map_tokens(&base, |t| left_alone(t, days)), device);
            let trained = predict_all(model, &map_tokens(&base, |t| practised(t, PRACTICE_REPS, days)), device);
            (w, alone, trained)
        })
        .collect();
    let fade_idx = HORIZONS.iter().position(|(d, _)| *d == FADE_DAYS);
    let mut table = GainTable::default();
    let mut total_cost = 0.0f64;
    let mut total_sd = 0.0f64;
    let mut n = 0usize;
    let mut key_sum: HashMap<KeyId, (f32, f32)> = HashMap::new();
    let mut pair_cost: HashMap<(Motor, Motor), (f32, f32, String)> = HashMap::new();
    let mut pair_fade: HashMap<(Motor, Motor), (f32, f32, String)> = HashMap::new();
    let (mut symbol_gain, mut all_gain) = (0.0f32, 0.0f32);
    for (li, text) in texts.iter().enumerate() {
        let chars: Vec<char> = text.chars().collect();
        for (i, a) in p0[li].iter().enumerate() {
            let prev = if i == 0 { Some(' ') } else { Some(chars[i - 1]) };
            let c = chars[i];
            total_cost += a.cost_ms as f64;
            total_sd += a.cost_sd as f64;
            n += 1;
            let Some(key) = pair_key(layout, prev, c) else { continue };
            let gain: f32 = horizons.iter().map(|(w, alone, trained)| w * (alone[li][i].cost_ms - trained[li][i].cost_ms).max(0.0)).sum();
            *table.gain.entry(key).or_default() += gain;
            all_gain += gain;
            if !c.is_alphabetic() && c != ' ' {
                symbol_gain += gain;
            }
            *table.uncertainty.entry(key).or_default() += a.cost_sd;
            if let Some(stroke) = layout.main_stroke(c) {
                let e = key_sum.entry(stroke.key).or_default();
                e.0 += a.cost_ms;
                e.1 += 1.0;
            }
            if c != ' ' && prev != Some(' ') {
                let label = || format!("{}{}", prev.unwrap_or(' '), c);
                let e = pair_cost.entry(key).or_insert_with(|| (0.0, 0.0, label()));
                e.0 += a.cost_ms;
                e.1 += 1.0;
                let seen = base[li].tokens[i].cont[CONT_NEVER] == 0.0;
                if let (Some(f), true) = (fade_idx, seen) {
                    let e = pair_fade.entry(key).or_insert_with(|| (0.0, 0.0, label()));
                    e.0 += horizons[f].1[li][i].cost_ms - a.cost_ms;
                    e.1 += 1.0;
                }
            }
        }
    }
    if n == 0 {
        return table;
    }
    for g in table.gain.values_mut().chain(table.uncertainty.values_mut()) {
        *g /= n as f32;
    }
    table.real_wpm = 12_000.0 / (total_cost / n as f64);
    table.uncertainty_ratio = (total_sd / total_cost) as f32;
    table.symbol_share = if all_gain > 0.0 { symbol_gain / all_gain } else { 0.0 };
    table.key_cost = key_sum.into_iter().map(|(k, (s, c))| (k, s / c)).collect();
    table.slow_pairs = top_pairs(pair_cost, 2.0, 12);
    table.fading_pairs = top_pairs(pair_fade, 1.0, 12).into_iter().filter(|(_, ms)| *ms > 1.0).collect();
    table
}

/// Mean per pair (at least `min_n` occurrences), highest first.
fn top_pairs(sums: HashMap<(Motor, Motor), (f32, f32, String)>, min_n: f32, limit: usize) -> Vec<(String, f32)> {
    let mut v: Vec<(String, f32)> = sums.into_values().filter(|(_, c, _)| *c >= min_n).map(|(s, c, t)| (t, s / c)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(limit);
    v
}

/// Predicted real speed if every pair got `extra(pair)` more repetitions, `days` from now.
pub fn projected_wpm(
    model: &TypistModel,
    history: &PracticeHistory,
    layout: &Layout,
    pool: &Pool,
    now_ms: i64,
    extra: &dyn Fn((Motor, Motor)) -> f32,
    device: &Device,
) -> (f64, f64) {
    let texts = eval_texts(pool, layout);
    let ctx = EncodeContext { layout, history, now_ms, session_sec: 0.0 };
    let lines: Vec<EncodedLine> = texts
        .iter()
        .map(|t| {
            let mut tokens = encode_text(t, Some(' '), &ctx);
            let chars: Vec<char> = t.chars().collect();
            for (i, tok) in tokens.iter_mut().enumerate() {
                let prev = if i == 0 { Some(' ') } else { Some(chars[i - 1]) };
                if let Some(key) = pair_key(layout, prev, chars[i]) {
                    let add = extra(key);
                    if add > 0.0 {
                        *tok = practised(tok, add, 0.0);
                    }
                }
            }
            EncodedLine { tokens, labels: None }
        })
        .collect();
    let preds = predict_all(model, &lines, device);
    let (mut cost, mut sd, mut n) = (0.0f64, 0.0f64, 0usize);
    for p in preds.iter().flatten() {
        cost += p.cost_ms as f64;
        sd += p.cost_sd as f64;
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let wpm = 12_000.0 / (cost / n as f64);
    // Relative cost spread translates to the same relative WPM spread.
    (wpm, wpm * sd / cost)
}
