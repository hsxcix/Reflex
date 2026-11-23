//! What practice is worth: the typist model predicts how much faster normal text would
//! become if a movement pair were practised a little more. This "gain table" drives the
//! planner; its total is also the basis of the real-speed estimate and the forecast.

use crate::content::{Corpus, Pool, code_lines, real_world_line};
use crate::history::{PracticeHistory, pair_key};
use crate::layout::{KeyId, Layout, Motor};
use crate::model::encode::{CONT_LOG_COUNT, CONT_LOG_DAYS, CONT_NEVER, EncodeContext, EncodedLine, encode_text};
use crate::model::{TokenPrediction, TypistModel};
use burn::tensor::Device;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;

/// Extra repetitions assumed when asking "what if this pair were practised more?".
const PRACTICE_REPS: f32 = 12.0;

#[derive(Debug, Clone, Default)]
pub struct GainTable {
    /// Expected ms saved per keystroke of normal text, per practised movement pair.
    pub gain: HashMap<(Motor, Motor), f32>,
    /// Model uncertainty per pair, weighted by frequency like `gain` (ms per keystroke).
    pub uncertainty: HashMap<(Motor, Motor), f32>,
    /// Predicted real speed on normal text (WPM).
    pub real_wpm: f64,
    /// Mean predicted cost per key (ms), for the heatmap.
    pub key_cost: HashMap<KeyId, f32>,
    /// Pairs with the highest predicted cost (text form, ms), slowest first.
    pub slow_pairs: Vec<(String, f32)>,
    /// Mean ensemble disagreement relative to cost (0 = certain).
    pub uncertainty_ratio: f32,
    /// Share of the total gain on digits and symbols.
    pub symbol_share: f32,
}

/// A fixed sample of normal text for the active pool: what "real speed" means.
/// Sentences come from the held-out reference set, so the trainer never practises the
/// exact text it is measured on.
pub fn eval_texts(pool: &Pool, layout: &Layout) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut texts = Vec::new();
    for lang in &pool.languages {
        let sentences = Corpus::get().reference_sentences(*lang);
        let mut picked = 0;
        while picked < 90 / pool.languages.len() {
            if let Some(s) = sentences.choose(&mut rng)
                && layout.can_type(s) && s.chars().count() < 90 {
                    texts.push(format!("{s} "));
                    picked += 1;
                }
        }
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

/// Encode texts with the current history, and a copy with every pair's practice bumped.
fn encode_pair(texts: &[String], layout: &Layout, history: &PracticeHistory, now_ms: i64) -> (Vec<EncodedLine>, Vec<EncodedLine>) {
    let ctx = EncodeContext { layout, history, now_ms, session_sec: 0.0 };
    let base: Vec<EncodedLine> = texts
        .iter()
        .map(|t| EncodedLine { tokens: encode_text(t, Some(' '), &ctx), labels: None })
        .collect();
    let bumped = base
        .iter()
        .map(|line| {
            let mut line = line.clone();
            for tok in &mut line.tokens {
                let count = (tok.cont[CONT_LOG_COUNT] * 8.0).exp_m1();
                tok.cont[CONT_LOG_COUNT] = (count + PRACTICE_REPS).ln_1p() / 8.0;
                tok.cont[CONT_LOG_DAYS] = 0.0;
                tok.cont[CONT_NEVER] = 0.0;
            }
            line
        })
        .collect();
    (base, bumped)
}

fn predict_all(model: &TypistModel, lines: &[EncodedLine], device: &Device) -> Vec<Vec<TokenPrediction>> {
    lines.chunks(48).flat_map(|chunk| model.predict(chunk, device)).collect()
}

pub fn compute_gains(model: &TypistModel, history: &PracticeHistory, layout: &Layout, pool: &Pool, now_ms: i64, device: &Device) -> GainTable {
    let texts = eval_texts(pool, layout);
    let (base, bumped) = encode_pair(&texts, layout, history, now_ms);
    let p0 = predict_all(model, &base, device);
    let p1 = predict_all(model, &bumped, device);
    let mut table = GainTable::default();
    let mut total_cost = 0.0f64;
    let mut total_sd = 0.0f64;
    let mut n = 0usize;
    let mut key_sum: HashMap<KeyId, (f32, f32)> = HashMap::new();
    let mut pair_cost: HashMap<(Motor, Motor), (f32, f32, String)> = HashMap::new();
    let (mut symbol_gain, mut all_gain) = (0.0f32, 0.0f32);
    for (text, (line0, line1)) in texts.iter().zip(p0.iter().zip(p1.iter())) {
        let chars: Vec<char> = text.chars().collect();
        for (i, (a, b)) in line0.iter().zip(line1.iter()).enumerate() {
            let prev = if i == 0 { Some(' ') } else { Some(chars[i - 1]) };
            let c = chars[i];
            total_cost += a.cost_ms as f64;
            total_sd += a.cost_sd as f64;
            n += 1;
            let Some(key) = pair_key(layout, prev, c) else { continue };
            let gain = (a.cost_ms - b.cost_ms).max(0.0);
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
                let e = pair_cost.entry(key).or_insert((0.0, 0.0, format!("{}{}", prev.unwrap_or(' '), c)));
                e.0 += a.cost_ms;
                e.1 += 1.0;
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
    let mut slow: Vec<(String, f32)> = pair_cost.into_values().filter(|(_, c, _)| *c >= 2.0).map(|(s, c, t)| (t, s / c)).collect();
    slow.sort_by(|a, b| b.1.total_cmp(&a.1));
    slow.truncate(12);
    table.slow_pairs = slow;
    table
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
                        let count = (tok.cont[CONT_LOG_COUNT] * 8.0).exp_m1();
                        tok.cont[CONT_LOG_COUNT] = (count + add).ln_1p() / 8.0;
                        tok.cont[CONT_LOG_DAYS] = 0.0;
                        tok.cont[CONT_NEVER] = 0.0;
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
