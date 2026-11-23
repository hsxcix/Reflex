//! Measured real speed: every few lines a held-out reference sentence is typed as a probe.
//! Its observed speed (errors included, same cost definition as the model) is adjusted for
//! how hard that sentence is, using the *frozen pretrained* model with no personal history:
//! a generic typist's difficulty, independent of what Reflex has learnt about you. So the
//! result is an observation, not the personal model's belief.

use super::gains::{ReferenceSet, reference_half, reference_texts};
use super::lines::LINE_CHARS;
use crate::content::{ContentKind, Lang, Pool};
use crate::history::PracticeHistory;
use crate::layout::{Geometry, Layout, LayoutId};
use crate::model::TypistModel;
use crate::model::encode::{EncodeContext, EncodedLine, encode_text};
use crate::model::typist::ERROR_COST_FACTOR;
use crate::record::TypedLine;
use crate::session::PlannedLine;
use burn::tensor::Device;
use rand::Rng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// One probe line every this many lines (~10 % of typing).
pub const PROBE_EVERY_LINES: usize = 8;
/// A probe needs at least this many timed presses to count.
const MIN_TIMED: usize = 15;
/// Measured speed shown = mean of the last this many probes on the layout.
const SHOWN_PROBES: usize = 5;

/// A probe sentence for `pool` that can be typed on `layout`.
pub fn probe_line(pool: &Pool, layout: &Layout, rng: &mut StdRng) -> PlannedLine {
    let lang = pool.pick_language(rng);
    let fits = |s: &&String| s.chars().count() <= LINE_CHARS + 20 && layout.can_type(s);
    let candidates: Vec<&String> = reference_half(lang, ReferenceSet::Probe).filter(fits).collect();
    match candidates.get(rng.random_range(0..candidates.len().max(1))) {
        Some(s) => PlannedLine::new(s, ContentKind::Probe, lang),
        // No usable sentence (tiny custom corpus): fall back to the other language or a word.
        None => {
            let other = Lang::ALL.into_iter().find(|l| *l != lang).unwrap_or(lang);
            reference_half(other, ReferenceSet::Probe)
                .find(fits)
                .map_or_else(|| PlannedLine::new("reflex", ContentKind::Probe, lang), |s| PlannedLine::new(s, ContentKind::Probe, other))
        }
    }
}

/// Observed cost per character (ms, errors included) of a typed line; `None` if too few
/// presses were timed (pauses, very short line).
pub fn observed_cost_ms(line: &TypedLine) -> Option<f64> {
    let timed: Vec<f64> = line.chars.iter().filter(|c| !c.error).filter_map(|c| c.latency_ms).map(f64::from).collect();
    if timed.len() < MIN_TIMED || line.chars.is_empty() {
        return None;
    }
    let mean = timed.iter().sum::<f64>() / timed.len() as f64;
    let error_rate = line.chars.iter().filter(|c| c.error).count() as f64 / line.chars.len() as f64;
    Some(mean * (1.0 + ERROR_COST_FACTOR as f64 * error_rate))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeResult {
    pub ended_ms: i64,
    pub layout: LayoutId,
    pub geometry: Geometry,
    /// Observed speed adjusted to standard-text difficulty (WPM).
    pub wpm: f64,
}

/// Scores probe lines with a frozen generic model and keeps the results.
pub struct ProbeScorer {
    baseline: TypistModel,
    device: Device,
    /// Generic cost per char on the standard text sample, per layout.
    reference: HashMap<(LayoutId, Geometry), f64>,
    results: Vec<ProbeResult>,
}

impl ProbeScorer {
    pub fn new(baseline: TypistModel, device: Device) -> ProbeScorer {
        ProbeScorer { baseline, device, reference: HashMap::new(), results: Vec::new() }
    }

    /// Generic (no personal history) predicted cost per character of `texts`.
    fn generic_cost(&self, texts: &[String], layout: &Layout) -> f64 {
        let history = PracticeHistory::default();
        let ctx = EncodeContext { layout, history: &history, now_ms: 0, session_sec: 0.0 };
        let lines: Vec<EncodedLine> = texts.iter().map(|t| EncodedLine { tokens: encode_text(t, Some(' '), &ctx), labels: None }).collect();
        let (mut sum, mut n) = (0.0f64, 0usize);
        for chunk in lines.chunks(48) {
            for p in self.baseline.predict(chunk, &self.device).iter().flatten() {
                sum += p.cost_ms as f64;
                n += 1;
            }
        }
        if n == 0 { f64::NAN } else { sum / n as f64 }
    }

    fn reference_cost(&mut self, layout: &Layout) -> f64 {
        let key = (layout.id, layout.geometry);
        if let Some(c) = self.reference.get(&key) {
            return *c;
        }
        let texts = reference_texts(&Pool::default(), layout, ReferenceSet::Probe, 7);
        let c = self.generic_cost(&texts, layout);
        self.reference.insert(key, c);
        c
    }

    /// Score a typed probe line; stores and returns the result.
    pub fn score(&mut self, line: &TypedLine, layout: &Layout) -> Option<ProbeResult> {
        if line.kind != ContentKind::Probe {
            return None;
        }
        let observed = observed_cost_ms(line)?;
        let difficulty = self.generic_cost(std::slice::from_ref(&line.text), layout);
        let reference = self.reference_cost(layout);
        if !(difficulty > 0.0 && reference > 0.0) {
            return None;
        }
        // Harder sentence than usual → the same observed speed counts as faster.
        let adjusted = observed * reference / difficulty;
        let result = ProbeResult {
            ended_ms: line.chars.last().map_or(0, |c| c.wall_ms),
            layout: line.layout,
            geometry: line.geometry,
            wpm: 12_000.0 / adjusted,
        };
        self.results.push(result);
        Some(result)
    }

    pub fn results(&self) -> &[ProbeResult] {
        &self.results
    }

    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Measured real speed on `layout`: mean of the last few probes typed on it.
    pub fn measured(&self, layout: LayoutId) -> Option<f64> {
        let recent: Vec<f64> = self.results.iter().rev().filter(|r| r.layout == layout).take(SHOWN_PROBES).map(|r| r.wpm).collect();
        (!recent.is_empty()).then(|| recent.iter().sum::<f64>() / recent.len() as f64)
    }

    /// Latest single probe on `layout`.
    pub fn latest(&self, layout: LayoutId) -> Option<f64> {
        self.results.iter().rev().find(|r| r.layout == layout).map(|r| r.wpm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CharRecord;
    use rand::SeedableRng;

    fn typed(text: &str, ms: f32, errors: usize) -> TypedLine {
        let chars = text
            .chars()
            .enumerate()
            .map(|(i, _)| CharRecord { pos: i as u16, latency_ms: (i > 0).then_some(ms), error: i < errors, wrong_key: None, wall_ms: i as i64 * 200, session_sec: 0.0 })
            .collect();
        TypedLine { text: text.into(), kind: ContentKind::Probe, lang: Lang::En, layout: LayoutId::DeIso, geometry: Geometry::Iso, prev_char: Some(' '), chars }
    }

    #[test]
    fn observed_cost_counts_errors() {
        let text = "the quick brown fox jumps over the dog ";
        let clean = observed_cost_ms(&typed(text, 200.0, 0)).unwrap();
        assert!((clean - 200.0).abs() < 1e-9);
        let sloppy = observed_cost_ms(&typed(text, 200.0, 4)).unwrap();
        assert!(sloppy > clean * 1.15);
        assert!(observed_cost_ms(&typed("short ", 200.0, 0)).is_none());
    }

    #[test]
    fn probes_come_from_the_probe_half() {
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let mut rng = StdRng::seed_from_u64(1);
        let model_half: std::collections::HashSet<&String> = Lang::ALL.iter().flat_map(|l| reference_half(*l, ReferenceSet::Model)).collect();
        for _ in 0..50 {
            let line = probe_line(&Pool::default(), &layout, &mut rng);
            assert_eq!(line.kind, ContentKind::Probe);
            assert!(layout.can_type(&line.text));
            assert!(!model_half.contains(&line.text.trim_end().to_string()));
        }
    }

    #[test]
    fn faster_typing_scores_higher_and_difficulty_is_adjusted() {
        let device = crate::model::cpu();
        let mut scorer = ProbeScorer::new(TypistModel::new(&device), device);
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let text = "the quick brown fox jumps over the lazy dog ";
        let slow = scorer.score(&typed(text, 250.0, 0), &layout).unwrap().wpm;
        let fast = scorer.score(&typed(text, 180.0, 0), &layout).unwrap().wpm;
        assert!(fast > slow * 1.3);
        assert_eq!(scorer.count(), 2);
        assert!((scorer.measured(LayoutId::DeIso).unwrap() - (slow + fast) / 2.0).abs() < 1e-9);
        assert!(scorer.measured(LayoutId::UsAnsi).is_none());
    }
}
