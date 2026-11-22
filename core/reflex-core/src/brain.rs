//! The learning loop for one person: observe typed lines, fine-tune the typist model
//! (with replay, validation and rollback), recompute what practice is worth, and let the
//! bandit pick the planner settings for the next batch. Used by the app's background
//! worker and by the benchmarks alike.

use crate::content::Pool;
use crate::history::PracticeHistory;
use crate::layout::{Layout, LayoutId};
use crate::model::encode::{EncodedLine, encode_typed_line, record_line};
use crate::model::train::{TrainParams, evaluate, train};
use crate::model::TypistModel;
use crate::record::TypedLine;
use crate::sim::Family;
use crate::sim::data::simulate_history;
use crate::trainer::bandit::{Bandit, Trial, context};
use crate::trainer::gains::{GainTable, compute_gains, projected_wpm};
use crate::trainer::planner::{ARMS, Planner};
use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::{Bytes, Device};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;
use std::sync::Arc;

/// Keep at most this many encoded lines in memory for fine-tuning.
const MAX_TRAIN_LINES: usize = 6000;
/// Every n-th line goes to the validation split (never trained on).
const VALID_EVERY: usize = 8;

pub struct LearnReport {
    pub valid_before: f32,
    pub valid_after: f32,
    pub kept: bool,
}

pub struct Forecast {
    /// (days from now, predicted real WPM, uncertainty ±WPM)
    pub points: Vec<(u32, f64, f64)>,
}

pub struct Brain {
    device: Device,
    model: TypistModel,
    history: PracticeHistory,
    train_lines: Vec<EncodedLine>,
    valid_lines: Vec<EncodedLine>,
    replay: Vec<EncodedLine>,
    bandit: Bandit,
    gains: Arc<GainTable>,
    arm: usize,
    seen_lines: usize,
    rng: StdRng,
    layout_cache: Option<Layout>,
    pub fine_tune_steps: usize,
}

impl Brain {
    /// `model` is a trained (or fresh) typist model; it is moved to an autodiff CPU device.
    pub fn new(model: TypistModel, seed: u64) -> Brain {
        let device = crate::model::cpu();
        let model = model.fork(&device.clone().autodiff()).train();
        let rng = StdRng::seed_from_u64(seed);
        let replay = replay_set(seed);
        let bandit = Bandit::new(&device);
        Brain {
            device,
            model,
            history: PracticeHistory::default(),
            train_lines: Vec::new(),
            valid_lines: Vec::new(),
            replay,
            bandit,
            gains: Arc::new(GainTable::default()),
            arm: 1,
            seen_lines: 0,
            rng,
            layout_cache: None,
            fine_tune_steps: 24,
        }
    }

    /// Load weights from the first existing path, or start untrained. Files are read fully
    /// into memory, so they can be replaced while the model is in use.
    pub fn load_model(paths: &[&Path]) -> (TypistModel, bool) {
        let device = crate::model::cpu();
        for path in paths {
            let Ok(bytes) = std::fs::read(path) else { continue };
            let Ok(record) = ModuleRecord::from_bytes(Bytes::from_bytes_vec(bytes)) else { continue };
            if let Ok(model) = TypistModel::new(&device).try_load_record(record) {
                return (model, true);
            }
        }
        (TypistModel::new(&device), false)
    }

    /// Save atomically (write a temp file, then rename over the old one).
    pub fn save_model(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let bytes = self.model.valid().into_record().into_bytes().map_err(|e| std::io::Error::other(e.to_string()))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes[..])?;
        std::fs::rename(tmp, path)
    }

    /// An inference copy of the current model.
    pub fn model(&self) -> TypistModel {
        self.model.valid()
    }

    pub fn history(&self) -> &PracticeHistory {
        &self.history
    }

    pub fn gains(&self) -> Arc<GainTable> {
        self.gains.clone()
    }

    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn trials(&self) -> &[Trial] {
        self.bandit.trials()
    }

    /// Restore the person's bandit outcomes plus the simulation prior.
    pub fn restore_trials(&mut self, trials: Vec<Trial>, prior: Vec<Trial>) {
        self.bandit.restore(trials, prior, &mut self.rng);
    }

    /// Read a bandit prior (JSON list of trials) if present.
    pub fn load_prior(path: &Path) -> Vec<Trial> {
        std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    fn layout_for(&mut self, id: LayoutId, geometry: crate::layout::Geometry) -> Layout {
        match &self.layout_cache {
            Some(l) if l.id == id && l.geometry == geometry => l.clone(),
            _ => {
                let l = Layout::new(id, geometry);
                self.layout_cache = Some(l.clone());
                l
            }
        }
    }

    /// Add typed lines (chronological) to history and the training/validation data.
    pub fn observe(&mut self, lines: &[TypedLine]) {
        for line in lines {
            let layout = self.layout_for(line.layout, line.geometry);
            let encoded = encode_typed_line(line, &layout, &self.history);
            record_line(&mut self.history, &layout, line);
            self.seen_lines += 1;
            if self.seen_lines.is_multiple_of(VALID_EVERY) {
                self.valid_lines.push(encoded);
            } else {
                self.train_lines.push(encoded);
            }
        }
        let excess = self.train_lines.len().saturating_sub(MAX_TRAIN_LINES);
        self.train_lines.drain(..excess);
        let excess = self.valid_lines.len().saturating_sub(MAX_TRAIN_LINES / VALID_EVERY);
        self.valid_lines.drain(..excess);
    }

    /// Fine-tune on the person's data; keep the update only if held-out predictions improve.
    pub fn learn(&mut self) -> LearnReport {
        let device = self.device.clone().autodiff();
        let valid: Vec<EncodedLine> = self.valid_lines.iter().rev().take(150).cloned().collect();
        let before = evaluate(&self.model, &valid, &device);
        // Recent lines matter most: train on the newest ~1500.
        let start = self.train_lines.len().saturating_sub(1500);
        let params = TrainParams { steps: self.fine_tune_steps, batch_size: 32, lr: 4e-4, replay_share: 0.3 };
        let candidate = train(self.model.clone(), &self.train_lines[start..], &self.replay, &params, &device, &mut self.rng, |_, _| {});
        let after = evaluate(&candidate, &valid, &device);
        // NaN (no validation data yet) counts as an improvement.
        let kept = valid.is_empty() || after.is_nan() || after <= before * 1.002;
        if kept {
            self.model = candidate;
        }
        LearnReport { valid_before: before, valid_after: if kept { after } else { before }, kept }
    }

    /// Recompute the gain table for the current layout and pool.
    pub fn refresh(&mut self, layout: &Layout, pool: &Pool, now_ms: i64) -> Arc<GainTable> {
        let model = self.model.valid();
        let table = compute_gains(&model, &self.history, layout, pool, now_ms, &self.device);
        self.gains = Arc::new(table);
        self.gains.clone()
    }

    /// Close a batch: reward the arm that was played and choose the next one.
    pub fn end_batch(&mut self, real_before: f64, real_after: f64, minutes: f64) -> usize {
        let ctx = context(self.history.total(), real_before, self.gains.uncertainty_ratio);
        if minutes > 0.05 && real_before > 0.0 {
            let reward = ((real_after - real_before) / minutes) as f32;
            self.bandit.record(Trial { context: ctx, arm: self.arm, reward }, &mut self.rng);
        }
        let next_ctx = context(self.history.total(), real_after, self.gains.uncertainty_ratio);
        self.arm = self.bandit.choose(next_ctx, &mut self.rng);
        self.arm
    }

    pub fn planner(&self) -> Planner {
        Planner::new(self.gains.clone(), ARMS[self.arm])
    }

    /// Project real speed forward assuming ~15 minutes of planned practice per day.
    pub fn forecast(&self, layout: &Layout, pool: &Pool, now_ms: i64) -> Forecast {
        let model = self.model.valid();
        let keystrokes_per_day = 15.0 * 60.0 * 5.0;
        let total_gain: f32 = self.gains.gain.values().sum::<f32>().max(1e-9);
        let pairs = self.gains.gain.len().max(1) as f32;
        let mut points = Vec::new();
        for days in [0u32, 7, 14, 28, 56] {
            let budget = keystrokes_per_day * days as f32;
            let gains = self.gains.clone();
            // Practice spreads over pairs: half by predicted value, half evenly (natural text).
            let extra = move |key| {
                let g = gains.gain.get(&key).copied().unwrap_or(0.0);
                budget * (0.5 * g / total_gain + 0.5 / pairs)
            };
            let (wpm, sd) = projected_wpm(&model, &self.history, layout, pool, now_ms + days as i64 * 86_400_000, &extra, &self.device);
            points.push((days, wpm, sd));
        }
        Forecast { points }
    }

    /// Mean held-out loss (for reporting model quality).
    pub fn valid_loss(&self) -> f32 {
        let device = self.device.clone().autodiff();
        let valid: Vec<EncodedLine> = self.valid_lines.iter().rev().take(150).cloned().collect();
        evaluate(&self.model, &valid, &device)
    }
}

/// Simulated lines mixed into fine-tuning so the model doesn't forget general typing.
fn replay_set(seed: u64) -> Vec<EncodedLine> {
    let mut out = Vec::new();
    for (i, family) in Family::TRAIN.iter().enumerate() {
        let lines = simulate_history(*family, seed.wrapping_mul(31).wrapping_add(i as u64 + 900_000), 2);
        let mut history = PracticeHistory::default();
        out.extend(crate::model::encode::encode_history(&lines, &mut history));
    }
    out
}
