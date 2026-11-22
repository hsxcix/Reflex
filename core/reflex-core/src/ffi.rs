//! The C-ABI surface for UIs (Swift today, anything UniFFI supports later).
//! Thin by design: conversion only, all logic lives in the core.

use crate::content::{Lang, Pool, Vocabulary};
use crate::engine::{self, Engine};
use crate::layout::{Geometry, KEY_CODES, Layer, LayoutId, key_id};
use crate::session::KeyInput;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum ReflexError {
    #[error("{0}")]
    Storage(String),
}

impl From<crate::store::StoreError> for ReflexError {
    fn from(e: crate::store::StoreError) -> Self {
        ReflexError::Storage(e.to_string())
    }
}

#[derive(uniffi::Record)]
pub struct KeyEvent {
    /// W3C physical key code by position (e.g. "KeyA"); empty if unknown.
    pub code: String,
    /// Characters produced (after dead-key composition).
    pub chars: String,
    pub modifiers: u32,
    /// Monotonic hardware timestamp in microseconds.
    pub timestamp_us: i64,
    pub is_dead: bool,
}

#[derive(uniffi::Record)]
pub struct BatchSummary {
    pub wpm: f64,
    pub accuracy: f64,
    pub trend: f64,
    pub real_wpm: Option<f64>,
    pub fatigue_hint: bool,
}

#[derive(uniffi::Record)]
pub struct KeyResult {
    pub correct: bool,
    pub advanced: bool,
    pub line_finished: bool,
    pub batch: Option<BatchSummary>,
}

#[derive(uniffi::Record)]
pub struct TypingView {
    pub lines: Vec<String>,
    pub cursor: u32,
    pub error_positions: Vec<u32>,
    pub live_wpm: f64,
    pub live_accuracy: f64,
    pub today_minutes: f64,
    pub layout_name: String,
    /// Model estimate of speed on normal text, once available.
    pub real_wpm: Option<f64>,
}

#[derive(uniffi::Record)]
pub struct PoolSettings {
    /// "de" and/or "en".
    pub languages: Vec<String>,
    /// "common" | "broad" | "full".
    pub vocabulary: String,
    pub code: bool,
}

/// One key cap for drawing the on-screen keyboard.
#[derive(uniffi::Record)]
pub struct KeyCap {
    pub code: String,
    pub row: u8,
    pub x: f32,
    pub width: f32,
    pub label: String,
    pub finger: u8,
}

/// Which key(s) to press for the next character.
#[derive(uniffi::Record)]
pub struct KeyHint {
    pub codes: Vec<String>,
    pub finger: u8,
    pub shift: bool,
    pub option: bool,
}

#[derive(uniffi::Record)]
pub struct DayPoint {
    /// Start of the day (ms since epoch, UTC).
    pub day_ms: i64,
    pub wpm: f64,
    pub accuracy: f64,
    pub real_wpm: Option<f64>,
    pub minutes: f64,
}

#[derive(uniffi::Record)]
pub struct ForecastPoint {
    pub days: u32,
    pub wpm: f64,
    pub plus_minus: f64,
}

#[derive(uniffi::Record)]
pub struct KeyHeat {
    pub code: String,
    /// 0 = fastest key, 1 = slowest key.
    pub value: f64,
    pub ms: f64,
}

#[derive(uniffi::Record)]
pub struct PairStat {
    pub text: String,
    pub ms: f64,
}

#[derive(uniffi::Record)]
pub struct Confusion {
    pub expected: String,
    pub typed: String,
    pub count: u32,
}

#[derive(uniffi::Record)]
pub struct Stats {
    pub days: Vec<DayPoint>,
    pub forecast: Vec<ForecastPoint>,
    pub key_heat: Vec<KeyHeat>,
    pub slow_pairs: Vec<PairStat>,
    pub confusions: Vec<Confusion>,
    pub real_wpm: Option<f64>,
    /// 0…1: how sure the typist model is about you.
    pub confidence: f64,
    pub model_ready: bool,
    pub streak_days: u32,
    pub total_minutes: f64,
    pub today_minutes: f64,
    pub total_keystrokes: i64,
}

#[derive(uniffi::Object)]
pub struct ReflexEngine {
    engine: Mutex<Engine>,
}

impl ReflexEngine {
    fn lock(&self) -> MutexGuard<'_, Engine> {
        self.engine.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[uniffi::export]
impl ReflexEngine {
    /// Open (or create) Reflex's data in `data_dir`.
    /// `resources_dir` holds the shipped model weights (the app bundle's Resources).
    #[uniffi::constructor]
    pub fn open(data_dir: String, resources_dir: String) -> Result<Arc<ReflexEngine>, ReflexError> {
        let engine = Engine::open(Path::new(&data_dir), Path::new(&resources_dir))?;
        Ok(Arc::new(ReflexEngine { engine: Mutex::new(engine) }))
    }

    /// Tell the core which layout ("de-iso", "us-ansi", "eurkey") and hardware ("iso"/"ansi") is active.
    pub fn set_layout(&self, layout: String, geometry: String) {
        let id = LayoutId::parse(&layout).unwrap_or(LayoutId::DeIso);
        let geometry = Geometry::parse(&geometry).unwrap_or(Geometry::Iso);
        // German needs the ISO-only key; on ANSI hardware use the closest full layout.
        let id = if id == LayoutId::DeIso && geometry == Geometry::Ansi { LayoutId::UsAnsi } else { id };
        self.lock().set_layout(id, geometry);
    }

    pub fn key_event(&self, event: KeyEvent) -> Result<KeyResult, ReflexError> {
        let input = KeyInput {
            key: key_id(&event.code),
            chars: event.chars,
            modifiers: event.modifiers,
            timestamp_us: event.timestamp_us,
            wall_ms: engine::now_ms(),
            is_dead: event.is_dead,
        };
        let (outcome, summary) = self.lock().key(input)?;
        Ok(KeyResult {
            correct: outcome.correct,
            advanced: outcome.advanced,
            line_finished: outcome.line_finished,
            batch: summary.map(|s| BatchSummary {
                wpm: s.wpm,
                accuracy: s.accuracy,
                trend: s.trend,
                real_wpm: s.real_wpm,
                fatigue_hint: s.fatigue_hint,
            }),
        })
    }

    pub fn view(&self) -> TypingView {
        let mut engine = self.lock();
        let v = engine.view();
        TypingView {
            lines: v.lines,
            cursor: v.cursor as u32,
            error_positions: v.error_positions.into_iter().map(|p| p as u32).collect(),
            live_wpm: v.live_wpm,
            live_accuracy: v.live_accuracy,
            today_minutes: v.today_minutes,
            layout_name: engine.layout().id.display_name().to_string(),
            real_wpm: v.real_wpm,
        }
    }

    /// The user stepped away (window hidden, stats opened): don't time the gap.
    pub fn pause(&self) {
        self.lock().pause();
    }

    pub fn pool(&self) -> PoolSettings {
        let engine = self.lock();
        let p = engine.pool();
        PoolSettings {
            languages: p.languages.iter().map(|l| l.as_str().to_string()).collect(),
            vocabulary: p.vocabulary.as_str().to_string(),
            code: p.code,
        }
    }

    pub fn set_pool(&self, settings: PoolSettings) -> Result<(), ReflexError> {
        let languages: Vec<Lang> = settings.languages.iter().filter_map(|l| Lang::parse(l)).collect();
        let pool = Pool {
            languages: if languages.is_empty() { Lang::ALL.to_vec() } else { languages },
            vocabulary: Vocabulary::parse(&settings.vocabulary).unwrap_or(Vocabulary::Broad),
            code: settings.code,
        };
        Ok(self.lock().set_pool(pool)?)
    }

    /// Key caps of the active layout, for the on-screen keyboard.
    pub fn keyboard(&self) -> Vec<KeyCap> {
        let engine = self.lock();
        let layout = engine.layout();
        layout
            .geometry
            .keys()
            .filter(|k| KEY_CODES[*k as usize] != "Enter")
            .map(|k| {
                let g = layout.key(k);
                let code = KEY_CODES[k as usize];
                KeyCap {
                    code: code.to_string(),
                    row: g.row,
                    x: g.x,
                    width: if code == "Space" { 6.0 } else { 1.0 },
                    label: layout.label(k, Layer::Base).unwrap_or_default(),
                    finger: g.finger.index() as u8,
                }
            })
            .collect()
    }

    /// Keys for the next expected character (for the hesitation hint).
    pub fn hint(&self) -> Option<KeyHint> {
        let engine = self.lock();
        let c = engine.expected()?;
        let layout = engine.layout();
        let strokes = layout.strokes(c)?;
        let main = *strokes.last()?;
        Some(KeyHint {
            codes: strokes.iter().map(|s| KEY_CODES[s.key as usize].to_string()).collect(),
            finger: layout.key(main.key).finger.index() as u8,
            shift: main.layer.shift(),
            option: strokes.iter().any(|s| s.layer.option()),
        })
    }

    /// Everything the progress screen shows. Also asks the learning thread for a fresh
    /// forecast, which shows up in a later call.
    pub fn stats(&self) -> Result<Stats, ReflexError> {
        let mut engine = self.lock();
        engine.request_forecast();
        let batches = engine.store().batches()?;
        let confusions = engine.store().confusions(8)?;
        let total_keystrokes = engine.store().total_chars()?;
        let today_minutes = engine.today_minutes();
        let layout = engine.layout().clone();
        let insights = engine.insights();
        let gains = insights.gains.clone();
        let mut days: Vec<DayPoint> = Vec::new();
        for b in &batches {
            let day_ms = b.ended_ms / 86_400_000 * 86_400_000;
            match days.last_mut() {
                Some(d) if d.day_ms == day_ms => {
                    let w = d.minutes / (d.minutes + b.seconds / 60.0);
                    d.wpm = d.wpm * w + b.wpm * (1.0 - w);
                    d.accuracy = d.accuracy * w + b.accuracy * (1.0 - w);
                    d.minutes += b.seconds / 60.0;
                    d.real_wpm = b.real_wpm.or(d.real_wpm);
                }
                _ => days.push(DayPoint { day_ms, wpm: b.wpm, accuracy: b.accuracy, real_wpm: b.real_wpm, minutes: b.seconds / 60.0 }),
            }
        }
        let key_heat = gains.as_ref().map_or_else(Vec::new, |g| {
            let (lo, hi) = g.key_cost.values().fold((f32::MAX, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            g.key_cost
                .iter()
                .map(|(k, v)| KeyHeat {
                    code: KEY_CODES[*k as usize].to_string(),
                    value: if hi > lo { ((v - lo) / (hi - lo)) as f64 } else { 0.0 },
                    ms: *v as f64,
                })
                .collect()
        });
        let label = |k: u8| layout.label(k, Layer::Base).filter(|l| !l.is_empty()).unwrap_or_else(|| KEY_CODES[k as usize].to_string());
        Ok(Stats {
            forecast: insights
                .forecast
                .as_ref()
                .map_or_else(Vec::new, |f| f.points.iter().map(|(d, w, pm)| ForecastPoint { days: *d, wpm: *w, plus_minus: *pm }).collect()),
            key_heat,
            slow_pairs: gains.as_ref().map_or_else(Vec::new, |g| {
                g.slow_pairs.iter().map(|(t, ms)| PairStat { text: t.clone(), ms: *ms as f64 }).collect()
            }),
            confusions: confusions
                .into_iter()
                .map(|(e, k, n)| Confusion { expected: e, typed: label(k), count: n })
                .collect(),
            real_wpm: gains.as_ref().map(|g| g.real_wpm).filter(|w| *w > 0.0),
            confidence: gains.as_ref().map_or(0.0, |g| (1.0 - 4.0 * g.uncertainty_ratio as f64).clamp(0.0, 1.0)),
            model_ready: insights.model_ready,
            streak_days: crate::stats::streak_days(&batches, engine::now_ms()),
            total_minutes: batches.iter().map(|b| b.seconds).sum::<f64>() / 60.0,
            today_minutes,
            total_keystrokes,
            days,
        })
    }

    /// Write all data as JSON to `path`.
    pub fn export(&self, path: String) -> Result<(), ReflexError> {
        let json = self.lock().store().export_json()?;
        std::fs::write(path, json).map_err(|e| ReflexError::Storage(e.to_string()))
    }
}
