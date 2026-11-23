//! The C-ABI surface for UIs (Swift today, anything UniFFI supports later).
//! Thin by design: conversion only, all logic lives in the core.

use crate::content::{Lang, Pool, Vocabulary};
use crate::engine::{self, Engine};
use crate::layout::{Geometry, KEY_CODES, Layer, Layout, LayoutId, key_id};
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
    /// Measured speed on recent probe lines (held-out sentences), once typed.
    pub measured_wpm: Option<f64>,
    /// The first visible line is a measurement line.
    pub probe_line: bool,
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
    /// Local midnight starting the day, in the current time zone (ms since epoch).
    pub day_ms: i64,
    pub wpm: f64,
    pub accuracy: f64,
    pub real_wpm: Option<f64>,
    /// Mean measured speed of that day's probe lines.
    pub measured_wpm: Option<f64>,
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

/// Measured numbers for one key of the active layout (first attempts, last 30 days).
#[derive(uniffi::Record)]
pub struct KeyMeasured {
    pub code: String,
    pub attempts: u32,
    pub errors: u32,
    /// errors / attempts.
    pub error_rate: f64,
    /// Mean time to a correct press (ms); `None` without timed presses.
    pub mean_ms: Option<f64>,
}

#[derive(uniffi::Record)]
pub struct PairStat {
    pub text: String,
    pub ms: f64,
}

/// Real speed on one layout the person has typed on (transition view).
#[derive(uniffi::Record)]
pub struct LayoutSpeedStat {
    pub layout: String,
    pub name: String,
    pub estimated_wpm: f64,
    pub measured_wpm: Option<f64>,
    pub keystrokes: f64,
    pub active: bool,
}

/// A character whose movement differs between the active layout and the one you came from.
#[derive(uniffi::Record)]
pub struct MovedKey {
    pub ch: String,
    /// Measured mean time on the active layout (ms), first attempts.
    pub now_ms: Option<f64>,
    pub now_attempts: u32,
    pub now_error_rate: f64,
    /// Same character on the old layout.
    pub before_ms: Option<f64>,
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
    /// Model-predicted cost per key.
    pub key_heat: Vec<KeyHeat>,
    /// Measured speed and errors per key.
    pub key_measured: Vec<KeyMeasured>,
    pub slow_pairs: Vec<PairStat>,
    /// Practised pairs the model expects to fade most within a week without practice
    /// (ms slower per press); the trainer schedules them before that happens.
    pub fading_pairs: Vec<PairStat>,
    pub confusions: Vec<Confusion>,
    pub real_wpm: Option<f64>,
    /// Measured speed on recent probe lines (active layout).
    pub measured_wpm: Option<f64>,
    /// Every layout typed on, with estimate and measurement.
    pub layouts: Vec<LayoutSpeedStat>,
    /// Display name of the layout you are switching from, while in transition.
    pub transition_from: Option<String>,
    /// Characters that moved, in transition (most practised on the old layout first).
    pub moved_keys: Vec<MovedKey>,
    /// The learning thread stopped (bug); typing still works but nothing is learnt.
    pub learning_stopped: bool,
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
    /// `utc_offset_min`: local time zone offset, so days end at local midnight.
    #[uniffi::constructor]
    pub fn open(data_dir: String, resources_dir: String, utc_offset_min: i32) -> Result<Arc<ReflexEngine>, ReflexError> {
        let engine = Engine::open(Path::new(&data_dir), Path::new(&resources_dir), utc_offset_min)?;
        Ok(Arc::new(ReflexEngine { engine: Mutex::new(engine) }))
    }

    /// The time zone changed (daylight saving time, travel).
    pub fn set_utc_offset(&self, utc_offset_min: i32) -> Result<(), ReflexError> {
        Ok(self.lock().set_utc_offset(utc_offset_min)?)
    }

    /// Tell the core which layout ("de-iso", "us-ansi", "eurkey") and hardware ("iso"/"ansi") is active.
    pub fn set_layout(&self, layout: String, geometry: String) -> Result<(), ReflexError> {
        let id = LayoutId::parse(&layout).unwrap_or(LayoutId::DeIso);
        let geometry = Geometry::parse(&geometry).unwrap_or(Geometry::Iso);
        // German needs the ISO-only key; on ANSI hardware use the closest full layout.
        let id = if id == LayoutId::DeIso && geometry == Geometry::Ansi { LayoutId::UsAnsi } else { id };
        Ok(self.lock().set_layout(id, geometry)?)
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
            measured_wpm: v.measured_wpm,
            probe_line: v.probe_line,
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
            .filter(|k| KEY_CODES.get(*k as usize).is_some_and(|c| *c != "Enter"))
            .map(|k| {
                let g = layout.key(k);
                let code = KEY_CODES.get(k as usize).copied().unwrap_or("");
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
            codes: strokes.iter().map(|s| code_name(s.key)).collect(),
            finger: layout.key(main.key).finger.index() as u8,
            shift: main.layer.shift(),
            option: strokes.iter().any(|s| s.layer.option()),
        })
    }

    /// Everything the progress screen shows. Also asks the learning thread for a fresh
    /// forecast, which shows up in a later call. Slow queries run on a separate read-only
    /// connection after the engine lock is released, so typing never waits for them.
    pub fn stats(&self) -> Result<Stats, ReflexError> {
        let (reader, today_minutes, offset, layout, gains, forecast, model_ready, measured_wpm, speeds, learning_stopped) = {
            let mut engine = self.lock();
            engine.request_forecast();
            let today_minutes = engine.today_minutes();
            let offset = engine.utc_offset_min();
            let layout = engine.layout().clone();
            let reader = engine.reader()?;
            let i = engine.insights();
            (reader, today_minutes, offset, layout, i.gains.clone(), i.forecast.clone(), i.model_ready, i.measured_wpm, i.layout_speeds.clone(), i.learning_stopped)
        };
        let batches = reader.batches()?;
        let confusions = reader.confusions(layout.id, layout.geometry, 8)?;
        let total_keystrokes = reader.total_chars()?;
        let char_stats = reader.char_stats(layout.id, layout.geometry)?;
        let mut days: Vec<DayPoint> = Vec::new();
        let mut probe_n: Vec<f64> = Vec::new();
        for b in &batches {
            // Each batch keeps the calendar day it was typed on; the point sits at that
            // day's midnight in the current zone, so the chart shows the right date.
            let day_ms = crate::stats::day_start_ms(b.local_day(offset), offset);
            match days.last_mut() {
                Some(d) if d.day_ms == day_ms => {
                    let w = d.minutes / (d.minutes + b.seconds / 60.0);
                    d.wpm = d.wpm * w + b.wpm * (1.0 - w);
                    d.accuracy = d.accuracy * w + b.accuracy * (1.0 - w);
                    d.minutes += b.seconds / 60.0;
                    d.real_wpm = b.real_wpm.or(d.real_wpm);
                }
                _ => {
                    days.push(DayPoint { day_ms, wpm: b.wpm, accuracy: b.accuracy, real_wpm: b.real_wpm, measured_wpm: None, minutes: b.seconds / 60.0 });
                    probe_n.push(0.0);
                }
            }
            if let (Some(p), Some(d), Some(n)) = (b.probe_wpm, days.last_mut(), probe_n.last_mut()) {
                d.measured_wpm = Some((d.measured_wpm.unwrap_or(0.0) * *n + p) / (*n + 1.0));
                *n += 1.0;
            }
        }
        let key_heat = gains.as_ref().map_or_else(Vec::new, |g| {
            let (lo, hi) = g.key_cost.values().fold((f32::MAX, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            g.key_cost
                .iter()
                .map(|(k, v)| KeyHeat {
                    code: code_name(*k),
                    value: if hi > lo { ((v - lo) / (hi - lo)) as f64 } else { 0.0 },
                    ms: *v as f64,
                })
                .collect()
        });
        let key_measured = crate::stats::key_measured(&char_stats, &layout)
            .into_iter()
            .map(|m| KeyMeasured {
                code: code_name(m.key),
                attempts: m.attempts,
                errors: m.errors,
                error_rate: m.error_rate,
                mean_ms: m.mean_ms,
            })
            .collect();
        let confusions = confusions.into_iter().map(|(e, k, n)| Confusion { typed: slip_label(&layout, &e, k), expected: e, count: n }).collect();
        // Transition: you came from another layout with clearly more practice.
        let total_on = |id: LayoutId| speeds.iter().filter(|s| s.layout == id).map(|s| s.keystrokes).sum::<f64>();
        let from = speeds
            .iter()
            .filter(|s| s.layout != layout.id)
            .max_by(|a, b| a.keystrokes.total_cmp(&b.keystrokes))
            .filter(|s| s.keystrokes >= 2.0 * total_on(layout.id).max(500.0))
            .copied();
        let moved_keys = match from {
            Some(f) => moved_keys(&reader, &layout, &Layout::new(f.layout, f.geometry), &char_stats)?,
            None => Vec::new(),
        };
        let pairs = |v: &[(String, f32)]| v.iter().map(|(t, ms)| PairStat { text: t.clone(), ms: *ms as f64 }).collect::<Vec<_>>();
        Ok(Stats {
            forecast: forecast
                .as_ref()
                .map_or_else(Vec::new, |f| f.points.iter().map(|(d, w, pm)| ForecastPoint { days: *d, wpm: *w, plus_minus: *pm }).collect()),
            key_heat,
            key_measured,
            slow_pairs: gains.as_ref().map_or_else(Vec::new, |g| pairs(&g.slow_pairs)),
            fading_pairs: gains.as_ref().map_or_else(Vec::new, |g| pairs(&g.fading_pairs)),
            confusions,
            real_wpm: gains.as_ref().map(|g| g.real_wpm).filter(|w| *w > 0.0),
            measured_wpm,
            layouts: speeds
                .iter()
                .map(|s| LayoutSpeedStat {
                    layout: s.layout.as_str().to_string(),
                    name: s.layout.display_name().to_string(),
                    estimated_wpm: s.estimated_wpm,
                    measured_wpm: s.measured_wpm,
                    keystrokes: s.keystrokes,
                    active: s.layout == layout.id,
                })
                .collect(),
            transition_from: from.map(|f| f.layout.display_name().to_string()),
            moved_keys,
            learning_stopped,
            confidence: gains.as_ref().map_or(0.0, |g| (1.0 - 4.0 * g.uncertainty_ratio as f64).clamp(0.0, 1.0)),
            model_ready,
            streak_days: crate::stats::streak_days(&batches, engine::now_ms(), offset),
            total_minutes: batches.iter().map(|b| b.seconds).sum::<f64>() / 60.0,
            today_minutes,
            total_keystrokes,
            days,
        })
    }

    /// Write everything as JSON to `path` (keystrokes, lines, batches; archive included).
    pub fn export_json(&self, path: String) -> Result<(), ReflexError> {
        // Read-only connection: building a large export never blocks typing.
        let reader = self.lock().reader()?;
        let json = reader.export_json()?;
        write_atomic(&path, json.as_bytes())
    }

    /// Write every keystroke as one CSV row to `path` (archive included).
    pub fn export_csv(&self, path: String) -> Result<(), ReflexError> {
        let reader = self.lock().reader()?;
        let csv = reader.export_csv()?;
        write_atomic(&path, csv.as_bytes())
    }
}

/// Write to a temporary file next to `path`, then rename: never leaves a half-written export.
fn write_atomic(path: &str, bytes: &[u8]) -> Result<(), ReflexError> {
    let target = Path::new(path);
    let name = target.file_name().and_then(|n| n.to_str()).unwrap_or("export");
    let tmp = target.with_file_name(format!(".{name}.reflex-tmp"));
    let result = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, target));
    if result.is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    result.map_err(|e| ReflexError::Storage(format!("could not write {path}: {e}")))
}

/// W3C code of a key id (ids come from the database too, so never index blindly).
fn code_name(k: u8) -> String {
    KEY_CODES.get(k as usize).map_or_else(String::new, |c| c.to_string())
}

/// How to show the key hit instead of `expected`: its label, or the missing modifier when
/// the right key was hit without Shift/Option.
fn slip_label(layout: &Layout, expected: &str, wrong: u8) -> String {
    let stroke = expected.chars().next().and_then(|c| layout.main_stroke(c));
    if let Some(s) = stroke.filter(|s| s.key == wrong) {
        return if s.layer.shift() { "no ⇧".into() } else if s.layer.option() { "no ⌥".into() } else { "same key".into() };
    }
    layout.label(wrong, Layer::Base).filter(|l| !l.is_empty()).unwrap_or_else(|| code_name(wrong))
}

/// Characters typed on `old` whose movement is different on `new`, with measured times on
/// both (first attempts). Most practised on the old layout first.
fn moved_keys(reader: &crate::store::Store, new: &Layout, old: &Layout, new_stats: &[crate::store::CharStat]) -> Result<Vec<MovedKey>, ReflexError> {
    let old_stats = reader.char_stats(old.id, old.geometry)?;
    let mean = |s: &crate::store::CharStat| (s.latency_n > 0).then(|| s.latency_sum_ms / s.latency_n as f64);
    let mut out: Vec<(u32, MovedKey)> = old_stats
        .iter()
        .filter(|o| o.expected != ' ' && new.motor(o.expected).is_some() && new.motor(o.expected) != old.motor(o.expected))
        .map(|o| {
            let now = new_stats.iter().find(|n| n.expected == o.expected);
            (
                o.attempts,
                MovedKey {
                    ch: o.expected.to_string(),
                    now_ms: now.and_then(mean),
                    now_attempts: now.map_or(0, |n| n.attempts),
                    now_error_rate: now.map_or(0.0, |n| n.errors as f64 / n.attempts.max(1) as f64),
                    before_ms: mean(o),
                },
            )
        })
        .collect();
    out.sort_by_key(|(attempts, _)| std::cmp::Reverse(*attempts));
    Ok(out.into_iter().take(16).map(|(_, m)| m).collect())
}
