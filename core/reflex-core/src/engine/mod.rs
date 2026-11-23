//! The engine ties everything together: typing session, storage, practice history, the
//! active trainer and the background learning worker. UIs talk to it through `ffi`; all
//! calls are cheap and never wait for ML.

mod worker;

use crate::brain::Forecast;
use crate::content::Pool;
use crate::layout::{Geometry, Layout, LayoutId};
use crate::record::TypedLine;
use crate::session::{BatchTotals, KeyInput, KeyOutcome, TypingSession};
use crate::store::{BatchRecord, OpenLine, Store, StoreError};
use crate::trainer::bandit::Trial;
use crate::trainer::gains::GainTable;
use crate::trainer::lines::RecentWords;
use crate::trainer::random::RandomSource;
use crate::trainer::{LineSource, PlanContext};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};
use worker::{Job, Update, WorkerSetup};

const VISIBLE_LINES: usize = 3;
const QUEUED_LINES: usize = 4;
const TRIALS_KEY: &str = "bandit_trials";

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchSummary {
    pub wpm: f64,
    pub accuracy: f64,
    /// Difference to the average of the previous batches (WPM).
    pub trend: f64,
    pub real_wpm: Option<f64>,
    pub fatigue_hint: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct View {
    pub lines: Vec<String>,
    pub cursor: usize,
    pub error_positions: Vec<usize>,
    pub live_wpm: f64,
    pub live_accuracy: f64,
    pub today_minutes: f64,
    pub real_wpm: Option<f64>,
}

/// What the learning side currently knows (for stats).
#[derive(Default)]
pub struct Insights {
    pub gains: Option<Arc<GainTable>>,
    pub forecast: Option<Forecast>,
    pub model_ready: bool,
}

pub struct Engine {
    store: Store,
    layout: Layout,
    pool: Pool,
    session: TypingSession,
    session_id: Option<i64>,
    source: Box<dyn LineSource>,
    recent: RecentWords,
    rng: StdRng,
    batch_lines: Vec<TypedLine>,
    recent_wpms: VecDeque<f64>,
    /// Active seconds on local day `today`.
    today_sec: f64,
    today: i64,
    utc_offset_min: i32,
    jobs: Sender<Job>,
    updates: Receiver<Update>,
    insights: Insights,
    pending_real_wpm: bool,
    /// Exclusive lock on the data folder for as long as the engine lives: a second
    /// instance on the same data would mix up open keys.
    _lock: std::fs::File,
}

impl Engine {
    /// Open the data in `data_dir`; `resources_dir` holds the shipped pretrained model.
    /// `utc_offset_min` is the local time zone offset; days (streak, minutes today) follow it.
    pub fn open(data_dir: &Path, resources_dir: &Path, utc_offset_min: i32) -> Result<Engine, StoreError> {
        std::fs::create_dir_all(data_dir)?;
        let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(data_dir.join("reflex.lock"))?;
        lock.try_lock().map_err(|e| match e {
            std::fs::TryLockError::WouldBlock => StoreError::Locked,
            std::fs::TryLockError::Error(e) => StoreError::Io(e),
        })?;
        let mut store = Store::open(&data_dir.join("reflex.sqlite"))?;
        // Keys of a line left open by the last run (quit or crash mid-line).
        store.abandon_open_keys()?;
        store.archive_old_sessions(now_ms())?;
        let pool = store.setting("pool")?.and_then(|s| crate::settings::parse_pool(&s)).unwrap_or_default();
        let trials: Vec<Trial> = store
            .setting(TRIALS_KEY)?
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let batches = store.batches()?;
        let recent_wpms = batches.iter().rev().take(10).map(|b| b.wpm).collect();
        let today = crate::stats::day_index(now_ms(), utc_offset_min);
        let today_sec = day_seconds(&batches, today, utc_offset_min);
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let personal = data_dir.join(format!("models/typist-v{}.bpk", crate::model::MODEL_VERSION));
        let (jobs, updates) = worker::spawn(WorkerSetup {
            model_paths: vec![personal.clone(), resources_dir.join("models/typist.bpk")],
            prior_path: resources_dir.join("models/bandit_prior.json"),
            save_path: personal,
            history: store.typed_lines()?,
            trials,
            layout: layout.clone(),
            pool: pool.clone(),
            now_ms: now_ms(),
        });
        let session = TypingSession::new(layout.id, layout.geometry);
        let mut engine = Engine {
            store,
            layout,
            pool,
            session,
            session_id: None,
            source: Box::new(RandomSource::default()),
            recent: RecentWords::default(),
            rng: StdRng::from_os_rng(),
            batch_lines: Vec::new(),
            recent_wpms,
            today_sec,
            today,
            utc_offset_min,
            jobs,
            updates,
            insights: Insights::default(),
            pending_real_wpm: false,
            _lock: lock,
        };
        engine.fill_queue();
        Ok(engine)
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn utc_offset_min(&self) -> i32 {
        self.utc_offset_min
    }

    /// The time zone changed (travel, daylight saving time).
    pub fn set_utc_offset(&mut self, utc_offset_min: i32) -> Result<(), StoreError> {
        if utc_offset_min != self.utc_offset_min {
            self.utc_offset_min = utc_offset_min;
            self.today = crate::stats::day_index(now_ms(), utc_offset_min);
            self.today_sec = day_seconds(&self.store.batches()?, self.today, utc_offset_min);
        }
        Ok(())
    }

    /// Start counting from zero once local midnight has passed.
    fn roll_day(&mut self) {
        let today = crate::stats::day_index(now_ms(), self.utc_offset_min);
        if today != self.today {
            self.today = today;
            self.today_sec = 0.0;
        }
    }

    pub fn insights(&mut self) -> &Insights {
        self.poll();
        &self.insights
    }

    /// Switch layout. A line in progress is dropped: its keys were typed on the old layout,
    /// stay in the export as abandoned (with that layout) and never count for the new one.
    pub fn set_layout(&mut self, id: LayoutId, geometry: Geometry) -> Result<(), StoreError> {
        if self.layout.id == id && self.layout.geometry == geometry {
            return Ok(());
        }
        if self.session.drop_all_lines() {
            self.store.abandon_open_keys()?;
        }
        self.layout = Layout::new(id, geometry);
        self.session.set_layout(id, geometry);
        self.session_id = None;
        self.fill_queue();
        self.send_context();
        Ok(())
    }

    pub fn set_pool(&mut self, pool: Pool) -> Result<(), StoreError> {
        if pool == self.pool {
            return Ok(());
        }
        self.store.set_setting("pool", &crate::settings::format_pool(&pool))?;
        self.pool = pool;
        self.session.clear_upcoming();
        self.fill_queue();
        self.send_context();
        Ok(())
    }

    /// Ask the learning thread for a fresh forecast (arrives with a later poll).
    pub fn request_forecast(&self) {
        self.jobs.send(Job::Forecast { layout: self.layout.clone(), pool: self.pool.clone(), now_ms: now_ms() }).ok();
    }

    pub fn pause(&mut self) {
        self.session.pause();
    }

    /// Handle a key. Returns the outcome and a summary if this key closed a batch.
    pub fn key(&mut self, input: KeyInput) -> Result<(KeyOutcome, Option<BatchSummary>), StoreError> {
        self.poll();
        let open_line = self.session.lines().next().map(|l| OpenLine {
            text: l.text.clone(),
            kind: l.kind,
            lang: l.lang,
            layout: self.layout.id,
            geometry: self.layout.geometry,
        });
        let keys_before = self.session.open_keys().len();
        let (outcome, finished) = self.session.key(&input);
        if finished.is_none()
            && self.session.open_keys().len() > keys_before
            && let (Some(line), Some(key)) = (open_line, self.session.open_keys().last().cloned()) {
                let session_id = self.ensure_session()?;
                self.store.insert_open_key(Some(session_id), &line, &key)?;
            }
        if let Some(f) = finished {
            let session_id = self.ensure_session()?;
            self.store.insert_line(session_id, &f.line, &f.keys)?;
            self.batch_lines.push(f.line);
            self.fill_queue();
        }
        if !outcome.batch_finished {
            return Ok((outcome, None));
        }
        let totals = self.session.take_batch();
        let summary = self.close_batch(totals)?;
        self.jobs
            .send(Job::Batch {
                lines: std::mem::take(&mut self.batch_lines),
                minutes: totals.active_sec / 60.0,
                layout: self.layout.clone(),
                pool: self.pool.clone(),
                now_ms: now_ms(),
            })
            .ok();
        self.pending_real_wpm = true;
        Ok((outcome, Some(summary)))
    }

    fn ensure_session(&mut self) -> Result<i64, StoreError> {
        if let Some(id) = self.session_id {
            return Ok(id);
        }
        let id = self.store.begin_session(self.layout.id, self.layout.geometry, now_ms())?;
        self.session_id = Some(id);
        Ok(id)
    }

    fn close_batch(&mut self, totals: BatchTotals) -> Result<BatchSummary, StoreError> {
        let wpm = totals.wpm();
        let avg = if self.recent_wpms.is_empty() {
            wpm
        } else {
            self.recent_wpms.iter().sum::<f64>() / self.recent_wpms.len() as f64
        };
        let fatigue_hint = crate::stats::fatigue(&self.recent_wpms, wpm, self.session.session_sec());
        self.recent_wpms.push_front(wpm);
        self.recent_wpms.truncate(10);
        self.roll_day();
        self.today_sec += totals.active_sec;
        let real_wpm = self.real_wpm();
        self.store.insert_batch(&BatchRecord {
            session_id: self.session_id.unwrap_or(0),
            ended_ms: now_ms(),
            seconds: totals.active_sec,
            chars: totals.chars,
            errors: totals.errors,
            wpm,
            accuracy: totals.accuracy(),
            real_wpm,
            arm: None,
            utc_offset_min: Some(self.utc_offset_min),
        })?;
        Ok(BatchSummary { wpm, accuracy: totals.accuracy(), trend: wpm - avg, real_wpm, fatigue_hint })
    }

    fn real_wpm(&self) -> Option<f64> {
        self.insights.gains.as_ref().map(|g| g.real_wpm).filter(|w| *w > 0.0)
    }

    /// Apply results from the learning thread.
    fn poll(&mut self) {
        while let Ok(update) = self.updates.try_recv() {
            if let Some(planner) = update.planner {
                // Swap the trainer for future lines; visible lines stay as they are.
                self.source = Box::new(planner);
                self.insights.model_ready = true;
            }
            if let Some(trials) = update.trials
                && let Ok(json) = serde_json::to_string(&trials) {
                    self.store.set_setting(TRIALS_KEY, &json).ok();
                }
            self.insights.gains = Some(update.gains);
            if update.forecast.is_some() {
                self.insights.forecast = update.forecast;
            }
            if self.pending_real_wpm {
                if let Some(real) = self.real_wpm() {
                    self.store.update_last_batch(real, Some(update.arm as u32)).ok();
                }
                self.pending_real_wpm = false;
            }
        }
    }

    pub fn view(&mut self) -> View {
        self.poll();
        self.roll_day();
        let totals = self.session.batch();
        View {
            lines: self.session.lines().take(VISIBLE_LINES).map(|l| l.text.clone()).collect(),
            cursor: self.session.cursor(),
            error_positions: self.session.error_positions().to_vec(),
            live_wpm: totals.wpm(),
            live_accuracy: totals.accuracy(),
            today_minutes: (self.today_sec + totals.active_sec) / 60.0,
            real_wpm: self.real_wpm(),
        }
    }

    pub fn expected(&self) -> Option<char> {
        self.session.expected()
    }

    pub fn today_minutes(&mut self) -> f64 {
        self.roll_day();
        (self.today_sec + self.session.batch().active_sec) / 60.0
    }

    fn send_context(&self) {
        self.jobs.send(Job::Context { layout: self.layout.clone(), pool: self.pool.clone(), now_ms: now_ms() }).ok();
    }

    fn fill_queue(&mut self) {
        while self.session.queued() < QUEUED_LINES {
            let ctx = PlanContext { pool: &self.pool, layout: &self.layout, recent: &self.recent };
            let line = self.source.next_line(&ctx, &mut self.rng);
            self.recent.note_line(&line.text);
            self.session.push_line(line);
        }
    }
}

fn day_seconds(batches: &[BatchRecord], day: i64, utc_offset_min: i32) -> f64 {
    batches
        .iter()
        .filter(|b| b.local_day(utc_offset_min) == day)
        .map(|b| b.seconds)
        .sum()
}
