//! The engine ties everything together: typing session, storage, practice history and
//! the trainer. UIs talk to it through `ffi`; all calls are cheap and never block on ML.

use crate::content::Pool;
use crate::history::PracticeHistory;
use crate::layout::{Geometry, Layout, LayoutId};
use crate::record::TypedLine;
use crate::session::{BatchTotals, KeyInput, KeyOutcome, TypingSession};
use crate::store::{BatchRecord, Store, StoreError};
use crate::trainer::lines::RecentWords;
use crate::trainer::random::RandomSource;
use crate::trainer::{LineSource, PlanContext};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const VISIBLE_LINES: usize = 3;
const QUEUED_LINES: usize = 4;

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

pub struct Engine {
    data_dir: PathBuf,
    resources_dir: PathBuf,
    real_wpm: Option<f64>,
    store: Store,
    layout: Layout,
    pool: Pool,
    session: TypingSession,
    session_id: Option<i64>,
    source: Box<dyn LineSource>,
    recent: RecentWords,
    rng: StdRng,
    history: PracticeHistory,
    batch_lines: Vec<TypedLine>,
    recent_wpms: VecDeque<f64>,
    today_sec: f64,
}

impl Engine {
    pub fn open(data_dir: &Path, resources_dir: &Path) -> Result<Engine, StoreError> {
        std::fs::create_dir_all(data_dir).ok();
        let mut store = Store::open(&data_dir.join("reflex.sqlite"))?;
        store.archive_old_sessions(now_ms())?;
        let pool = store
            .setting("pool")?
            .and_then(|s| crate::settings::parse_pool(&s))
            .unwrap_or_default();
        let mut history = PracticeHistory::default();
        for line in store.typed_lines()? {
            replay(&mut history, &line);
        }
        let batches = store.batches()?;
        let recent_wpms = batches.iter().rev().take(10).map(|b| b.wpm).collect();
        let today_sec = today_seconds(&batches);
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let session = TypingSession::new(layout.id, layout.geometry);
        let mut engine = Engine {
            data_dir: data_dir.to_path_buf(),
            resources_dir: resources_dir.to_path_buf(),
            real_wpm: None,
            store,
            layout,
            pool,
            session,
            session_id: None,
            source: Box::new(RandomSource::default()),
            recent: RecentWords::default(),
            rng: StdRng::from_os_rng(),
            history,
            batch_lines: Vec::new(),
            recent_wpms,
            today_sec,
        };
        engine.fill_queue();
        Ok(engine)
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn resources_dir(&self) -> &Path {
        &self.resources_dir
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn history(&self) -> &PracticeHistory {
        &self.history
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn set_source(&mut self, source: Box<dyn LineSource>) {
        self.source = source;
        self.session.clear_upcoming();
        self.fill_queue();
    }

    pub fn set_layout(&mut self, id: LayoutId, geometry: Geometry) {
        if self.layout.id == id && self.layout.geometry == geometry {
            return;
        }
        self.layout = Layout::new(id, geometry);
        self.session.set_layout(id, geometry);
        self.session_id = None;
        self.session.clear_upcoming();
        self.fill_queue();
    }

    pub fn set_pool(&mut self, pool: Pool) -> Result<(), StoreError> {
        if pool == self.pool {
            return Ok(());
        }
        self.store.set_setting("pool", &crate::settings::format_pool(&pool))?;
        self.pool = pool;
        self.session.clear_upcoming();
        self.fill_queue();
        Ok(())
    }

    pub fn pause(&mut self) {
        self.session.pause();
    }

    /// Handle a key. Returns the outcome and a summary if this key closed a batch.
    pub fn key(&mut self, input: KeyInput) -> Result<(KeyOutcome, Option<BatchSummary>, Option<Vec<TypedLine>>), StoreError> {
        let (outcome, finished) = self.session.key(&input);
        if let Some(f) = finished {
            let session_id = match self.session_id {
                Some(id) => id,
                None => {
                    let id = self.store.begin_session(self.layout.id, self.layout.geometry, now_ms())?;
                    self.session_id = Some(id);
                    id
                }
            };
            self.store.insert_line(session_id, &f.line, &f.keys)?;
            replay(&mut self.history, &f.line);
            self.batch_lines.push(f.line);
            self.fill_queue();
        }
        if !outcome.batch_finished {
            return Ok((outcome, None, None));
        }
        let totals = self.session.take_batch();
        let lines = std::mem::take(&mut self.batch_lines);
        let summary = self.close_batch(totals, None)?;
        Ok((outcome, Some(summary), Some(lines)))
    }

    fn close_batch(&mut self, totals: BatchTotals, real_wpm: Option<f64>) -> Result<BatchSummary, StoreError> {
        let wpm = totals.wpm();
        let avg = if self.recent_wpms.is_empty() {
            wpm
        } else {
            self.recent_wpms.iter().sum::<f64>() / self.recent_wpms.len() as f64
        };
        let fatigue_hint = crate::stats::fatigue(&self.recent_wpms, wpm, self.session.session_sec());
        self.recent_wpms.push_front(wpm);
        self.recent_wpms.truncate(10);
        self.today_sec += totals.active_sec;
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
        })?;
        Ok(BatchSummary { wpm, accuracy: totals.accuracy(), trend: wpm - avg, real_wpm, fatigue_hint })
    }

    /// Attach the model's estimate to the most recent batch (computed in the background).
    pub fn record_real_wpm(&self, real_wpm: f64, arm: Option<u32>) -> Result<(), StoreError> {
        self.store.update_last_batch(real_wpm, arm)
    }

    pub fn view(&self) -> View {
        let totals = self.session.batch();
        View {
            lines: self.session.lines().take(VISIBLE_LINES).map(|l| l.text.clone()).collect(),
            cursor: self.session.cursor(),
            error_positions: self.session.error_positions().to_vec(),
            live_wpm: totals.wpm(),
            live_accuracy: totals.accuracy(),
            today_minutes: (self.today_sec + totals.active_sec) / 60.0,
            real_wpm: self.real_wpm,
        }
    }

    pub fn expected(&self) -> Option<char> {
        self.session.expected()
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

fn replay(history: &mut PracticeHistory, line: &TypedLine) {
    let mut prev = line.prev_char;
    for (c, rec) in line.text.chars().zip(line.chars.iter()) {
        history.record(prev, c, rec.wall_ms);
        prev = Some(c);
    }
}

fn today_seconds(batches: &[BatchRecord]) -> f64 {
    let day = now_ms() / 86_400_000;
    batches.iter().filter(|b| b.ended_ms / 86_400_000 == day).map(|b| b.seconds).sum()
}
