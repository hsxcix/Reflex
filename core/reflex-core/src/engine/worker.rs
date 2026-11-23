//! Background learning thread. Owns the typist model and the bandit; receives finished
//! batches, learns, and sends back a new planner plus fresh insights. The UI thread never
//! waits for it.

use crate::brain::{BatchOutcome, Brain, Forecast, LayoutSpeed};
use crate::content::Pool;
use crate::layout::Layout;
use crate::record::TypedLine;
use crate::trainer::bandit::Trial;
use crate::trainer::gains::GainTable;
use crate::trainer::planner::Planner;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

pub enum Job {
    /// A batch was finished: learn from it. `batch_id` is the stored batch row.
    Batch { batch_id: i64, lines: Vec<TypedLine>, minutes: f64, layout: Layout, pool: Pool, now_ms: i64 },
    /// Layout or pool changed: re-plan without learning.
    Context { layout: Layout, pool: Pool, now_ms: i64 },
    /// Stats screen opened: compute a fresh forecast.
    Forecast { layout: Layout, pool: Pool, now_ms: i64 },
}

/// What an update answers. Only a `Batch` reply carries that batch's results; the engine
/// must not attach a context or forecast reply to the last batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reply {
    Startup,
    Batch { batch_id: i64, outcome: BatchOutcome },
    Context,
    Forecast,
}

pub struct Update {
    pub reply: Reply,
    pub planner: Option<Planner>,
    pub gains: Arc<GainTable>,
    pub trials: Option<Vec<Trial>>,
    pub forecast: Option<Forecast>,
    pub measured_wpm: Option<f64>,
    pub layout_speeds: Vec<LayoutSpeed>,
}

pub struct WorkerSetup {
    /// Personal model first, then the shipped pretrained one (also the probe baseline).
    pub model_paths: Vec<PathBuf>,
    pub prior_path: PathBuf,
    pub save_path: PathBuf,
    pub history: Vec<TypedLine>,
    pub trials: Vec<Trial>,
    pub layout: Layout,
    pub pool: Pool,
    pub now_ms: i64,
}

pub fn spawn(setup: WorkerSetup) -> (Sender<Job>, Receiver<Update>) {
    let (job_tx, job_rx) = channel::<Job>();
    let (up_tx, up_rx) = channel::<Update>();
    thread::Builder::new()
        .name("reflex-learning".into())
        .spawn(move || {
            // A panic here would otherwise end learning silently; the engine notices the
            // closed channel and reports that learning stopped.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(setup, job_rx, up_tx)));
            if let Err(e) = result {
                let msg = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default();
                eprintln!("reflex: learning thread stopped: {msg}");
            }
        })
        .expect("spawn learning thread");
    (job_tx, up_rx)
}

/// Collapse queued jobs: every batch is kept (in order), but only the latest context and
/// forecast request matter.
fn coalesce(first: Job, jobs: &Receiver<Job>) -> Vec<Job> {
    let mut out = vec![first];
    while let Ok(job) = jobs.try_recv() {
        out.push(job);
    }
    let last_context = out.iter().rposition(|j| matches!(j, Job::Context { .. }));
    let last_forecast = out.iter().rposition(|j| matches!(j, Job::Forecast { .. }));
    out.into_iter()
        .enumerate()
        .filter(|(i, j)| match j {
            Job::Batch { .. } => true,
            Job::Context { .. } => Some(*i) == last_context,
            Job::Forecast { .. } => Some(*i) == last_forecast,
        })
        .map(|(_, j)| j)
        .collect()
}

fn run(setup: WorkerSetup, jobs: Receiver<Job>, updates: Sender<Update>) {
    let paths: Vec<&std::path::Path> = setup.model_paths.iter().map(PathBuf::as_path).collect();
    let (model, trained) = Brain::load_model(&paths);
    let (baseline, _) = Brain::load_model(&paths[paths.len().saturating_sub(1)..]);
    let mut brain = Brain::new(model, baseline, setup.now_ms as u64);
    brain.observe(&setup.history);
    brain.restore_trials(setup.trials, Brain::load_prior(&setup.prior_path));
    let mut layout = setup.layout.clone();
    let mut pool = setup.pool.clone();
    let send = |brain: &mut Brain, reply: Reply, trials: bool, forecast: Option<Forecast>, layout: &Layout, pool: &Pool, now_ms: i64| {
        let planner = trained.then(|| {
            let mut planner = brain.planner();
            // Score the word lists here, not on the UI thread when the first line is needed.
            planner.prepare(pool, layout);
            planner
        });
        updates
            .send(Update {
                reply,
                planner,
                gains: brain.gains(),
                trials: trials.then(|| brain.trials().to_vec()),
                forecast,
                measured_wpm: brain.measured_wpm(layout.id),
                layout_speeds: if trained { brain.layout_speeds(pool, now_ms) } else { Vec::new() },
            })
            .is_ok()
    };
    if trained {
        brain.refresh(&layout, &pool, setup.now_ms);
    }
    if !send(&mut brain, Reply::Startup, false, None, &layout, &pool, setup.now_ms) {
        return;
    }
    while let Ok(first) = jobs.recv() {
        for job in coalesce(first, &jobs) {
            let ok = match job {
                Job::Batch { batch_id, lines, minutes, layout: l, pool: p, now_ms } => {
                    (layout, pool) = (l, p);
                    let outcome = if trained {
                        let outcome = brain.batch(&lines, minutes, &layout, &pool, now_ms);
                        if outcome.kept {
                            brain.save_model(&setup.save_path).ok();
                        }
                        outcome
                    } else {
                        brain.observe(&lines);
                        BatchOutcome { kept: false, played_arm: brain.arm(), real_wpm: 0.0, probe_wpm: None }
                    };
                    send(&mut brain, Reply::Batch { batch_id, outcome }, true, None, &layout, &pool, now_ms)
                }
                Job::Context { layout: l, pool: p, now_ms } => {
                    (layout, pool) = (l, p);
                    if trained {
                        brain.refresh(&layout, &pool, now_ms);
                    }
                    send(&mut brain, Reply::Context, false, None, &layout, &pool, now_ms)
                }
                Job::Forecast { layout: l, pool: p, now_ms } => {
                    let forecast = trained.then(|| brain.forecast(&l, &p, now_ms));
                    send(&mut brain, Reply::Forecast, false, forecast, &layout, &pool, now_ms)
                }
            };
            if !ok {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Geometry, LayoutId};

    #[test]
    fn coalescing_keeps_batches_and_latest_requests() {
        let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
        let pool = Pool::default();
        let (tx, rx) = channel();
        let ctx = |t| Job::Context { layout: layout.clone(), pool: pool.clone(), now_ms: t };
        let fc = |t| Job::Forecast { layout: layout.clone(), pool: pool.clone(), now_ms: t };
        let batch = |id| Job::Batch { batch_id: id, lines: Vec::new(), minutes: 1.0, layout: layout.clone(), pool: pool.clone(), now_ms: 0 };
        for job in [fc(1), batch(1), fc(2), ctx(3), batch(2), fc(4)] {
            tx.send(job).unwrap();
        }
        let first = rx.recv().unwrap();
        let kinds: Vec<String> = coalesce(first, &rx)
            .iter()
            .map(|j| match j {
                Job::Batch { batch_id, .. } => format!("b{batch_id}"),
                Job::Context { now_ms, .. } => format!("c{now_ms}"),
                Job::Forecast { now_ms, .. } => format!("f{now_ms}"),
            })
            .collect();
        assert_eq!(kinds, ["b1", "c3", "b2", "f4"]);
    }
}
