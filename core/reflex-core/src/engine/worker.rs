//! Background learning thread. Owns the typist model and the bandit; receives finished
//! batches, learns, and sends back a new planner plus fresh insights. The UI thread never
//! waits for it.

use crate::brain::{Brain, Forecast};
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
    /// A batch was finished: learn from it.
    Batch { lines: Vec<TypedLine>, minutes: f64, layout: Layout, pool: Pool, now_ms: i64 },
    /// Layout or pool changed: re-plan without learning.
    Context { layout: Layout, pool: Pool, now_ms: i64 },
    /// Stats screen opened: compute a fresh forecast.
    Forecast { layout: Layout, pool: Pool, now_ms: i64 },
}

pub struct Update {
    pub planner: Option<Planner>,
    pub gains: Arc<GainTable>,
    pub arm: usize,
    pub trials: Option<Vec<Trial>>,
    pub forecast: Option<Forecast>,
}

pub struct WorkerSetup {
    pub model_paths: Vec<PathBuf>,
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
        .spawn(move || run(setup, job_rx, up_tx))
        .expect("spawn learning thread");
    (job_tx, up_rx)
}

fn run(setup: WorkerSetup, jobs: Receiver<Job>, updates: Sender<Update>) {
    let paths: Vec<&std::path::Path> = setup.model_paths.iter().map(PathBuf::as_path).collect();
    let (model, trained) = Brain::load_model(&paths);
    let mut brain = Brain::new(model, setup.now_ms as u64);
    brain.observe(&setup.history);
    brain.restore_trials(setup.trials);
    let send = |brain: &Brain, trials: bool, forecast: Option<Forecast>| {
        updates
            .send(Update {
                planner: trained.then(|| brain.planner()),
                gains: brain.gains(),
                arm: brain.arm(),
                trials: trials.then(|| brain.trials().to_vec()),
                forecast,
            })
            .is_ok()
    };
    if trained {
        brain.refresh(&setup.layout, &setup.pool, setup.now_ms);
    }
    if !send(&brain, false, None) {
        return;
    }
    while let Ok(job) = jobs.recv() {
        let ok = match job {
            Job::Batch { lines, minutes, layout, pool, now_ms } => {
                let before = brain.gains().real_wpm;
                brain.observe(&lines);
                if trained {
                    if brain.learn().kept {
                        brain.save_model(&setup.save_path).ok();
                    }
                    let after = brain.refresh(&layout, &pool, now_ms).real_wpm;
                    brain.end_batch(before, after, minutes);
                }
                send(&brain, true, None)
            }
            Job::Context { layout, pool, now_ms } => {
                if trained {
                    brain.refresh(&layout, &pool, now_ms);
                }
                send(&brain, false, None)
            }
            Job::Forecast { layout, pool, now_ms } => {
                let forecast = trained.then(|| brain.forecast(&layout, &pool, now_ms));
                send(&brain, false, forecast)
            }
        };
        if !ok {
            break;
        }
    }
}
