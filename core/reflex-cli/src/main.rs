//! reflex-cli: pretraining, simulation and benchmarks for the Reflex core (no UI).

use clap::{Parser, Subcommand};
use reflex_core::history::PracticeHistory;
use reflex_core::model::encode::{EncodedLine, encode_history};
use reflex_core::model::train::{TrainParams, evaluate, train};
use reflex_core::model::{self, TypistModel};
use reflex_core::bench::{BenchConfig, TrainerKind, keystrokes_to, prediction_quality, run_trainer};
use reflex_core::brain::Brain;
use reflex_core::sim::Family;
use reflex_core::sim::data::simulate_history;
use burn::module::Module;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Reflex core tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Pretrain the typist model on simulated typists and save the weights.
    Train {
        #[arg(long, default_value_t = 400)]
        typists: usize,
        #[arg(long, default_value_t = 8)]
        sessions: usize,
        #[arg(long, default_value_t = 3000)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value = "data/models/typist.bpk")]
        out: PathBuf,
        /// Train on the CPU instead of the GPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Learn the bandit's starting belief from simulated people (training families only).
    TrainBandit {
        #[arg(long, default_value = "data/models/typist.bpk")]
        model: PathBuf,
        #[arg(long, default_value_t = 10)]
        typists: usize,
        #[arg(long, default_value_t = 30)]
        batches: usize,
        #[arg(long, default_value = "data/models/bandit_prior.json")]
        out: PathBuf,
    },
    /// Check the model's gain estimates against the simulator's ground truth.
    Diagnose {
        #[arg(long, default_value = "data/models/typist.bpk")]
        model: PathBuf,
        #[arg(long, default_value_t = 12)]
        warmup: usize,
    },
    /// Benchmark trainers and model quality on held-out simulated typists.
    Bench {
        #[arg(long, default_value = "data/models/typist.bpk")]
        model: PathBuf,
        #[arg(long, default_value_t = 2)]
        per_family: usize,
        #[arg(long, default_value_t = 32)]
        batches: usize,
        #[arg(long, default_value_t = 8)]
        lines: usize,
        /// Skip the prediction-quality part.
        #[arg(long)]
        skip_quality: bool,
    },
}

/// Simulate and encode typists in parallel (one chunk per CPU core).
fn simulate(typists: std::ops::Range<usize>, sessions: usize, families: &[Family]) -> Vec<EncodedLine> {
    let ids: Vec<usize> = typists.collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = ids.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = ids
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for &i in part {
                        let family = families[i % families.len()];
                        let lines = simulate_history(family, i as u64 * 7919 + 13, sessions);
                        let mut history = PracticeHistory::default();
                        out.extend(encode_history(&lines, &mut history));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("simulation thread")).collect()
    })
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Train { typists, sessions, steps, batch, out, cpu } => {
            let t0 = Instant::now();
            let train_set = simulate(0..typists, sessions, &Family::TRAIN);
            let valid_set = simulate(100_000..100_000 + typists / 10 + 4, sessions, &Family::TRAIN);
            let held_out = simulate(200_000..200_000 + typists / 10 + 4, sessions, &Family::HELD_OUT);
            let tokens: usize = train_set.iter().map(|l| l.tokens.len()).sum();
            println!("simulated {} lines ({tokens} keystrokes) in {:.1}s", train_set.len(), t0.elapsed().as_secs_f32());
            let device = if cpu { model::cpu() } else { model::accelerator() }.autodiff();
            let mut rng = StdRng::seed_from_u64(1);
            let fresh = TypistModel::new(&device);
            println!("before: valid {:.4}", evaluate(&fresh, &valid_set, &device));
            let t1 = Instant::now();
            let params = TrainParams { steps, batch_size: batch, lr: 1e-3, replay_share: 0.0 };
            let trained = train(fresh, &train_set, &[], &params, &device, &mut rng, |step, loss| {
                println!("step {step:>5}  loss {loss:.4}  ({:.0}s)", t1.elapsed().as_secs_f32());
            });
            println!(
                "after: valid {:.4}  held-out families {:.4}",
                evaluate(&trained, &valid_set, &device),
                evaluate(&trained, &held_out, &device)
            );
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir).expect("create output dir");
            }
            trained.valid().save_file(&out).expect("save weights");
            println!("saved {}", out.display());
        }
        Command::Bench { model, per_family, batches, lines, skip_quality } => bench(&model, per_family, batches, lines, skip_quality),
        Command::TrainBandit { model, typists, batches, out } => {
            let (m, loaded) = Brain::load_model(&[model.as_path()]);
            assert!(loaded, "no model");
            let cfg = BenchConfig { typists_per_family: 1, batches, lines_per_batch: 8, batches_per_day: 8 };
            let mut trials = Vec::new();
            for i in 0..typists {
                let family = Family::TRAIN[i % Family::TRAIN.len()];
                let curve = run_trainer(TrainerKind::Ml, family, 900_000 + i as u64, &cfg, &m, &[]);
                println!("{family:?} #{i}: {} trials, {:.1} → {:.1} wpm", curve.trials.len(), curve.start_wpm, curve.points.last().map_or(0.0, |p| p.1));
                trials.extend(curve.trials);
            }
            std::fs::write(&out, serde_json::to_string(&trials).expect("serialise trials")).expect("write prior");
            println!("saved {} trials to {}", trials.len(), out.display());
        }
        Command::Diagnose { model, warmup } => {
            let (m, loaded) = Brain::load_model(&[model.as_path()]);
            assert!(loaded, "no model");
            for (i, family) in Family::HELD_OUT.iter().enumerate() {
                let (rho, overlap) = reflex_core::bench::gain_accuracy(&m, *family, 42 + i as u64, warmup);
                println!("{family:?}: rank correlation {rho:.3}, top-30 overlap {:.0}%", overlap * 100.0);
            }
        }
    }
}

fn bench(path: &std::path::Path, per_family: usize, batches: usize, lines: usize, skip_quality: bool) {
    let (model, loaded) = Brain::load_model(&[path]);
    assert!(loaded, "no model at {}", path.display());
    let prior = Brain::load_prior(&path.with_file_name("bandit_prior.json"));
    let cfg = BenchConfig { typists_per_family: per_family, batches, lines_per_batch: lines, batches_per_day: 8 };
    if !skip_quality {
        let t = Instant::now();
        let (base, pre, fine) = prediction_quality(&model, 3 * per_family);
        println!("prediction quality on held-out typists ({:.0}s)", t.elapsed().as_secs_f32());
        println!("  per-pair averages   rmse(log latency) {:.4}  brier {:.5}", base.rmse_log_latency, base.brier);
        println!("  pretrained model    rmse(log latency) {:.4}  brier {:.5}", pre.rmse_log_latency, pre.brier);
        println!("  fine-tuned model    rmse(log latency) {:.4}  brier {:.5}", fine.rmse_log_latency, fine.brier);
    }
    let kinds = [TrainerKind::Random, TrainerKind::Heuristic, TrainerKind::Ml, TrainerKind::Oracle];
    let mut savings_vs = [Vec::new(), Vec::new()];
    let mut oracle_savings = Vec::new();
    let mut gains = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for family in Family::HELD_OUT {
        for i in 0..per_family {
            let seed = 700_000 + i as u64 * 101 + family as u64 * 7;
            let t = Instant::now();
            let curves: Vec<_> = kinds.iter().map(|k| run_trainer(*k, family, seed, &cfg, &model, &prior)).collect();
            let start = curves[0].start_wpm;
            let finals: Vec<f64> = curves.iter().map(|c| c.points.last().map_or(start, |p| p.1)).collect();
            for (g, f) in gains.iter_mut().zip(&finals) {
                g.push(f - start);
            }
            let budget = curves[0].points.last().map_or(1, |p| p.0) as f64;
            for (j, base) in [0usize, 1].iter().enumerate() {
                let level = finals[*base];
                let needed = keystrokes_to(&curves[2], level).map_or(budget * 1.5, |k| k as f64);
                savings_vs[j].push(1.0 - needed / budget);
            }
            let needed = keystrokes_to(&curves[3], finals[0]).map_or(budget * 1.5, |k| k as f64);
            oracle_savings.push(1.0 - needed / budget);
            println!(
                "{family:?} #{i}: start {start:.1} wpm → random {:.1} · heuristic {:.1} · ML {:.1} · oracle {:.1}  ({:.0}s)",
                finals[0], finals[1], finals[2], finals[3], t.elapsed().as_secs_f32()
            );
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    for (k, g) in kinds.iter().zip(&gains) {
        println!("mean gain {:<13} {:+.2} wpm", k.name(), mean(g));
    }
    println!("ML keystrokes saved vs random text:  {:.0}%", 100.0 * mean(&savings_vs[0]));
    println!("ML keystrokes saved vs weakest keys: {:.0}%", 100.0 * mean(&savings_vs[1]));
    println!("oracle keystrokes saved vs random (upper bound): {:.0}%", 100.0 * mean(&oracle_savings));
}
