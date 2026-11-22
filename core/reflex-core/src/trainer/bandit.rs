//! Learned tuning: a small neural contextual bandit that learns which planner settings
//! (arms) actually raise *this* person's real speed. An ensemble of value heads gives
//! Thompson sampling: pick a random head, play its best arm.

use super::planner::ARMS;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const HEADS: usize = 4;
const N_CONTEXT: usize = 3;
const HIDDEN: usize = 24;

/// One observed outcome.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub context: [f32; N_CONTEXT],
    pub arm: usize,
    /// Real-speed change per minute of practice (WPM/min).
    pub reward: f32,
}

#[derive(Module, Debug)]
struct ValueNet {
    hidden: Linear,
    out: Linear,
}

impl ValueNet {
    fn new(device: &Device) -> ValueNet {
        ValueNet {
            hidden: LinearConfig::new(N_CONTEXT + ARMS.len(), HIDDEN).init(device),
            out: LinearConfig::new(HIDDEN, HEADS).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.out.forward(relu(self.hidden.forward(x)))
    }
}

pub struct Bandit {
    net: ValueNet,
    device: Device,
    trials: Vec<Trial>,
    /// Outcomes from simulated people: a starting belief, outweighed by own trials over time.
    prior: Vec<Trial>,
}

/// Weight of one prior trial relative to one of the person's own trials.
const PRIOR_WEIGHT: f32 = 0.25;

/// Context the bandit conditions on.
pub fn context(total_keystrokes: f64, real_wpm: f64, uncertainty: f32) -> [f32; N_CONTEXT] {
    [(total_keystrokes as f32).ln_1p() / 12.0, real_wpm as f32 / 100.0, uncertainty]
}

impl Bandit {
    pub fn new(device: &Device) -> Bandit {
        Bandit { net: ValueNet::new(device), device: device.clone(), trials: Vec::new(), prior: Vec::new() }
    }

    pub fn trials(&self) -> &[Trial] {
        &self.trials
    }

    fn features(context: &[f32; N_CONTEXT], arm: usize) -> Vec<f32> {
        let mut x = context.to_vec();
        x.extend((0..ARMS.len()).map(|a| (a == arm) as u8 as f32));
        x
    }

    /// Thompson sampling over ensemble heads; explores uniformly while data is scarce.
    pub fn choose(&self, context: [f32; N_CONTEXT], rng: &mut StdRng) -> usize {
        let evidence = self.trials.len() + self.prior.len();
        if evidence < ARMS.len() * 2 || rng.random::<f64>() < 0.08 {
            return rng.random_range(0..ARMS.len());
        }
        let head = rng.random_range(0..HEADS);
        let x: Vec<f32> = (0..ARMS.len()).flat_map(|a| Self::features(&context, a)).collect();
        let x = Tensor::<2>::from_data(TensorData::new(x, [ARMS.len(), N_CONTEXT + ARMS.len()]), &self.device);
        let values = self.net.valid().forward(x).narrow(1, head, 1);
        let values = values.into_data().convert::<f32>().try_into_vec::<f32>().unwrap_or_default();
        values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
    }

    /// Add an outcome and refit (tiny network, a few dozen steps).
    pub fn record(&mut self, trial: Trial, rng: &mut StdRng) {
        self.trials.push(trial);
        self.fit(rng);
    }

    pub fn restore(&mut self, trials: Vec<Trial>, prior: Vec<Trial>, rng: &mut StdRng) {
        self.trials = trials;
        self.prior = prior;
        if !self.trials.is_empty() || !self.prior.is_empty() {
            self.fit(rng);
        }
    }

    fn fit(&mut self, rng: &mut StdRng) {
        let device = self.device.clone().autodiff();
        let mut net = self.net.clone().fork(&device).train();
        let mut optim = AdamConfig::new().init();
        let own = self.trials.iter().rev().take(400).map(|t| (t, 1.0));
        let prior = self.prior.iter().map(|t| (t, PRIOR_WEIGHT));
        let data: Vec<(&Trial, f32)> = own.chain(prior).collect();
        let n = data.len();
        let x: Vec<f32> = data.iter().flat_map(|(t, _)| Self::features(&t.context, t.arm)).collect();
        let y: Vec<f32> = data.iter().map(|(t, _)| t.reward.clamp(-5.0, 5.0)).collect();
        let weight: Vec<f32> = data.iter().map(|(_, w)| *w).collect();
        let x = Tensor::<2>::from_data(TensorData::new(x, [n, N_CONTEXT + ARMS.len()]), &device);
        let y = Tensor::<2>::from_data(TensorData::new(y, [n, 1]), &device);
        for _ in 0..40 {
            // Bootstrap: each head sees a different resample of the trials.
            let w: Vec<f32> = (0..n * HEADS)
                .map(|i| if rng.random::<f64>() < 0.632 { weight[i / HEADS] } else { 0.0 })
                .collect();
            let w = Tensor::<2>::from_data(TensorData::new(w, [n, HEADS]), &device);
            let pred = net.forward(x.clone());
            let loss = ((pred - y.clone()).powf_scalar(2.0) * w.clone()).sum() / (w.sum() + 1.0);
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = optim.step(3e-3, net, grads);
        }
        self.net = net.valid().to_device(&self.device);
    }
}
