//! Training loops: pretraining on simulated typists and on-device fine-tuning with
//! replay, validation and rollback.

use super::ENSEMBLE;
use super::encode::EncodedLine;
use super::typist::{TypistModel, make_batch};
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Poisson};

pub struct TrainParams {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Share of each batch drawn from `replay` (guards against forgetting).
    pub replay_share: f64,
}

/// Run `steps` optimisation steps on `data` (+ replay). The model must live on an
/// autodiff device. Calls `progress(step, loss)` every 50 steps.
pub fn train(
    mut model: TypistModel,
    data: &[EncodedLine],
    replay: &[EncodedLine],
    params: &TrainParams,
    device: &Device,
    rng: &mut StdRng,
    mut progress: impl FnMut(usize, f32),
) -> TypistModel {
    if data.is_empty() {
        return model;
    }
    let mut optimizer = AdamConfig::new().with_epsilon(1e-6).init();
    let poisson = Poisson::new(1.0).expect("valid rate");
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut cursor = order.len();
    for step in 0..params.steps {
        let mut lines: Vec<EncodedLine> = Vec::with_capacity(params.batch_size);
        for _ in 0..params.batch_size {
            if !replay.is_empty() && rng.random::<f64>() < params.replay_share {
                lines.push(replay[rng.random_range(0..replay.len())].clone());
            } else {
                if cursor >= order.len() {
                    order.shuffle(rng);
                    cursor = 0;
                }
                lines.push(data[order[cursor]].clone());
                cursor += 1;
            }
        }
        let batch = make_batch(&lines, device);
        let [b, t] = batch.pad.dims();
        let boot: Vec<f32> = (0..b * t * ENSEMBLE).map(|_| poisson.sample(rng) as f32).collect();
        let boot = Tensor::<3>::from_data(TensorData::new(boot, [b, t, ENSEMBLE]), device);
        let loss = model.loss(&batch, boot);
        if step % 50 == 0 || step + 1 == params.steps {
            progress(step, scalar(&loss));
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(params.lr, model, grads);
    }
    model
}

/// Mean prediction loss on `data` without bootstrap noise or regulariser (lower = better
/// predictions). Can be negative (Gaussian NLL).
pub fn evaluate(model: &TypistModel, data: &[EncodedLine], device: &Device) -> f32 {
    if data.is_empty() {
        return f32::NAN;
    }
    let model = model.valid();
    let mut total = 0.0;
    let mut n = 0.0;
    for chunk in data.chunks(64) {
        let batch = make_batch(chunk, device);
        let [b, t] = batch.pad.dims();
        let ones = Tensor::<3>::ones([b, t, ENSEMBLE], device);
        total += scalar(&model.data_loss(&batch, ones)) * chunk.len() as f32;
        n += chunk.len() as f32;
    }
    total / n
}

fn scalar(t: &Tensor<1>) -> f32 {
    t.to_data().convert::<f32>().try_into_vec::<f32>().map_or(f32::NAN, |v| v[0])
}
