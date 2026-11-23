//! The typist model: a small causal Transformer that predicts, for every upcoming
//! keystroke, a latency distribution, the chance of an error and which key gets hit
//! instead. An ensemble of heads (with randomised priors and a per-person correction
//! table) provides uncertainty.

use super::ENSEMBLE;
use super::encode::{EncodedLine, N_CONT, N_FINGERS, N_KEYS, N_LAYERS};
use crate::charset;
use crate::layout::LayoutId;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, relu, sigmoid};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

const D_MODEL: usize = 64;
const D_CHAR: usize = 32;
const D_KEY: usize = 16;
const D_SMALL: usize = 8;
/// log(150 ms): typical latency, the heads learn offsets from it.
const LOG_LATENCY_BIAS: f32 = 5.0;
const ERROR_LOGIT_BIAS: f32 = -3.2;

#[derive(Module, Debug)]
pub struct TypistModel {
    char_emb: Embedding,
    key_emb: Embedding,
    finger_emb: Embedding,
    layer_emb: Embedding,
    layout_emb: Embedding,
    pos_emb: Embedding,
    input: Linear,
    encoder: TransformerEncoder,
    mu_head: Linear,
    sigma_head: Linear,
    err_head: Linear,
    confusion_head: Linear,
    /// Per-person, per-character-pair corrections for each head (latency, error).
    personal: Embedding,
}

/// Model outputs for a padded batch.
pub struct Output {
    /// Mean of log latency, [B, T, E].
    pub mu: Tensor<3>,
    pub log_sigma: Tensor<3>,
    pub err_logit: Tensor<3>,
    /// Logits over keys for a wrong press, [B, T, K].
    pub confusion: Tensor<3>,
}

/// A batch as tensors (built from [`EncodedLine`]s).
pub struct Batch {
    pub chars: [Tensor<2, Int>; 3],
    pub keys: [Tensor<2, Int>; 2],
    pub fingers: [Tensor<2, Int>; 2],
    pub layer: Tensor<2, Int>,
    pub layout: Tensor<2, Int>,
    pub bigram: Tensor<2, Int>,
    pub positions: Tensor<2, Int>,
    pub cont: Tensor<3>,
    pub prior: Tensor<3>,
    pub pad: Tensor<2, Bool>,
    pub labels: Option<Labels>,
    pub lengths: Vec<usize>,
}

pub struct Labels {
    pub log_latency: Tensor<2>,
    pub latency_mask: Tensor<2>,
    pub error: Tensor<2>,
    pub label_mask: Tensor<2>,
    pub wrong_key: Tensor<2, Int>,
    pub wrong_mask: Tensor<2>,
}

impl TypistModel {
    pub fn new(device: &Device) -> TypistModel {
        let n_bigrams = charset::size() * charset::size();
        TypistModel {
            char_emb: EmbeddingConfig::new(charset::size(), D_CHAR).init(device),
            key_emb: EmbeddingConfig::new(N_KEYS, D_KEY).init(device),
            finger_emb: EmbeddingConfig::new(N_FINGERS, D_SMALL).init(device),
            layer_emb: EmbeddingConfig::new(N_LAYERS, D_SMALL).init(device),
            layout_emb: EmbeddingConfig::new(LayoutId::ALL.len(), D_SMALL).init(device),
            pos_emb: EmbeddingConfig::new(super::encode::MAX_LEN, D_MODEL).init(device),
            input: LinearConfig::new(3 * D_CHAR + 2 * D_KEY + 4 * D_SMALL + N_CONT, D_MODEL).init(device),
            encoder: TransformerEncoderConfig::new(D_MODEL, 2 * D_MODEL, 4, 2)
                .with_dropout(0.1)
                .with_norm_first(true)
                .init(device),
            mu_head: LinearConfig::new(D_MODEL, ENSEMBLE).init(device),
            sigma_head: LinearConfig::new(D_MODEL, ENSEMBLE).init(device),
            err_head: LinearConfig::new(D_MODEL, ENSEMBLE).init(device),
            confusion_head: LinearConfig::new(D_MODEL, N_KEYS).init(device),
            personal: EmbeddingConfig::new(n_bigrams, 2 * ENSEMBLE)
                .with_initializer(burn::nn::Initializer::Zeros)
                .init(device),
        }
    }

    pub fn forward(&self, b: &Batch) -> Output {
        let [batch, len] = b.pad.dims();
        let x = Tensor::cat(
            vec![
                self.char_emb.forward(b.chars[0].clone()),
                self.char_emb.forward(b.chars[1].clone()),
                self.char_emb.forward(b.chars[2].clone()),
                self.key_emb.forward(b.keys[0].clone()),
                self.key_emb.forward(b.keys[1].clone()),
                self.finger_emb.forward(b.fingers[0].clone()),
                self.finger_emb.forward(b.fingers[1].clone()),
                self.layer_emb.forward(b.layer.clone()),
                self.layout_emb.forward(b.layout.clone()),
                b.cont.clone(),
            ],
            2,
        );
        let x = self.input.forward(x) + self.pos_emb.forward(b.positions.clone());
        let device = x.device();
        let input = TransformerEncoderInput::new(x)
            .mask_pad(b.pad.clone())
            .mask_attn(generate_autoregressive_mask(batch, len, &device));
        let h = self.encoder.forward(input);
        let personal = self.personal.forward(b.bigram.clone()) + b.prior.clone();
        let p_mu = personal.clone().narrow(2, 0, ENSEMBLE);
        let p_err = personal.narrow(2, ENSEMBLE, ENSEMBLE);
        Output {
            mu: self.mu_head.forward(h.clone()) + p_mu + LOG_LATENCY_BIAS,
            log_sigma: (self.sigma_head.forward(h.clone()) - 1.1).clamp(-3.0, 1.0),
            err_logit: self.err_head.forward(h.clone()) + p_err + ERROR_LOGIT_BIAS,
            confusion: self.confusion_head.forward(h),
        }
    }

    /// Training loss: prediction loss plus the regulariser on personal weights.
    pub fn loss(&self, b: &Batch, boot: Tensor<3>) -> Tensor<1> {
        let l2 = self.personal.weight.val().powf_scalar(2.0).mean();
        self.data_loss(b, boot) + l2 * 50.0
    }

    /// Prediction loss only (latency NLL, error BCE, confusion CE). `boot` [B, T, E] are
    /// bootstrap weights so heads see different data. Model selection compares this, not
    /// the regulariser.
    pub fn data_loss(&self, b: &Batch, boot: Tensor<3>) -> Tensor<1> {
        let labels = b.labels.as_ref().expect("training batch has labels");
        let out = self.forward(b);
        let y = labels.log_latency.clone().unsqueeze_dim::<3>(2);
        let lat_w = labels.latency_mask.clone().unsqueeze_dim::<3>(2) * boot.clone();
        let z = (y - out.mu.clone()) / out.log_sigma.clone().exp();
        let nll = (z.powf_scalar(2.0) * 0.5 + out.log_sigma.clone()) * lat_w.clone();
        let nll = nll.sum() / (lat_w.sum() + 1.0);

        let err_y = labels.error.clone().unsqueeze_dim::<3>(2);
        let err_w = labels.label_mask.clone().unsqueeze_dim::<3>(2) * boot;
        let logit = out.err_logit;
        let bce = relu(logit.clone()) - logit.clone() * err_y + (logit.abs().neg().exp() + 1.0).log();
        let bce = (bce * err_w.clone()).sum() / (err_w.sum() + 1.0);

        let logp = log_softmax(out.confusion, 2);
        let picked = logp.gather(2, labels.wrong_key.clone().unsqueeze_dim::<3>(2)).squeeze_dim::<2>(2);
        let ce = (picked * labels.wrong_mask.clone()).sum().neg() / (labels.wrong_mask.clone().sum() + 1.0);

        nll + bce * 2.0 + ce * 0.2
    }
}

/// Per-token prediction summary (means and spread over the ensemble).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenPrediction {
    /// Expected time cost in ms including error correction.
    pub cost_ms: f32,
    /// Standard deviation of `cost_ms` across ensemble heads (model uncertainty).
    pub cost_sd: f32,
    pub latency_ms: f32,
    /// Mean predicted log latency (median latency = exp of this).
    pub log_latency: f32,
    pub error_p: f32,
}

/// Expected cost of an error: noticing it and pressing the right key (≈ two keystrokes).
pub const ERROR_COST_FACTOR: f32 = 2.0;

impl TypistModel {
    /// Predict all tokens of `lines` (no gradients). Returns one Vec per line.
    pub fn predict(&self, lines: &[EncodedLine], device: &Device) -> Vec<Vec<TokenPrediction>> {
        if lines.is_empty() {
            return Vec::new();
        }
        let batch = make_batch(lines, device);
        let out = self.forward(&batch);
        let [b, t, e] = out.mu.dims();
        let sigma2 = out.log_sigma.exp().powf_scalar(2.0);
        let latency = (out.mu.clone() + sigma2 * 0.5).exp();
        let err = sigmoid(out.err_logit);
        let cost = latency.clone() * (err.clone() * ERROR_COST_FACTOR + 1.0);
        let cost = to_vec(cost);
        let mu = to_vec(out.mu);
        let latency = to_vec(latency);
        let err = to_vec(err);
        let mut result = Vec::with_capacity(b);
        for (bi, len) in batch.lengths.iter().enumerate() {
            let mut line = Vec::with_capacity(*len);
            for ti in 0..*len {
                let base = (bi * t + ti) * e;
                let costs = &cost[base..base + e];
                let mean = costs.iter().sum::<f32>() / e as f32;
                let var = costs.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / e as f32;
                line.push(TokenPrediction {
                    cost_ms: mean,
                    cost_sd: var.sqrt(),
                    latency_ms: latency[base..base + e].iter().sum::<f32>() / e as f32,
                    log_latency: mu[base..base + e].iter().sum::<f32>() / e as f32,
                    error_p: err[base..base + e].iter().sum::<f32>() / e as f32,
                });
            }
            result.push(line);
        }
        result
    }
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 tensor data")
}

/// Pad and convert encoded lines into a batch.
pub fn make_batch(lines: &[EncodedLine], device: &Device) -> Batch {
    let b = lines.len();
    let t = lines.iter().map(|l| l.tokens.len()).max().unwrap_or(1).max(1);
    let n = b * t;
    let mut chars = [vec![0i64; n], vec![0i64; n], vec![0i64; n]];
    let mut keys = [vec![(N_KEYS - 1) as i64; n], vec![(N_KEYS - 1) as i64; n]];
    let mut fingers = [vec![10i64; n], vec![10i64; n]];
    let mut layer = vec![0i64; n];
    let mut layout = vec![0i64; n];
    let mut bigram = vec![0i64; n];
    let mut positions = vec![0i64; n];
    let mut cont = vec![0f32; n * N_CONT];
    let mut prior = vec![0f32; n * 2 * ENSEMBLE];
    let mut pad = vec![true; n];
    let labeled = lines.iter().all(|l| l.labels.is_some());
    let mut log_latency = vec![0f32; n];
    let mut latency_mask = vec![0f32; n];
    let mut error = vec![0f32; n];
    let mut label_mask = vec![0f32; n];
    let mut wrong_key = vec![0i64; n];
    let mut wrong_mask = vec![0f32; n];
    for (bi, line) in lines.iter().enumerate() {
        for (ti, tok) in line.tokens.iter().enumerate().take(t) {
            let i = bi * t + ti;
            for k in 0..3 {
                chars[k][i] = tok.chars[k];
            }
            for k in 0..2 {
                keys[k][i] = tok.keys[k];
                fingers[k][i] = tok.fingers[k];
            }
            layer[i] = tok.layer;
            layout[i] = tok.layout;
            bigram[i] = tok.bigram;
            positions[i] = ti as i64;
            cont[i * N_CONT..(i + 1) * N_CONT].copy_from_slice(&tok.cont);
            prior[i * 2 * ENSEMBLE..(i + 1) * 2 * ENSEMBLE].copy_from_slice(&tok.prior);
            pad[i] = false;
            if let Some(Some(label)) = line.labels.as_ref().map(|l| l[ti]) {
                label_mask[i] = 1.0;
                error[i] = label.error as u8 as f32;
                if let (Some(y), false) = (label.log_latency, label.error) {
                    log_latency[i] = y;
                    latency_mask[i] = 1.0;
                }
                if let (true, Some(k)) = (label.error, label.wrong_key) {
                    wrong_key[i] = k as i64;
                    wrong_mask[i] = 1.0;
                }
            }
        }
    }
    let int2 = |v: Vec<i64>| Tensor::<2, Int>::from_data(TensorData::new(v, [b, t]), device);
    let f2 = |v: Vec<f32>| Tensor::<2>::from_data(TensorData::new(v, [b, t]), device);
    let [c0, c1, c2] = chars;
    let [k0, k1] = keys;
    let [f0, f1] = fingers;
    Batch {
        chars: [int2(c0), int2(c1), int2(c2)],
        keys: [int2(k0), int2(k1)],
        fingers: [int2(f0), int2(f1)],
        layer: int2(layer),
        layout: int2(layout),
        bigram: int2(bigram),
        positions: int2(positions),
        cont: Tensor::<3>::from_data(TensorData::new(cont, [b, t, N_CONT]), device),
        prior: Tensor::<3>::from_data(TensorData::new(prior, [b, t, 2 * ENSEMBLE]), device),
        pad: Tensor::<2, Bool>::from_data(TensorData::new(pad, [b, t]), device),
        labels: labeled.then(|| Labels {
            log_latency: f2(log_latency),
            latency_mask: f2(latency_mask),
            error: f2(error),
            label_mask: f2(label_mask),
            wrong_key: int2(wrong_key),
            wrong_mask: f2(wrong_mask),
        }),
        lengths: lines.iter().map(|l| l.tokens.len().min(t)).collect(),
    }
}
