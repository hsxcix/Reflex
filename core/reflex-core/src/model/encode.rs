//! Turning text + practice history into model inputs (and typed lines into labels).

use super::ENSEMBLE;
use crate::charset;
use crate::history::PracticeHistory;
use crate::layout::{KEY_CODES, Layout};
use crate::record::TypedLine;

pub const MAX_LEN: usize = 96;
pub const N_CONT: usize = 19;
/// Key vocabulary: all physical keys plus "unknown".
pub const N_KEYS: usize = KEY_CODES.len() + 1;
pub const N_FINGERS: usize = 11;
pub const N_LAYERS: usize = 8;

/// Inputs for one token (one character to type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token {
    pub chars: [i64; 3],
    pub keys: [i64; 2],
    pub fingers: [i64; 2],
    pub layer: i64,
    pub layout: i64,
    pub bigram: i64,
    pub cont: [f32; N_CONT],
    pub prior: [f32; 2 * ENSEMBLE],
}

/// Training target for one token.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Label {
    pub log_latency: Option<f32>,
    pub error: bool,
    pub wrong_key: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedLine {
    pub tokens: Vec<Token>,
    /// `None` for prediction-only lines; otherwise one entry per token (`None` = unlabeled).
    pub labels: Option<Vec<Option<Label>>>,
}

/// Context needed to encode a line.
pub struct EncodeContext<'a> {
    pub layout: &'a Layout,
    pub history: &'a PracticeHistory,
    pub now_ms: i64,
    pub session_sec: f32,
}

/// Index of the history columns inside `Token::cont` (the planner edits these).
pub const CONT_LOG_COUNT: usize = 10;
pub const CONT_LOG_DAYS: usize = 11;
pub const CONT_NEVER: usize = 12;
const CONT_SESSION: usize = 15;

pub fn encode_text(text: &str, prev_char: Option<char>, ctx: &EncodeContext) -> Vec<Token> {
    let chars: Vec<char> = text.chars().take(MAX_LEN).collect();
    let mut tokens = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        let p1 = if i == 0 { prev_char } else { Some(chars[i - 1]) };
        let p2 = match i {
            0 => None,
            1 => prev_char,
            _ => Some(chars[i - 2]),
        };
        tokens.push(token(c, p1, p2, i, ctx.session_sec, ctx));
    }
    tokens
}

fn token(c: char, p1: Option<char>, p2: Option<char>, pos: usize, session_sec: f32, ctx: &EncodeContext) -> Token {
    let layout = ctx.layout;
    let strokes = layout.strokes(c);
    let main = strokes.and_then(|s| s.last().copied());
    let prev_main = p1.and_then(|p| layout.main_stroke(p));
    let g = main.map(|s| layout.key(s.key));
    let pg = prev_main.map(|s| layout.key(s.key));
    let key_id = |s: Option<crate::layout::Stroke>| s.map_or(KEY_CODES.len() as i64, |s| s.key as i64);
    let finger_id = |g: Option<crate::layout::KeyGeometry>| g.map_or(10, |g| g.finger.index() as i64);
    let dead = strokes.is_some_and(|s| s.len() > 1);
    let layer = main.map_or(0, |s| s.layer as i64) + if dead { 4 } else { 0 };

    let (same_finger, same_hand, same_key, row_dist, x_dist) = match (g, pg, main, prev_main) {
        (Some(g), Some(pg), Some(m), Some(pm)) => (
            (g.finger == pg.finger && m.key != pm.key) as u8 as f32,
            (g.finger.hand() == pg.finger.hand()) as u8 as f32,
            (m.key == pm.key) as u8 as f32,
            g.row.abs_diff(pg.row) as f32 / 4.0,
            (g.x - pg.x).abs() / 8.0,
        ),
        _ => (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let h = ctx.history.features(layout, p1, c, ctx.now_ms);
    let cont = [
        same_finger,
        same_hand,
        same_key,
        row_dist,
        x_dist,
        (p1 == Some(' ')) as u8 as f32,
        c.is_uppercase() as u8 as f32,
        c.is_ascii_digit() as u8 as f32,
        (c.is_ascii_punctuation() || "€§°".contains(c)) as u8 as f32,
        pos as f32 / 80.0,
        h.log_bigram_count / 8.0,
        h.log_days_since / 6.0,
        h.never_seen,
        h.log_char_count / 10.0,
        h.log_total / 16.0,
        session_sec / 1800.0,
        h.relative_latency,
        h.error_rate * 5.0,
        h.evidence,
    ];
    let c_id = charset::id(c) as i64;
    let p1_id = charset::id_or_start(p1) as i64;
    let bigram = p1_id * charset::size() as i64 + c_id;
    Token {
        chars: [c_id, p1_id, charset::id_or_start(p2) as i64],
        keys: [key_id(main), key_id(prev_main)],
        fingers: [finger_id(g), finger_id(pg)],
        layer,
        layout: layout.id.index() as i64,
        bigram,
        cont,
        prior: prior_for(bigram),
    }
}

/// Fixed random prior per (bigram, ensemble head): heads disagree on pairs the person
/// hasn't typed yet, and agree once data overrides the prior ("randomised priors").
fn prior_for(bigram: i64) -> [f32; 2 * ENSEMBLE] {
    let mut out = [0.0; 2 * ENSEMBLE];
    for (i, v) in out.iter_mut().enumerate() {
        let z = gaussian(splitmix(bigram as u64 * 31 + i as u64));
        *v = if i < ENSEMBLE { 0.12 * z } else { 0.45 * z };
    }
    out
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn gaussian(seed: u64) -> f32 {
    let u1 = ((seed >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
    let u2 = (splitmix(seed) >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Encode a person's typed lines in order, replaying their practice history so each
/// line sees exactly what had been practised before it.
pub fn encode_history(lines: &[TypedLine], history: &mut PracticeHistory) -> Vec<EncodedLine> {
    let mut out = Vec::with_capacity(lines.len());
    let mut layout_cache: Option<Layout> = None;
    for line in lines {
        if layout_cache.as_ref().is_none_or(|l| l.id != line.layout || l.geometry != line.geometry) {
            layout_cache = Some(Layout::new(line.layout, line.geometry));
        }
        let layout = layout_cache.as_ref().unwrap();
        out.push(encode_typed_line(line, layout, history));
        record_line(history, layout, line);
    }
    out
}

pub fn encode_typed_line(line: &TypedLine, layout: &Layout, history: &PracticeHistory) -> EncodedLine {
    let session_sec = line.chars.first().map_or(0.0, |c| c.session_sec);
    let ctx = EncodeContext { layout, history, now_ms: line.started_ms(), session_sec };
    let mut tokens = encode_text(&line.text, line.prev_char, &ctx);
    let mut labels = vec![None; tokens.len()];
    for rec in &line.chars {
        let pos = rec.pos as usize;
        if pos >= tokens.len() {
            continue;
        }
        tokens[pos].cont[CONT_SESSION] = rec.session_sec / 1800.0;
        labels[pos] = Some(Label {
            log_latency: rec.latency_ms.filter(|ms| *ms > 20.0 && *ms < 2000.0).map(f32::ln),
            error: rec.error,
            wrong_key: rec.wrong_key,
        });
    }
    EncodedLine { tokens, labels: Some(labels) }
}

/// Add a typed line's keystrokes to `history`.
pub fn record_line(history: &mut PracticeHistory, layout: &Layout, line: &TypedLine) {
    let mut prev = line.prev_char;
    let mut recs = line.chars.iter().peekable();
    for (pos, c) in line.text.chars().enumerate() {
        while recs.peek().is_some_and(|r| (r.pos as usize) < pos) {
            recs.next();
        }
        if let Some(rec) = recs.peek().filter(|r| r.pos as usize == pos) {
            history.record(layout, prev, c, rec.wall_ms, rec.latency_ms, rec.error);
        }
        prev = Some(c);
    }
}
