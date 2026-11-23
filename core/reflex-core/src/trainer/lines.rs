//! Assembling readable lines from weighted words, with variety guardrails.

use crate::content::{ContentKind, Corpus, Lang, Pool, Word, code_lines, real_world_line};
use crate::layout::Layout;
use crate::session::PlannedLine;
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use std::collections::{HashMap, VecDeque};

/// Target visible line length in characters.
pub const LINE_CHARS: usize = 54;
/// A word may not reappear within this many recently shown words.
const RECENT_WINDOW: usize = 120;

/// Words shown recently (guardrail against repetition).
#[derive(Debug, Default, Clone)]
pub struct RecentWords {
    words: VecDeque<String>,
    /// How often each word occurs in `words` (O(1) lookups when filtering 20k words).
    counts: HashMap<String, u32>,
}

impl RecentWords {
    pub fn contains(&self, w: &str) -> bool {
        self.counts.contains_key(w)
    }

    pub fn note_line(&mut self, text: &str) {
        for w in text.split_whitespace() {
            self.words.push_back(w.to_string());
            *self.counts.entry(w.to_string()).or_default() += 1;
            if self.words.len() > RECENT_WINDOW
                && let Some(old) = self.words.pop_front()
                && let Some(n) = self.counts.get_mut(&old)
            {
                *n -= 1;
                if *n == 0 {
                    self.counts.remove(&old);
                }
            }
        }
    }
}

/// Build a line by repeatedly sampling words proportional to `weight(index, word)`.
/// Words that can't be typed on `layout` or were shown recently are skipped.
pub fn weighted_word_line(
    words: &[Word],
    weight: impl Fn(usize, &Word) -> f64,
    layout: &Layout,
    recent: &RecentWords,
    lang: Lang,
    rng: &mut StdRng,
) -> PlannedLine {
    let candidates: Vec<(&Word, f64)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| layout.can_type(&w.text) && !recent.contains(&w.text))
        .map(|(i, w)| (w, weight(i, w).max(0.0)))
        .filter(|(_, wt)| *wt > 0.0)
        .collect();
    let mut line = String::new();
    let mut used: Vec<&str> = Vec::new();
    let total: f64 = candidates.iter().map(|(_, w)| w).sum();
    let mut guard = 0;
    while line.chars().count() < LINE_CHARS - 4 && guard < 200 && total > 0.0 {
        guard += 1;
        let mut r = rng.random::<f64>() * total;
        let mut pick = candidates[0].0;
        for (w, wt) in &candidates {
            r -= wt;
            if r <= 0.0 {
                pick = w;
                break;
            }
        }
        if used.contains(&pick.text.as_str()) {
            continue;
        }
        if line.chars().count() + pick.text.chars().count() + 1 > LINE_CHARS {
            if line.chars().count() > LINE_CHARS * 2 / 3 {
                break;
            }
            continue;
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&pick.text);
        used.push(&pick.text);
    }
    if line.is_empty() {
        line = fallback_words(layout);
    }
    PlannedLine::new(&line, ContentKind::Words, lang)
}

/// A natural sentence, chosen by `score` among a random sample (higher = better).
pub fn sentence_line(
    lang: Lang,
    layout: &Layout,
    score: impl Fn(&str) -> f64,
    sample: usize,
    rng: &mut StdRng,
) -> PlannedLine {
    let all = Corpus::get().sentences(lang);
    let best = (0..sample)
        .filter_map(|_| all.choose(rng))
        .filter(|s| s.chars().count() <= LINE_CHARS + 20 && layout.can_type(s))
        .max_by(|a, b| score(a).total_cmp(&score(b)));
    match best {
        Some(s) => PlannedLine::new(s, ContentKind::Sentence, lang),
        None => real_world(lang, layout, rng),
    }
}

/// Last resort when nothing else can be typed on `layout`: plain letters every layout has.
fn fallback_words(layout: &Layout) -> String {
    ["reflex", "asdf jkl", "test"].into_iter().find(|w| layout.can_type(w)).unwrap_or("a").to_string()
}

pub fn real_world(lang: Lang, layout: &Layout, rng: &mut StdRng) -> PlannedLine {
    for _ in 0..20 {
        let text = real_world_line(lang, rng);
        if layout.can_type(&text) {
            return PlannedLine::new(&text, ContentKind::RealWorld, lang);
        }
    }
    let text = if layout.can_type("12 + 30 = 42") { "12 + 30 = 42".to_string() } else { fallback_words(layout) };
    PlannedLine::new(&text, ContentKind::RealWorld, lang)
}

pub fn code_line(lang: Lang, layout: &Layout, score: impl Fn(&str) -> f64, rng: &mut StdRng) -> PlannedLine {
    let lines = code_lines();
    let best = (0..8)
        .filter_map(|_| lines.choose(rng))
        .filter(|l| layout.can_type(l))
        .max_by(|a, b| score(a).total_cmp(&score(b)))
        .map(|l| l.to_string())
        .unwrap_or_else(|| if layout.can_type("let x = 1;") { "let x = 1;".into() } else { fallback_words(layout) });
    PlannedLine::new(&best, ContentKind::Code, lang)
}

/// Mix of content kinds a trainer wants, as probabilities.
#[derive(Debug, Clone, Copy)]
pub struct Mix {
    pub sentence: f64,
    pub real_world: f64,
    pub code: f64,
}

impl Mix {
    pub fn pick(&self, pool: &Pool, rng: &mut StdRng) -> ContentKind {
        let code = if pool.code { self.code } else { 0.0 };
        let r = rng.random::<f64>();
        if r < self.sentence {
            ContentKind::Sentence
        } else if r < self.sentence + self.real_world {
            ContentKind::RealWorld
        } else if r < self.sentence + self.real_world + code {
            ContentKind::Code
        } else {
            ContentKind::Words
        }
    }
}
