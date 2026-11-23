//! Bundled text: word frequency lists, natural sentences, real-world snippets
//! (numbers, dates, addresses) and code lines.

mod code;
mod realworld;

use rand::Rng;
use rand::seq::IndexedRandom;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    De,
    En,
}

impl Lang {
    pub const ALL: [Lang; 2] = [Lang::De, Lang::En];

    pub fn as_str(self) -> &'static str {
        match self {
            Lang::De => "de",
            Lang::En => "en",
        }
    }

    pub fn parse(s: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.as_str() == s)
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// How deep into the frequency list words may be drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vocabulary {
    Common,
    Broad,
    Full,
}

impl Vocabulary {
    pub fn size(self) -> usize {
        match self {
            Vocabulary::Common => 1_000,
            Vocabulary::Broad => 5_000,
            Vocabulary::Full => 20_000,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Vocabulary::Common => "common",
            Vocabulary::Broad => "broad",
            Vocabulary::Full => "full",
        }
    }

    pub fn parse(s: &str) -> Option<Vocabulary> {
        [Vocabulary::Common, Vocabulary::Broad, Vocabulary::Full].into_iter().find(|v| v.as_str() == s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    Words,
    Sentence,
    RealWorld,
    Code,
}

impl ContentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ContentKind::Words => "words",
            ContentKind::Sentence => "sentence",
            ContentKind::RealWorld => "realworld",
            ContentKind::Code => "code",
        }
    }

    pub fn parse(s: &str) -> Option<ContentKind> {
        [ContentKind::Words, ContentKind::Sentence, ContentKind::RealWorld, ContentKind::Code]
            .into_iter()
            .find(|k| k.as_str() == s)
    }

    /// Natural text is what "real speed" is measured on.
    pub fn is_natural(self) -> bool {
        matches!(self, ContentKind::Sentence)
    }
}

/// The user's restriction of the content pool (the only "settings" Reflex has).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pool {
    pub languages: Vec<Lang>,
    pub vocabulary: Vocabulary,
    pub code: bool,
}

impl Default for Pool {
    fn default() -> Pool {
        Pool { languages: Lang::ALL.to_vec(), vocabulary: Vocabulary::Broad, code: false }
    }
}

impl Pool {
    pub fn pick_language(&self, rng: &mut impl Rng) -> Lang {
        *self.languages.choose(rng).unwrap_or(&Lang::En)
    }
}

#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    /// Relative frequency within its language (sums to 1 over the full list).
    pub freq: f64,
}

pub struct Corpus {
    words: [Vec<Word>; 2],
    /// Sentences the trainer may serve as practice.
    sentences: [Vec<String>; 2],
    /// Held-out sentences that only measure real speed (never practised).
    reference: [Vec<String>; 2],
}

/// Roughly one sentence in [`REFERENCE_EVERY`] is held out for measuring real speed,
/// chosen by a stable hash of its text so the split never changes between runs.
const REFERENCE_EVERY: u64 = 25;

fn is_reference(sentence: &str) -> bool {
    // FNV-1a: stable across Rust versions, unlike `DefaultHasher`.
    let hash = sentence.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    hash % REFERENCE_EVERY == 0
}

impl Corpus {
    pub fn get() -> &'static Corpus {
        static CORPUS: OnceLock<Corpus> = OnceLock::new();
        CORPUS.get_or_init(Corpus::load)
    }

    fn load() -> Corpus {
        let words = |bytes: &[u8]| {
            let text = decompress(bytes);
            let raw: Vec<(String, f64)> = text
                .lines()
                .filter_map(|l| l.split_once('\t'))
                .map(|(w, c)| (w.to_string(), c.parse::<f64>().unwrap_or(1.0)))
                .collect();
            let total: f64 = raw.iter().map(|(_, c)| c).sum();
            raw.into_iter().map(|(text, c)| Word { text, freq: c / total }).collect()
        };
        let split = |bytes: &[u8]| -> (Vec<String>, Vec<String>) {
            decompress(bytes).lines().map(str::to_string).partition(|s| !is_reference(s))
        };
        let (de, de_ref) = split(include_bytes!("../../../data/content/de_sentences.txt.zst"));
        let (en, en_ref) = split(include_bytes!("../../../data/content/en_sentences.txt.zst"));
        Corpus {
            words: [
                words(include_bytes!("../../../data/content/de_words.tsv.zst")),
                words(include_bytes!("../../../data/content/en_words.tsv.zst")),
            ],
            sentences: [de, en],
            reference: [de_ref, en_ref],
        }
    }

    pub fn words(&self, lang: Lang, vocabulary: Vocabulary) -> &[Word] {
        let all = &self.words[lang.index()];
        &all[..vocabulary.size().min(all.len())]
    }

    /// Practice sentences (the held-out reference set is excluded).
    pub fn sentences(&self, lang: Lang) -> &[String] {
        &self.sentences[lang.index()]
    }

    /// Held-out sentences for measuring real speed; never served as practice.
    pub fn reference_sentences(&self, lang: Lang) -> &[String] {
        &self.reference[lang.index()]
    }
}

fn decompress(bytes: &[u8]) -> String {
    let raw = zstd::decode_all(bytes).expect("bundled content is valid zstd");
    String::from_utf8(raw).expect("bundled content is UTF-8")
}

pub use code::code_lines;
pub use realworld::real_world_line;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_loads() {
        let c = Corpus::get();
        assert_eq!(c.words(Lang::De, Vocabulary::Common).len(), 1000);
        assert!(c.sentences(Lang::En).len() > 10_000);
        let total: f64 = c.words(Lang::En, Vocabulary::Full).iter().map(|w| w.freq).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reference_sentences_are_held_out() {
        let c = Corpus::get();
        for lang in Lang::ALL {
            let reference = c.reference_sentences(lang);
            assert!(reference.len() > 300, "{lang:?}: {}", reference.len());
            let practice: std::collections::HashSet<&String> = c.sentences(lang).iter().collect();
            assert!(reference.iter().all(|s| !practice.contains(s)));
        }
    }
}
