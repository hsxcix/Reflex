//! (De)serialising the pool toggles, the only persisted user choices.

use crate::content::{Lang, Pool, Vocabulary};

pub fn format_pool(pool: &Pool) -> String {
    let langs: Vec<&str> = pool.languages.iter().map(|l| l.as_str()).collect();
    format!("{};{};{}", langs.join(","), pool.vocabulary.as_str(), pool.code)
}

pub fn parse_pool(s: &str) -> Option<Pool> {
    let mut parts = s.split(';');
    let languages: Vec<Lang> = parts.next()?.split(',').filter_map(Lang::parse).collect();
    let vocabulary = Vocabulary::parse(parts.next()?)?;
    let code = parts.next()? == "true";
    (!languages.is_empty()).then_some(Pool { languages, vocabulary, code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let p = Pool { languages: vec![Lang::En], vocabulary: Vocabulary::Full, code: true };
        assert_eq!(parse_pool(&format_pool(&p)), Some(p));
    }
}
