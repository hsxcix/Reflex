# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Build Reflex's bundled text content from raw downloads.

Inputs (in RAW dir): FrequencyWords {de,en}_50k.txt, Tatoeba {deu,eng}_sentences.tsv.bz2
Outputs (in core/data/content): {de,en}_words.tsv.zst (word<TAB>count, ranked),
{de,en}_sentences.txt.zst (one sentence per line).
"""

import bz2
import collections
import pathlib
import random
import re
import sys

from compression import zstd

RAW = pathlib.Path(sys.argv[1]).expanduser()
OUT = pathlib.Path(sys.argv[2]).expanduser()
WORDS_PER_LANG = 20_000
SENTENCES_PER_LANG = 30_000

LETTERS = {"de": "a-zäöüß", "en": "a-z"}
TYPOGRAPHY = str.maketrans({
    "„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‚": "'", "‘": "'", "’": "'",
    "–": "-", "—": "-", " ": " ", " ": " ",
})
SENTENCE_OK = re.compile(r"^[A-Za-z0-9ÄÖÜäöüß .,;:!?'\"()\-/%&€$@#+=*]+$")
# Kept deliberately short: filters the obvious, the frequency cut handles most of the rest.
BLOCKLIST = {
    "de": {"fick", "ficken", "fickt", "gefickt", "scheiße", "scheiß", "arsch", "arschloch", "hure",
           "nutte", "schlampe", "wichser", "fotze", "schwanz", "titten", "nigger", "neger", "schwuchtel",
           "verdammt", "kacke", "pisser", "missgeburt", "spast", "behindert", "hurensohn"},
    "en": {"fuck", "fucking", "fucked", "fucker", "shit", "bitch", "ass", "asshole", "dick", "cock",
           "pussy", "cunt", "whore", "slut", "bastard", "nigger", "nigga", "fag", "faggot", "damn",
           "retard", "motherfucker", "tits", "piss", "crap", "bullshit", "goddamn", "dumbass"},
}


def normalize(text: str) -> str:
    text = text.translate(TYPOGRAPHY).replace("…", "...")
    return re.sub(r"\s+", " ", text).strip()


def load_sentences(lang3: str) -> list[str]:
    with bz2.open(RAW / f"{lang3}_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        return [normalize(line.rstrip("\n").split("\t", 2)[2]) for line in f]


def casing_counts(sentences: list[str], letters: str) -> dict[str, collections.Counter]:
    """For each lowercase word, count the casings seen when not sentence-initial."""
    token = re.compile(rf"[{letters}A-ZÄÖÜ]+", re.IGNORECASE)
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for s in sentences:
        for i, m in enumerate(token.finditer(s)):
            if i == 0:
                continue
            w = m.group(0)
            counts[w.lower()][w] += 1
    return counts


def build_words(lang: str, sentences: list[str], names: set[str]) -> list[tuple[str, int]]:
    letters = LETTERS[lang]
    valid = re.compile(rf"^[{letters}]+$")
    casings = casing_counts(sentences, letters)
    words: list[tuple[str, int]] = []
    for line in (RAW / f"{lang}_50k.txt").read_text(encoding="utf-8").splitlines():
        word, _, count = line.partition(" ")
        if not valid.match(word) or word in BLOCKLIST[lang]:
            continue
        seen = casings.get(word)
        if not seen or sum(seen.values()) < 2:
            continue  # unseen in real sentences: likely subtitle noise
        if len(word) == 1 and word not in {"a", "i"}:
            continue
        form = seen.most_common(1)[0][0]
        if form.isupper() and len(form) > 1:
            continue  # acronyms
        if lang == "en" and form[0].isupper() and form != "I":
            continue  # proper nouns
        if lang == "de" and form[0].isupper():
            if word in names:
                continue  # capitalised in English too: a name, not a German noun
            form = form[0] + form[1:].lower()
        words.append((form, int(count)))
        if len(words) == WORDS_PER_LANG:
            break
    return words


def build_sentences(lang: str, sentences: list[str]) -> list[str]:
    block = BLOCKLIST[lang]
    token = re.compile(r"[^\W\d_]+")
    picked = []
    seen = set()
    for s in sentences:
        if not (18 <= len(s) <= 90) or not SENTENCE_OK.match(s) or s in seen:
            continue
        if any(t.lower() in block for t in token.findall(s)):
            continue
        seen.add(s)
        picked.append(s)
    random.Random(7).shuffle(picked)
    return picked[:SENTENCES_PER_LANG]


def write_zst(path: pathlib.Path, text: str) -> None:
    path.write_bytes(zstd.compress(text.encode("utf-8"), level=19))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    english = casing_counts(load_sentences("eng"), LETTERS["en"])
    names = {w for w, c in english.items() if c.most_common(1)[0][0][0].isupper()}
    for lang, lang3 in (("de", "deu"), ("en", "eng")):
        sentences = load_sentences(lang3)
        words = build_words(lang, sentences, names)
        picked = build_sentences(lang, sentences)
        write_zst(OUT / f"{lang}_words.tsv.zst", "".join(f"{w}\t{c}\n" for w, c in words))
        write_zst(OUT / f"{lang}_sentences.txt.zst", "".join(s + "\n" for s in picked))
        print(f"{lang}: {len(words)} words, {len(picked)} sentences; sample: {[w for w, _ in words[200:212]]}")


if __name__ == "__main__":
    main()
