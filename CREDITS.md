# Credits

## Bundled text data (`core/data/content`)

These files are not covered by Reflex's own LICENSE; their licenses apply.

- `de_words.tsv.zst`, `en_words.tsv.zst`: word frequency lists adapted (filtered,
  re-cased, truncated) from [FrequencyWords](https://github.com/hermitdave/FrequencyWords)
  by Hermit Dave, derived from OpenSubtitles 2018. Licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the adapted lists
  are shared under the same license.
- `de_sentences.txt.zst`, `en_sentences.txt.zst`: sentences from
  [Tatoeba](https://tatoeba.org) (filtered). Licensed
  [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).

## Everything else

Code, keyboard layout tables, code lines, real-world templates, the app icon and the
pretrained model weights (trained on simulated typists) were made for Reflex and fall
under its LICENSE.

Third-party Rust crates are dependencies only (not included in this repository); each
keeps its own license (mostly MIT/Apache-2.0, UniFFI under MPL-2.0).
