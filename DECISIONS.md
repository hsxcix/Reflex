# Decisions

Decisions made during the autonomous build (newest last), with reasons.

- **22.11.2026 · No external keystroke dataset.** The Aalto download was stopped on the user's request ("figure out how to do it yourself"). Pretraining uses Reflex's own simulated typists only; the models then learn from the user's real keystrokes. Upside: no third-party data license in the weights, so sharing later needs no retraining.
- **22.11.2026 · Content sources.** Word frequencies: FrequencyWords (OpenSubtitles 2018, CC BY-SA 4.0). Sentences: Tatoeba (CC BY 2.0 FR). Raw files in `~/Apps/Reflex-data/raw`, built by `tools/build_content.py`. German noun capitalisation recovered from Tatoeba casing, since the frequency lists are lowercase.
- **22.11.2026 · Correctness is judged on the produced character**, not on the expected key sequence. macOS composes dead keys itself; the core only needs the final character. The layout tables are used for features, hints and the on-screen keyboard.
