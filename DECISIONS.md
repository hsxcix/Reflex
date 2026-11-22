# Decisions

Decisions made during the autonomous build (newest last), with reasons.

- **22.11.2026 · No external keystroke dataset.** The Aalto download was stopped on the user's request ("figure out how to do it yourself"). Pretraining uses Reflex's own simulated typists only; the models then learn from the user's real keystrokes. Upside: no third-party data license in the weights, so sharing later needs no retraining.
- **22.11.2026 · Content sources.** Word frequencies: FrequencyWords (OpenSubtitles 2018, CC BY-SA 4.0). Sentences: Tatoeba (CC BY 2.0 FR). Raw files in `~/Apps/Reflex-data/raw`, built by `tools/build_content.py`. German noun capitalisation recovered from Tatoeba casing, since the frequency lists are lowercase.
- **22.11.2026 · Correctness is judged on the produced character**, not on the expected key sequence. macOS composes dead keys itself; the core only needs the final character. The layout tables are used for features, hints and the on-screen keyboard.
- **22.11.2026 · Build without Xcode.** SwiftPM + `build.sh` with the Command Line Tools. SwiftUI's `@State` macro needs Xcode's plugin, so view-local state uses `@StateObject` + a tiny `Local<T>` class. Everything else (Charts, `@Observable`) works.
- **22.11.2026 · Burn 0.22 with the `flex` CPU backend in the app, wgpu (Metal) for pretraining.** The model is small (~200k params); on-device fine-tuning on the CPU takes a few seconds per batch in the background and avoids GPU start-up cost. GPU pretraining measured ~4× faster than CPU.
- **22.11.2026 · Practice is tracked per movement pair (key × modifier layer), not per character.** Skill then transfers between layouts exactly where movements match (e.g. `as` on German and U.S.), and only moved keys need relearning.
- **22.11.2026 · Model inputs include per-pair running speed/error, frozen at session start.** First benchmark showed the model found only ~10% of the truly most valuable pairs: it couldn't see how weak a pair currently is. Freezing the averages per session while practice counts keep running lets the network learn the effect of *more practice* given current weakness, which is exactly the planner's question.
- **22.11.2026 · German word list drops names** (words capitalised in English text too, e.g. "Samantha").
- **22.11.2026 · Fatigue hint uses measured speed drop**, not model residuals: simpler, robust, and the model already sees time-in-session as an input.
- **22.11.2026 · Personal model is saved only when an update improves held-out predictions** (rollback otherwise); weights are read fully into memory so the file can be replaced safely.
- **22.11.2026 · Exploration bonus on the same scale as gains.** Model uncertainty is aggregated per pair exactly like the predicted gain (frequency-weighted, per keystroke); before, a raw ms value drowned out the gain signal.
- **22.11.2026 · Bandit starts from a simulation prior.** Its first dozen batches used to be uniformly random, which wastes a new user's first sessions. `reflex-cli train-bandit` collects outcomes on simulated typists (training families only) into `bandit_prior.json`; each prior trial counts ¼ of a real one, so the user's own outcomes take over quickly.
- **22.11.2026 · Benchmark includes an oracle** (planner fed the simulator's true gains) as the upper bound: it shows how much any trainer can gain over random text in the simulator, and how much of that the ML captures.
- **22.11.2026 · Sentences with names are skipped** (Tatoeba repeats Tom, Mary, Ziri constantly).
- **22.11.2026 · Model v3 shipped** (1000 simulated typists, 17.9 M keystrokes, 10k GPU steps). "What if practised more" is asked with +12 repetitions (clearer signal than +4: top-30 overlap with the true most valuable pairs rose from ~30% to 37–50%).
- **22.11.2026 · Benchmark results (honest).** Held-out typist families, 40 batches each:
  - Prediction: fine-tuned model RMSE(log latency) 0.329 vs 0.406 for per-pair averages (−19%); pretrained-only 0.373.
  - Training: ML +3.94 WPM vs +3.50 for random text and for the weakest-keys heuristic; ML ahead for every simulated person. Keystrokes saved vs random text: 23–28% across runs (target was ≥30%, **not reached yet**). Oracle upper bound: 48–52%.
  - The remaining gap is in the gain estimates, not the planner or bandit (fixed-arm and bandit runs are within noise). Next lever: a model head trained directly on future improvement after practice.
- **22.11.2026 · Not done in v1:** RL challenger policy (the neural bandit is the learned policy layer for now), layout-switch transfer benchmark, Linux test run (no Docker on this Mac; the core has no OS-specific code).
