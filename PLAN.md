# Reflex — Plan (v1: Typing)

Status: **v1 implemented** 22.11.2026 (plan rev. 4). Deviations and benchmark results: DECISIONS.md.

## Goal

The best typing trainer possible. You open it and type; a self-learning ML core decides everything (which words, keys, symbols, how much variety, how much natural text) so you get faster and more accurate on real text over months. No levels, no settings menu, only a few toggles on the main screen. Personal tool for now, built so it can be shared later.

Later (not v1): full-keyboard capture (shortcuts, F-keys, system keys), Linux UI, daily reminder, sharing/distribution (then: check Aalto terms with the authors, retrain without it if needed, notarisation).

## Autonomy rules for the implementation

- The build runs from start to finish **without asking**. Where something is unclear, decide yourself and log it with a reason in `DECISIONS.md`.
- Pre-approved: install Rust (rustup via Homebrew) and needed Rust/Python tooling (`uv run`, no global pip); download the Aalto 136M Keystrokes dataset (if the page requires a form or accepting terms, skip it and continue with simulation-only pretraining; note it in DECISIONS.md); create the private GitHub repo `hsxcix/Reflex` and push to it.
- Not needed now: Xcode (see Mac app build). Never use `open` without `-g` for test launches.
- Long jobs (pretraining, simulation) run in the background; keep the Mac awake while they run.
- Commit in logical steps; git identity per global CLAUDE.md.

## Architecture: portable core + native shell

```
┌──────────────────────────────┐      ┌───────────────────────────────┐
│ ReflexMac (Swift, SwiftUI)   │      │ later: Linux UI (GTK/Qt/TUI)  │
│ window, rendering, key input,│      │ same core; only UI + key-code │
│ charts, menus, animations    │      │ translation is new            │
└──────────────┬───────────────┘      └───────────────┬───────────────┘
               │ UniFFI-generated bindings (C ABI)    │
┌──────────────┴──────────────────────────────────────┴───────────────┐
│ reflex-core (Rust: std + SQLite + Burn, Unix-portable)              │
│ typist model · trainer · content · layouts · storage · stats ·      │
│ forecast · simulated typists · CLI (reflex-cli)                     │
└─────────────────────────────────────────────────────────────────────┘
```

Rule: **everything that computes lives in the core.** The UI shows what the core says, forwards key events with timestamps, and draws stats the core has already computed. The core decides whether a key was correct.

### reflex-core (Rust)
- No macOS APIs, no GUI deps. Dependencies: `burn` (ML; Metal/Vulkan/CUDA via wgpu/CubeCL, CPU fallback), `rusqlite` (bundled SQLite), `serde`, `rand`, `uniffi`, `zstd`. Builds and tests on macOS and Linux.
- The UI passes in the data folder; the core never looks up OS paths.
- Keys identified by **platform-neutral physical codes** (W3C/HID names: `KeyA`, `Semicolon`, `IntlBackslash`…) plus the produced character. Each UI maps its native key codes to these.
- Layouts are data files in the core: **`de-iso`, `us-ansi`, `eurkey`** (char ↔ code ↔ finger ↔ hand ↔ row ↔ modifiers needed, dead keys).
- Content embedded in the binary (compressed). Model weights shipped as files, fine-tuned copies stored in the data folder with checkpoints.
- `reflex-cli`: `sim`, `train`, `bench`, `stats`, `export`. Works without any UI.

### ReflexMac (Swift 6, SwiftUI, macOS 26+)
- **Build without Xcode for now:** SwiftPM package + `build.sh` (the SDK in the Command Line Tools has SwiftUI, Charts and AppKit; `actool`/`iconutil` handle the icon). The Rust core is linked as a static library (`aarch64-apple-darwin`) through a SwiftPM system-library target + the UniFFI-generated Swift file. An Xcode project can be added later if it ever helps (Instruments, previews); nothing in the design depends on it.
- **Layout auto-detection:** reads the active macOS input source (`TISCopyCurrentKeyboardInputSource`) and tells the core the layout id, live, when you switch. Unknown layout → fall back to the closest known one and show its name in the top bar.
- Key input through a custom `NSView` (first responder): `keyCode`, `characters`, modifiers, **`event.timestamp`** (hardware timing). Dead keys via `interpretKeyEvents`/`NSTextInputClient`.
- Native feel: window material, full screen / focus mode, Swift Charts, standard menus with shortcuts, window restoration, light/dark automatic, SF Mono, smooth caret animation, respects "Bewegung reduzieren".
- Ad-hoc signed `.app` in `~/Applications`. LICENSE "All rights reserved" + warranty disclaimer (same wording as AIUsageBar), no open-source license.

## Data

- **Every keystroke is stored forever** (SQLite): time (µs), expected/typed char, physical code, modifiers incl. which Shift, layout id, correct flag, latency, context, word, language, content type, time in session. ~30–50 bytes each → roughly 300 MB per year at heavy use. Sessions older than 30 days are compressed (zstd) into archive tables. Never deleted automatically.
- Pauses > 2 s are cut out of timing.
- Export: Ablage → Daten exportieren… (JSON + CSV).

## The algorithm (crown jewel): machine learning

All learning components are neural networks in **Burn**, running on the M5 GPU via Metal (Vulkan/CUDA on Linux, CPU fallback). Not MLX (Mac-only) and not the Neural Engine (needs Core ML, Mac-only, unnecessary at this model size).

### Data problem → three-stage training
1. **Pretraining on population data:** Aalto 136M Keystrokes (168k people, English) teaches general human typing: transitions, errors, speed distributions. Used privately for this personal tool.
2. **Pretraining on simulated typists:** a generator of synthetic typists with varied weaknesses, learning/forgetting rates and fatigue, on German-ISO, US-ANSI and EurKEY geometry. Covers German, umlauts, symbols and layouts the Aalto data lacks.
3. **On-device fine-tuning on you:** after every batch, a few gradient steps on your keystrokes (small learning rate + replay of pretraining data against forgetting). Runs in a background thread, a few seconds, never blocks typing.

### Typist model: "how will I type this?"
- Small **Transformer** (~0.5–2M parameters) reading the upcoming text and your recent history. Predicts per next keystroke: **latency distribution**, **error probability**, **which wrong key** is likely.
- Inputs: learned embeddings for character, physical key, finger, hand, row; **layout-specific embedding** on top of shared finger/position features (skill transfers when you switch layouts, only the new mappings get relearned); context window; session/fatigue signals; practice history per unit (counts, recency).
- **Deep ensemble** (5 heads on a shared trunk) for uncertainty: where heads disagree, the model doesn't know you yet → exploration.
- Learning and forgetting are **learned from data** via the history inputs, not hand-coded formulas.

### Objective
**Real speed** = the typist model's predicted time for a held-out sample of normal text in the active languages and layout, error corrections included. Shown as real WPM. Every decision maximises *expected gain in real speed per minute of practice*.

### Trainer: model-based planner + learned tuning (RL as challenger)
- **Planner:** for candidate lines (real words / sentences from the pool), the typist model simulates "you after practising this line" and estimates the real-speed gain. The line with the best gain (plus ensemble-based exploration bonus, Thompson-style) is shown next.
- **Learned tuning:** a contextual bandit (small neural value net) adjusts the planner's knobs (focus width, repetition spacing, word length, targeted vs. natural ratio) based on measured transfer to natural text. Over weeks it learns how *you* learn best.
- **RL challenger:** a policy trained with RL on simulated typists. It's only switched on if it beats the planner on typist families it never trained on (see Benchmarks). Otherwise it stays in the code as an experiment.
- **No stages:** capitals, punctuation, digits, symbols, umlauts appear when they pay off. Never a "level up".
- **Guardrails in code:** no repeat word within N lines, no single-bigram spam, readable lines, a fixed minimum share of natural text for honest measurement.
- **Start:** no calibration screen. Broad natural text at first; the pretrained models start targeting within minutes.
- **Fatigue:** when actual latency drifts above prediction over ~3 batches → subtle "break?" hint; tired data down-weighted in fine-tuning.

### Forecast
Typist model rolled forward with the trainer's expected practice → "~95 WPM in 4 weeks at your current pace" with an uncertainty band from the ensemble.

### Validation
Every fine-tune is checked on held-out recent keystrokes; worse → rollback to the previous checkpoint. A per-key-average baseline runs invisibly as a yardstick.

## Content
- Word frequency lists, top ~20k German + ~20k English, natural sentences (e.g. Tatoeba, CC BY), licenses checked and listed in `CREDITS.md`. Offensive words filtered, German noun capitalisation kept.
- **Built-in real-world text** (small share, chosen by the trainer): numbers, dates, prices, times, email addresses, URLs, common abbreviations, generated from templates.
- **Code snippets:** toggle on the main screen, off by default (short real-looking snippets, generated/bundled with a permissive license).

## UI / Design
Inspired by the Monkeytype feel (dark, calm, monospace, text centred, almost nothing else), but its own identity:
- Own palette (deep graphite + warm amber accent, light variant too), own caret, own type scale, native material background. App icon designed in the build (simple, matches the palette).
- **Main screen = typing screen.** 3 lines, current one centred; typed text dims, soft underline caret, a wrong key gives a short red flash + tiny shake. Typing starts with the first key.
- **Stop on error:** the cursor waits for the right key, the wrong one is recorded, no backspace.
- **Top bar** (fades while typing, back when you pause or move the mouse):
  - Language: `DE` `EN` (one or both; default both)
  - Vocabulary: `common` (1k) · `broad` (5k) · `full` (20k); default broad
  - `code` toggle (default off)
  - Detected layout name, live real WPM + accuracy
  The toggles only limit the pool; the trainer decides inside it.
- **Hesitation hint:** pause > 1.2 s on a key → faint keyboard of the detected layout fades in with the target key in its finger colour, then fades out.
- **Batch line** every ~45 s: WPM · accuracy · trend arrow. No popups. Small daily-minutes ring (target 15 min, informational only).
- **Stats (`Esc` / ⌘2):** real WPM history + forecast band, keyboard heatmap (speed/errors), slowest bigrams, top confusions, streak, practice time, current model confidence. `Tab` / ⌘1 back.
- No sounds, no reminders. UI text in English.

## Project structure
```
Reflex/
  core/
    reflex-core/        model/, trainer/, sim/, content/, layout/, store/, stats/, ffi/
    reflex-cli/
    data/               word lists, sentences, templates, layouts, model weights
  mac/                  SwiftPM package ReflexMac (Package.swift, Sources/, Resources/)
  tools/                content + Aalto preprocessing scripts (uv run)
  build.sh              cargo build → uniffi-bindgen → swift build → .app bundle → icon → sign
  CLAUDE.md, PLAN.md, DECISIONS.md, CREDITS.md, LICENSE, README.md
```
The Aalto raw data stays outside the repo (`~/Apps/Reflex-data/`); only trained weights go into `core/data`.

## Core API (sketch)
```
Engine::open(data_dir) -> Engine
engine.set_layout(layout_id)
engine.set_pool(languages, vocabulary, code)
engine.next_lines(n) -> [Line { text }]
engine.key_event(KeyEvent { code, chars, modifiers, timestamp_us }) -> KeyResult { correct, advance, flash }
engine.batch_summary() -> BatchSummary { wpm, accuracy, trend, fatigue_hint }
engine.stats() -> Stats { history, forecast, heatmap, slow_bigrams, confusions, streak, confidence }
engine.hint_for_current() -> Option<KeyHint { code, finger }>
engine.export(path)
```
Fine-tuning and planning run on core-owned background threads; all UI calls return immediately.

## Testing and benchmarks (no real typing needed)
- `cargo test`: shapes, losses, fine-tune lowers loss, ensemble uncertainty shrinks with data, checkpoint save/load/rollback, planner guardrails, layout maps (all three), dead keys, store + archive compression, FFI round trip.
- **Simulated typists**, with benchmark typists from **simulator families the models never trained on** (avoids circular tests). `reflex-cli bench` must show:
  1. the typist model beats the per-key baseline on held-out data (real Aalto sessions if available, plus simulated);
  2. the typist model finds planted weaknesses within N batches;
  3. the trainer reaches the target real speed with **≥30% fewer keystrokes** than random natural text;
  4. the trainer beats a hand-written weakest-keys-first heuristic;
  5. after a simulated layout switch, it recovers faster than a fresh model (transfer works).
  Tune until all pass; results go into `DECISIONS.md`.
- Swift: build check, key-code mapping test, layout detection, then a test launch with `open -g` and a scripted smoke test (synthetic key events) to check the typing loop end to end.
- Core built and tested once on Linux (Docker, CPU backend) if Docker is available; otherwise note it in DECISIONS.md.

## Implementation order (one run)
1. Rust toolchain, repo skeleton, LICENSE, CLAUDE.md, README, DECISIONS.md, private GitHub repo.
2. Core: layouts (de-iso, us-ansi, eurkey), store + archive, keystroke ingestion, content pipeline (downloads, cleaning, licenses).
3. **Playable early:** UniFFI + minimal Swift typing view with a simple interim selector, so the app is usable while the ML trains and your keystrokes are already recorded. Build into `~/Applications` at this point.
4. Core: simulated typist generator; Aalto download + preprocessing.
5. Core: typist model (Transformer + ensemble), pretraining (Aalto + sim), on-device fine-tuning, checkpoints/rollback.
6. Core: objective, planner, bandit tuning, guardrails, fatigue, forecast; RL challenger.
7. Benchmarks + baselines; tune until they pass. `reflex-cli` complete.
8. Swift: full UI (top bar, hesitation keyboard, batch line, stats with Charts, menus, full screen, icon, layout auto-detection).
9. Linux portability check, final build into `~/Applications`, smoke test, push, update CLAUDE.md with build/run instructions.
