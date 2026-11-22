# Reflex

Self-learning typing trainer: portable Rust core (all logic + ML) and a native SwiftUI Mac app.
Plan: PLAN.md · decisions log: DECISIONS.md (append with date + reason when deciding something).

## Layout
- `core/reflex-core` — library (layouts, content, store, session, trainer, ML model, simulator, FFI via UniFFI). No macOS APIs.
- `core/reflex-cli` — `reflex-cli` for pretraining, simulation, benchmarks (GPU via Burn/wgpu).
- `core/uniffi-bindgen` — generates Swift bindings.
- `mac/` — SwiftPM app `ReflexMac` (no Xcode needed). Generated files (`Sources/ReflexMac/Generated`, FFI header/modulemap) are gitignored.
- `tools/build_content.py` — rebuilds `core/data/content` from raw downloads in `~/Apps/Reflex-data/raw` (`uv run tools/build_content.py ~/Apps/Reflex-data/raw core/data/content`).

## Build / run
- `./build.sh` builds core + app, installs `~/Applications/Reflex.app`, launches it with `open -g` (no focus steal). `--no-open` to skip launch.
- Core tests: `cd core && cargo test -p reflex-core`.
- Rust lives at `/opt/homebrew/opt/rustup/bin` (rustup via Homebrew; not on PATH by default).
- User data: `~/Library/Application Support/Reflex/reflex.sqlite` (every keystroke, never auto-deleted).

## ML workflow
- Pretrain: `cd core && ./target/release/reflex-cli train --typists 1000 --sessions 10 --steps 10000 --out data/models/typist.bpk` (GPU via Metal, ~15 min). Bump `MODEL_VERSION` (model/mod.rs) when inputs/architecture change.
- Bandit prior: `reflex-cli train-bandit` → `data/models/bandit_prior.json` (redo after a new model).
- Check: `reflex-cli diagnose` (gain estimates vs simulator truth), `reflex-cli bench --per-family 2 --batches 40` (trainers vs random/heuristic/oracle, ~10 min). Results go into DECISIONS.md.
- Simulator families: `Family::TRAIN` for pretraining only, `Family::HELD_OUT` for benchmarks only.

## Gotchas
- SwiftUI's `@State` is a macro whose plugin ships only with Xcode: use `@StateObject private var x = Local(value)` (LocalState.swift) instead. `@Observable`, `@StateObject`, `@Environment` work.
- ISO Macs report the key left of "1" as `kVK_ISO_Section` and the key left of "Z" as `kVK_ANSI_Grave`; `PhysicalKey` maps them by position.
- Correctness is judged on the produced character (macOS composes dead keys), not on key codes.
- Burn 0.22: move a model to a training device with `fork(&device.autodiff()).train()` (not `to_device`); load weights from bytes (`ModuleRecord::from_bytes`), files are read lazily otherwise.
- Window screenshots for checks: `screencapture -l <id>` (id via CGWindowList); may need screen-recording permission.
