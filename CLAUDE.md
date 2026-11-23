# Reflex

Self-learning typing trainer: portable Rust core (all logic + ML) and a native SwiftUI Mac app.
Plan: PLAN.md · decisions log: DECISIONS.md (append with date + reason when deciding something).

## Layout
- `core/reflex-core` — library (layouts, content, store, session, trainer, ML model, simulator, FFI via UniFFI). No macOS APIs.
- `core/reflex-cli` — `reflex-cli` for pretraining, simulation, benchmarks (GPU via Burn/wgpu).
- `core/uniffi-bindgen` — generates Swift bindings.
- `mac/` — SwiftPM app `ReflexMac` (no Xcode needed). Generated files (`Sources/ReflexMac/Generated`, FFI header/modulemap) are gitignored.
- `tools/build_content.py` — rebuilds `core/data/content` from raw downloads (URLs in its docstring; download into a temp folder, then `uv run tools/build_content.py <raw> core/data/content`). The raw folder `~/Apps/Reflex-data` was deleted 23.11.2026.

## Build / run
- `./build.sh` builds core + app, installs `~/Applications/Reflex.app`, launches it with `open -g` (no focus steal). `--no-open` to skip launch, `--smoke` to run the app check instead. Env: `REFLEX_INSTALL_DIR`, `CARGO_HOME`.
- App check: `tools/smoke.sh [app]` starts the app with `--smoke-test` against a temp store (synthetic NSEvents through `KeyCaptureNSView`: one line incl. a wrong press, an ignored auto-repeat, stats load, CSV+JSON export verified). `REFLEX_DATA=<dir>` points the app at another data folder.
- Core tests: `cd core && cargo test -p reflex-core`.
- Layout tables vs real macOS layouts: `tools/check_layouts.sh` (run after editing `core/data/layouts`).
- Rust lives at `/opt/homebrew/opt/rustup/bin` (rustup via Homebrew; not on PATH by default).
- User data: `~/Library/Application Support/Reflex/reflex.sqlite` (every keystroke, never auto-deleted; schema v4: `open_keys` while typing, `abandoned_keys` after a quit or layout switch mid-line, per-batch `utc_offset_min` and `probe_wpm`). Each migration step runs in one transaction with its version bump. Stats and exports use `Store::reader()` (read-only connection) outside the engine lock. `reflex.lock` allows one instance per data folder.
- Real speed: reference sentences (`Corpus::reference_sentences`) are never practised and split in halves (`gains::ReferenceSet`): `Model` half = the model's estimate/gain table, `Probe` half = measurement lines (every 8th line, `trainer::probe`, scored against the frozen pretrained model's difficulty) and the simulator's ground truth in benchmarks. Days use the local UTC offset passed in from the app.
- Gain table values practice at horizons 1/7/30 days (practised now vs. left alone), so pairs about to fade get scheduled; `fading_pairs` feeds the stats panel. Bandit arms switch per block (ends at a probe or after 6 min); reward = model change on the same model before/after + observed probe change, weighted n/(n+300).
- Worker replies are tagged (`Reply::Batch { batch_id }`); only those touch a batch row. Forecast/context jobs are coalesced.

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
