//! End to end: type a full batch through the engine, then check storage and the learning
//! worker's update (planner + real-speed estimate).

use reflex_core::brain::Brain;
use reflex_core::content::{ContentKind, Pool};
use reflex_core::engine::Engine;
use reflex_core::history::{PracticeHistory, pair_key};
use reflex_core::layout::Layout;
use reflex_core::trainer::gains::compute_gains;
use reflex_core::layout::{Geometry, LayoutId, key_id};
use reflex_core::session::KeyInput;
use reflex_core::store::{KeyStatus, StoreError};
use std::path::Path;
use std::time::{Duration, Instant};

#[test]
fn typing_a_batch_trains_and_plans() {
    let dir = tempfile::tempdir().unwrap();
    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut engine = Engine::open(dir.path(), &resources, 60).unwrap();
    engine.set_layout(LayoutId::DeIso, Geometry::Iso).unwrap();
    let mut t_us = 1_000_000i64;
    let mut batch = None;
    let mut keys = 0;
    while batch.is_none() && keys < 5000 {
        let c = engine.expected().expect("there is always a next character");
        let layout = engine.layout().clone();
        let code = layout.main_stroke(c).map(|s| s.key);
        // Every 15th key starts with a wrong press.
        if keys % 15 == 7 {
            engine.key(input('#', key_id("Backslash"), t_us)).unwrap();
            t_us += 90_000;
        }
        let (outcome, summary) = engine.key(input(c, code, t_us)).unwrap();
        assert!(outcome.correct && outcome.advanced);
        t_us += 110_000;
        keys += 1;
        batch = summary;
    }
    let summary = batch.expect("a batch closes after ~45 s of typing");
    assert!(summary.wpm > 80.0 && summary.wpm < 120.0, "wpm {}", summary.wpm);
    assert!(summary.accuracy < 1.0 && summary.accuracy > 0.85);
    let lines = engine.store().typed_lines().unwrap();
    assert!(lines.len() >= 5);
    // Every key of a finished line is exported exactly once, none as abandoned.
    let rows = engine.store().keystrokes().unwrap();
    assert!(rows.iter().all(|r| r.status == KeyStatus::Finished && r.line_no.is_some()));
    // Exactly one correct key per typed character, no duplicates.
    let correct: Vec<(u64, u16)> = rows.iter().filter(|r| r.correct).map(|r| (r.line_no.unwrap(), r.pos)).collect();
    let unique: std::collections::HashSet<&(u64, u16)> = correct.iter().collect();
    assert_eq!(unique.len(), correct.len());
    assert_eq!(correct.len(), lines.iter().map(|l| l.chars.len()).sum::<usize>());
    assert_eq!(rows.iter().filter(|r| !r.correct).count(), keys / 15 + usize::from(keys % 15 > 7));

    // A measurement (probe) line was served among the practice lines.
    assert!(lines.iter().any(|l| l.kind == ContentKind::Probe), "no probe line in {} lines", lines.len());

    // Forecast requests queued around the batch must not be mistaken for its result.
    engine.request_forecast();
    engine.request_forecast();
    // The learning thread fine-tunes, re-plans and reports a real-speed estimate, and the
    // batch row gets the arm it was typed with.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let ready = engine.view().real_wpm.is_some() && engine.insights().model_ready;
        let batch = engine.store().batches().unwrap().pop().unwrap();
        if ready && batch.arm.is_some() {
            assert!(batch.real_wpm.is_some());
            break;
        }
        assert!(Instant::now() < deadline, "no update from the learning thread");
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(!engine.insights().learning_stopped);
}

/// The trainer values practice by what it prevents too: the same pair, practised equally
/// often, is worth more when it was last typed weeks ago (about to be forgotten) than
/// when it was typed today. Uses the shipped model.
#[test]
fn stale_pairs_are_worth_more_than_fresh_ones() {
    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let path = resources.join("models/typist.bpk");
    let (model, loaded) = Brain::load_model(&[path.as_path()]);
    assert!(loaded);
    let layout = Layout::new(LayoutId::DeIso, Geometry::Iso);
    let pool = Pool::default();
    let now = 1_800_000_000_000i64;
    let day = 86_400_000i64;
    let history_at = |last_ms: i64| {
        let mut h = PracticeHistory::default();
        for i in 0..60 {
            h.record(&layout, Some('e'), 'n', last_ms - (60 - i) * 1000, Some(180.0), false);
        }
        h
    };
    let device = reflex_core::model::cpu();
    let key = pair_key(&layout, Some('e'), 'n').unwrap();
    let fresh = compute_gains(&model, &history_at(now - 1000 * 60), &layout, &pool, now, &device).gain[&key];
    let stale = compute_gains(&model, &history_at(now - 25 * day), &layout, &pool, now, &device).gain[&key];
    assert!(stale > fresh, "stale {stale} vs fresh {fresh}");
}

fn input(c: char, key: Option<u8>, t_us: i64) -> KeyInput {
    KeyInput { key, chars: c.to_string(), modifiers: 0, timestamp_us: t_us, wall_ms: t_us / 1000, is_dead: false }
}

#[test]
fn keys_survive_a_quit_mid_line() {
    let dir = tempfile::tempdir().unwrap();
    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut t_us = 1_000_000i64;
    {
        let mut engine = Engine::open(dir.path(), &resources, 60).unwrap();
        engine.set_layout(LayoutId::DeIso, Geometry::Iso).unwrap();
        // A wrong press keeps the cursor; the right one moves it on.
        let c = engine.expected().unwrap();
        let wrong = if c == '#' { '+' } else { '#' };
        let (miss, _) = engine.key(input(wrong, key_id("Backslash"), t_us)).unwrap();
        assert!(!miss.correct && !miss.advanced);
        t_us += 100_000;
        for _ in 0..3 {
            let c = engine.expected().unwrap();
            let code = engine.layout().main_stroke(c).map(|s| s.key);
            engine.key(input(c, code, t_us)).unwrap();
            t_us += 100_000;
        }
        // Exported mid-line: the keys show up as the open line.
        let open = engine.store().keystrokes().unwrap();
        assert_eq!(open.len(), 4);
        assert!(open.iter().all(|r| r.status == KeyStatus::Open));
        // A second instance on the same data is refused.
        assert!(matches!(Engine::open(dir.path(), &resources, 60), Err(StoreError::Locked)));
    } // quit mid-line
    let engine = Engine::open(dir.path(), &resources, 60).unwrap();
    let rows = engine.store().keystrokes().unwrap();
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r.status == KeyStatus::Abandoned));
    assert!(!rows[0].correct && rows[1..].iter().all(|r| r.correct));
    assert_eq!(rows.iter().map(|r| r.pos).collect::<Vec<_>>(), [0, 0, 1, 2]);
    assert!(engine.store().typed_lines().unwrap().is_empty());
}

#[test]
fn layout_switch_mid_line_keeps_keys_on_their_layout() {
    let dir = tempfile::tempdir().unwrap();
    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut engine = Engine::open(dir.path(), &resources, 60).unwrap();
    engine.set_layout(LayoutId::DeIso, Geometry::Iso).unwrap();
    let mut t_us = 1_000_000i64;
    let mut type_one = |engine: &mut Engine| {
        let c = engine.expected().unwrap();
        let code = engine.layout().main_stroke(c).map(|s| s.key);
        let (outcome, _) = engine.key(input(c, code, t_us)).unwrap();
        t_us += 100_000;
        outcome
    };
    for _ in 0..3 {
        type_one(&mut engine);
    }
    engine.set_layout(LayoutId::UsAnsi, Geometry::Ansi).unwrap();
    // The line restarts on the new layout; finish it there.
    while !type_one(&mut engine).line_finished {}
    let rows = engine.store().keystrokes().unwrap();
    let (old, new): (Vec<_>, Vec<_>) = rows.iter().partition(|r| r.layout == "de-iso");
    assert_eq!(old.len(), 3);
    assert!(old.iter().all(|r| r.status == KeyStatus::Abandoned && r.geometry == "iso"));
    assert!(!new.is_empty() && new.iter().all(|r| r.layout == "us-ansi" && r.status == KeyStatus::Finished));
    assert_eq!(new[0].pos, 0);
    // Heatmap data: only the finished line on U.S., nothing on German.
    let us: u32 = engine.store().char_stats(LayoutId::UsAnsi, Geometry::Ansi).unwrap().iter().map(|c| c.attempts).sum();
    assert_eq!(us as usize, new.len());
    assert!(engine.store().char_stats(LayoutId::DeIso, Geometry::Iso).unwrap().is_empty());
}
