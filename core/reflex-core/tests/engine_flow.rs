//! End to end: type a full batch through the engine, then check storage and the learning
//! worker's update (planner + real-speed estimate).

use reflex_core::engine::Engine;
use reflex_core::layout::{Geometry, LayoutId, key_id};
use reflex_core::session::KeyInput;
use std::path::Path;
use std::time::{Duration, Instant};

#[test]
fn typing_a_batch_trains_and_plans() {
    let dir = tempfile::tempdir().unwrap();
    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut engine = Engine::open(dir.path(), &resources).unwrap();
    engine.set_layout(LayoutId::DeIso, Geometry::Iso);
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
    assert!(engine.store().typed_lines().unwrap().len() >= 5);

    // The learning thread fine-tunes, re-plans and reports a real-speed estimate.
    let deadline = Instant::now() + Duration::from_secs(120);
    while engine.view().real_wpm.is_none() || !engine.insights().model_ready {
        assert!(Instant::now() < deadline, "no update from the learning thread");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn input(c: char, key: Option<u8>, t_us: i64) -> KeyInput {
    KeyInput { key, chars: c.to_string(), modifiers: 0, timestamp_us: t_us, wall_ms: t_us / 1000, is_dead: false }
}
