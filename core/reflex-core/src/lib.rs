//! Reflex core: everything that computes. Portable Rust (Unix), no GUI or OS-specific APIs.

uniffi::setup_scaffolding!();

pub mod bench;
pub mod brain;
pub mod charset;
pub mod content;
pub mod engine;
pub mod ffi;
pub mod history;
pub mod layout;
pub mod model;
pub mod record;
pub mod session;
pub mod settings;
pub mod sim;
pub mod stats;
pub mod store;
pub mod trainer;
