//! Reflex core: everything that computes. Portable Rust (Unix), no GUI or OS-specific APIs.

uniffi::setup_scaffolding!();

pub mod charset;
pub mod content;
pub mod engine;
pub mod ffi;
pub mod history;
pub mod layout;
pub mod record;
pub mod session;
pub mod settings;
pub mod stats;
pub mod store;
pub mod trainer;
