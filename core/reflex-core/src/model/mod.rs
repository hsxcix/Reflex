//! Machine learning: the typist model, its training, and compute-device selection.

pub mod encode;
pub mod train;
pub mod typist;

pub use typist::{TokenPrediction, TypistModel};

use burn::tensor::Device;

/// Number of ensemble heads.
pub const ENSEMBLE: usize = 5;

/// CPU device for on-device inference and fine-tuning (small model, low latency).
pub fn cpu() -> Device {
    Device::flex()
}

/// Best available accelerator for offline pretraining (Metal on macOS, Vulkan/CUDA
/// elsewhere), falling back to the CPU.
pub fn accelerator() -> Device {
    #[cfg(feature = "gpu")]
    {
        Device::wgpu(Default::default())
    }
    #[cfg(not(feature = "gpu"))]
    {
        cpu()
    }
}
