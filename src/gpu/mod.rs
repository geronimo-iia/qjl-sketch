// GPU acceleration via WGPU compute shaders.
//
// GPU is used only by `KeyStore::scores` which batches all
// vectors across all entries into a single GPU dispatch. Individual
// `score()` and `score_compressed()` calls always use CPU — single
// entries (32 vectors) are too small for GPU to beat CPU.
//
// Override threshold: `QJL_GPU_MIN_BATCH` env var (default 5000).

#[cfg(feature = "gpu")]
mod wgpu_backend;

#[cfg(feature = "gpu")]
pub use wgpu_backend::GpuContext;

#[cfg(feature = "gpu")]
const DEFAULT_GPU_MIN_BATCH: usize = 5_000;

/// Minimum total vectors across all entries for GPU dispatch in `scores`.
///
/// Override: `QJL_GPU_MIN_BATCH=0` to always use GPU.
#[cfg(feature = "gpu")]
pub fn gpu_min_batch() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("QJL_GPU_MIN_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GPU_MIN_BATCH)
    })
}
