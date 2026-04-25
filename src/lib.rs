//! QJL sketch — fast approximate attention scoring via sign-based vector compression.
//!
//! Compresses key/value vectors using random projection sign hashing (QJL) and
//! min-max scalar quantization, then stores them in append-only mmap-backed stores.
//! Scoring is approximate inner product via packed sign bits; batched store-level
//! scoring can be GPU-accelerated with the `gpu` feature.
//!
//! # Feature flags
//!
//! - `serde` — enables `Serialize`/`Deserialize` on all public structs and
//!   streaming store export/import.
//! - `gpu` — enables WGPU GPU-accelerated `KeyStore::scores` (batched float × sign).

pub mod codebook;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod math;
pub mod mse_quant;
pub mod outliers;
pub mod quantize;
pub mod quantizer;
pub mod rotation;
pub mod score;
pub mod sketch;
pub mod store;
pub mod values;
