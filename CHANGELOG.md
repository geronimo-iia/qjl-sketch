# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] — 2026-04-26

### Changed (Breaking)

- Renamed public API identifiers to use domain-correct terminology:
  - `head_dim` → `dim` (structs, method parameters, error messages)
  - `query` → `token` (all scoring function parameters and local variables)
  - `slug` / `slug_hash` → `entry_id` / `content_hash` (store API and index structures)
- Replaced `page` / `pages` domain term (from llm-wiki origin) with `entry` / `entries`
  throughout source code, tests, benchmarks, bench script, and all documentation

### Fixed

- `benches/store.rs`: benchmark name `"append_single_page"` → `"append_single_entry"`
- `scripts/bench.sh`: corrected mismatched benchmark paths
  (`cold_start_100_entrys` → `cold_start_100_entries`,
  `get_entry_from_100` → `get_entry_from_100_entries`)

### Documentation

- Rewrote `docs/api-overview.md` from source — complete method signatures, all
  public fields, error variant table, feature flag behaviour
- Added `ValueEntryView` row (was missing from the Store table)
- Updated all `docs/design/` files: `store.md`, `persistence.md`, `testing.md`,
  `algorithms/08-compressed-scoring.md`, `algorithms/11-gpu-scoring.md`
- Updated `docs/decisions/001-gpu-scoring-architecture.md`
- Bumped test count heading in `docs/design/testing.md` from v0.5.0 to v0.6.0
- Added crate-level `//!` documentation to `src/lib.rs`
- Added doc comments on all previously undocumented public items:
  - `KeysConfig`, `ValuesConfig`, `IndexEntry`, `IndexMeta` — structs, fields,
    and all `write_to` / `read_from` / `build_sketch` methods
  - `KeyQuantizer` — `new`, `build_sketch`, `update`, `score_token`,
    `compressed_len`, `residual_len`, `residual_is_empty`
  - `KeyStore` — `live_bytes`, `dead_bytes`
  - `KeyEntryView` — `outlier_indices`, `key_quant`, `key_outlier_quant`,
    `key_norms`, `outlier_norms`
  - `ValueStore` — `len`, `is_empty`, `live_bytes`, `dead_bytes`
  - `ValueEntryView` — `packed`, `scale`, `mn`
- Fixed stale `"Head dimension"` → `"Vector dimension"` in `CompressedKeys::dim`

## [0.5.0] — TBD

### Added

- `gpu` feature flag — WGPU-based GPU acceleration for store-level scoring
  - Float×sign compute shader (`score_float_sign.wgsl`)
  - `GpuContext` — lazy singleton with runtime adapter detection
  - `KeyStore::scores` batches all vectors across all entries
    into a single GPU dispatch; falls back to CPU path
    when GPU unavailable or below threshold
  - Individual `score()` and `score_compressed()` always use CPU
  - `QJL_GPU_MIN_BATCH` env var for `scores` GPU threshold (default 5000)
  - 5 GPU tests (`#[ignore]` — require GPU adapter)
  - `benches/gpu_score.rs` — CPU baseline vs GPU benchmarks
  - `scripts/bench.sh` — benchmark runner with report collection
- `log` crate for structured logging (GPU fallback warnings)

## [0.4.0] — 2025-07-24

### Added

- `rotation` module — `RandomRotation` d×d orthogonal matrix via
  Gaussian QR with sign correction (Haar-uniform construction)
- `mse_quant` module — MSE-optimal vector quantization (TurboQuant Stage 1)
  - `mse_quantize` — rotate + Lloyd-Max per-coordinate quantization
  - `mse_dequantize` — centroid lookup + inverse rotation
  - `mse_score` — score token against quantized vectors (rotate token
    once, dot with dequantized rotated coordinates)
  - `MseQuantized` struct
- `serde` feature flag — optional `Serialize`/`Deserialize` on public structs
  - Simple derives on `Codebook`, `MseQuantized`, `CompressedKeys`,
    `CompressedValues`, `KeysConfig`, `ValuesConfig`, `IndexEntry`, `IndexMeta`
  - Params-only custom serde on `QJLSketch` and `RandomRotation`
    (serialize seed + dims, reconstruct matrices on deserialize)
  - `KeyExportEntry` / `ValueExportEntry` for store export/import
  - `KeyStore::iter_entries()` / `ValueStore::iter_entries()` — streaming export
  - `KeyStore::import_entry()` / `ValueStore::import_entry()` — streaming import
- `seed` field on `QJLSketch` and `RandomRotation` (stored for serialization)
- 14 new tests (7 rotation + 6 mse_quant + 1 quality)
- 12 serde round-trip tests (feature-gated)
- 2 examples: `serde_roundtrip`, `store_export_import`
- 3 examples: `basic_qjl`, `compressed_scoring`, `mse_quantization`

## [0.3.0] — 2025-07-24

### Added

- `codebook` module — Lloyd-Max optimal scalar quantization codebook
  for the Beta(1/2, (d-1)/2) coordinate marginal of unit-sphere vectors
  - `Codebook` struct with `quantize` / `dequantize` (f32 storage)
  - `generate_codebook(dim, bit_width, iterations)` — 1–8 bit support
  - `CodebookCache` — memoizing cache keyed by (dim, bit_width)
- `math` module — numerical helpers (all f64 internals)
  - `lgamma` — Lanczos approximation
  - `beta_pdf` — coordinate marginal PDF with Gaussian path for dim > 50
  - `normal_icdf` — Beasley-Springer-Moro rational approximation
  - `sample_beta_marginal` — inverse CDF via bisection / Gaussian
  - `simpson_integrate` — paired Simpson's rule
- `hamming_similarity` — standalone Hamming similarity on packed sign bits
- `QJLSketch::score_compressed` — batch compressed-vs-compressed scoring
  via Hamming-based cosine estimation with outlier separation
- `QJLSketch::score_compressed_pair` — single-pair variant for
  cross-index comparison (e.g. entry-to-entry similarity)
- `QjlError::InvalidCodebookBitWidth`, `InvalidDimension`,
  `SketchParamMismatch`, and `IndexOutOfBounds` error variants
- `THIRD_PARTY_NOTICES` file (TurboQuant MIT attribution)
- 41 new tests (14 codebook + 12 math + 4 error + 4 hamming +
  3 score_compressed + 3 score_compressed_pair + 1 quality)

### Changed

- Upgrade to Rust 2024 edition (`edition = "2024"` in Cargo.toml and rustfmt.toml)
- Split `docs/design/algorithms.md` into per-algorithm files under
  `docs/design/algorithms/`

### Fixed

- 15 rustdoc `broken_intra_doc_links` warnings across existing modules
  (escaped `[brackets]` in doc comments)
- `benches/score.rs`: missing `.unwrap()` on `sketch.score()` Result

## [0.2.0] — 2025-07-23

**Breaking change:** all public functions now return `Result<T, QjlError>`
instead of panicking on invalid input.

### Added

- `QjlError` enum with variants: `DimensionMismatch`, `InvalidSketchDim`,
  `InvalidBitWidth`, `NonFiniteInput`, `OutlierIndexOutOfRange`,
  `StoreMagicMismatch`, `StoreVersionMismatch`, `Io`
- `qjl_sketch::error::Result<T>` type alias
- `validate_finite` helper for NaN/infinity detection
- Input validation on all public API boundaries
- 18 new tests (7 error unit + 5 negative sketch/outlier + 6 negative integration)

### Changed

- `QJLSketch::new` returns `Result<Self>`
- `QJLSketch::quantize` returns `Result<CompressedKeys>`
- `QJLSketch::score` returns `Result<Vec<f32>>`
- `detect_outliers` returns `Result<Vec<u8>>`
- `quantize_values` returns `Result<CompressedValues>`
- `quantized_dot` returns `Result<f32>`
- `KeyQuantizer::new`, `build_sketch`, `update`, `score_token` return `Result`
- `KeyStore` and `ValueStore` methods return `qjl_sketch::error::Result`
- Store config `read_from` returns `QjlError::StoreMagicMismatch` /
  `StoreVersionMismatch` instead of `io::Error`

## [0.1.0] — 2025-07-23

First release. CPU-only QJL compression, scoring, and persistence.

### Core Algorithms

- `QJLSketch` — random projection matrix with QR orthogonalization,
  deterministic from seed
- QJL sign-based key quantization with outlier/inlier separation
- Score computation via signed dot product with `sqrt(π/2)/s` scale
  factor — matches the QJL CUDA kernel exactly
- Outlier detection — top-k dimension norms across group
- Min-max value quantization (2-bit and 4-bit) with i32 bit-packing
- Fused dequantize + weighted dot product
- Streaming `KeyQuantizer` — batch and one-at-a-time compression

### Persistence

- `KeyStore` — append-only compressed key storage with mmap loading
- `ValueStore` — append-only compressed value storage with mmap loading
- Two independent stores per directory (keys.bin/idx + values.bin/idx)
- Sketch params in index header — no config.bin, matrix recomputed
  from seed
- Zero-copy `KeyEntryView` / `ValueEntryView` into mmap'd data
- Content-hash staleness detection (blake3)
- Compaction with atomic rename
- Crash recovery: truncated tail detection, index-ahead-of-store
  filtering

### Quality

- 75 tests (54 unit + 12 persistence integration + 9 quality)
- Distortion < 0.35 at sketch_dim = 2 × dim
- Ranking preservation: Kendall's tau > 0.70, top-10 recall ≥ 0.55
- Outlier separation reduces distortion ≥ 20% on spiky vectors
- Score survives persistence round-trip (bit-exact)

### Benchmarks

- Score: 18 µs/entry (32 vectors, d=128, s=256)
- Key quantize: 38 µs/vector
- Cold start: 221 µs for 100 entries
- Entry lookup: 5 ns (binary search + mmap slice)
