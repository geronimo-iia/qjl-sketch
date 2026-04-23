# Roadmap

## Goal

A standalone Rust crate (`turboquant`) that compresses vectors via the
TurboQuant pipeline and scores queries against compressed stores —
no LLM, no GPU, CPU-only. Testable end-to-end: compress vectors,
persist, reload, score, verify ranking preservation.

## Phase 0 — Project scaffold

Set up the crate, CI, and test harness.

- [x] `cargo init --lib turboquant` in `projects/turboquant/`
- [x] Cargo.toml: edition 2021, rust-version 1.95, MIT OR Apache-2.0
- [x] Dependencies: `nalgebra`, `rand`, `rand_distr`, `bytemuck`, `rayon`, `memmap2`, `blake3`
- [x] Dev-dependencies: `tempfile`, `approx`
- [x] `src/lib.rs` with empty module declarations
- [x] `.github/workflows/ci.yml` — cargo fmt, clippy, test, audit
- [x] `.github/workflows/release.yml` — test + publish to crates.io
- [x] `.github/workflows/dependabot.yml` — auto-merge patch/minor
- [x] `.github/dependabot.yml` — cargo + github-actions weekly
- [x] `.github/ISSUE_TEMPLATE/` — bug report, feature request, config
- [x] `.github/pull_request_template.md`
- [x] `rust-toolchain.toml` — pinned to 1.95
- [x] `rustfmt.toml`, `clippy.toml`, `audit.toml`
- [x] `.tool-versions` — rust 1.95.0
- [x] `LICENSE-MIT`, `LICENSE-APACHE`
- [x] `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `README.md`
- [x] `docs/` — README, roadmap, release guide
- [x] `docs/study/` — article, overview
- [x] `docs/design/` — algorithms, pipeline, persistence, testing
- [x] `tests/` directory structure matching `design/testing.md`
- [x] `cargo build` — clean
- [x] `cargo test` — passes (zero tests)


### Milestone: `cargo test` passes (zero tests, clean build)

## Phase 1 — Core algorithms (CPU, f32)

Implement each algorithm from `design/algorithms.md` with unit tests.
One module per concern. No persistence, no pipeline — pure math.

### 1a — Random projection

- [x] `src/sketch.rs`: `QJLSketch` struct
- [x] `QJLSketch::new(head_dim, sketch_dim, seed)` — Gaussian init
- [x] QR orthogonalization per chunk (Algorithm 1)
- [x] `proj_dir_score` and `proj_dir_quant` stored as `Vec<f32>`
- [x] Tests: dimensions, orthogonality, determinism, different seeds (6 tests)

### 1b — Outlier detection

- [x] `src/outliers.rs`: `detect_outliers(keys, count) → Vec<u8>`
- [x] L2 norm per dimension across group, top-k selection (Algorithm 4)
- [x] Tests: known outlier picked, count respected, mask (4 tests)

### 1c — QJL quantization

- [x] `src/quantize.rs`: `CompressedKeys` struct
- [x] `QJLSketch::quantize(keys, outlier_indices) → CompressedKeys`
- [x] Sign extraction, bit-packing 8 signs per u8 (Algorithm 2)
- [x] Outlier/inlier separation, outlier norms
- [x] Tests: output shape, bit-packing, outlier separation, norms (5 tests)

### 1d — Score computation

- [x] `src/score.rs`: `QJLSketch::score(query, compressed) → Vec<f32>`
- [x] Query sketch projection via `proj_dir_quant`
- [x] Outlier query sketch subtraction (matches CUDA kernel)
- [x] Signed dot: float query sketch × packed sign bits
- [x] Scale factor: `sqrt(π/2) / sketch_dim`
- [x] Tests: signed dot, identical vectors, sign preserved, multiple vectors (5 tests)

### 1e — Value quantization

- [x] `src/values.rs`: `CompressedValues` struct
- [x] `quantize_values(values, group_size, bits) → CompressedValues`
- [x] Min-max scalar quantization + i32 bit-packing (Algorithm 5)
- [x] `quantized_dot(weights, compressed) → f32` (Algorithm 6)
- [x] Tests: round-trip error bound, 4-bit/2-bit range, matmul accuracy (6 tests)

### 1f — Streaming quantizer

- [x] `src/quantizer.rs`: `KeyQuantizer` struct
- [x] `build_sketch(keys)` — batch compress (Algorithm 7 init)
- [x] `update(key)` — append one vector, flush on buffer full (Algorithm 7)
- [x] `attention_score(query) → Vec<f32>` — score against full state
- [x] Tests: stream matches batch, residual buffer, buffer flush (6 tests)

### Milestone: 32 unit tests passing ✓

## Phase 2 — Quality validation

Statistical tests proving our implementation preserves the properties
that TurboQuant guarantees. All self-contained, no external fixtures.
All run by default (~7 seconds total).

### 2a — Rotation preserves geometry

- [x] `test_rotation_preserves_norm` — 1K vectors, norm ratio ∈ [0.90, 1.10]
- [x] `test_rotation_preserves_inner_product` — 1K pairs, mean error < 0.15

### 2b — Sign quantization distortion

- [x] `test_distortion_rate` — 10K pairs, distortion < 0.35 at s=2d
- [x] `test_distortion_decreases_with_sketch_dim` — monotonic d > 2d > 4d

### 2c — Ranking preservation

- [x] `test_top_k_recall` — 200 keys, mean recall ≥ 0.55 over 100 trials
- [x] `test_kendall_tau` — 100 keys, mean tau > 0.70 over 50 trials

### 2d — Value quantization accuracy

- [x] `test_value_quantized_matmul_error_4bit` — mean relative error < 0.20
- [x] `test_value_quantized_matmul_error_2bit` — mean relative error < 1.0

### 2e — Outlier separation benefit

- [x] `test_outlier_vs_no_outlier` — ≥ 20% distortion reduction with 10x outliers

### Milestone: 9 quality tests passing ✓

## Phase 3 — Persistence

Implement the two-store design from `design/persistence.md`.
Keys and values in separate files with independent indexes.

### 3a — Index format + sketch reconstruction

- [x] `src/store/config.rs`: `KeysConfig`, `ValuesConfig`, `IndexEntry`, `IndexMeta`
- [x] Binary read/write for all config structs (little-endian)
- [x] Magic bytes + version validation on read
- [x] Reconstruct `QJLSketch` from `KeysConfig` (deterministic seed)
- [x] Tests: round-trip for all structs, sketch reconstruction, bad magic rejected (6 tests)

### 3b — Key store

- [x] `src/store/key_store.rs`: `KeyStore` struct
- [x] `KeyStore::create(dir, config)` — create empty keys.bin + keys.idx
- [x] `KeyStore::open(dir)` — read keys.idx header, construct QJLSketch,
      mmap keys.bin, load index
- [x] `append(slug_hash, content_hash, compressed_keys)` — serialize
      entry, append to keys.bin, fsync, atomic index rewrite
- [x] `get_page(slug_hash) → Option<KeyPageView>` — binary search,
      zero-copy slice into mmap
- [x] `KeyPageView` — accessors for key_quant, key_norms,
      outlier_norms, outlier_indices (unaligned f32 read)
- [x] `to_compressed_keys()` — reconstruct `CompressedKeys` from view
- [x] `is_fresh(slug_hash, content_hash)` — staleness check
- [x] Update with higher generation, dead space tracking
- [x] Tests: create/open, append/get, page not found, score survives
      persistence, multiple pages, reopen, staleness, update overwrites (8 tests)

### 3c — Value store

- [x] `src/store/value_store.rs`: `ValueStore` struct
- [x] `ValueStore::create(dir, bits, group_size)`
- [x] `ValueStore::open(dir)`
- [x] `append(slug_hash, content_hash, compressed_values)`
- [x] `get_page(slug_hash) → Option<ValuePageView>`
- [x] `ValuePageView` — accessors for packed, scale, mn
- [x] Tests: create/open, append/get, page not found, quantized_dot
      survives persistence, multiple pages, reopen, staleness,
      update overwrites (8 tests)

### 3d — Update and staleness

- [x] `is_fresh(slug_hash, content_hash) → bool` on both stores
- [x] Append with higher generation, old entry becomes dead
- [x] `dead_bytes()` / `live_bytes()` in index header
- [x] Tests: keys fresh but values stale, independent lifecycle,
      dead bytes tracked after reopen (3 integration tests)

### 3e — Compaction

- [x] `compact()` on KeyStore — rewrite live entries, rebuild index,
      atomic rename
- [x] `compact()` on ValueStore — same pattern
- [x] Tests: key store reclaims space, preserves scores, survives
      reopen; value store reclaims space, preserves quantized_dot
      (5 integration tests)

### 3f — Crash recovery

- [x] Detect truncated tail on open (magic + entry_len walk), truncate
      — both KeyStore and ValueStore
- [x] Drop index entries pointing beyond .bin length — both stores
- [x] Rebuild index from .bin scan if index missing (KeyStore falls
      back to keys.idx.tmp from interrupted atomic rename)
- [x] Tests: truncated tail recovery (keys + values), index ahead of
      store (keys + values) (4 integration tests)

### Milestone: 75 tests — compress → persist → reload → score = same result ✓

## Phase 4 — Performance

Not needed for correctness, but needed for practical use.

- [ ] SIMD popcount: `std::arch` for `_popcnt64` on x86, fallback
      to `u8::count_ones()`
- [ ] Batch projection as GEMM via `nalgebra` BLAS
- [ ] `rayon` parallelism for multi-head score computation
- [ ] Benchmark suite: `benches/` with `criterion`
      — score latency vs. page count (100, 1K, 10K)
      — compress throughput (pages/sec)
      — cold start time (mmap open + first query)

### Milestone: benchmark numbers documented, no regressions in CI

## Future

- GPU score kernel via `wgpu` compute shaders
- W_q / W_k / W_v weight loading from GGUF or safetensors

Pipeline integration (BM25 pre-filter + TurboQuant rerank) lives in
the [llm-wiki](https://github.com/geronimo-iia/llm-wiki) project.

## Project Structure (target)

```
projects/turboquant/
├── study/                  ← source material
│   ├── README.md
│   └── article.md
├── design/                 ← architecture decisions
│   ├── algorithms.md
│   ├── persistence.md
│   ├── pipeline.md
│   └── testing.md
├── src/
│   ├── lib.rs
│   ├── sketch.rs           ← QJLSketch (projection, quantize, score)
│   ├── outliers.rs          ← outlier detection
│   ├── quantize.rs          ← CompressedKeys, sign hashing
│   ├── score.rs             ← score computation (Hamming + norms)
│   ├── values.rs            ← CompressedValues, quantized matmul
│   ├── quantizer.rs         ← KeyQuantizer (batch + streaming)
│   ├── store/
│   │   ├── mod.rs
│   │   ├── config.rs        ← index headers, sketch params
│   │   ├── key_store.rs     ← KeyStore, append, compact, mmap
│   │   └── value_store.rs   ← ValueStore, append, compact, mmap
├── tests/
│   ├── quality/
│   └── persistence/
├── benches/
│   └── score.rs
├── article.md
├── Cargo.toml
├── LICENSE-APACHE
├── LICENSE-MIT
└── README.md
```
