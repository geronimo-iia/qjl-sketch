# Roadmap

## Goal

A standalone Rust crate (`qjl-sketch`) that compresses vectors via
QJL sign-based hashing and scores queries against compressed stores —
no LLM, no GPU, CPU-only.

## Completed

| Phase | What | Tests |
|-------|------|-------|
| 0 | Project scaffold — CI, release, dependabot, docs, licenses | — |
| 1 | Core algorithms — sketch, outliers, quantize, score, values, streaming quantizer | 32 |
| 2 | Quality validation — distortion, ranking, outlier benefit, value accuracy | 9 |
| 3 | Persistence — KeyStore, ValueStore, staleness, compaction, crash recovery | 34 |
| 4 | Benchmarks — score, compress, store latency with criterion | — |

75 tests. Published on [crates.io](https://crates.io/crates/qjl-sketch).

## Active

### Error types

Replace panics and raw `io::Error` with a dedicated `QjlError` enum.

- [ ] `src/error.rs`: `QjlError` enum with variants:
      `DimensionMismatch { expected, got }`,
      `InvalidSketchDim(usize)` (not divisible by 8),
      `InvalidBitWidth(u8)` (not 2 or 4),
      `NonFiniteInput { context }`,
      `StoreMagicMismatch`,
      `StoreVersionMismatch { expected, got }`,
      `Io(std::io::Error)`
- [ ] `pub type Result<T> = std::result::Result<T, QjlError>`
- [ ] Replace `assert!` / `assert_eq!` in public API with `Result` returns:
      `QJLSketch::new`, `quantize`, `score`, `detect_outliers`,
      `quantize_values`, `quantized_dot`
- [ ] Replace `io::Error::new(InvalidData, ...)` in store with `QjlError` variants
- [ ] `impl From<std::io::Error> for QjlError`
- [ ] `impl std::fmt::Display for QjlError`
- [ ] `impl std::error::Error for QjlError`
- [ ] Update all tests to use `.unwrap()` or `?` on Result returns

### Input validation

Reject bad inputs at API boundaries instead of producing silent garbage.

- [ ] `QJLSketch::new`: head_dim > 0, sketch_dim > 0
- [ ] `QJLSketch::quantize`: keys.len() == num_vectors * head_dim,
      all values finite, outlier indices < head_dim
- [ ] `QJLSketch::score`: query.len() == head_dim, all values finite
- [ ] `detect_outliers`: keys.len() == group_size * head_dim,
      count <= head_dim, head_dim <= 256
- [ ] `quantize_values`: values.len() divisible by group_size,
      bits == 2 or 4, all values finite
- [ ] `quantized_dot`: weights.len() == num_elements, all finite
- [ ] `KeyQuantizer::new`: buffer_size divisible by group_size
- [ ] `KeyStore::append` / `ValueStore::append`: compressed data
      internally consistent (lengths match)
- [ ] All validation returns `QjlError`

## Future

### Performance optimization

- [ ] SIMD: restructure `signed_dot` to process 8 bits per iteration
- [ ] Batch projection as GEMM via nalgebra BLAS
- [ ] `rayon` parallelism for multi-page scoring
- [ ] Batch append (amortize fsync cost)

### Serde support

Add `Serialize`/`Deserialize` on public structs for debug dumps and interop.

- [ ] Add `serde = { version = "1", features = ["derive"] }` to dependencies
- [ ] Derive `Serialize, Deserialize` on `CompressedKeys`
- [ ] Derive `Serialize, Deserialize` on `CompressedValues`
- [ ] Derive `Serialize, Deserialize` on `KeysConfig`, `ValuesConfig`
- [ ] Derive `Serialize, Deserialize` on `IndexEntry`, `IndexMeta`
- [ ] `QJLSketch`: serialize as params only (head_dim, sketch_dim,
      outlier_sketch_dim, seed) — not the matrices
- [ ] Tests: serde round-trip for each struct (serialize → deserialize → equal)
- [ ] Feature-gate behind `serde` feature flag to keep default deps minimal

### Other

- GPU score kernel via `wgpu` compute shaders
- W_q / W_k / W_v weight loading from GGUF or safetensors

Pipeline integration (BM25 pre-filter + QJL rerank) lives in
the [llm-wiki](https://github.com/geronimo-iia/llm-wiki) project.
