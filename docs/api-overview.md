# API Overview

Public API surface of `qjl-sketch`, organized by use case.

---

## Compress

### QJL sign-based compression

```rust
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::outliers::detect_outliers;

let sketch = QJLSketch::new(dim, sketch_dim, outlier_sketch_dim, seed)?;
let outlier_indices = detect_outliers(&keys, group_size, dim, count)?;
let compressed = sketch.quantize(&keys, num_vectors, &outlier_indices)?;
```

| Item | Module | Description |
| --- | --- | --- |
| `QJLSketch` | `sketch` | Random projection matrix (QR-orthogonalized, Johnson-Lindenstrauss) |
| `QJLSketch::new(dim, sketch_dim, outlier_sketch_dim, seed)` | `sketch` | Create sketch; `sketch_dim` and `outlier_sketch_dim` must be multiples of 8 |
| `QJLSketch::quantize(keys, num_vectors, outlier_indices)` | `quantize` | Compress `num_vectors` key vectors to `CompressedKeys` |
| `CompressedKeys` | `quantize` | Packed sign bits + L2 norms + outlier signs + outlier norms |
| `detect_outliers(keys, group_size, dim, count)` | `outliers` | Return top-`count` outlier dimension indices (by group L2 norm) |
| `outlier_mask(indices, dim)` | `outliers` | Convert outlier index list to a `Vec<bool>` mask of length `dim` |
| `pack_signs(signs: &[bool])` | `quantize` | Pack a bool slice into bytes (MSB-first) |
| `unpack_signs(bytes: &[u8], n: usize)` | `quantize` | Unpack `n` sign bits from bytes |

**`QJLSketch` public fields**

| Field | Type | Description |
| --- | --- | --- |
| `dim` | `usize` | Vector dimension |
| `sketch_dim` | `usize` | Number of random projections |
| `outlier_sketch_dim` | `usize` | Projection count for outlier components |
| `seed` | `u64` | RNG seed (deterministic reconstruction) |
| `proj_dir_score` | `Vec<f32>` | Projection matrix `[dim × sketch_dim]`, row-major |
| `proj_dir_quant` | `Vec<f32>` | Transposed projection `[sketch_dim × dim]`, row-major |

---

### Value quantization (min-max)

```rust
use qjl_sketch::values::{quantize_values, dequantize_all, quantized_dot};

let compressed = quantize_values(&values, group_size, bits)?;
let reconstructed = dequantize_all(&compressed);
let dot = quantized_dot(&weights, &compressed)?;
```

| Item | Module | Description |
| --- | --- | --- |
| `quantize_values(values, group_size, bits)` | `values` | Min-max quantization; `bits` must be 2 or 4 |
| `CompressedValues` | `values` | Bit-packed values + per-group `scale` and `min` |
| `dequantize_all(compressed)` | `values` | Reconstruct all values as `Vec<f32>` |
| `quantized_dot(weights, compressed)` | `values` | Fused dequantize + weighted dot product (scalar) |

---

### MSE-optimal quantization (rotation + Lloyd-Max)

```rust
use qjl_sketch::rotation::RandomRotation;
use qjl_sketch::codebook::{generate_codebook, CodebookCache};
use qjl_sketch::mse_quant::{mse_quantize, mse_dequantize, mse_score};

let rot = RandomRotation::new(dim, seed)?;
let cb = generate_codebook(dim, bit_width, iterations)?;
let quantized = mse_quantize(&vectors, num_vectors, &rot, &cb)?;
let reconstructed = mse_dequantize(&quantized, &rot, &cb)?;
let scores = mse_score(&token, &quantized, &rot, &cb)?;
```

| Item | Module | Description |
| --- | --- | --- |
| `RandomRotation` | `rotation` | `d × d` Haar-uniform orthogonal matrix |
| `RandomRotation::new(dim, seed)` | `rotation` | Create rotation (deterministic from seed) |
| `RandomRotation::rotate(x)` | `rotation` | Apply rotation: returns `Vec<f32>` |
| `RandomRotation::rotate_inverse(y)` | `rotation` | Apply inverse rotation: returns `Vec<f32>` |
| `Codebook` | `codebook` | Lloyd-Max centroids + decision boundaries (1–8 bit) |
| `generate_codebook(dim, bit_width, iterations)` | `codebook` | Fit optimal scalar codebook |
| `Codebook::quantize(value)` | `codebook` | Scalar `f32` → codebook index |
| `Codebook::dequantize(index)` | `codebook` | Codebook index → centroid `f32` |
| `CodebookCache` | `codebook` | Thread-safe memoizing cache keyed by `(dim, bits)` |
| `MseQuantized` | `mse_quant` | Per-coordinate codebook indices after rotation |
| `mse_quantize(vectors, num_vectors, rot, cb)` | `mse_quant` | Rotate then quantize each coordinate |
| `mse_dequantize(quantized, rot, cb)` | `mse_quant` | Dequantize then inverse-rotate |
| `mse_score(token, quantized, rot, cb)` | `mse_quant` | Score a float token against quantized vectors |

**`MseQuantized` public fields**

| Field | Type | Description |
| --- | --- | --- |
| `indices` | `Vec<u8>` | Flattened `[num_vectors × dim]` codebook indices |
| `num_vectors` | `usize` | Number of quantized vectors |
| `dim` | `usize` | Vector dimension |
| `bit_width` | `u8` | Bits per coordinate |

---

### Streaming compression

```rust
use qjl_sketch::quantizer::KeyQuantizer;

let mut kq = KeyQuantizer::new(&sketch, group_size, outlier_count, buffer_size)?;
kq.build_sketch(&keys, num_vectors)?;  // compress a batch
kq.update(&single_key)?;              // compress one vector at a time
let scores = kq.score_token(&token)?;
```

| Item | Module | Description |
| --- | --- | --- |
| `KeyQuantizer` | `quantizer` | Streaming wrapper: buffer, compress, score without materializing all keys at once |

---

## Score

```rust
// Float x sign: token (f32) vs compressed keys
let scores: Vec<f32> = sketch.score(&token, &compressed)?;

// Compressed x compressed: entry-to-entry Hamming similarity
let scores: Vec<f32> = sketch.score_compressed(&a, &b)?;
let score: f32 = sketch.score_compressed_pair(&a, i, &b, j)?;

// Standalone Hamming similarity
let sim: f32 = hamming_similarity(&a_bytes, &b_bytes, total_bits);
```

| Item | Module | Description |
| --- | --- | --- |
| `QJLSketch::score(token, compressed)` | `score` | Float × sign inner product estimate; returns `Vec<f32>` of length `num_vectors` |
| `QJLSketch::score_compressed(a, b)` | `score` | Hamming cosine between two `CompressedKeys`; returns `Vec<f32>` |
| `QJLSketch::score_compressed_pair(a, i, b, j)` | `score` | Single pair score from two `CompressedKeys` at indices `i` and `j` |
| `hamming_similarity(a, b, total_bits)` | `score` | Fraction of matching bits in `[0, 1]` |

---

## Store

```rust
use qjl_sketch::store::key_store::KeyStore;
use qjl_sketch::store::value_store::ValueStore;
use qjl_sketch::store::config::{KeysConfig, ValuesConfig};

// Create / open
let mut store = KeyStore::create(dir, config)?;
let store     = KeyStore::open(dir)?;

// Read / write
store.append(entry_id, content_hash, &compressed)?;
let entry = store.get_entry(entry_id);          // Option<KeyEntryView>
let fresh = store.is_fresh(entry_id, content_hash);

// Capacity
let n     = store.len();
let empty = store.is_empty();
let live  = store.live_bytes();
let dead  = store.dead_bytes();

// Maintenance
store.compact()?;

// Score all entries (float x sign; GPU-accelerated with `gpu` feature)
let results: Vec<(u64, Vec<f32>)> = store.scores(&token, &sketch, &outlier_indices)?;

// Export / import (requires `serde` feature)
for entry in store.iter_entries() { /* KeyExportEntry */ }
store.import_entry(&export_entry)?;
```

### `KeyStore`

| Item | Module | Description |
| --- | --- | --- |
| `KeyStore` | `store::key_store` | Append-only mmap-backed compressed key storage |
| `KeyStore::create(dir, config)` | | Create new store; writes `keys.bin` + `keys.idx` |
| `KeyStore::open(dir)` | | Open existing store; recovers from partial writes and missing index |
| `KeyStore::append(entry_id, content_hash, compressed)` | | Append or overwrite a `CompressedKeys` entry |
| `KeyStore::get_entry(entry_id)` | | Zero-copy lookup; returns `Option<KeyEntryView<'_>>` |
| `KeyStore::is_fresh(entry_id, content_hash)` | | `true` if stored hash matches |
| `KeyStore::len()` | | Number of live entries |
| `KeyStore::is_empty()` | | `true` when store has no entries |
| `KeyStore::live_bytes()` | | Bytes used by live entries |
| `KeyStore::dead_bytes()` | | Bytes from overwritten entries (reclaimed by `compact`) |
| `KeyStore::compact()` | | Rewrite store with only live entries; reclaims dead space |
| `KeyStore::scores(token, sketch, outlier_indices)` | | Score token against all entries; returns `Vec<(entry_id, Vec<f32>)>` |
| `KeyStore::iter_entries()` | | Streaming iterator over `KeyExportEntry` (serde) |
| `KeyStore::import_entry(entry)` | | Append from a `KeyExportEntry` (serde) |
| `KeyEntryView<'a>` | `store::key_store` | Zero-copy view into a mmap'd key entry |
| `KeyEntryView::num_vectors` | | Number of key vectors in this entry |
| `KeyEntryView::outlier_count` | | Number of outlier dimensions |
| `KeyEntryView::to_compressed(dim)` | | Materialize as `CompressedKeys` |
| `KeyExportEntry` | `store::key_store` | Serializable snapshot: `entry_id`, `content_hash`, `compressed` |

**`KeysConfig` fields**

| Field | Type | Description |
| --- | --- | --- |
| `dim` | `u16` | Key vector dimension |
| `sketch_dim` | `u16` | Number of sign projections |
| `outlier_sketch_dim` | `u16` | Sign projections for outlier components |
| `seed` | `u64` | Sketch RNG seed |

`KeysConfig::build_sketch()` reconstructs the `QJLSketch` from stored parameters.

---

### `ValueStore`

Same append-only, mmap-backed design as `KeyStore`, but stores `CompressedValues`.

```rust
let mut store  = ValueStore::create(dir, config)?;
let store      = ValueStore::open(dir)?;
store.append(entry_id, content_hash, &compressed)?;
let entry      = store.get_entry(entry_id);     // Option<ValueEntryView>
let fresh      = store.is_fresh(entry_id, content_hash);
store.compact()?;
for entry in store.iter_entries() { /* ValueExportEntry */ }
store.import_entry(&export_entry)?;
```

| Item | Module | Description |
| --- | --- | --- |
| `ValueStore` | `store::value_store` | Append-only mmap-backed compressed value storage |
| `ValueStore::create(dir, config)` | | Create new store; writes `values.bin` + `values.idx` |
| `ValueStore::open(dir)` | | Open existing store with crash recovery |
| `ValueStore::append(entry_id, content_hash, compressed)` | | Append or overwrite a `CompressedValues` entry |
| `ValueStore::get_entry(entry_id)` | | Zero-copy lookup; returns `Option<ValueEntryView<'_>>` |
| `ValueStore::is_fresh(entry_id, content_hash)` | | `true` if stored hash matches |
| `ValueStore::len()` | | Number of live entries |
| `ValueStore::is_empty()` | | `true` when store has no entries |
| `ValueStore::live_bytes()` | | Bytes used by live entries |
| `ValueStore::dead_bytes()` | | Bytes from overwritten entries |
| `ValueStore::compact()` | | Reclaim dead space |
| `ValueStore::iter_entries()` | | Streaming iterator over `ValueExportEntry` (serde) |
| `ValueStore::import_entry(entry)` | | Append from a `ValueExportEntry` (serde) |
| `ValueEntryView<'a>` | `store::value_store` | Zero-copy view into a mmap'd value entry |
| `ValueEntryView::num_elements` | | Number of quantized elements |
| `ValueEntryView::num_groups` | | Number of quantization groups |
| `ValueEntryView::to_compressed()` | | Materialize as `CompressedValues` |
| `ValueExportEntry` | `store::value_store` | Serializable snapshot: `entry_id`, `content_hash`, `compressed` |

**`ValuesConfig` fields**

| Field | Type | Description |
| --- | --- | --- |
| `bits` | `u8` | Bits per value (2 or 4) |
| `group_size` | `u16` | Elements per quantization group |

---

### Shared index structures (`store::config`)

| Item | Description |
| --- | --- |
| `IndexEntry` | On-disk record: `entry_id`, `offset`, `entry_len`, `generation`, `content_hash` |
| `IndexMeta` | Store-level counters: `entry_count`, `live_bytes`, `dead_bytes` |

---

## Error handling

```rust
use qjl_sketch::error::{QjlError, Result, validate_finite};
```

All public functions return `Result<T, QjlError>`. Error variants:

| Variant | Meaning |
| --- | --- |
| `DimensionMismatch { expected, got }` | Vector length does not match configured dimension |
| `InvalidSketchDim(usize)` | `sketch_dim` is 0 or not a multiple of 8 |
| `InvalidBitWidth(u8)` | Bit width outside the supported range |
| `NonFiniteInput { context }` | Input contains NaN or ±Inf |
| `InvalidCodebookBitWidth(u8)` | Codebook bit width must be 1–8 |
| `InvalidDimension(usize)` | Dimension must be > 0 |
| `SketchParamMismatch { context }` | Two `CompressedKeys` have incompatible sketch parameters |
| `IndexOutOfBounds { index, len }` | Vector index out of bounds |
| `StoreMagicMismatch` | File does not start with the expected magic bytes |
| `StoreVersionMismatch { expected, got }` | Index file version is not supported |
| `OutlierIndexOutOfRange { index, dim }` | Outlier dimension index ≥ vector dimension |
| `Io(std::io::Error)` | Underlying filesystem error |

`validate_finite(values, context)` returns `Err(NonFiniteInput)` if any element is NaN or Inf.

---

## Feature flags

| Flag | What it enables |
| --- | --- |
| `serde` | `Serialize`/`Deserialize` on all public structs; store `iter_entries` / `import_entry` |
| `gpu` | WGPU GPU-accelerated `KeyStore::scores` (batched float × sign, single dispatch) |

GPU dispatch activates in `KeyStore::scores` only when total vectors across all entries
≥ `QJL_GPU_MIN_BATCH` (default 5000) and a GPU adapter is available. All other scoring
paths always use CPU. Override: `QJL_GPU_MIN_BATCH=0` to always use GPU.

---

## Utilities

| Item | Module | Description |
| --- | --- | --- |
| `matvec(mat, rows, cols, vec)` | `sketch` | Matrix-vector multiply; returns `Vec<f32>` |
| `l2_norm(v)` | `sketch` | L2 norm of a `f32` slice |
| `validate_finite(values, context)` | `error` | Reject NaN/Inf; used at all public entry points |
| `gpu_min_batch()` | `gpu` | Read `QJL_GPU_MIN_BATCH` env var (gpu feature only) |
