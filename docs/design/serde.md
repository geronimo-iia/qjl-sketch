# Serde Support

Optional serialization/deserialization for debug dumps, interop,
and store export/import. Gated behind the `serde` feature flag.

## Usage

```toml
[dependencies]
qjl-sketch = { version = "0.4", features = ["serde"] }
```

## Serializable structs

### Simple derives (data-only)

These serialize all fields directly:

| Struct | Module | Fields |
|--------|--------|--------|
| `Codebook` | `codebook` | centroids, boundaries, bit_width |
| `MseQuantized` | `mse_quant` | indices, num_vectors, dim, bit_width |
| `CompressedKeys` | `quantize` | sign bits, norms, outlier indices |
| `CompressedValues` | `values` | packed, scale, mn, etc. |
| `KeysConfig` | `store::config` | head_dim, sketch_dim, seed, etc. |
| `ValuesConfig` | `store::config` | bits, group_size |
| `IndexEntry` | `store::config` | slug_hash, offset, etc. |
| `IndexMeta` | `store::config` | entry_count, live_bytes, dead_bytes |
| `KeyExportEntry` | `store::key_store` | slug_hash, content_hash, CompressedKeys |
| `ValueExportEntry` | `store::value_store` | slug_hash, content_hash, CompressedValues |

### Params-only (reconstructible)

These contain large derived matrices. Only construction params are
serialized; matrices are reconstructed on deserialize.

| Struct | Serialized fields | Reconstructed |
|--------|-------------------|---------------|
| `QJLSketch` | head_dim, sketch_dim, outlier_sketch_dim, seed | proj_dir_score, proj_dir_quant |
| `RandomRotation` | dim, seed | matrix, matrix_t |

### Not serializable

| Struct | Reason |
|--------|--------|
| `KeyStore` / `ValueStore` | File handles, mmap |
| `KeyPageView` / `ValuePageView` | Borrowed references |
| `KeyQuantizer` | Borrowed `&QJLSketch` |
| `CodebookCache` | Runtime cache |

## Store export/import

Streaming export/import avoids loading entire stores into memory.

```rust
// Export: one entry at a time → JSONL
let mut out = BufWriter::new(File::create("keys.jsonl")?);
for entry in store.iter_pages() {
    serde_json::to_writer(&mut out, &entry)?;
    out.write_all(b"\n")?;
}

// Import: one line at a time
let reader = BufReader::new(File::open("keys.jsonl")?);
for line in reader.lines() {
    let entry: KeyExportEntry = serde_json::from_str(&line?)?;
    store.import_entry(&entry)?;
}
```

The crate is format-agnostic — use JSON for debugging, bincode for
compact transfer, or any other serde-compatible format.
