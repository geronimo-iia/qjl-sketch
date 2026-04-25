# Algorithm 11: GPU-Accelerated Scoring

Impl: `src/gpu/mod.rs`, `src/gpu/wgpu_backend.rs`,
`src/gpu/score_float_sign.wgsl`

WGPU compute shader for float x sign scoring. GPU is used **only** by
`KeyStore::scores` which batches all vectors across all
entries into a single GPU dispatch.

Individual `score()` and `score_compressed()` calls always use CPU.

## Kernel: Float x Sign (token sketch x sign bits)

File: `src/gpu/score_float_sign.wgsl`

Each thread: one key vector. Unpacks 32 sign bits per u32 word,
multiplies by token sketch floats, accumulates dot product.

## Where GPU activates

```
score(token, compressed)           -> always CPU
score_compressed(a, b)             -> always CPU
score_compressed_pair(a, i, b, j)  -> always CPU

KeyStore::scores(token, sketch, outlier_indices):
  total_vectors = sum of all entry vector counts
  if gpu feature + adapter + total_vectors >= QJL_GPU_MIN_BATCH (5000):
    -> batch ALL sign bits + norms into contiguous buffers
    -> compute token sketch once on CPU
    -> single GPU dispatch for all vectors
    -> split scores back by entry boundaries
  else:
    -> CPU float x sign per entry (sketch.score() loop)
```

Both paths produce the same float x sign scores. GPU just makes
it faster for large stores.

## Environment Variable

| Variable | Default | Controls |
|----------|---------|----------|
| `QJL_GPU_MIN_BATCH` | 5000 | Total vectors for `scores` GPU dispatch |

Set to `0` to force GPU. Value read once and cached.


## Performance

See [benchmarks.md](../../benchmarks.md) for GPU vs CPU results
across dimensions and entry counts.

## Feature Flag

```toml
[features]
gpu = ["dep:wgpu", "dep:pollster"]
```

Opt-in at compile time. Default builds have zero GPU dependencies.
GPU tests are `#[ignore]` -- run with `cargo test --features gpu -- --ignored`.
