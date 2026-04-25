#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qjl_sketch::store::config::KeysConfig;
use qjl_sketch::store::key_store::KeyStore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use tempfile::tempdir;

fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let normal: StandardNormal = StandardNormal;
    (0..d)
        .map(|_| {
            let v: f64 = normal.sample(rng);
            v as f32
        })
        .collect()
}

const VECS_PER_ENTRY: usize = 32;

struct BenchConfig {
    d: usize,
    s: usize,
    label: &'static str,
}

const CONFIGS: &[BenchConfig] = &[
    BenchConfig {
        d: 64,
        s: 128,
        label: "d64",
    },
    BenchConfig {
        d: 128,
        s: 256,
        label: "d128",
    },
];

const ENTRY_COUNTS: &[usize] = &[10, 100, 1_000, 10_000];

fn make_store(
    d: usize,
    s: usize,
    num_entries: usize,
    rng: &mut ChaCha20Rng,
) -> (KeyStore, KeysConfig, tempfile::TempDir) {
    let config = KeysConfig {
        dim: d as u16,
        sketch_dim: s as u16,
        outlier_sketch_dim: s as u16,
        seed: 42,
    };
    let sketch = config.build_sketch();
    let dir = tempdir().unwrap();
    let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
    let outlier_indices = vec![0u8];

    for eid in 0..num_entries as u64 {
        let keys = random_vec(VECS_PER_ENTRY * d, rng);
        let compressed = sketch
            .quantize(&keys, VECS_PER_ENTRY, &outlier_indices)
            .unwrap();
        store.append(eid, eid, &compressed).unwrap();
    }

    (store, config, dir)
}

/// CPU baseline: explicit sketch.score() per entry (never GPU)
fn bench_cpu_per_entry(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    for cfg in CONFIGS {
        let mut group = c.benchmark_group(format!("cpu_per_entry_{}", cfg.label));

        for &num_entries in ENTRY_COUNTS {
            let (store, config, _dir) = make_store(cfg.d, cfg.s, num_entries, &mut rng);
            let sketch = config.build_sketch();
            let token = random_vec(cfg.d, &mut rng);

            group.bench_with_input(
                BenchmarkId::new("entries", num_entries),
                &num_entries,
                |b, _| {
                    b.iter(|| {
                        let mut total = 0.0f32;
                        for eid in 0..num_entries as u64 {
                            if let Some(view) = store.get_entry(eid) {
                                let keys = view.to_compressed(cfg.d);
                                let scores =
                                    sketch.score(black_box(&token), black_box(&keys)).unwrap();
                                total += scores.iter().sum::<f32>();
                            }
                        }
                        black_box(total)
                    });
                },
            );
        }

        group.finish();
    }
}

/// scores with auto-dispatch (GPU when >= GPU_MIN_BATCH)
fn bench_scores(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    for cfg in CONFIGS {
        let mut group = c.benchmark_group(format!("scores_{}", cfg.label));

        for &num_entries in ENTRY_COUNTS {
            let (store, config, _dir) = make_store(cfg.d, cfg.s, num_entries, &mut rng);
            let sketch = config.build_sketch();
            let token = random_vec(cfg.d, &mut rng);
            let outlier_indices = vec![0u8];

            group.bench_with_input(
                BenchmarkId::new("entries", num_entries),
                &num_entries,
                |b, _| {
                    b.iter(|| {
                        store.scores(
                            black_box(&token),
                            black_box(&sketch),
                            black_box(&outlier_indices),
                        )
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench_cpu_per_entry, bench_scores);
criterion_main!(benches);
