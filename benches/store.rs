use criterion::{criterion_group, criterion_main, Criterion};
use qjl_sketch::outliers::detect_outliers;
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

fn keys_config() -> KeysConfig {
    KeysConfig {
        dim: 128,
        sketch_dim: 256,
        outlier_sketch_dim: 64,
        seed: 42,
    }
}

fn bench_cold_start(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = keys_config();
    let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
    let sketch = config.build_sketch();
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    // Pre-populate with 100 entries
    for eid in 0u64..100 {
        let keys = random_vec(32 * 128, &mut rng);
        let outlier_indices = detect_outliers(&keys, 32, 128, 4).unwrap();
        let compressed = sketch.quantize(&keys, 32, &outlier_indices).unwrap();
        store.append(eid, eid * 10, &compressed).unwrap();
    }
    drop(store);

    c.bench_function("cold_start_100_entries", |b| {
        b.iter(|| {
            let store = KeyStore::open(black_box(dir.path())).unwrap();
            black_box(store.len());
        });
    });
}

fn bench_append(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = keys_config();
    let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
    let sketch = config.build_sketch();
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    let keys = random_vec(32 * 128, &mut rng);
    let outlier_indices = detect_outliers(&keys, 32, 128, 4).unwrap();
    let compressed = sketch.quantize(&keys, 32, &outlier_indices).unwrap();

    let mut eid = 0u64;
    c.bench_function("append_single_page", |b| {
        b.iter(|| {
            store.append(eid, eid * 10, black_box(&compressed)).unwrap();
            eid += 1;
        });
    });
}

fn bench_get_entry(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = keys_config();
    let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
    let sketch = config.build_sketch();
    let mut rng = ChaCha20Rng::seed_from_u64(300);

    for eid in 0u64..100 {
        let keys = random_vec(32 * 128, &mut rng);
        let outlier_indices = detect_outliers(&keys, 32, 128, 4).unwrap();
        let compressed = sketch.quantize(&keys, 32, &outlier_indices).unwrap();
        store.append(eid, eid * 10, &compressed).unwrap();
    }

    c.bench_function("get_entry_from_100_entries", |b| {
        let mut eid = 0u64;
        b.iter(|| {
            let view = store.get_entry(black_box(eid % 100)).unwrap();
            black_box(view.num_vectors);
            eid += 1;
        });
    });
}

criterion_group!(benches, bench_cold_start, bench_append, bench_get_entry);
criterion_main!(benches);
