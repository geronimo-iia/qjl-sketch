#![cfg(feature = "serde")]

use qjl_sketch::codebook::{generate_codebook, Codebook};
use qjl_sketch::mse_quant::{mse_quantize, MseQuantized};
use qjl_sketch::quantize::CompressedKeys;
use qjl_sketch::rotation::RandomRotation;
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::store::config::{IndexEntry, IndexMeta, KeysConfig, ValuesConfig};
use qjl_sketch::store::key_store::{KeyExportEntry, KeyStore};
use qjl_sketch::store::value_store::{ValueExportEntry, ValueStore};
use qjl_sketch::values::{quantize_values, CompressedValues};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
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

// ── Data struct round-trips ──────────────────────────────────────────────────

#[test]
fn codebook_roundtrip() {
    let cb = generate_codebook(64, 4, 50).unwrap();
    let json = serde_json::to_string(&cb).unwrap();
    let cb2: Codebook = serde_json::from_str(&json).unwrap();
    assert_eq!(cb.centroids, cb2.centroids);
    assert_eq!(cb.boundaries, cb2.boundaries);
    assert_eq!(cb.bit_width, cb2.bit_width);
}

#[test]
fn compressed_keys_roundtrip() {
    let sketch = QJLSketch::new(16, 32, 16, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let keys = random_vec(4 * 16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8, 1]).unwrap();

    let json = serde_json::to_string(&compressed).unwrap();
    let c2: CompressedKeys = serde_json::from_str(&json).unwrap();
    assert_eq!(compressed.key_quant, c2.key_quant);
    assert_eq!(compressed.key_norms, c2.key_norms);
    assert_eq!(compressed.num_vectors, c2.num_vectors);
}

#[test]
fn compressed_values_roundtrip() {
    let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let compressed = quantize_values(&values, 8, 4).unwrap();

    let json = serde_json::to_string(&compressed).unwrap();
    let c2: CompressedValues = serde_json::from_str(&json).unwrap();
    assert_eq!(compressed.packed, c2.packed);
    assert_eq!(compressed.scale, c2.scale);
    assert_eq!(compressed.mn, c2.mn);
    assert_eq!(compressed.num_elements, c2.num_elements);
}

#[test]
fn mse_quantized_roundtrip() {
    let rot = RandomRotation::new(32, 42).unwrap();
    let cb = generate_codebook(32, 3, 50).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let v = random_vec(32, &mut rng);
    let q = mse_quantize(&v, 1, &rot, &cb).unwrap();

    let json = serde_json::to_string(&q).unwrap();
    let q2: MseQuantized = serde_json::from_str(&json).unwrap();
    assert_eq!(q.indices, q2.indices);
    assert_eq!(q.dim, q2.dim);
    assert_eq!(q.bit_width, q2.bit_width);
}

#[test]
fn keys_config_roundtrip() {
    let config = KeysConfig {
        head_dim: 128,
        sketch_dim: 256,
        outlier_sketch_dim: 64,
        seed: 42,
    };
    let json = serde_json::to_string(&config).unwrap();
    let c2: KeysConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config, c2);
}

#[test]
fn values_config_roundtrip() {
    let config = ValuesConfig {
        bits: 4,
        group_size: 32,
    };
    let json = serde_json::to_string(&config).unwrap();
    let c2: ValuesConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config, c2);
}

#[test]
fn index_entry_roundtrip() {
    let entry = IndexEntry {
        slug_hash: 0xDEAD,
        offset: 1024,
        entry_len: 500,
        generation: 3,
        content_hash: 0xBEEF,
    };
    let json = serde_json::to_string(&entry).unwrap();
    let e2: IndexEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(entry, e2);
}

#[test]
fn index_meta_roundtrip() {
    let meta = IndexMeta {
        entry_count: 42,
        live_bytes: 10000,
        dead_bytes: 500,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let m2: IndexMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(meta.entry_count, m2.entry_count);
    assert_eq!(meta.live_bytes, m2.live_bytes);
    assert_eq!(meta.dead_bytes, m2.dead_bytes);
}

// ── Reconstructible struct round-trips ───────────────────────────────────────

#[test]
fn qjl_sketch_roundtrip() {
    let sketch = QJLSketch::new(64, 128, 32, 42).unwrap();
    let json = serde_json::to_string(&sketch).unwrap();
    let s2: QJLSketch = serde_json::from_str(&json).unwrap();
    assert_eq!(sketch.head_dim, s2.head_dim);
    assert_eq!(sketch.sketch_dim, s2.sketch_dim);
    assert_eq!(sketch.seed, s2.seed);
    assert_eq!(sketch.proj_dir_score, s2.proj_dir_score);
    assert_eq!(sketch.proj_dir_quant, s2.proj_dir_quant);
}

#[test]
fn random_rotation_roundtrip() {
    let rot = RandomRotation::new(32, 99).unwrap();
    let json = serde_json::to_string(&rot).unwrap();
    let r2: RandomRotation = serde_json::from_str(&json).unwrap();
    assert_eq!(rot.dim, r2.dim);
    assert_eq!(rot.seed, r2.seed);
    // Matrices are reconstructed from seed — verify via rotate
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let y1 = rot.rotate(&x).unwrap();
    let y2 = r2.rotate(&x).unwrap();
    assert_eq!(y1, y2);
}

// ── Store export/import round-trips ──────────────────────────────────────────

#[test]
fn key_store_export_import_roundtrip() {
    let dir_src = tempdir().unwrap();
    let dir_dst = tempdir().unwrap();
    let config = KeysConfig {
        head_dim: 16,
        sketch_dim: 32,
        outlier_sketch_dim: 16,
        seed: 42,
    };
    let sketch = config.build_sketch();
    let mut store_src = KeyStore::create(dir_src.path(), config.clone()).unwrap();

    let mut rng = ChaCha20Rng::seed_from_u64(500);
    for slug in 0u64..5 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]).unwrap();
        store_src.append(slug, slug * 100, &compressed).unwrap();
    }

    // Export → serialize → deserialize → import
    let mut store_dst = KeyStore::create(dir_dst.path(), config.clone()).unwrap();
    for entry in store_src.iter_pages() {
        let json = serde_json::to_string(&entry).unwrap();
        let entry2: KeyExportEntry = serde_json::from_str(&json).unwrap();
        store_dst.import_entry(&entry2).unwrap();
    }

    assert_eq!(store_dst.len(), 5);

    // Verify scores match
    let query = random_vec(16, &mut rng);
    for slug in 0u64..5 {
        let page_src = store_src.get_page(slug).unwrap();
        let page_dst = store_dst.get_page(slug).unwrap();
        let ck_src = page_src.to_compressed_keys(16);
        let ck_dst = page_dst.to_compressed_keys(16);
        let scores_src = sketch.score(&query, &ck_src).unwrap();
        let scores_dst = sketch.score(&query, &ck_dst).unwrap();
        assert_eq!(scores_src, scores_dst);
    }
}

#[test]
fn value_store_export_import_roundtrip() {
    let dir_src = tempdir().unwrap();
    let dir_dst = tempdir().unwrap();
    let config = ValuesConfig {
        bits: 4,
        group_size: 8,
    };
    let mut store_src = ValueStore::create(dir_src.path(), config.clone()).unwrap();

    for slug in 0u64..5 {
        let values: Vec<f32> = (0..8).map(|i| (slug as f32) + i as f32).collect();
        let compressed = quantize_values(&values, 8, 4).unwrap();
        store_src.append(slug, slug * 10, &compressed).unwrap();
    }

    let mut store_dst = ValueStore::create(dir_dst.path(), config).unwrap();
    for entry in store_src.iter_pages() {
        let json = serde_json::to_string(&entry).unwrap();
        let entry2: ValueExportEntry = serde_json::from_str(&json).unwrap();
        store_dst.import_entry(&entry2).unwrap();
    }

    assert_eq!(store_dst.len(), 5);
    for slug in 0u64..5 {
        let p_src = store_src.get_page(slug).unwrap();
        let p_dst = store_dst.get_page(slug).unwrap();
        assert_eq!(p_src.packed(), p_dst.packed());
        assert_eq!(p_src.scale(), p_dst.scale());
        assert_eq!(p_src.mn(), p_dst.mn());
    }
}
