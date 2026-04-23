use turboquant::store::config::{KeysConfig, ValuesConfig};
use turboquant::store::key_store::KeyStore;
use turboquant::store::value_store::ValueStore;
use turboquant::values::quantize_values;

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

fn keys_config() -> KeysConfig {
    KeysConfig {
        head_dim: 16,
        sketch_dim: 32,
        outlier_sketch_dim: 16,
        seed: 42,
    }
}

fn values_config() -> ValuesConfig {
    ValuesConfig {
        bits: 4,
        group_size: 8,
    }
}

#[test]
fn test_keys_fresh_values_stale() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let vc = values_config();
    let mut key_store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let mut val_store = ValueStore::create(dir.path(), vc).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let slug: u64 = 0xAA;
    let content_v1: u64 = 0x11;
    let content_v2: u64 = 0x22;

    // Write v1 to both stores
    let keys_v1 = random_vec(4 * 16, &mut rng);
    let compressed_keys_v1 = sketch.quantize(&keys_v1, 4, &[0u8]);
    key_store
        .append(slug, content_v1, &compressed_keys_v1)
        .unwrap();

    let values_v1: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let compressed_values_v1 = quantize_values(&values_v1, 8, 4);
    val_store
        .append(slug, content_v1, &compressed_values_v1)
        .unwrap();

    // Both fresh at v1
    assert!(key_store.is_fresh(slug, content_v1));
    assert!(val_store.is_fresh(slug, content_v1));

    // Update keys to v2, leave values at v1
    let keys_v2 = random_vec(4 * 16, &mut rng);
    let compressed_keys_v2 = sketch.quantize(&keys_v2, 4, &[0u8]);
    key_store
        .append(slug, content_v2, &compressed_keys_v2)
        .unwrap();

    // Keys fresh at v2, values stale
    assert!(key_store.is_fresh(slug, content_v2));
    assert!(!key_store.is_fresh(slug, content_v1));
    assert!(!val_store.is_fresh(slug, content_v2));
    assert!(val_store.is_fresh(slug, content_v1));

    // Score still works with updated keys
    let query = random_vec(16, &mut rng);
    let page = key_store.get_page(slug).unwrap();
    let reloaded = page.to_compressed_keys(kc.head_dim as usize);
    let scores = sketch.score(&query, &reloaded);
    assert_eq!(scores.len(), 4);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn test_both_stores_independent_lifecycle() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let vc = values_config();
    let mut key_store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let mut val_store = ValueStore::create(dir.path(), vc).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(200);

    // Add 3 pages to keys, only 2 to values
    for slug in 0u64..3 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        key_store.append(slug, slug * 10, &compressed).unwrap();
    }
    for slug in 0u64..2 {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let compressed = quantize_values(&values, 8, 4);
        val_store.append(slug, slug * 10, &compressed).unwrap();
    }

    assert_eq!(key_store.len(), 3);
    assert_eq!(val_store.len(), 2);

    // Page 2 has keys but no values — valid state
    assert!(key_store.get_page(2).is_some());
    assert!(val_store.get_page(2).is_none());
}

#[test]
fn test_dead_bytes_tracked_after_reopen() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(300);
    let keys = random_vec(4 * 16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8]);

    store.append(0xAA, 0x11, &compressed).unwrap();
    store.append(0xAA, 0x22, &compressed).unwrap();

    let dead_before = store.dead_bytes();
    assert!(dead_before > 0);

    // Reopen and verify dead_bytes persisted
    let store2 = KeyStore::open(dir.path()).unwrap();
    assert_eq!(store2.dead_bytes(), dead_before);
    assert_eq!(store2.len(), 1);
}
