//! Store export/import example: streaming JSONL export from one KeyStore,
//! import into another, verify scores match.
//!
//! Run with: `cargo run --example store_export_import --features serde`

use qjl_sketch::store::config::KeysConfig;
use qjl_sketch::store::key_store::{KeyExportEntry, KeyStore};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::io::{BufRead, BufReader, BufWriter, Write};
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

fn main() {
    let config = KeysConfig {
        dim: 16,
        sketch_dim: 32,
        outlier_sketch_dim: 16,
        seed: 42,
    };
    let sketch = config.build_sketch();

    // Create source store with 10 entries
    let dir_src = tempdir().unwrap();
    let mut store_src = KeyStore::create(dir_src.path(), config.clone()).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    for eid in 0u64..10 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]).unwrap();
        store_src.append(eid, eid * 100, &compressed).unwrap();
    }
    println!("Source store: entries: {}", store_src.len());

    // Export to JSONL file (streaming — one line per entry)
    let dump_path = dir_src.path().join("keys.jsonl");
    {
        let file = std::fs::File::create(&dump_path).unwrap();
        let mut writer = BufWriter::new(file);
        for entry in store_src.iter_entries() {
            serde_json::to_writer(&mut writer, &entry).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.flush().unwrap();
    }
    let dump_size = std::fs::metadata(&dump_path).unwrap().len();
    println!("Exported to JSONL: {} bytes", dump_size);

    // Import from JSONL into a new store (streaming — one line at a time)
    let dir_dst = tempdir().unwrap();
    let mut store_dst = KeyStore::create(dir_dst.path(), config).unwrap();
    {
        let file = std::fs::File::open(&dump_path).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let entry: KeyExportEntry = serde_json::from_str(&line.unwrap()).unwrap();
            store_dst.import_entry(&entry).unwrap();
        }
    }
    println!(
        "Imported into destination store: entries: {}",
        store_dst.len()
    );

    // Verify scores match
    let token = random_vec(16, &mut rng);
    for eid in 0u64..10 {
        let ck_src = store_src.get_entry(eid).unwrap().to_compressed(16);
        let ck_dst = store_dst.get_entry(eid).unwrap().to_compressed(16);
        let s1 = sketch.score(&token, &ck_src).unwrap();
        let s2 = sketch.score(&token, &ck_dst).unwrap();
        assert_eq!(s1, s2, "score mismatch for entry {eid}");
    }
    println!("✓ All scores match between source and destination stores");
}
