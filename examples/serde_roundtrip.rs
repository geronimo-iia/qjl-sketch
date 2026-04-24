//! Serde round-trip example: serialize and deserialize core structs to JSON.
//!
//! Run with: `cargo run --example serde_roundtrip --features serde`

use qjl_sketch::codebook::generate_codebook;
use qjl_sketch::rotation::RandomRotation;
use qjl_sketch::sketch::QJLSketch;

fn main() {
    // Codebook
    let cb = generate_codebook(64, 4, 100).unwrap();
    let json = serde_json::to_string_pretty(&cb).unwrap();
    println!(
        "Codebook JSON ({} bytes):\n{}\n",
        json.len(),
        &json[..200.min(json.len())]
    );
    let cb2: qjl_sketch::codebook::Codebook = serde_json::from_str(&json).unwrap();
    assert_eq!(cb.centroids, cb2.centroids);
    println!("✓ Codebook round-trip OK\n");

    // QJLSketch — serializes as params only, reconstructs matrices
    let sketch = QJLSketch::new(64, 128, 32, 42).unwrap();
    let json = serde_json::to_string_pretty(&sketch).unwrap();
    println!("QJLSketch JSON ({} bytes):\n{}\n", json.len(), json);
    let s2: QJLSketch = serde_json::from_str(&json).unwrap();
    assert_eq!(sketch.proj_dir_score, s2.proj_dir_score);
    println!("✓ QJLSketch round-trip OK (matrices reconstructed from seed)\n");

    // RandomRotation — same pattern
    let rot = RandomRotation::new(32, 99).unwrap();
    let json = serde_json::to_string_pretty(&rot).unwrap();
    println!("RandomRotation JSON ({} bytes):\n{}\n", json.len(), json);
    let r2: RandomRotation = serde_json::from_str(&json).unwrap();
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    assert_eq!(rot.rotate(&x).unwrap(), r2.rotate(&x).unwrap());
    println!("✓ RandomRotation round-trip OK\n");
}
