//! Basic QJL compression and scoring example.
//!
//! Run with: `cargo run --example basic_qjl`

use qjl_sketch::outliers::detect_outliers;
use qjl_sketch::sketch::QJLSketch;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

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
    let d = 128;
    let s = 256;
    let num_keys = 32;
    let outlier_count = 4;

    // Create a sketch (deterministic from seed)
    let sketch = QJLSketch::new(d, s, 64, 42).unwrap();
    println!("Sketch: d={d}, s={s}, seed={}", sketch.seed);

    // Generate random key vectors
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let keys: Vec<f32> = (0..num_keys)
        .flat_map(|_| random_vec(d, &mut rng))
        .collect();

    // Detect outlier dimensions and compress
    let outlier_indices = detect_outliers(&keys, num_keys, d, outlier_count).unwrap();
    println!("Outlier dims: {outlier_indices:?}");

    let compressed = sketch.quantize(&keys, num_keys, &outlier_indices).unwrap();
    let bits_per_vector = s; // 1 bit per projection
    let bytes_per_vector = bits_per_vector / 8;
    let original_bytes = d * 4; // f32
    println!(
        "Compressed {num_keys} vectors: {} bytes/vec (was {} bytes/vec, {:.0}x compression)",
        bytes_per_vector,
        original_bytes,
        original_bytes as f32 / bytes_per_vector as f32
    );

    // Score a token against compressed keys
    let token = random_vec(d, &mut rng);
    let scores = sketch.score(&token, &compressed).unwrap();

    // Compare with exact dot products
    let exact_scores: Vec<f32> = (0..num_keys)
        .map(|i| {
            token
                .iter()
                .zip(keys[i * d..(i + 1) * d].iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();

    println!("\nTop-5 by approximate score:");
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(i, approx) in ranked.iter().take(5) {
        println!(
            "  vec {i:3}: approx={approx:+.4}, exact={:+.4}",
            exact_scores[i]
        );
    }
}
