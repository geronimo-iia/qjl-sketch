//! Compressed-vs-compressed scoring: compare two vector groups without decompressing.
//!
//! Run with: `cargo run --example compressed_scoring`

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
    let d = 64;
    let s = 256;
    let sketch = QJLSketch::new(d, s, s, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    // Compress two sets of vectors (e.g. two vector groups)
    let num = 8;
    let group_a: Vec<f32> = (0..num).flat_map(|_| random_vec(d, &mut rng)).collect();
    let group_b: Vec<f32> = (0..num).flat_map(|_| random_vec(d, &mut rng)).collect();

    let outliers = vec![0u8];
    let ca = sketch.quantize(&group_a, num, &outliers).unwrap();
    let cb = sketch.quantize(&group_b, num, &outliers).unwrap();

    // Batch scoring: a[i] vs b[i]
    let batch_scores = sketch.score_compressed(&ca, &cb).unwrap();
    println!("Batch scores (a[i] vs b[i]):");
    for (i, s) in batch_scores.iter().enumerate() {
        println!("  pair {i}: {s:+.4}");
    }

    // Cross-pair scoring: any a[i] vs any b[j]
    println!("\nCross-pair scores (a[0] vs each b[j]):");
    for j in 0..num {
        let s = sketch.score_compressed_pair(&ca, 0, &cb, j).unwrap();
        println!("  a[0] vs b[{j}]: {s:+.4}");
    }

    // Self-similarity: a[i] vs a[i] should be high (≈ ||v||²)
    let self_scores = sketch.score_compressed(&ca, &ca).unwrap();
    println!("\nSelf-similarity (a[i] vs a[i]):");
    for (i, s) in self_scores.iter().enumerate() {
        let exact_norm_sq: f32 = group_a[i * d..(i + 1) * d].iter().map(|x| x * x).sum();
        println!("  vec {i}: score={s:.4}, ||v||²={exact_norm_sq:.4}");
    }
}
