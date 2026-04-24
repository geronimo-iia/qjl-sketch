//! MSE-optimal quantization: rotation + Lloyd-Max per-coordinate.
//!
//! Run with: `cargo run --example mse_quantization`

use qjl_sketch::codebook::generate_codebook;
use qjl_sketch::mse_quant::{mse_dequantize, mse_quantize, mse_score};
use qjl_sketch::rotation::RandomRotation;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn random_unit_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let normal: StandardNormal = StandardNormal;
    let mut v: Vec<f32> = (0..d)
        .map(|_| {
            let x: f64 = normal.sample(rng);
            x as f32
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn main() {
    let dim = 128;
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let rot = RandomRotation::new(dim, 77).unwrap();
    println!("RandomRotation: dim={dim}, seed={}", rot.seed);

    // Compare 2-bit vs 4-bit quantization
    for bits in [2, 4] {
        let cb = generate_codebook(dim, bits, 100).unwrap();
        println!("\n--- {bits}-bit codebook ({} levels) ---", cb.num_levels());

        // Quantize a unit vector
        let v = random_unit_vec(dim, &mut rng);
        let quantized = mse_quantize(&v, 1, &rot, &cb).unwrap();
        let reconstructed = mse_dequantize(&quantized, &rot, &cb).unwrap();

        // MSE
        let mse: f32 = v
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;
        println!("MSE per dimension: {mse:.6}");

        // Score accuracy
        let query = random_unit_vec(dim, &mut rng);
        let exact: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let scores = mse_score(&query, &quantized, &rot, &cb).unwrap();
        let error = (scores[0] - exact).abs();
        println!(
            "Score: approx={:.4}, exact={exact:.4}, error={error:.4}",
            scores[0]
        );

        // Storage
        let bytes = quantized.indices.len() * bits as usize / 8;
        let original = dim * 4;
        println!(
            "Storage: {bytes} bytes ({:.0}x compression)",
            original as f32 / bytes as f32
        );
    }
}
