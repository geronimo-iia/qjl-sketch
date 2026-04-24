use super::helpers::*;
use qjl_sketch::codebook::generate_codebook;
use qjl_sketch::mse_quant::{mse_quantize, mse_score};
use qjl_sketch::rotation::RandomRotation;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[test]
fn test_mse_ranking_preservation() {
    let dim = 64;
    let num_keys = 100;
    let num_trials = 50;
    let rot = RandomRotation::new(dim, 42).unwrap();
    let cb = generate_codebook(dim, 4, 100).unwrap();

    let mut tau_sum = 0.0f32;

    for trial in 0..num_trials {
        let mut rng = ChaCha20Rng::seed_from_u64(800 + trial);
        let mut q = random_vec(dim, &mut rng);
        normalize(&mut q);

        let mut keys = Vec::with_capacity(num_keys * dim);
        for _ in 0..num_keys {
            let mut k = random_vec(dim, &mut rng);
            normalize(&mut k);
            keys.extend_from_slice(&k);
        }

        let exact_scores: Vec<f32> = (0..num_keys)
            .map(|i| dot(&q, &keys[i * dim..(i + 1) * dim]))
            .collect();
        let exact_ranking = argsort_desc(&exact_scores);

        let quantized = mse_quantize(&keys, num_keys, &rot, &cb).unwrap();
        let approx_scores = mse_score(&q, &quantized, &rot, &cb).unwrap();
        let approx_ranking = argsort_desc(&approx_scores);

        tau_sum += kendall_tau(&exact_ranking, &approx_ranking);
    }

    let mean_tau = tau_sum / num_trials as f32;
    assert!(
        mean_tau > 0.60,
        "mean Kendall's tau {mean_tau:.3} <= 0.60 over {num_trials} trials"
    );
}
