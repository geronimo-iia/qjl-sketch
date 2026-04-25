// MSE-optimal vector quantization: rotate → Lloyd-Max per-coordinate.
//
// This is Stage 1 of TurboQuant. The random rotation decorrelates
// coordinates so that per-coordinate scalar quantization achieves
// the optimal rate-distortion trade-off for MSE.

use crate::codebook::Codebook;
use crate::error::{validate_finite, QjlError, Result};
use crate::rotation::RandomRotation;

/// MSE-quantized vectors: per-coordinate codebook indices after rotation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MseQuantized {
    /// Codebook indices, flattened \[num_vectors × dim\].
    pub indices: Vec<u8>,
    /// Number of vectors.
    pub num_vectors: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Bits per coordinate.
    pub bit_width: u8,
}

/// Quantize vectors via rotation + Lloyd-Max scalar quantization.
///
/// For each vector: rotate → quantize each coordinate via codebook.
///
/// - `vectors`: flattened \[num_vectors, dim\] row-major
pub fn mse_quantize(
    vectors: &[f32],
    num_vectors: usize,
    rotation: &RandomRotation,
    codebook: &Codebook,
) -> Result<MseQuantized> {
    let dim = rotation.dim;
    if vectors.len() != num_vectors * dim {
        return Err(QjlError::DimensionMismatch {
            expected: num_vectors * dim,
            got: vectors.len(),
        });
    }
    validate_finite(vectors, "mse_quantize input")?;

    let mut indices = Vec::with_capacity(num_vectors * dim);

    for v in 0..num_vectors {
        let x = &vectors[v * dim..(v + 1) * dim];
        let y = rotation.rotate(x)?;
        for &val in &y {
            indices.push(codebook.quantize(val));
        }
    }

    Ok(MseQuantized {
        indices,
        num_vectors,
        dim,
        bit_width: codebook.bit_width,
    })
}

/// Reconstruct vectors from MSE-quantized representation.
///
/// For each vector: dequantize each coordinate → inverse rotate.
pub fn mse_dequantize(
    quantized: &MseQuantized,
    rotation: &RandomRotation,
    codebook: &Codebook,
) -> Result<Vec<f32>> {
    if rotation.dim != quantized.dim {
        return Err(QjlError::DimensionMismatch {
            expected: quantized.dim,
            got: rotation.dim,
        });
    }

    let dim = quantized.dim;
    let mut out = Vec::with_capacity(quantized.num_vectors * dim);

    for v in 0..quantized.num_vectors {
        let idx_slice = &quantized.indices[v * dim..(v + 1) * dim];
        let y_approx: Vec<f32> = idx_slice.iter().map(|&i| codebook.dequantize(i)).collect();
        let x_approx = rotation.rotate_inverse(&y_approx)?;
        out.extend_from_slice(&x_approx);
    }

    Ok(out)
}

/// Score a token against MSE-quantized vectors.
///
/// Rotates the token once, then dots with dequantized rotated
/// coordinates directly: dot(Π·q, ỹ) = dot(q, Πᵀ·ỹ) by orthogonality.
pub fn mse_score(
    token: &[f32],
    quantized: &MseQuantized,
    rotation: &RandomRotation,
    codebook: &Codebook,
) -> Result<Vec<f32>> {
    let dim = rotation.dim;
    if token.len() != dim {
        return Err(QjlError::DimensionMismatch {
            expected: dim,
            got: token.len(),
        });
    }
    if quantized.dim != dim {
        return Err(QjlError::DimensionMismatch {
            expected: dim,
            got: quantized.dim,
        });
    }
    validate_finite(token, "mse_score token")?;

    let q_rot = rotation.rotate(token)?;

    let scores = (0..quantized.num_vectors)
        .map(|v| {
            let idx_slice = &quantized.indices[v * dim..(v + 1) * dim];
            q_rot
                .iter()
                .zip(idx_slice.iter())
                .map(|(&q, &i)| q * codebook.dequantize(i))
                .sum()
        })
        .collect();

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::generate_codebook;
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

    fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    #[test]
    fn roundtrip_reconstruction() {
        let dim = 64;
        let rot = RandomRotation::new(dim, 42).unwrap();
        let cb = generate_codebook(dim, 4, 100).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(100);

        let mut v = random_vec(dim, &mut rng);
        normalize(&mut v);

        let q = mse_quantize(&v, 1, &rot, &cb).unwrap();
        let recon = mse_dequantize(&q, &rot, &cb).unwrap();

        // MSE should be small for 4-bit quantization of a unit vector
        let mse: f32 = v
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;
        assert!(mse < 0.1, "MSE too high: {mse}");
    }

    #[test]
    fn score_vs_exact() {
        let dim = 64;
        let rot = RandomRotation::new(dim, 42).unwrap();
        let cb = generate_codebook(dim, 4, 100).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(200);

        let mut q_vec = random_vec(dim, &mut rng);
        normalize(&mut q_vec);
        let mut k_vec = random_vec(dim, &mut rng);
        normalize(&mut k_vec);

        let exact: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();

        let quantized = mse_quantize(&k_vec, 1, &rot, &cb).unwrap();
        let scores = mse_score(&q_vec, &quantized, &rot, &cb).unwrap();

        let error = (scores[0] - exact).abs();
        assert!(
            error < 0.3,
            "score error too high: approx={}, exact={exact}, error={error}",
            scores[0]
        );
    }

    #[test]
    fn mse_decreases_with_bits() {
        let dim = 128;
        let rot = RandomRotation::new(dim, 42).unwrap();
        let cb2 = generate_codebook(dim, 2, 100).unwrap();
        let cb4 = generate_codebook(dim, 4, 100).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let mut v = random_vec(dim, &mut rng);
        normalize(&mut v);

        let q2 = mse_quantize(&v, 1, &rot, &cb2).unwrap();
        let r2 = mse_dequantize(&q2, &rot, &cb2).unwrap();
        let mse2: f32 = v
            .iter()
            .zip(r2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        let q4 = mse_quantize(&v, 1, &rot, &cb4).unwrap();
        let r4 = mse_dequantize(&q4, &rot, &cb4).unwrap();
        let mse4: f32 = v
            .iter()
            .zip(r4.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        assert!(
            mse4 < mse2,
            "4-bit MSE ({mse4}) should be less than 2-bit MSE ({mse2})"
        );
    }

    #[test]
    fn dimension_mismatch_quantize() {
        let rot = RandomRotation::new(16, 1).unwrap();
        let cb = generate_codebook(16, 2, 50).unwrap();
        assert!(mse_quantize(&[1.0; 10], 1, &rot, &cb).is_err());
    }

    #[test]
    fn dimension_mismatch_score() {
        let rot = RandomRotation::new(16, 1).unwrap();
        let cb = generate_codebook(16, 2, 50).unwrap();
        let v = vec![0.1f32; 16];
        let q = mse_quantize(&v, 1, &rot, &cb).unwrap();
        assert!(mse_score(&[1.0; 8], &q, &rot, &cb).is_err());
    }

    #[test]
    fn multiple_vectors() {
        let dim = 32;
        let rot = RandomRotation::new(dim, 42).unwrap();
        let cb = generate_codebook(dim, 3, 50).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(400);

        let num = 5;
        let vecs: Vec<f32> = (0..num).flat_map(|_| random_vec(dim, &mut rng)).collect();
        let token = random_vec(dim, &mut rng);

        let q = mse_quantize(&vecs, num, &rot, &cb).unwrap();
        assert_eq!(q.indices.len(), num * dim);
        assert_eq!(q.num_vectors, num);

        let scores = mse_score(&token, &q, &rot, &cb).unwrap();
        assert_eq!(scores.len(), num);
        for s in &scores {
            assert!(s.is_finite());
        }

        let recon = mse_dequantize(&q, &rot, &cb).unwrap();
        assert_eq!(recon.len(), num * dim);
    }
}
