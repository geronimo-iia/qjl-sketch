// Random orthogonal rotation for TurboQuant decorrelation.
//
// Generates a d×d orthogonal matrix via QR decomposition of a
// random Gaussian matrix, with sign correction to ensure a proper
// rotation (det = +1). This is the standard Haar-uniform random
// orthogonal matrix construction.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{validate_finite, QjlError, Result};

/// A d×d random orthogonal rotation matrix.
///
/// Used as the decorrelation step before Lloyd-Max scalar quantization.
/// After rotation, coordinates of a unit-sphere vector are approximately
/// i.i.d. Beta(1/2, (d-1)/2), making per-coordinate quantization optimal.
pub struct RandomRotation {
    /// d×d orthogonal matrix, row-major.
    matrix: Vec<f32>,
    /// d×d transposed matrix (cached for inverse rotation).
    matrix_t: Vec<f32>,
    /// Vector dimension.
    pub dim: usize,
}

impl RandomRotation {
    /// Create a random orthogonal rotation for dimension `dim`.
    ///
    /// Algorithm:
    /// 1. Sample d×d Gaussian matrix
    /// 2. QR decompose → Q is orthogonal
    /// 3. Correct column signs so R diagonal is positive (ensures proper rotation)
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QjlError::InvalidDimension(dim));
        }

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let normal = StandardNormal;

        let data: Vec<f64> = (0..dim * dim).map(|_| normal.sample(&mut rng)).collect();
        let g = DMatrix::from_vec(dim, dim, data);

        let qr = g.qr();
        let q = qr.q();
        let r = qr.r();

        // Sign correction: flip columns where R diagonal is negative
        let signs: Vec<f64> = (0..dim)
            .map(|i| if r[(i, i)] < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let sign_diag = DMatrix::from_diagonal(&DVector::from_vec(signs));
        let mat = q * sign_diag;

        // Convert to f32 row-major
        let mut matrix = vec![0.0f32; dim * dim];
        let mut matrix_t = vec![0.0f32; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                let v = mat[(r, c)] as f32;
                matrix[r * dim + c] = v;
                matrix_t[c * dim + r] = v;
            }
        }

        Ok(Self {
            matrix,
            matrix_t,
            dim,
        })
    }

    /// Apply rotation: y = Π·x
    pub fn rotate(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.dim {
            return Err(QjlError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        validate_finite(x, "rotation input")?;
        Ok(matvec_square(&self.matrix, self.dim, x))
    }

    /// Apply inverse rotation: x = Πᵀ·y
    pub fn rotate_inverse(&self, y: &[f32]) -> Result<Vec<f32>> {
        if y.len() != self.dim {
            return Err(QjlError::DimensionMismatch {
                expected: self.dim,
                got: y.len(),
            });
        }
        validate_finite(y, "inverse rotation input")?;
        Ok(matvec_square(&self.matrix_t, self.dim, y))
    }
}

/// Matrix-vector product for a d×d row-major matrix.
fn matvec_square(mat: &[f32], dim: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for r in 0..dim {
        let row = &mat[r * dim..(r + 1) * dim];
        let mut acc = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            acc += a * b;
        }
        out[r] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonality() {
        let dim = 16;
        let rot = RandomRotation::new(dim, 42).unwrap();
        // Π·Πᵀ ≈ I
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += rot.matrix[i * dim + k] * rot.matrix[j * dim + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "[{i},{j}]: expected {expected}, got {dot}"
                );
            }
        }
    }

    #[test]
    fn roundtrip() {
        let dim = 32;
        let rot = RandomRotation::new(dim, 123).unwrap();
        let x: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let y = rot.rotate(&x).unwrap();
        let x2 = rot.rotate_inverse(&y).unwrap();
        for (a, b) in x.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-4, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn preserves_norm() {
        let dim = 64;
        let rot = RandomRotation::new(dim, 7).unwrap();
        let x: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y = rot.rotate(&x).unwrap();
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_x - norm_y).abs() < 1e-3,
            "norm not preserved: {norm_x} vs {norm_y}"
        );
    }

    #[test]
    fn deterministic() {
        let a = RandomRotation::new(16, 42).unwrap();
        let b = RandomRotation::new(16, 42).unwrap();
        assert_eq!(a.matrix, b.matrix);
    }

    #[test]
    fn different_seeds() {
        let a = RandomRotation::new(16, 42).unwrap();
        let b = RandomRotation::new(16, 99).unwrap();
        assert_ne!(a.matrix, b.matrix);
    }

    #[test]
    fn dimension_mismatch() {
        let rot = RandomRotation::new(4, 1).unwrap();
        assert!(rot.rotate(&[1.0, 2.0]).is_err());
        assert!(rot.rotate_inverse(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn invalid_dimension_zero() {
        assert!(RandomRotation::new(0, 1).is_err());
    }
}
