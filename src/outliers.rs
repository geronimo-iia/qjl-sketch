use crate::error::{QjlError, Result};

/// Detect outlier dimensions within a group of vectors.
pub fn detect_outliers(
    keys: &[f32],
    group_size: usize,
    dim: usize,
    count: usize,
) -> Result<Vec<u8>> {
    if keys.len() != group_size * dim {
        return Err(QjlError::DimensionMismatch {
            expected: group_size * dim,
            got: keys.len(),
        });
    }
    if count > dim {
        return Err(QjlError::DimensionMismatch {
            expected: dim,
            got: count,
        });
    }
    if dim > 256 {
        return Err(QjlError::InvalidSketchDim(dim));
    }

    // L2 norm per dimension across the group
    let mut dim_norms = vec![0.0f32; dim];
    for t in 0..group_size {
        let row = &keys[t * dim..(t + 1) * dim];
        for (d, &val) in row.iter().enumerate() {
            dim_norms[d] += val * val;
        }
    }
    for n in &mut dim_norms {
        *n = n.sqrt();
    }

    // Top-k by norm (partial sort)
    let mut indices: Vec<u8> = (0..dim as u8).collect();
    indices.select_nth_unstable_by(count.saturating_sub(1), |&a, &b| {
        dim_norms[b as usize]
            .partial_cmp(&dim_norms[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(count);
    indices.sort_unstable();
    Ok(indices)
}

/// Build an outlier mask from indices. mask\[i\] = true if i is an outlier.
pub fn outlier_mask(indices: &[u8], dim: usize) -> Vec<bool> {
    let mut mask = vec![false; dim];
    for &idx in indices {
        mask[idx as usize] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_known_spike() {
        let dim = 4;
        let group_size = 8;
        let mut keys = vec![0.1f32; group_size * dim];
        // Make dimension 2 a massive outlier
        for t in 0..group_size {
            keys[t * dim + 2] = 100.0;
        }
        let indices = detect_outliers(&keys, group_size, dim, 1).unwrap();
        assert_eq!(indices, vec![2]);
    }

    #[test]
    fn test_outlier_count_respected() {
        let dim = 8;
        let group_size = 4;
        let keys = vec![1.0f32; group_size * dim];
        let indices = detect_outliers(&keys, group_size, dim, 3).unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_outlier_multiple() {
        let dim = 4;
        let group_size = 4;
        let mut keys = vec![0.1f32; group_size * dim];
        // Dims 1 and 3 are outliers
        for t in 0..group_size {
            keys[t * dim + 1] = 50.0;
            keys[t * dim + 3] = 80.0;
        }
        let indices = detect_outliers(&keys, group_size, dim, 2).unwrap();
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_outlier_mask() {
        let mask = outlier_mask(&[1, 3], 5);
        assert_eq!(mask, vec![false, true, false, true, false]);
    }

    #[test]
    fn test_outlier_dimension_mismatch() {
        let keys = vec![1.0f32; 10];
        assert!(detect_outliers(&keys, 4, 3, 1).is_err());
    }

    #[test]
    fn test_outlier_count_exceeds_dim() {
        let keys = vec![1.0f32; 16];
        assert!(detect_outliers(&keys, 4, 4, 5).is_err());
    }
}
