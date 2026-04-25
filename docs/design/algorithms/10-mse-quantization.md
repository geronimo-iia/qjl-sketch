# Algorithm 10: MSE-Optimal Quantization

Source: TurboQuant paper (Stage 1), arXiv:2504.19874.
Impl: `src/rotation.rs` → `RandomRotation`, `src/mse_quant.rs` →
`mse_quantize`, `mse_dequantize`, `mse_score`.

Achieves the optimal rate-distortion trade-off for MSE by combining
a random orthogonal rotation (decorrelation) with Lloyd-Max scalar
quantization per coordinate.

## Random Rotation

```
Input:
  dim  = vector dimension
  seed = RNG seed

Output:
  Π : d×d orthogonal matrix (proper rotation, det = +1)

Algorithm:
  1. Sample d×d Gaussian matrix G
  2. QR decompose: G = QR
  3. Sign-correct: flip columns of Q where R diagonal < 0
  4. Π = Q · diag(signs)
```

After rotation, coordinates of a unit-sphere vector are approximately
i.i.d. Beta(1/2, (d-1)/2), making per-coordinate quantization optimal.

## Quantize

```
Input:
  vectors  : [num_vectors, d] f32
  rotation : RandomRotation (d×d)
  codebook : Codebook (from Algorithm 7)

Output:
  indices : [num_vectors × d] u8  — codebook index per coordinate

Algorithm:
  For each vector x:
    1. y = Π · x                    (rotate)
    2. For each j: idx_j = codebook.quantize(y_j)  (scalar quantize)
```

## Dequantize

```
Input:
  indices  : [num_vectors × d] u8
  rotation : RandomRotation
  codebook : Codebook

Output:
  vectors : [num_vectors, d] f32

Algorithm:
  For each quantized vector:
    1. ỹ_j = codebook.dequantize(idx_j)   (centroid lookup)
    2. x̃ = Πᵀ · ỹ                         (inverse rotate)
```

## Score

```
Input:
  token     : [d] f32
  quantized : MseQuantized
  rotation  : RandomRotation
  codebook  : Codebook

Output:
  scores : [num_vectors] f32

Algorithm:
  1. q_rot = Π · token                    (rotate token once)
  2. For each quantized vector v:
       score = Σ_j q_rot[j] · codebook.dequantize(indices[v,j])
```

The scoring avoids inverse rotation of every stored vector.
By orthogonality: dot(q, Πᵀ·ỹ) = dot(Π·q, ỹ). The token is
rotated once (O(d²)), then each score is O(d) centroid lookups + dot.

## Theoretical MSE bound

Δ_MSE(b) = √3 · π/2 · (1/4)^b

For b=4: ≈ 0.011 per dimension. MSE decreases exponentially with bits.
