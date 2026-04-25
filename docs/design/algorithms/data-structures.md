# Data Structures

```rust
// src/sketch.rs
pub struct QJLSketch {
    pub dim: usize,
    pub sketch_dim: usize,
    pub outlier_sketch_dim: usize,
    pub proj_dir_score: Vec<f32>,     // [dim, sketch_dim] row-major
    pub proj_dir_quant: Vec<f32>,     // [sketch_dim, dim] row-major
}

// src/quantize.rs
pub struct CompressedKeys {
    pub key_quant: Vec<u8>,           // [num_vectors, sketch_dim/8]
    pub key_outlier_quant: Vec<u8>,   // [num_vectors, outlier_sketch_dim/8]
    pub key_norms: Vec<f32>,          // [num_vectors]
    pub outlier_norms: Vec<f32>,      // [num_vectors]
    pub outlier_indices: Vec<u8>,     // [outlier_count] per group
    pub num_vectors: usize,
    pub dim: usize,
}

// src/codebook.rs
pub struct Codebook {
    pub centroids: Vec<f32>,          // [2^bit_width] reconstruction values
    pub boundaries: Vec<f32>,         // [2^bit_width - 1] decision thresholds
    pub bit_width: u8,                // 1..=8
}

// src/rotation.rs
pub struct RandomRotation {
    matrix: Vec<f32>,                 // d×d row-major orthogonal matrix
    matrix_t: Vec<f32>,               // d×d transposed (cached)
    pub dim: usize,
}

// src/mse_quant.rs
pub struct MseQuantized {
    pub indices: Vec<u8>,             // [num_vectors × dim] codebook indices
    pub num_vectors: usize,
    pub dim: usize,
    pub bit_width: u8,
}

// src/values.rs
pub struct CompressedValues {
    pub packed: Vec<i32>,             // bit-packed quantized values
    pub scale: Vec<f32>,              // [num_groups]
    pub mn: Vec<f32>,                 // [num_groups]
    pub num_elements: usize,
    pub bits: u8,                     // 2 or 4
    pub group_size: usize,
}
```
