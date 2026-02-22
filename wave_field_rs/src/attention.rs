//! Wave Field Attention V3.5 — Rust port
//!
//! Implements the same forward pass as the Python `WaveFieldAttention` class:
//!
//! 1. QKV projection
//! 2. Absolute position mapping  (token i → field position i × stride)
//! 3. Bilinear scatter  (deposit values onto the continuous 1-D field)
//! 4. Wave convolution via FFT  — O(n log n)
//! 5. Static multi-field coupling  (cross-head interactions)
//! 6. Content-dependent gating
//! 7. Bilinear gather  (read from field)
//! 8. Output projection

use ndarray::{Array1, Array2};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

// ─── activation helpers ──────────────────────────────────────────────────────

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    // Tanh approximation of GELU (matches PyTorch's default)
    0.5 * x
        * (1.0
            + ((2.0_f32 / PI).sqrt() * (x + 0.044715 * x * x * x))
                .tanh())
}

// ─── low-level linear algebra ─────────────────────────────────────────────────

pub fn linear_on_slice(
    data: &[f32],      // length B*N*in_dim, row-major
    rows: usize,       // B * N
    in_dim: usize,
    weight: &Array2<f32>, // (out_dim, in_dim)
    bias: &Array1<f32>,   // (out_dim,)
) -> Vec<f32> {
    let _out_dim = weight.shape()[0];
    let wt: Array2<f32> = weight.t().to_owned();

    // Build (rows, in_dim) view
    let x = ndarray::ArrayView2::from_shape((rows, in_dim), data)
        .expect("linear_on_slice: bad shape");
    let mut out = x.dot(&wt);
    for mut row in out.rows_mut() {
        row += bias;
    }
    out.into_raw_vec_and_offset().0
}

/// Layer normalisation over the last dimension.
/// `x_flat`: `(tokens, dim)`, in-place normalisation.
pub fn layer_norm(x: &mut [f32], tokens: usize, dim: usize,
                  weight: &[f32], bias: &[f32], eps: f32) {
    for i in 0..tokens {
        let slice = &mut x[i * dim..(i + 1) * dim];
        let mean = slice.iter().sum::<f32>() / dim as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let std = (var + eps).sqrt();
        for (j, v) in slice.iter_mut().enumerate() {
            *v = (*v - mean) / std * weight[j] + bias[j];
        }
    }
}

/// Softmax over the last axis of a 2-D matrix (rows × cols), in-place.
pub fn softmax_rows(x: &mut Array2<f32>) {
    for mut row in x.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
}

// ─── WaveFieldAttention ───────────────────────────────────────────────────────

/// Parameters for a single `WaveFieldAttention` layer.
pub struct WaveFieldAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub field_size: usize,
    pub max_seq_len: usize,

    /// (3*embedding_dim, embedding_dim)
    pub qkv_weight: Array2<f32>,
    pub qkv_bias: Array1<f32>,
    /// (embedding_dim, embedding_dim)
    pub out_weight: Array2<f32>,
    pub out_bias: Array1<f32>,
    /// (embedding_dim, embedding_dim)
    pub gate_weight: Array2<f32>,
    pub gate_bias: Array1<f32>,

    /// Wave kernel parameters — shape `(num_heads,)`
    pub wave_frequency: Array1<f32>,
    pub wave_damping: Array1<f32>,
    pub wave_phase: Array1<f32>,

    /// Multi-head coupling — shape `(num_heads, num_heads)`
    pub field_coupling: Array2<f32>,

    /// field_stride = (field_size - 1) / max(max_seq_len - 1, 1)
    pub field_stride: f32,
}

impl WaveFieldAttention {
    /// Initialise with xavier-like random weights (for benchmarking).
    pub fn new_random(
        embedding_dim: usize,
        num_heads: usize,
        field_size: usize,
        max_seq_len: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        use rand_distr::{Distribution, Normal};

        let head_dim = embedding_dim / num_heads;
        let std = 0.02_f32;
        let normal = Normal::new(0.0_f32, std).unwrap();

        let mut rand_vec = |n: usize| -> Vec<f32> {
            (0..n).map(|_| normal.sample(&mut *rng)).collect()
        };

        // Wave frequency: linspace(0.3, 4.0, num_heads)
        let wave_frequency: Array1<f32> = Array1::linspace(0.3, 4.0, num_heads);
        // Wave damping: linspace(-3.0, 0.5, num_heads)
        let wave_damping: Array1<f32> = Array1::linspace(-3.0, 0.5, num_heads);
        // Wave phase: linspace(0, π, num_heads)
        let wave_phase: Array1<f32> = Array1::linspace(0.0, PI, num_heads);

        // Field coupling ≈ identity + small noise
        let coupling_v = rand_vec(num_heads * num_heads);
        let coupling_noise = Array2::from_shape_vec((num_heads, num_heads), coupling_v).unwrap();
        let mut field_coupling = Array2::eye(num_heads);
        field_coupling.scaled_add(0.01, &coupling_noise);

        // Gate bias starts at +2.0 (sigmoid(2) ≈ 0.88 — mostly open)
        let mut gate_bias = Array1::from_vec(rand_vec(embedding_dim));
        gate_bias.fill(2.0);

        let qkv_weight = Array2::from_shape_vec(
            (3 * embedding_dim, embedding_dim), rand_vec(3 * embedding_dim * embedding_dim)).unwrap();
        let out_weight = Array2::from_shape_vec(
            (embedding_dim, embedding_dim), rand_vec(embedding_dim * embedding_dim)).unwrap();
        let gate_weight = Array2::from_shape_vec(
            (embedding_dim, embedding_dim), rand_vec(embedding_dim * embedding_dim)).unwrap();

        let field_stride = (field_size as f32 - 1.0)
            / (max_seq_len.max(2) - 1) as f32;

        Self {
            embedding_dim,
            num_heads,
            head_dim,
            field_size,
            max_seq_len,
            qkv_weight,
            qkv_bias: Array1::zeros(3 * embedding_dim),
            out_weight,
            out_bias: Array1::zeros(embedding_dim),
            gate_weight,
            gate_bias,
            wave_frequency,
            wave_damping,
            wave_phase,
            field_coupling,
            field_stride,
        }
    }

    // ── wave kernels ──────────────────────────────────────────────────────────

    /// Build left-aligned causal wave kernels and return their rfft.
    /// Returns a `Vec<Vec<Complex<f32>>>` of shape `[H][pad_size/2+1]`.
    fn build_wave_kernels(&self, planner: &mut FftPlanner<f32>) -> Vec<Vec<Complex<f32>>> {
        let g = self.field_size;
        let pad = 2 * g;
        let h = self.num_heads;

        let mut kernels_fft: Vec<Vec<Complex<f32>>> = Vec::with_capacity(h);

        for hi in 0..h {
            let alpha = softplus(self.wave_damping[hi]);
            let omega = self.wave_frequency[hi];
            let phi = self.wave_phase[hi];

            // kernel[t] = exp(-alpha*t) * cos(omega*t + phi),  t = 0..G
            let mut k: Vec<f32> = (0..g)
                .map(|t| {
                    let tf = t as f32;
                    (-alpha * tf).exp() * (omega * tf + phi).cos()
                })
                .collect();

            // normalise by L1 norm
            let norm: f32 = k.iter().map(|v| v.abs()).sum::<f32>().max(1e-8);
            k.iter_mut().for_each(|v| *v /= norm);

            // zero-pad to `pad` for linear (non-circular) convolution
            k.resize(pad, 0.0);

            // real-to-complex FFT
            let fft = planner.plan_fft_forward(pad);
            let mut buf: Vec<Complex<f32>> =
                k.iter().map(|&v| Complex::new(v, 0.0)).collect();
            fft.process(&mut buf);

            // We only need the positive-frequency half (rfft analogue)
            buf.truncate(pad / 2 + 1);
            kernels_fft.push(buf);
        }

        kernels_fft
    }

    // ── wave convolution ─────────────────────────────────────────────────────

    /// FFT-based wave convolution of the field.
    ///
    /// `field` shape: `(B, H, G, D)` stored as a flat `Vec<f32>`.
    /// Returns a new Vec of the same shape.
    fn wave_convolve(
        &self,
        field: &[f32],       // B × H × G × D, row-major
        batch: usize,
        kernel_ffts: &[Vec<Complex<f32>>],
        planner: &mut FftPlanner<f32>,
    ) -> Vec<f32> {
        let b = batch;
        let h = self.num_heads;
        let g = self.field_size;
        let d = self.head_dim;
        let pad = 2 * g;
        let half = pad / 2 + 1;

        let mut out = vec![0.0_f32; b * h * g * d];

        let fft_fwd = planner.plan_fft_forward(pad);
        let fft_inv = planner.plan_fft_inverse(pad);

        // Iterate over (batch, head, feature-dim) and do 1-D convolution
        for bi in 0..b {
            for hi in 0..h {
                let kern = &kernel_ffts[hi];
                for di in 0..d {
                    // Extract the G-length field slice for (bi, hi, :, di)
                    let mut signal: Vec<Complex<f32>> = (0..g)
                        .map(|gi| {
                            let idx = bi * h * g * d + hi * g * d + gi * d + di;
                            Complex::new(field[idx], 0.0)
                        })
                        .collect();
                    signal.resize(pad, Complex::new(0.0, 0.0));

                    // Forward FFT
                    fft_fwd.process(&mut signal);

                    // Multiply by kernel (positive-freq half)
                    for fi in 0..half {
                        signal[fi] *= kern[fi];
                    }
                    // Mirror conjugate for negative frequencies
                    for fi in 1..pad / 2 {
                        signal[pad - fi] = signal[fi].conj();
                    }

                    // Inverse FFT
                    fft_inv.process(&mut signal);

                    // Scale and write first G elements back
                    let scale = 1.0 / pad as f32;
                    for gi in 0..g {
                        let idx = bi * h * g * d + hi * g * d + gi * d + di;
                        out[idx] = signal[gi].re * scale;
                    }
                }
            }
        }

        out
    }

    // ── bilinear scatter ─────────────────────────────────────────────────────

    /// Deposit `deposit` onto the field using bilinear interpolation.
    ///
    /// `deposit` shape: `(B, H, N, head_dim)` flat Vec.
    /// `field_pos`: `(N,)` continuous field positions.
    /// Returns field `(B, H, G, head_dim)`.
    fn bilinear_scatter(
        &self,
        deposit: &[f32],
        field_pos: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let b = batch;
        let h = self.num_heads;
        let g = self.field_size;
        let d = self.head_dim;
        let n = seq_len;

        let mut field = vec![0.0_f32; b * h * g * d];

        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    let fp = field_pos[ni].min((g - 2) as f32).max(0.0);
                    let lo = fp as usize;
                    let hi_idx = lo + 1;
                    let frac = fp - lo as f32;
                    let w_lo = 1.0 - frac;
                    let w_hi = frac;

                    for di in 0..d {
                        let dep_idx = bi * h * n * d + hi * n * d + ni * d + di;
                        let v = deposit[dep_idx];

                        let f_lo = bi * h * g * d + hi * g * d + lo * d + di;
                        let f_hi = bi * h * g * d + hi * g * d + hi_idx * d + di;
                        field[f_lo] += v * w_lo;
                        field[f_hi] += v * w_hi;
                    }
                }
            }
        }

        field
    }

    // ── bilinear gather ──────────────────────────────────────────────────────

    /// Read from field using bilinear interpolation.
    ///
    /// Returns gathered values `(B, H, N, head_dim)`.
    fn bilinear_gather(
        &self,
        field: &[f32],
        field_pos: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let b = batch;
        let h = self.num_heads;
        let g = self.field_size;
        let d = self.head_dim;
        let n = seq_len;

        let mut gathered = vec![0.0_f32; b * h * n * d];

        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    let fp = field_pos[ni].min((g - 2) as f32).max(0.0);
                    let lo = fp as usize;
                    let hi_idx = lo + 1;
                    let frac = fp - lo as f32;
                    let w_lo = 1.0 - frac;
                    let w_hi = frac;

                    for di in 0..d {
                        let f_lo = bi * h * g * d + hi * g * d + lo * d + di;
                        let f_hi = bi * h * g * d + hi * g * d + hi_idx * d + di;
                        let val = field[f_lo] * w_lo + field[f_hi] * w_hi;
                        let g_idx = bi * h * n * d + hi * n * d + ni * d + di;
                        gathered[g_idx] = val;
                    }
                }
            }
        }

        gathered
    }

    // ── field coupling ───────────────────────────────────────────────────────

    /// Static multi-field coupling: soft-max over coupling matrix, then
    /// batch-matmul over heads.
    ///
    /// `field` shape: `(B, H, G, D)` flat Vec.
    fn apply_field_coupling(&self, field: &[f32], batch: usize) -> Vec<f32> {
        let b = batch;
        let h = self.num_heads;
        let g = self.field_size;
        let d = self.head_dim;

        // Softmax of coupling matrix row-wise: (H, H)
        let mut coupling = self.field_coupling.clone();
        softmax_rows(&mut coupling);

        let mut out = vec![0.0_f32; b * h * g * d];

        // For each batch element: out[b, h2, g, d] = Σ_h1 coupling[h2, h1] * field[b, h1, g, d]
        for bi in 0..b {
            for h2 in 0..h {
                for h1 in 0..h {
                    let c = coupling[[h2, h1]];
                    if c.abs() < 1e-8 {
                        continue;
                    }
                    for gi in 0..g {
                        for di in 0..d {
                            let src = bi * h * g * d + h1 * g * d + gi * d + di;
                            let dst = bi * h * g * d + h2 * g * d + gi * d + di;
                            out[dst] += c * field[src];
                        }
                    }
                }
            }
        }

        out
    }

    // ── forward pass ─────────────────────────────────────────────────────────

    /// Forward pass: `x` shape `(batch, seq_len, embedding_dim)` flat Vec.
    /// Returns output of the same shape.
    pub fn forward(
        &self,
        x: &[f32],
        batch: usize,
        seq_len: usize,
        planner: &mut FftPlanner<f32>,
    ) -> Vec<f32> {
        let b = batch;
        let n = seq_len;
        let d = self.embedding_dim;
        let h = self.num_heads;
        let hd = self.head_dim;
        let g = self.field_size;

        // ── 1. QKV projection: (B*N, D) → (B*N, 3*D) ────────────────────────
        let qkv = linear_on_slice(x, b * n, d, &self.qkv_weight, &self.qkv_bias);

        // Split into Q, K, V each (B*N, D)
        let _q_raw: Vec<f32> = (0..b * n)
            .flat_map(|i| qkv[i * 3 * d..i * 3 * d + d].iter().cloned())
            .collect();
        let k_raw: Vec<f32> = (0..b * n)
            .flat_map(|i| qkv[i * 3 * d + d..i * 3 * d + 2 * d].iter().cloned())
            .collect();
        let v_raw: Vec<f32> = (0..b * n)
            .flat_map(|i| qkv[i * 3 * d + 2 * d..i * 3 * d + 3 * d].iter().cloned())
            .collect();

        // Reshape to (B, H, N, hd): q[b, h, n, hd]
        // Original layout: (B, N, H, hd) after view; we transpose 1↔2
        // q_raw is (B*N, D) = (B*N, H*hd) in row-major;
        // index q[bi, ni, hi, di] = q_raw[(bi*N + ni)*D + hi*hd + di]
        // We want q_bhnd[bi, hi, ni, di] — just re-index on access.
        let idx_k = |bi: usize, hi: usize, ni: usize, di: usize| -> f32 {
            k_raw[(bi * n + ni) * d + hi * hd + di]
        };
        let idx_v = |bi: usize, hi: usize, ni: usize, di: usize| -> f32 {
            v_raw[(bi * n + ni) * d + hi * hd + di]
        };

        // ── 2. Absolute position mapping ──────────────────────────────────────
        let field_pos: Vec<f32> = (0..n)
            .map(|i| (i as f32 * self.field_stride).min((g - 2) as f32))
            .collect();

        // ── 3. K magnitude and deposit = V * K_mag ───────────────────────────
        // deposit shape: (B, H, N, hd) — flat Vec
        let mut deposit = vec![0.0_f32; b * h * n * hd];
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    // k magnitude (L2 norm over hd)
                    let k_mag: f32 = (0..hd)
                        .map(|di| idx_k(bi, hi, ni, di).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    for di in 0..hd {
                        let dep_idx = bi * h * n * hd + hi * n * hd + ni * hd + di;
                        deposit[dep_idx] = idx_v(bi, hi, ni, di) * k_mag;
                    }
                }
            }
        }

        // ── 4. Bilinear scatter ───────────────────────────────────────────────
        let field = self.bilinear_scatter(&deposit, &field_pos, b, n);

        // ── 5. Wave convolution ───────────────────────────────────────────────
        let kernel_ffts = self.build_wave_kernels(planner);
        let field = self.wave_convolve(&field, b, &kernel_ffts, planner);

        // ── 6. Field coupling ─────────────────────────────────────────────────
        let field = self.apply_field_coupling(&field, b);

        // ── 7. Gating: sigmoid(gate_proj(x))  shape (B, N, D) ────────────────
        let mut gate = linear_on_slice(x, b * n, d, &self.gate_weight, &self.gate_bias);
        gate.iter_mut().for_each(|v| *v = sigmoid(*v));

        // ── 8. Bilinear gather ────────────────────────────────────────────────
        let gathered = self.bilinear_gather(&field, &field_pos, b, n);

        // ── 9. output = gathered * gate, then out_proj ────────────────────────
        // gathered: (B, H, N, hd),  gate: (B, N, D) = (B, N, H*hd)
        // transpose gathered to (B, N, D)
        let mut out_pre = vec![0.0_f32; b * n * d];
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    for di in 0..hd {
                        let g_idx = bi * h * n * hd + hi * n * hd + ni * hd + di;
                        let gate_idx = (bi * n + ni) * d + hi * hd + di;
                        let out_idx = (bi * n + ni) * d + hi * hd + di;
                        out_pre[out_idx] += gathered[g_idx] * gate[gate_idx];
                    }
                }
            }
        }

        linear_on_slice(&out_pre, b * n, d, &self.out_weight, &self.out_bias)
    }


}

// ── Query parameter count ─────────────────────────────────────────────────────

impl WaveFieldAttention {
    pub fn param_count(&self) -> usize {
        let d = self.embedding_dim;
        let h = self.num_heads;
        3 * d * d + 3 * d   // qkv weight + bias
        + d * d + d          // out weight + bias
        + d * d + d          // gate weight + bias
        + 3 * h              // wave params
        + h * h              // coupling
    }
}
