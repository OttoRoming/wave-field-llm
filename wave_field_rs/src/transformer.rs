//! Wave Field Transformer — Rust port
//!
//! Full model: embeddings → wave-field layers → field-interference modules →
//! layer-norm → output projection (weight-tied to embeddings).

use crate::attention::{gelu, layer_norm, linear_on_slice, sigmoid, softplus, WaveFieldAttention};
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

// ─── sinusoidal positional encoding ──────────────────────────────────────────

fn make_sinusoidal_pe(seq_len: usize, dim: usize) -> Vec<f32> {
    let mut pe = vec![0.0_f32; seq_len * dim];
    for pos in 0..seq_len {
        for i in (0..dim).step_by(2) {
            let div_term = (-(i as f32) * (10000.0_f32.ln() / dim as f32)).exp();
            pe[pos * dim + i] = ((pos as f32) * div_term).sin();
            if i + 1 < dim {
                pe[pos * dim + i + 1] = ((pos as f32) * div_term).cos();
            }
        }
    }
    pe
}

// ─── Feed-Forward Network ─────────────────────────────────────────────────────

pub struct Ffn {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl Ffn {
    pub fn new_random(embedding_dim: usize, ffn_dim: usize, rng: &mut impl rand::Rng) -> Self {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f32, 0.02).unwrap();
        let mut rand_vec = |n: usize| -> Vec<f32> {
            (0..n).map(|_| normal.sample(&mut *rng)).collect()
        };
        Self {
            w1: Array2::from_shape_vec((ffn_dim, embedding_dim), rand_vec(ffn_dim * embedding_dim)).unwrap(),
            b1: Array1::zeros(ffn_dim),
            w2: Array2::from_shape_vec((embedding_dim, ffn_dim), rand_vec(embedding_dim * ffn_dim)).unwrap(),
            b2: Array1::zeros(embedding_dim),
        }
    }

    /// `x`: `(tokens, embedding_dim)` flat slice → same shape.
    pub fn forward(&self, x: &[f32], tokens: usize, embed_dim: usize) -> Vec<f32> {
        let ffn_dim = self.w1.shape()[0];
        let hidden = linear_on_slice(x, tokens, embed_dim, &self.w1, &self.b1);
        let hidden: Vec<f32> = hidden.into_iter().map(gelu).collect();
        // apply w2
        let out = linear_on_slice(&hidden, tokens, ffn_dim, &self.w2, &self.b2);
        out
    }
}

// ─── WaveFieldTransformerLayer ────────────────────────────────────────────────

pub struct WaveFieldTransformerLayer {
    pub attention: WaveFieldAttention,
    pub ffn: Ffn,
    /// norm1 weight/bias (pre-attention)
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    /// norm2 weight/bias (pre-FFN)
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub embedding_dim: usize,
}

impl WaveFieldTransformerLayer {
    pub fn new_random(
        embedding_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        field_size: usize,
        max_seq_len: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            attention: WaveFieldAttention::new_random(
                embedding_dim,
                num_heads,
                field_size,
                max_seq_len,
                rng,
            ),
            ffn: Ffn::new_random(embedding_dim, ffn_dim, rng),
            norm1_w: vec![1.0; embedding_dim],
            norm1_b: vec![0.0; embedding_dim],
            norm2_w: vec![1.0; embedding_dim],
            norm2_b: vec![0.0; embedding_dim],
            embedding_dim,
        }
    }

    /// Forward: `x` flat `(B, N, D)` → same shape.
    pub fn forward(
        &self,
        x: &[f32],
        batch: usize,
        seq_len: usize,
        planner: &mut FftPlanner<f32>,
    ) -> Vec<f32> {
        let d = self.embedding_dim;
        let tokens = batch * seq_len;

        // Pre-norm 1
        let mut normed = x.to_vec();
        layer_norm(&mut normed, tokens, d, &self.norm1_w, &self.norm1_b, 1e-5);

        // Attention + residual
        let attn = self.attention.forward(&normed, batch, seq_len, planner);
        let mut x2: Vec<f32> = x.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm 2
        let mut normed2 = x2.clone();
        layer_norm(&mut normed2, tokens, d, &self.norm2_w, &self.norm2_b, 1e-5);

        // FFN + residual
        let ffn_out = self.ffn.forward(&normed2, tokens, d);
        for (v, f) in x2.iter_mut().zip(ffn_out.iter()) {
            *v += f;
        }

        x2
    }

    pub fn param_count(&self) -> usize {
        self.attention.param_count()
            + self.ffn.w1.len() + self.ffn.b1.len()
            + self.ffn.w2.len() + self.ffn.b2.len()
            + 4 * self.embedding_dim // two layer-norms
    }
}

// ─── FieldInterferenceModule ──────────────────────────────────────────────────

pub struct FieldInterferenceModule {
    /// compress weight (embedding_dim/4, embedding_dim)
    pub compress_w: Array2<f32>,
    pub compress_b: Array1<f32>,
    /// expand weight (embedding_dim, embedding_dim/4)
    pub expand_w: Array2<f32>,
    pub expand_b: Array1<f32>,
    /// local phase proj (embedding_dim, embedding_dim)
    pub local_w: Array2<f32>,
    pub local_b: Array1<f32>,
    /// global phase proj (embedding_dim, embedding_dim)
    pub global_w: Array2<f32>,
    pub global_b: Array1<f32>,
    /// interference gate (embedding_dim*2, embedding_dim)
    pub gate_w: Array2<f32>,
    pub gate_b: Array1<f32>,
    /// interference temperature (scalar, stored as raw pre-softplus value)
    pub temperature_raw: f32,
    pub embedding_dim: usize,
    pub compressed_dim: usize,
}

impl FieldInterferenceModule {
    pub fn new_random(embedding_dim: usize, rng: &mut impl rand::Rng) -> Self {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f32, 0.02).unwrap();
        let mut rand_vec = |n: usize| -> Vec<f32> {
            (0..n).map(|_| normal.sample(&mut *rng)).collect()
        };
        let compressed_dim = embedding_dim / 4;
        Self {
            compress_w: Array2::from_shape_vec((compressed_dim, embedding_dim), rand_vec(compressed_dim * embedding_dim)).unwrap(),
            compress_b: Array1::zeros(compressed_dim),
            expand_w: Array2::from_shape_vec((embedding_dim, compressed_dim), rand_vec(embedding_dim * compressed_dim)).unwrap(),
            expand_b: Array1::zeros(embedding_dim),
            local_w: Array2::from_shape_vec((embedding_dim, embedding_dim), rand_vec(embedding_dim * embedding_dim)).unwrap(),
            local_b: Array1::zeros(embedding_dim),
            global_w: Array2::from_shape_vec((embedding_dim, embedding_dim), rand_vec(embedding_dim * embedding_dim)).unwrap(),
            global_b: Array1::zeros(embedding_dim),
            gate_w: Array2::from_shape_vec((embedding_dim, embedding_dim * 2), rand_vec(embedding_dim * embedding_dim * 2)).unwrap(),
            gate_b: Array1::zeros(embedding_dim),
            temperature_raw: -2.0,
            embedding_dim,
            compressed_dim,
        }
    }

    /// L2-normalise rows of a flat `(tokens, dim)` array in-place.
    fn normalize_rows(data: &mut [f32], tokens: usize, dim: usize) {
        for i in 0..tokens {
            let row = &mut data[i * dim..(i + 1) * dim];
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            row.iter_mut().for_each(|v| *v /= norm);
        }
    }

    /// Forward: `x` flat `(B, N, D)` → same shape.
    pub fn forward(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let b = batch;
        let n = seq_len;
        let d = self.embedding_dim;
        let cd = self.compressed_dim;
        let tokens = b * n;

        // ── compress x ────────────────────────────────────────────────────────
        let compressed = linear_on_slice(x, tokens, d, &self.compress_w, &self.compress_b);

        // ── causal cumulative mean (position i sees avg of 0..=i, per batch) ──
        let mut causal = vec![0.0_f32; tokens * cd];
        for bi in 0..b {
            let mut running = vec![0.0_f32; cd];
            for ni in 0..n {
                let src = (bi * n + ni) * cd;
                for ci in 0..cd {
                    running[ci] += compressed[src + ci];
                }
                let dst = (bi * n + ni) * cd;
                for ci in 0..cd {
                    causal[dst + ci] = running[ci] / (ni + 1) as f32;
                }
            }
        }

        // ── expand causal summary ─────────────────────────────────────────────
        let global_ctx = linear_on_slice(&causal, tokens, cd, &self.expand_w, &self.expand_b);

        // ── local and global phase projections (normalised) ────────────────────
        let mut local_phase =
            linear_on_slice(x, tokens, d, &self.local_w, &self.local_b);
        Self::normalize_rows(&mut local_phase, tokens, d);

        let mut global_phase =
            linear_on_slice(&global_ctx, tokens, d, &self.global_w, &self.global_b);
        Self::normalize_rows(&mut global_phase, tokens, d);

        // ── cosine similarity per token → (tokens, 1) ─────────────────────────
        let phase_align: Vec<f32> = (0..tokens)
            .map(|i| {
                let l = &local_phase[i * d..(i + 1) * d];
                let g = &global_phase[i * d..(i + 1) * d];
                l.iter().zip(g.iter()).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect();

        // ── interference strength ─────────────────────────────────────────────
        let temp = softplus(self.temperature_raw) + 0.05;
        let interference: Vec<f32> = phase_align.iter().map(|&v| sigmoid(v / temp)).collect();

        // ── gate ──────────────────────────────────────────────────────────────
        // concatenate x and global_ctx → (tokens, 2*D)
        let concat: Vec<f32> = (0..tokens)
            .flat_map(|i| {
                x[i * d..(i + 1) * d]
                    .iter()
                    .chain(global_ctx[i * d..(i + 1) * d].iter())
                    .cloned()
            })
            .collect();

        let gate_out = linear_on_slice(&concat, tokens, 2 * d, &self.gate_w, &self.gate_b);
        let gate: Vec<f32> = gate_out.into_iter().map(sigmoid).collect();

        // ── output = x + gate * global_ctx * interference ─────────────────────
        let mut out = x.to_vec();
        for i in 0..tokens {
            let s = interference[i];
            for di in 0..d {
                let idx = i * d + di;
                out[idx] += gate[idx] * global_ctx[idx] * s;
            }
        }

        out
    }

    pub fn param_count(&self) -> usize {
        let d = self.embedding_dim;
        let cd = self.compressed_dim;
        cd * d + cd         // compress
        + d * cd + d        // expand
        + d * d + d         // local_phase
        + d * d + d         // global_phase
        + d * 2 * d + d     // gate
        + 1                 // temperature
    }
}

// ─── WaveFieldTransformer ─────────────────────────────────────────────────────

pub struct WaveFieldTransformer {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub interference_interval: usize,

    /// Token embedding weight: (vocab_size, embedding_dim)
    pub token_embedding: Array2<f32>,
    /// Output projection weight (tied to token_embedding): same pointer conceptually;
    /// we use token_embedding.t() directly.

    pub layers: Vec<WaveFieldTransformerLayer>,
    pub interference_modules: Vec<FieldInterferenceModule>,

    /// Final layer-norm
    pub final_norm_w: Vec<f32>,
    pub final_norm_b: Vec<f32>,
}

impl WaveFieldTransformer {
    /// Construct with random weights (for benchmarking).
    pub fn new_random(
        vocab_size: usize,
        embedding_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ffn_dim: usize,
        field_size: usize,
        max_seq_len: usize,
        interference_interval: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f32, 0.02).unwrap();

        let emb_v: Vec<f32> = (0..vocab_size * embedding_dim)
            .map(|_| normal.sample(rng))
            .collect();
        let token_embedding =
            Array2::from_shape_vec((vocab_size, embedding_dim), emb_v).unwrap();

        let layers = (0..num_layers)
            .map(|_| {
                WaveFieldTransformerLayer::new_random(
                    embedding_dim,
                    num_heads,
                    ffn_dim,
                    field_size,
                    max_seq_len,
                    rng,
                )
            })
            .collect();

        let num_interference = num_layers / interference_interval;
        let interference_modules = (0..num_interference)
            .map(|_| FieldInterferenceModule::new_random(embedding_dim, rng))
            .collect();

        Self {
            vocab_size,
            embedding_dim,
            max_seq_len,
            interference_interval,
            token_embedding,
            layers,
            interference_modules,
            final_norm_w: vec![1.0; embedding_dim],
            final_norm_b: vec![0.0; embedding_dim],
        }
    }

    /// Count all parameters.
    pub fn param_count(&self) -> usize {
        let mut total = self.vocab_size * self.embedding_dim; // embedding (tied)
        for l in &self.layers {
            total += l.param_count();
        }
        for m in &self.interference_modules {
            total += m.param_count();
        }
        total + 2 * self.embedding_dim // final norm
    }

    /// Forward pass.
    ///
    /// `input_ids`: `(batch, seq_len)` token indices.
    /// Returns logits `(batch, seq_len, vocab_size)` as a flat Vec.
    pub fn forward(
        &self,
        input_ids: &[u32],
        batch: usize,
        seq_len: usize,
        planner: &mut FftPlanner<f32>,
    ) -> Vec<f32> {
        let b = batch;
        let n = seq_len;
        let d = self.embedding_dim;

        // ── Token embeddings ──────────────────────────────────────────────────
        let mut x: Vec<f32> = Vec::with_capacity(b * n * d);
        for &tok in input_ids.iter() {
            let tok = (tok as usize).min(self.vocab_size - 1);
            let row = self.token_embedding.row(tok);
            x.extend(row.iter().cloned());
        }

        // ── Sinusoidal positional encoding ────────────────────────────────────
        let pe = make_sinusoidal_pe(n, d);
        for bi in 0..b {
            for ni in 0..n {
                for di in 0..d {
                    x[(bi * n + ni) * d + di] += pe[ni * d + di];
                }
            }
        }

        // ── Transformer layers with interference ──────────────────────────────
        let mut interference_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, b, n, planner);

            if (layer_idx + 1) % self.interference_interval == 0
                && interference_idx < self.interference_modules.len()
            {
                x = self.interference_modules[interference_idx].forward(&x, b, n);
                interference_idx += 1;
            }
        }

        // ── Final layer-norm ──────────────────────────────────────────────────
        layer_norm(
            &mut x,
            b * n,
            d,
            &self.final_norm_w,
            &self.final_norm_b,
            1e-5,
        );

        // ── Output projection (weight-tied to token_embedding) ────────────────
        // logits = x @ token_embedding.T  →  (B*N, vocab_size)
        let x_view =
            ndarray::ArrayView2::from_shape((b * n, d), &x).expect("shape");
        let logits_2d = x_view.dot(&self.token_embedding.t()); // (B*N, V)
        logits_2d.into_raw_vec_and_offset().0
    }
}

// ─── unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_model(seq_len: usize) -> (WaveFieldTransformer, usize) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let model = WaveFieldTransformer::new_random(
            256,  // vocab_size
            64,   // embedding_dim
            2,    // num_layers
            4,    // num_heads
            128,  // ffn_dim
            256,  // field_size
            seq_len,
            3,    // interference_interval (no interference in 2-layer model)
            &mut rng,
        );
        let params = model.param_count();
        (model, params)
    }

    #[test]
    fn test_forward_shape() {
        let batch = 2;
        let seq_len = 16;
        let (model, params) = make_model(seq_len);
        println!("param_count: {}", params);

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let input: Vec<u32> = (0..batch * seq_len)
            .map(|_| rand::Rng::gen_range(&mut rng, 0u32..256))
            .collect();

        let mut planner = FftPlanner::new();
        let logits = model.forward(&input, batch, seq_len, &mut planner);

        assert_eq!(logits.len(), batch * seq_len * 256);
    }

    #[test]
    fn test_causality() {
        // Changing token at position N-1 must not affect logits at positions 0..N-2.
        let batch = 1;
        let seq_len = 10;
        let (model, _) = make_model(seq_len);

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let input_a: Vec<u32> = (0..seq_len)
            .map(|_| rand::Rng::gen_range(&mut rng, 0u32..256))
            .collect();
        let mut input_b = input_a.clone();
        input_b[seq_len - 1] = (input_b[seq_len - 1] + 50) % 256;

        let mut planner = FftPlanner::new();
        let la = model.forward(&input_a, batch, seq_len, &mut planner);
        let lb = model.forward(&input_b, batch, seq_len, &mut planner);

        // Positions 0..N-2 should be identical
        for pos in 0..seq_len - 1 {
            let slice_a = &la[pos * 256..(pos + 1) * 256];
            let slice_b = &lb[pos * 256..(pos + 1) * 256];
            let max_diff = slice_a
                .iter()
                .zip(slice_b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "causality violation at pos {pos}: max_diff={max_diff}"
            );
        }
    }

    #[test]
    fn test_deterministic() {
        let batch = 1;
        let seq_len = 8;
        let (model, _) = make_model(seq_len);
        let input: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

        let mut p1 = FftPlanner::new();
        let logits1 = model.forward(&input, batch, seq_len, &mut p1);
        let mut p2 = FftPlanner::new();
        let logits2 = model.forward(&input, batch, seq_len, &mut p2);
        assert_eq!(logits1, logits2);
    }
}
