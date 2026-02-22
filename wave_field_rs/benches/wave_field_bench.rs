//! Criterion benchmarks for Wave Field LLM (Rust port)
//!
//! Measures forward-pass throughput at several sequence lengths and
//! compares them against the theoretical O(n log n) scaling curve.
//!
//! Run with:
//!   cargo bench --bench wave_field_bench
//!
//! HTML reports are written to target/criterion/.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::SeedableRng;
use rustfft::FftPlanner;
use wave_field_rs::WaveFieldTransformer;

// ─── model configuration (≈ 6 M params, matches the Python baseline) ─────────

const VOCAB_SIZE: usize = 256;
const EMBEDDING_DIM: usize = 256;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;
const FFN_DIM: usize = 1024;
const FIELD_SIZE: usize = 1024;
const INTERFERENCE_INTERVAL: usize = 3;
const BATCH_SIZE: usize = 1;

fn build_model(max_seq_len: usize) -> WaveFieldTransformer {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    WaveFieldTransformer::new_random(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        FFN_DIM,
        FIELD_SIZE,
        max_seq_len,
        INTERFERENCE_INTERVAL,
        &mut rng,
    )
}

// ─── forward-pass benchmark at various sequence lengths ──────────────────────

fn bench_forward_pass(c: &mut Criterion) {
    let seq_lengths: &[usize] = &[64, 128, 256, 512, 1024, 2048];

    let mut group = c.benchmark_group("wave_field_forward");

    for &seq_len in seq_lengths {
        // Build model whose field_stride is calibrated to this sequence length
        let model = build_model(seq_len);

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let input: Vec<u32> = (0..BATCH_SIZE * seq_len)
            .map(|_| rand::Rng::gen_range(&mut rng, 0u32..VOCAB_SIZE as u32))
            .collect();

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &_s| {
                let mut planner = FftPlanner::new();
                b.iter(|| {
                    let _ = model.forward(&input, BATCH_SIZE, seq_len, &mut planner);
                });
            },
        );
    }

    group.finish();
}

// ─── per-component benchmarks ─────────────────────────────────────────────────

fn bench_attention_only(c: &mut Criterion) {
    use wave_field_rs::attention::WaveFieldAttention;

    let seq_lengths: &[usize] = &[64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("wave_field_attention_only");

    for &seq_len in seq_lengths {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let attn = WaveFieldAttention::new_random(
            EMBEDDING_DIM,
            NUM_HEADS,
            FIELD_SIZE,
            seq_len,
            &mut rng,
        );

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(7);
        let x: Vec<f32> = (0..BATCH_SIZE * seq_len * EMBEDDING_DIM)
            .map(|_| rand::Rng::gen::<f32>(&mut rng2))
            .collect();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &_s| {
                let mut planner = FftPlanner::new();
                b.iter(|| {
                    let _ = attn.forward(&x, BATCH_SIZE, seq_len, &mut planner);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_forward_pass, bench_attention_only);
criterion_main!(benches);
