//! Wave Field LLM — Rust Implementation
//!
//! O(n log n) physics-based language model using damped-wave attention kernels.
//!
//! Architecture mirrors the Python/PyTorch V3.5 implementation exactly:
//! - Wave-parameterised kernels: k(t) = exp(-α·t)·cos(ω·t + φ)  for t ≥ 0
//! - Bilinear scatter/gather onto a continuous 1-D field
//! - FFT convolution in O(n log n)
//! - Static multi-head field coupling
//! - Content-dependent gating
//! - Sinusoidal positional encoding
//! - Field Interference modules (causal pooling + phase-alignment gate)

pub mod attention;
pub mod transformer;

pub use transformer::WaveFieldTransformer;
