//! The SIMD backend must agree with the scalar reference: bit for bit when
//! decoding, and within rounding for sums (SIMD adds in a different order).
//! On a CPU without AVX2 `detect()` is scalar and these tests compare scalar
//! with itself, which still passes.
use biopsy::{
    fixtures::Rng,
    kernels::{self, Backend},
    quant::{BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES},
    stats::Summary,
};

fn simd() -> Backend {
    Backend::detect()
}

fn same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "index {i}: {x} vs {y}"
        );
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1e-300)
}

fn random_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.next_u64() as u8).collect()
}

type Decoder = fn(Backend, &[u8], &mut [f32]);

fn compare_decoder(decode: Decoder, bytes: &[u8], values: usize) {
    let (mut a, mut b) = (vec![0.0; values], vec![0.0; values]);
    decode(Backend::Scalar, bytes, &mut a);
    decode(simd(), bytes, &mut b);
    same_bits(&a, &b);
}

#[test]
fn every_half_precision_pattern_decodes_identically() {
    let all: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    compare_decoder(kernels::f16_to_f32, &all, 65536);
    compare_decoder(kernels::bf16_to_f32, &all, 65536);
    // Lengths that leave a scalar tail after the 8-wide loop.
    for n in 0..20 {
        compare_decoder(kernels::f16_to_f32, &all[..2 * n], n);
        compare_decoder(kernels::bf16_to_f32, &all[2 * n..4 * n], n);
    }
}

#[test]
fn random_quantized_blocks_decode_identically() {
    // Random bytes reach every code and every scale, including NaN/Inf scales.
    for blocks in [1, 2, 7, 300] {
        let q8 = random_bytes(blocks as u64, blocks * Q8_0_BLOCK_BYTES);
        compare_decoder(kernels::dequantize_q8_0, &q8, blocks * BLOCK_ELEMENTS);
        let q4 = random_bytes(99 + blocks as u64, blocks * Q4_0_BLOCK_BYTES);
        compare_decoder(kernels::dequantize_q4_0, &q4, blocks * BLOCK_ELEMENTS);
    }
}

/// Normal values with the awkward cases mixed in.
fn awkward(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| match rng.next_u64() % 16 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => 0.0,
            4 => -0.0,
            5 => f32::from_bits(1), // smallest subnormal
            _ => rng.normal(3.0) + 10.0,
        })
        .collect()
}

#[test]
fn summaries_agree_including_non_finite_values_and_tails() {
    for n in (0..40).chain([4095, 4096, 4097]) {
        let values = awkward(n as u64, n);
        let (a, b) = (
            Summary::of(&values, Backend::Scalar),
            Summary::of(&values, simd()),
        );
        assert_eq!(
            (a.count, a.finite, a.nan, a.inf, a.zeros),
            (b.count, b.finite, b.nan, b.inf, b.zeros),
            "n = {n}"
        );
        assert!(a.min == b.min && a.max == b.max, "n = {n}");
        assert!(
            close(a.mean, b.mean) && close(a.m2, b.m2),
            "n = {n}: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn diff_sums_agree_including_non_finite_values_and_tails() {
    for n in (0..40).chain([4097]) {
        let x = awkward(n as u64, n);
        let mut y = awkward(1000 + n as u64, n);
        for i in (0..n).step_by(3) {
            y[i] = x[i]; // some exact matches, including NaN-NaN and inf-inf
        }
        let a = kernels::diff_sums(Backend::Scalar, &x, &y);
        let b = kernels::diff_sums(simd(), &x, &y);
        assert_eq!(
            (
                a.finite_pairs,
                a.equal,
                a.nonfinite_mismatch,
                a.max_abs_diff
            ),
            (
                b.finite_pairs,
                b.equal,
                b.nonfinite_mismatch,
                b.max_abs_diff
            ),
            "n = {n}"
        );
        for (p, q) in [
            (a.sum_sq_diff, b.sum_sq_diff),
            (a.sum_abs_diff, b.sum_abs_diff),
            (a.sum_aa, b.sum_aa),
            (a.sum_bb, b.sum_bb),
            (a.sum_ab, b.sum_ab),
        ] {
            assert!(close(p, q), "n = {n}: {p} vs {q}");
        }
    }
}

#[test]
fn backend_names() {
    assert_eq!(Backend::Scalar.name(), "scalar");
    assert!(["scalar", "avx2"].contains(&simd().name()));
}
