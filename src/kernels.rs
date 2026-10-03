//! The hot loops: decoding stored bytes to F32 and reducing F32 chunks to sums.
//!
//! Every kernel has a portable scalar version, the reference for correctness.
//! Where measurement showed a gain (`examples/bench.rs`), x86-64 CPUs with AVX2,
//! F16C and FMA also get a SIMD version that processes eight floats per
//! instruction. Callers pass a [`Backend`] explicitly instead of reading global
//! state, so tests can run both side by side.
//!
//! Decoding results are bit-identical between backends. Sums are not: SIMD adds
//! eight lanes separately and combines them at the end, a different order of
//! floating-point additions, so totals can differ in the last few bits.

/// Which implementation of the kernels to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Plain Rust loops. Always available.
    Scalar,
    /// AVX2 + F16C + FMA. Only [`Backend::detect`] can create one, because the
    /// token's field is private: that is what makes the SIMD calls sound.
    Avx2(Avx2Token),
}

/// Proof that the running CPU supports the AVX2 kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Avx2Token(());

impl Backend {
    /// The fastest backend this CPU supports, checked at run time.
    pub fn detect() -> Backend {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            return Backend::Avx2(Avx2Token(()));
        }
        Backend::Scalar
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2(_) => "avx2",
        }
    }
}

/// Pass 1 of a chunk summary: counts, sum, and range of the finite values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scan {
    pub finite: u64,
    pub nan: u64,
    pub zeros: u64,
    pub sum: f64,
    pub min: f32,
    pub max: f32,
}

/// Elementwise sums for comparing two equally long chunks `a` and `b`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiffSums {
    /// Pairs where both values are finite; only these enter the numeric sums.
    pub finite_pairs: u64,
    /// Pairs that compare equal (`a == b`), plus pairs of identical non-finite values.
    pub equal: u64,
    /// Pairs where at least one side is NaN/Inf and the two do not match.
    pub nonfinite_mismatch: u64,
    pub sum_sq_diff: f64,
    pub sum_abs_diff: f64,
    pub max_abs_diff: f64,
    pub sum_aa: f64,
    pub sum_bb: f64,
    pub sum_ab: f64,
}

// Each dispatcher below matches on the backend. The `unsafe` blocks are sound
// because an `Avx2` value exists only after `detect` confirmed the CPU features.

/// Decode little-endian F16 bytes (two per value) into `out`.
pub fn f16_to_f32(backend: Backend, bytes: &[u8], out: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::f16_to_f32(bytes, out) },
        _ => scalar::f16_to_f32(bytes, out),
    }
}

/// Decode little-endian BF16 bytes (two per value) into `out`.
///
/// No SIMD version: BF16 decoding is a shift, and the compiler already turns the
/// scalar loop into vector code. A hand-written AVX2 kernel measured 1.06-1.34x
/// across runs, within the benchmark's noise, so it was removed (see README).
pub fn bf16_to_f32(_backend: Backend, bytes: &[u8], out: &mut [f32]) {
    scalar::bf16_to_f32(bytes, out);
}

/// Decode whole Q8_0 blocks into `out` (32 values per 34-byte block).
pub fn dequantize_q8_0(backend: Backend, bytes: &[u8], out: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::dequantize_q8_0(bytes, out) },
        _ => scalar::dequantize_q8_0(bytes, out),
    }
}

/// Decode whole Q4_0 blocks into `out` (32 values per 18-byte block).
pub fn dequantize_q4_0(backend: Backend, bytes: &[u8], out: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::dequantize_q4_0(bytes, out) },
        _ => scalar::dequantize_q4_0(bytes, out),
    }
}

pub fn scan(backend: Backend, values: &[f32]) -> Scan {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::scan(values) },
        _ => scalar::scan(values),
    }
}

/// Sum of squared deviations from `mean` over the finite values (pass 2).
pub fn sum_sq_dev(backend: Backend, values: &[f32], mean: f64) -> f64 {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::sum_sq_dev(values, mean) },
        _ => scalar::sum_sq_dev(values, mean),
    }
}

pub fn diff_sums(backend: Backend, a: &[f32], b: &[f32]) -> DiffSums {
    assert_eq!(a.len(), b.len());
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2(_) => unsafe { avx2::diff_sums(a, b) },
        _ => scalar::diff_sums(a, b),
    }
}

pub mod scalar {
    use super::{DiffSums, Scan};
    use crate::{half, quant};

    pub fn f16_to_f32(bytes: &[u8], out: &mut [f32]) {
        let (pairs, _) = bytes.as_chunks::<2>();
        for (value, pair) in out.iter_mut().zip(pairs) {
            *value = half::f16_to_f32(u16::from_le_bytes(*pair));
        }
    }

    pub fn bf16_to_f32(bytes: &[u8], out: &mut [f32]) {
        let (pairs, _) = bytes.as_chunks::<2>();
        for (value, pair) in out.iter_mut().zip(pairs) {
            *value = half::bf16_to_f32(u16::from_le_bytes(*pair));
        }
    }

    pub fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) {
        let (blocks, _) = bytes.as_chunks::<{ quant::Q8_0_BLOCK_BYTES }>();
        let (values, _) = out.as_chunks_mut::<{ quant::BLOCK_ELEMENTS }>();
        for (values, block) in values.iter_mut().zip(blocks) {
            quant::dequantize_q8_0_block(block, values);
        }
    }

    pub fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) {
        let (blocks, _) = bytes.as_chunks::<{ quant::Q4_0_BLOCK_BYTES }>();
        let (values, _) = out.as_chunks_mut::<{ quant::BLOCK_ELEMENTS }>();
        for (values, block) in values.iter_mut().zip(blocks) {
            quant::dequantize_q4_0_block(block, values);
        }
    }

    pub fn scan(values: &[f32]) -> Scan {
        let mut s = Scan {
            finite: 0,
            nan: 0,
            zeros: 0,
            sum: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        };
        for &x in values {
            if x.is_finite() {
                s.finite += 1;
                s.sum += f64::from(x);
                s.min = s.min.min(x);
                s.max = s.max.max(x);
                s.zeros += u64::from(x == 0.0);
            } else if x.is_nan() {
                s.nan += 1;
            }
        }
        s
    }

    pub fn sum_sq_dev(values: &[f32], mean: f64) -> f64 {
        let mut total = 0.0;
        for &x in values {
            if x.is_finite() {
                let d = f64::from(x) - mean;
                total += d * d;
            }
        }
        total
    }

    pub fn diff_sums(a: &[f32], b: &[f32]) -> DiffSums {
        let mut s = DiffSums::default();
        for (&x, &y) in a.iter().zip(b) {
            if x.is_finite() && y.is_finite() {
                let (x, y) = (f64::from(x), f64::from(y));
                let d = x - y;
                s.finite_pairs += 1;
                s.equal += u64::from(x == y);
                s.sum_sq_diff += d * d;
                s.sum_abs_diff += d.abs();
                s.max_abs_diff = s.max_abs_diff.max(d.abs());
                s.sum_aa += x * x;
                s.sum_bb += y * y;
                s.sum_ab += x * y;
            } else if x == y || (x.is_nan() && y.is_nan()) {
                s.equal += 1; // the same infinity, or NaN on both sides
            } else {
                s.nonfinite_mismatch += 1;
            }
        }
        s
    }
}

/// AVX2 versions. A `__m256` register holds eight f32 lanes; a `__m256d` holds
/// four f64 lanes, so each eight-float vector is widened as two halves before
/// being added to f64 accumulators. Comparisons produce lane masks (all ones or
/// all zeros) that select values without branching; `movemask` packs a mask's
/// sign bits into an integer, so `count_ones` counts the lanes that matched.
///
/// Every function here is `unsafe` because it requires the CPU features in its
/// `target_feature` attribute; the dispatchers above guarantee them. Loads and
/// stores are unaligned (`loadu`/`storeu`): memory-mapped tensor bytes and Vec
/// buffers carry no 32-byte alignment promise.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{DiffSums, Scan, scalar};
    use crate::{half, quant};

    #[inline]
    #[target_feature(enable = "avx2")]
    fn hsum_pd(v: __m256d) -> f64 {
        let mut lanes = [0.0f64; 4];
        // SAFETY: `lanes` has room for exactly four f64.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn lanes_ps(v: __m256) -> [f32; 8] {
        let mut lanes = [0.0f32; 8];
        // SAFETY: `lanes` has room for exactly eight f32.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
        lanes
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn hmax_pd(v: __m256d) -> f64 {
        let mut lanes = [0.0f64; 4];
        // SAFETY: `lanes` has room for exactly four f64.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
        lanes.into_iter().fold(0.0, f64::max)
    }

    /// Widen a 32-bit lane mask (four lanes) to a 64-bit mask for f64 lanes:
    /// sign extension turns each all-ones i32 into an all-ones i64.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen_mask(mask: __m128) -> __m256d {
        _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(mask)))
    }

    /// Lanes that are neither NaN nor infinite: |x| < inf is false for both.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn finite_mask(x: __m256) -> __m256 {
        let abs = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)));
        _mm256_cmp_ps::<_CMP_LT_OQ>(abs, _mm256_set1_ps(f32::INFINITY))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn count(mask: __m256) -> u64 {
        u64::from(_mm256_movemask_ps(mask).count_ones())
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn f16_to_f32(bytes: &[u8], out: &mut [f32]) {
        let (halves, _) = bytes.as_chunks::<16>();
        let (floats, _) = out.as_chunks_mut::<8>();
        let n = halves.len().min(floats.len());
        for (src, dst) in halves.iter().zip(floats.iter_mut()) {
            // SAFETY: src is 16 bytes (8 halves) and dst is 8 floats.
            unsafe {
                let h = _mm_loadu_si128(src.as_ptr().cast());
                _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_cvtph_ps(h));
            }
        }
        scalar::f16_to_f32(&bytes[n * 16..], &mut out[n * 8..]);
    }

    /// The low eight signed bytes of `codes`, widened to floats and scaled by `d`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn i8x8_times(codes: __m128i, d: __m256) -> __m256 {
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), d)
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) {
        let (blocks, _) = bytes.as_chunks::<{ quant::Q8_0_BLOCK_BYTES }>();
        let (values, _) = out.as_chunks_mut::<{ quant::BLOCK_ELEMENTS }>();
        for (block, dst) in blocks.iter().zip(values.iter_mut()) {
            let d = _mm256_set1_ps(half::f16_to_f32(u16::from_le_bytes([block[0], block[1]])));
            // SAFETY: the two 16-byte loads cover block[2..34]; the four 8-float
            // stores cover dst[0..32].
            unsafe {
                let first = _mm_loadu_si128(block[2..].as_ptr().cast());
                let second = _mm_loadu_si128(block[18..].as_ptr().cast());
                let p = dst.as_mut_ptr();
                _mm256_storeu_ps(p, i8x8_times(first, d));
                _mm256_storeu_ps(p.add(8), i8x8_times(_mm_srli_si128::<8>(first), d));
                _mm256_storeu_ps(p.add(16), i8x8_times(second, d));
                _mm256_storeu_ps(p.add(24), i8x8_times(_mm_srli_si128::<8>(second), d));
            }
        }
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) {
        let (blocks, _) = bytes.as_chunks::<{ quant::Q4_0_BLOCK_BYTES }>();
        let (values, _) = out.as_chunks_mut::<{ quant::BLOCK_ELEMENTS }>();
        let low_nibble = _mm_set1_epi8(0x0f);
        let eight = _mm_set1_epi8(8);
        for (block, dst) in blocks.iter().zip(values.iter_mut()) {
            let d = _mm256_set1_ps(half::f16_to_f32(u16::from_le_bytes([block[0], block[1]])));
            // SAFETY: the 16-byte load covers block[2..18]; the four 8-float stores
            // cover dst[0..32].
            unsafe {
                let packed = _mm_loadu_si128(block[2..].as_ptr().cast());
                // Low nibbles are elements 0..16, high nibbles 16..32 (not interleaved).
                // Subtracting 8 maps codes 0..=15 onto -8..=7 as signed bytes.
                let low = _mm_sub_epi8(_mm_and_si128(packed, low_nibble), eight);
                let high = _mm_sub_epi8(
                    _mm_and_si128(_mm_srli_epi16::<4>(packed), low_nibble),
                    eight,
                );
                let p = dst.as_mut_ptr();
                _mm256_storeu_ps(p, i8x8_times(low, d));
                _mm256_storeu_ps(p.add(8), i8x8_times(_mm_srli_si128::<8>(low), d));
                _mm256_storeu_ps(p.add(16), i8x8_times(high, d));
                _mm256_storeu_ps(p.add(24), i8x8_times(_mm_srli_si128::<8>(high), d));
            }
        }
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn scan(values: &[f32]) -> Scan {
        let (vectors, tail) = values.as_chunks::<8>();
        let inf = _mm256_set1_ps(f32::INFINITY);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let zero = _mm256_setzero_ps();
        let (mut sum_lo, mut sum_hi) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        let (mut min, mut max) = (inf, neg_inf);
        let (mut finite, mut nan, mut zeros) = (0, 0, 0);
        for v in vectors {
            // SAFETY: `v` is exactly eight f32.
            let x = unsafe { _mm256_loadu_ps(v.as_ptr()) };
            let ok = finite_mask(x);
            finite += count(ok);
            nan += count(_mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
            zeros += count(_mm256_cmp_ps::<_CMP_EQ_OQ>(x, zero));
            // Non-finite lanes become 0 for the sum, and +/-inf for min/max,
            // so they cannot affect any result.
            let kept = _mm256_and_ps(x, ok);
            sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(kept)));
            sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(kept)));
            min = _mm256_min_ps(min, _mm256_blendv_ps(inf, x, ok));
            max = _mm256_max_ps(max, _mm256_blendv_ps(neg_inf, x, ok));
        }
        let rest = scalar::scan(tail);
        Scan {
            finite: finite + rest.finite,
            nan: nan + rest.nan,
            zeros: zeros + rest.zeros,
            sum: hsum_pd(_mm256_add_pd(sum_lo, sum_hi)) + rest.sum,
            min: lanes_ps(min).into_iter().fold(rest.min, f32::min),
            max: lanes_ps(max).into_iter().fold(rest.max, f32::max),
        }
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn sum_sq_dev(values: &[f32], mean: f64) -> f64 {
        let (vectors, tail) = values.as_chunks::<8>();
        let mean_v = _mm256_set1_pd(mean);
        let (mut acc_lo, mut acc_hi) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        for v in vectors {
            // SAFETY: `v` is exactly eight f32.
            let x = unsafe { _mm256_loadu_ps(v.as_ptr()) };
            let ok = finite_mask(x);
            // Non-finite lanes give inf/NaN deviations; AND with the mask zeroes them.
            let d_lo = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)), mean_v);
            let d_hi = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x)), mean_v);
            let d_lo = _mm256_and_pd(d_lo, widen_mask(_mm256_castps256_ps128(ok)));
            let d_hi = _mm256_and_pd(d_hi, widen_mask(_mm256_extractf128_ps::<1>(ok)));
            acc_lo = _mm256_fmadd_pd(d_lo, d_lo, acc_lo);
            acc_hi = _mm256_fmadd_pd(d_hi, d_hi, acc_hi);
        }
        hsum_pd(_mm256_add_pd(acc_lo, acc_hi)) + scalar::sum_sq_dev(tail, mean)
    }

    /// Accumulators for one four-lane half of the eight pairs in a vector.
    struct Half {
        sq: __m256d,
        abs: __m256d,
        max: __m256d,
        aa: __m256d,
        bb: __m256d,
        ab: __m256d,
    }

    impl Half {
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn new() -> Half {
            let z = _mm256_setzero_pd();
            Half {
                sq: z,
                abs: z,
                max: z,
                aa: z,
                bb: z,
                ab: z,
            }
        }

        /// Add four masked pairs. Masked-out lanes are exactly zero on both sides.
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn add(&mut self, a: __m128, b: __m128, keep: __m128) {
            let mask = widen_mask(keep);
            let a = _mm256_and_pd(_mm256_cvtps_pd(a), mask);
            let b = _mm256_and_pd(_mm256_cvtps_pd(b), mask);
            let d = _mm256_sub_pd(a, b);
            let abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), d);
            self.sq = _mm256_fmadd_pd(d, d, self.sq);
            self.abs = _mm256_add_pd(self.abs, abs);
            self.max = _mm256_max_pd(self.max, abs);
            self.aa = _mm256_fmadd_pd(a, a, self.aa);
            self.bb = _mm256_fmadd_pd(b, b, self.bb);
            self.ab = _mm256_fmadd_pd(a, b, self.ab);
        }
    }

    #[target_feature(enable = "avx2,f16c,fma,popcnt")]
    pub unsafe fn diff_sums(a: &[f32], b: &[f32]) -> DiffSums {
        let (va, tail_a) = a.as_chunks::<8>();
        let (vb, tail_b) = b.as_chunks::<8>();
        let (mut lo, mut hi) = (Half::new(), Half::new());
        let (mut finite_pairs, mut equal, mut mismatch) = (0, 0, 0);
        let all_lanes = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (pa, pb) in va.iter().zip(vb) {
            // SAFETY: both are exactly eight f32.
            let (x, y) = unsafe { (_mm256_loadu_ps(pa.as_ptr()), _mm256_loadu_ps(pb.as_ptr())) };
            let ok = _mm256_and_ps(finite_mask(x), finite_mask(y));
            let both_nan = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x),
                _mm256_cmp_ps::<_CMP_UNORD_Q>(y, y),
            );
            let same = _mm256_or_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(x, y), both_nan);
            finite_pairs += count(ok);
            equal += count(same);
            // andnot(m, all) is "not m": lanes neither finite on both sides nor matching.
            mismatch += count(_mm256_andnot_ps(_mm256_or_ps(ok, same), all_lanes));
            lo.add(
                _mm256_castps256_ps128(x),
                _mm256_castps256_ps128(y),
                _mm256_castps256_ps128(ok),
            );
            hi.add(
                _mm256_extractf128_ps::<1>(x),
                _mm256_extractf128_ps::<1>(y),
                _mm256_extractf128_ps::<1>(ok),
            );
        }
        let rest = scalar::diff_sums(tail_a, tail_b);
        DiffSums {
            finite_pairs: finite_pairs + rest.finite_pairs,
            equal: equal + rest.equal,
            nonfinite_mismatch: mismatch + rest.nonfinite_mismatch,
            sum_sq_diff: hsum_pd(_mm256_add_pd(lo.sq, hi.sq)) + rest.sum_sq_diff,
            sum_abs_diff: hsum_pd(_mm256_add_pd(lo.abs, hi.abs)) + rest.sum_abs_diff,
            max_abs_diff: hmax_pd(_mm256_max_pd(lo.max, hi.max)).max(rest.max_abs_diff),
            sum_aa: hsum_pd(_mm256_add_pd(lo.aa, hi.aa)) + rest.sum_aa,
            sum_bb: hsum_pd(_mm256_add_pd(lo.bb, hi.bb)) + rest.sum_bb,
            sum_ab: hsum_pd(_mm256_add_pd(lo.ab, hi.ab)) + rest.sum_ab,
        }
    }
}
