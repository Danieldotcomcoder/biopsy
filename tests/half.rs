//! Exhaustive checks: there are only 65,536 sixteen-bit patterns, so test them all.
use biopsy::half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};

/// An independent decoder written from the IEEE formula instead of bit surgery.
fn reference_f16(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f64::from(bits & 0x03ff);
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        31 if mantissa == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(e - 15),
    }
}

#[test]
fn every_f16_pattern_matches_the_formula() {
    for bits in 0..=u16::MAX {
        let expected = reference_f16(bits);
        let got = f16_to_f32(bits);
        if expected.is_nan() {
            assert!(got.is_nan(), "{bits:#06x}");
        } else {
            assert_eq!(got.to_bits(), (expected as f32).to_bits(), "{bits:#06x}");
        }
    }
}

#[test]
fn every_f16_pattern_round_trips_through_f32() {
    for bits in 0..=u16::MAX {
        let value = f16_to_f32(bits);
        if value.is_nan() {
            assert!(f16_to_f32(f32_to_f16(value)).is_nan(), "{bits:#06x}");
        } else {
            assert_eq!(f32_to_f16(value), bits, "{bits:#06x}");
        }
    }
}

#[test]
fn f32_to_f16_rounds_to_nearest_even_between_every_pair() {
    for bits in 0u16..0x7c00 {
        let low = f64::from(f16_to_f32(bits));
        // Past the largest finite value (65504), rounding treats 65536 as the next step.
        let high = if bits == 0x7bff {
            65536.0
        } else {
            f64::from(f16_to_f32(bits + 1))
        };
        // F16 has 11 significant bits, so the midpoint needs 12: exact in F32.
        let mid = ((low + high) / 2.0) as f32;
        let even = if bits % 2 == 0 { bits } else { bits + 1 };
        assert_eq!(f32_to_f16(mid), even, "midpoint above {bits:#06x}");
        assert_eq!(
            f32_to_f16(mid.next_down()),
            bits,
            "below midpoint {bits:#06x}"
        );
        assert_eq!(
            f32_to_f16(mid.next_up()),
            bits + 1,
            "above midpoint {bits:#06x}"
        );
        assert_eq!(f32_to_f16(-mid), even | 0x8000, "negative {bits:#06x}");
    }
}

#[test]
fn f32_to_f16_edge_cases() {
    assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
    assert_eq!(f32_to_f16(1.0e6), 0x7c00);
    assert_eq!(f32_to_f16(f32::MIN_POSITIVE), 0); // F32 normal, far below F16 range
    assert_eq!(f32_to_f16(1.0e-45), 0); // F32 subnormal
    assert_eq!(f32_to_f16(-0.0), 0x8000);
    assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
}

#[test]
fn every_bf16_pattern_is_the_top_half_of_an_f32() {
    for bits in 0..=u16::MAX {
        let value = bf16_to_f32(bits);
        assert_eq!(value.to_bits(), u32::from(bits) << 16);
        if !value.is_nan() {
            assert_eq!(f32_to_bf16(value), bits);
        }
    }
}

#[test]
fn f32_to_bf16_rounds_to_nearest_even() {
    // 1.0 is 0x3f80_0000; the next BF16 is 0x3f81_0000. Their midpoint ties to even.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(f32_to_bf16(f32::MAX), 0x7f80); // rounds up to infinity
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}
