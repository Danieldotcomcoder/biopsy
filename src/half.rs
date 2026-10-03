//! Half-precision floats by hand. Both formats are 16 bits, laid out differently:
//!
//! ```text
//! F16  (IEEE binary16): [sign:1][exponent:5, bias 15 ][mantissa:10]
//! BF16 (bfloat16):      [sign:1][exponent:8, bias 127][mantissa:7]
//! F32  (IEEE binary32): [sign:1][exponent:8, bias 127][mantissa:23]
//! ```
//!
//! BF16 is literally the top half of an F32, so decoding is a shift. F16 has a
//! smaller exponent, so decoding must re-bias the exponent and handle subnormals.

/// Decode one IEEE binary16 value. Every F16 value is exactly representable in F32,
/// so this conversion never rounds.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        // Zero and subnormals have no implicit leading 1: value = mantissa × 2^-24.
        // Both factors are exact in F32, so the product is exact too.
        0 => (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits(),
        // All-ones exponent: infinity (mantissa 0) or NaN (payload kept).
        0x1f => 0x7f80_0000 | (mantissa << 13),
        // Normal: rebias the exponent (127 - 15 = 112) and widen the mantissa.
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Encode an F32 as the nearest IEEE binary16, ties to even (the IEEE default).
/// Values at or beyond 65520 become infinity; NaN stays NaN.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        // NaN: force the quiet bit so a truncated payload can never become infinity.
        return sign | 0x7e00 | ((abs >> 13) & 0x03ff) as u16;
    }
    if abs >= 0x4780_0000 {
        // |value| >= 65536 (or infinity) is beyond the largest finite F16 (65504).
        return sign | 0x7c00;
    }
    if abs >= 0x3880_0000 {
        // F16 normal range (|value| >= 2^-14). Rebias the exponent in place, then
        // round away the 13 low mantissa bits. Adding 0xfff (+1 if the kept part is
        // odd) carries exactly when the dropped bits are above, or tied at, half.
        // A carry may ripple into the exponent; that is the correct rounding.
        let rebiased = abs - (112 << 23);
        let rounded = rebiased + 0x0fff + ((rebiased >> 13) & 1);
        return sign | (rounded >> 13) as u16;
    }
    // F16 subnormal (or zero): the result is round(|value| / 2^-24).
    // |value| = significand × 2^(exponent - 150), so we shift right by 126 - exponent.
    let exponent = abs >> 23;
    let significand = (abs & 0x007f_ffff) | 0x0080_0000;
    let shift = 126 - exponent; // at least 14 here, because exponent <= 112
    if shift > 24 {
        return sign; // below half of the smallest subnormal: rounds to zero
    }
    let kept = significand >> shift;
    let dropped = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let round_up = dropped > half || (dropped == half && kept & 1 == 1);
    // If rounding reaches 0x400, that is the smallest normal number's encoding.
    sign | (kept + u32::from(round_up)) as u16
}

/// Decode one bfloat16 value: it is the upper 16 bits of an F32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encode an F32 as bfloat16 with ties-to-even rounding (used by test fixtures).
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x0040) as u16; // keep it a quiet NaN
    }
    let rounded = bits + 0x7fff + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc049), -3.140625);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // the tie rounds to even: infinity
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
    }
}
