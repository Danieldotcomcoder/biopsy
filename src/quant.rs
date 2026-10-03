//! GGML block quantization, the two simplest schemes. A block stores 32 values as
//! small integers plus one shared F16 scale `d`; decoding is `value = d × code`.
//!
//! ```text
//! Q8_0 block, 34 bytes: [d: f16 LE][q0 q1 ... q31: i8]           value[i] = d × q[i]
//! Q4_0 block, 18 bytes: [d: f16 LE][16 bytes, two 4-bit codes each]
//!     byte j low  nibble -> value[j]      = d × (low  - 8)
//!     byte j high nibble -> value[j + 16] = d × (high - 8)
//! ```
//!
//! Q4_0 does NOT interleave neighbours: byte j holds elements j and j + 16.
//! The quantizers mirror ggml's reference (`quantize_row_q8_0_ref`,
//! `quantize_row_q4_0_ref`) so fixtures match what llama.cpp would write.

use crate::half::{f16_to_f32, f32_to_f16};

pub const BLOCK_ELEMENTS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS;
pub const Q4_0_BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS / 2;

fn scale(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

/// Decode one 34-byte Q8_0 block into 32 values.
pub fn dequantize_q8_0_block(block: &[u8], out: &mut [f32]) {
    let d = scale(block);
    for (value, &code) in out[..BLOCK_ELEMENTS]
        .iter_mut()
        .zip(&block[2..Q8_0_BLOCK_BYTES])
    {
        *value = f32::from(code as i8) * d;
    }
}

/// Decode one 18-byte Q4_0 block into 32 values.
pub fn dequantize_q4_0_block(block: &[u8], out: &mut [f32]) {
    let d = scale(block);
    let (low, high) = out[..BLOCK_ELEMENTS].split_at_mut(BLOCK_ELEMENTS / 2);
    for (j, &byte) in block[2..Q4_0_BLOCK_BYTES].iter().enumerate() {
        low[j] = f32::from((byte & 0x0f) as i8 - 8) * d;
        high[j] = f32::from((byte >> 4) as i8 - 8) * d;
    }
}

/// Quantize values (a multiple of 32) to Q8_0: `d = max|x| / 127`, codes rounded.
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len().is_multiple_of(BLOCK_ELEMENTS));
    let mut out = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * Q8_0_BLOCK_BYTES);
    for block in values.as_chunks::<BLOCK_ELEMENTS>().0 {
        let amax = block.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(block.iter().map(|x| (x * inverse).round() as i8 as u8));
    }
    out
}

/// Quantize values (a multiple of 32) to Q4_0. ggml picks the value with the
/// largest magnitude, keeps its sign, and sets `d = max / -8` so that value maps
/// exactly onto code 0 (= -8), using the asymmetric 4-bit range -8..=7 fully.
pub fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len().is_multiple_of(BLOCK_ELEMENTS));
    let mut out = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * Q4_0_BLOCK_BYTES);
    for block in values.as_chunks::<BLOCK_ELEMENTS>().0 {
        let max = block
            .iter()
            .fold(0.0f32, |m, &x| if x.abs() > m.abs() { x } else { m });
        let d = max / -8.0;
        let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        let code = |x: f32| ((x * inverse + 8.5) as i8).min(15) as u8;
        let (low, high) = block.split_at(BLOCK_ELEMENTS / 2);
        out.extend(low.iter().zip(high).map(|(&a, &b)| code(a) | code(b) << 4));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_built_blocks_decode_exactly() {
        // d = 0.5 (F16 0x3800), codes -1, 0, 1, ...
        let mut q8 = vec![0x00, 0x38];
        q8.extend((0..32).map(|i| (i as i8 - 1) as u8));
        let mut out = [0.0; 32];
        dequantize_q8_0_block(&q8, &mut out);
        assert_eq!(out[0], -0.5);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[31], 15.0);

        // d = 2.0 (F16 0x4000); byte 0 = 0xF0: low nibble 0 -> -8, high 15 -> +7.
        let mut q4 = vec![0x00, 0x40, 0xf0];
        q4.extend([0x88; 15]); // code 8 means zero
        dequantize_q4_0_block(&q4, &mut out);
        assert_eq!(out[0], -16.0); // element 0 comes from byte 0's LOW nibble
        assert_eq!(out[16], 14.0); // element 16 comes from byte 0's HIGH nibble
        assert!(out[1..16].iter().chain(&out[17..]).all(|&x| x == 0.0));
    }
}
