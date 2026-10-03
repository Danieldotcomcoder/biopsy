use biopsy::{
    fixtures::Rng,
    half::{f16_to_f32, f32_to_f16},
    quant::{
        BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES, dequantize_q4_0_block,
        dequantize_q8_0_block, quantize_q4_0, quantize_q8_0,
    },
};

type BlockDecoder = fn(&[u8], &mut [f32]);

fn dequantize<const BYTES: usize>(bytes: &[u8], decode: BlockDecoder) -> Vec<f32> {
    let (blocks, _) = bytes.as_chunks::<BYTES>();
    let mut out = vec![0.0; blocks.len() * BLOCK_ELEMENTS];
    for (block, values) in blocks.iter().zip(out.as_chunks_mut::<BLOCK_ELEMENTS>().0) {
        decode(block, values);
    }
    out
}

fn stored_scale(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

/// Decode each block and check every value against `bound(stored scale)`.
fn check_error<const BYTES: usize>(
    values: &[f32],
    bytes: &[u8],
    decode: BlockDecoder,
    bound: impl Fn(f32) -> f32,
) {
    let (blocks, _) = bytes.as_chunks::<BYTES>();
    let (originals, _) = values.as_chunks::<BLOCK_ELEMENTS>();
    assert_eq!(blocks.len(), originals.len());
    for (block, original) in blocks.iter().zip(originals) {
        let mut decoded = [0.0; BLOCK_ELEMENTS];
        decode(block, &mut decoded);
        let bound = bound(stored_scale(block));
        for (x, y) in original.iter().zip(decoded) {
            assert!((x - y).abs() <= bound, "{x} -> {y}, bound {bound}");
        }
    }
}

#[test]
fn q8_0_error_is_within_half_a_step() {
    let values = Rng::new(1).normals(32 * 500, 1.0);
    let bytes = quantize_q8_0(&values);
    assert_eq!(bytes.len(), 500 * Q8_0_BLOCK_BYTES);
    // Half a quantization step, plus the scale's own F16 rounding times 127 codes.
    check_error::<Q8_0_BLOCK_BYTES>(&values, &bytes, dequantize_q8_0_block, |d| {
        d * (0.5 + 127.0 / 2048.0)
    });
}

#[test]
fn q4_0_error_is_within_one_step() {
    let values = Rng::new(2).normals(32 * 500, 1.0);
    let bytes = quantize_q4_0(&values);
    assert_eq!(bytes.len(), 500 * Q4_0_BLOCK_BYTES);
    // Codes span -8..=7, so the side opposite the extreme clamps at 7: a full
    // step of error is possible there, half a step elsewhere.
    check_error::<Q4_0_BLOCK_BYTES>(&values, &bytes, dequantize_q4_0_block, |d| {
        d.abs() * (1.0 + 8.0 / 1024.0)
    });
}

#[test]
fn values_on_the_grid_survive_exactly() {
    // amax = 63.5 gives d = 0.5, exact in F16, so every k/2 is reproduced.
    let mut q8: Vec<f32> = (0..32).map(|k| (k as f32 - 16.0) * 0.5).collect();
    q8[0] = 63.5;
    let decoded = dequantize::<Q8_0_BLOCK_BYTES>(&quantize_q8_0(&q8), dequantize_q8_0_block);
    assert_eq!(decoded, q8);

    // Q4_0: the largest-magnitude value (-4) maps to code 0, so d = 0.5 exactly.
    let q4: Vec<f32> = (0..32).map(|k| ((k % 16) as f32 - 8.0) * 0.5).collect();
    let decoded = dequantize::<Q4_0_BLOCK_BYTES>(&quantize_q4_0(&q4), dequantize_q4_0_block);
    assert_eq!(decoded, q4);
}

#[test]
fn zero_blocks_and_sign_conventions() {
    let zeros = quantize_q4_0(&[0.0; 32]);
    // d = 0 / -8 = -0.0 (F16 0x8000), exactly what ggml's reference writes.
    assert_eq!(&zeros[..2], &[0x00, 0x80]);
    assert!(zeros[2..].iter().all(|&b| b == 0x88)); // code 8 = zero, both nibbles
    assert_eq!(
        dequantize::<Q4_0_BLOCK_BYTES>(&zeros, dequantize_q4_0_block),
        vec![0.0; 32]
    );

    // A positive extreme gives a NEGATIVE Q4_0 scale (d = max / -8).
    let mut values = [0.0f32; 32];
    values[7] = 4.0;
    let bytes = quantize_q4_0(&values);
    assert_eq!(stored_scale(&bytes), -0.5);
    assert_eq!(bytes[2 + 7] & 0x0f, 0); // element 7: low nibble of byte 7, code 0
    assert_eq!(f32_to_f16(-0.5).to_le_bytes(), [bytes[0], bytes[1]]);
}
