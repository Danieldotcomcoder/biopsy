//! Turn a tensor's stored bytes into F32 values, a bounded chunk at a time.
//!
//! Nothing here allocates a whole tensor's worth of F32s. Work is split twice:
//! a parallel JOB covers 64 Ki values; inside a job, values are decoded 4 Ki at
//! a time into a reused buffer. Memory use depends on the thread count, never on
//! the model size.

use std::ops::Range;

use crate::{
    kernels::{self, Backend},
    tensor::{DType, TensorView},
};

/// Values decoded per buffer fill: 16 KiB of F32, which stays in a core's 32 KiB L1
/// data cache while it is decoded and then read twice by the summary. Measured
/// with `examples/bench.rs` (see README): 64 Ki-value buffers spill a 256 KiB L2
/// once two hyperthreads share a core, and ran the 16-thread pipeline 4x slower.
/// A multiple of every block size, so a buffer never splits a quantized block.
pub const CHUNK_ELEMENTS: usize = 1 << 12;

/// Values per parallel job. Much larger than a buffer so that a multi-GB model
/// yields thousands of jobs, not millions; a multiple of `CHUNK_ELEMENTS`.
pub const JOB_ELEMENTS: usize = 1 << 16;

/// Decode whole blocks of `dtype` from `bytes` into `out` (resized to fit).
pub fn decode_into(dtype: DType, bytes: &[u8], out: &mut Vec<f32>, backend: Backend) {
    debug_assert!(bytes.len().is_multiple_of(dtype.block_bytes()));
    let count = bytes.len() / dtype.block_bytes() * dtype.block_elements();
    // Resizing a reused buffer to the same length writes nothing, so the zero fill
    // only happens the first time each thread's buffer grows.
    out.resize(count, 0.0);
    match dtype {
        DType::F32 => {
            // `from_le_bytes` on 4-byte arrays never assumes the mapping is aligned.
            let (words, _) = bytes.as_chunks::<4>();
            for (value, word) in out.iter_mut().zip(words) {
                *value = f32::from_le_bytes(*word);
            }
        }
        DType::F16 => kernels::f16_to_f32(backend, bytes, out),
        DType::BF16 => kernels::bf16_to_f32(backend, bytes, out),
        DType::Q8_0 => kernels::dequantize_q8_0(backend, bytes, out),
        DType::Q4_0 => kernels::dequantize_q4_0(backend, bytes, out),
    }
}

/// Number of values a decodable view holds, derived from its byte length.
pub fn element_count(dtype: DType, bytes: &[u8]) -> usize {
    bytes.len() / dtype.block_bytes() * dtype.block_elements()
}

/// Split `0..elements` into consecutive ranges of at most `chunk` elements.
/// Fixed boundaries make every parallel reduction reproducible: the same chunks
/// are merged in the same order whatever the thread count.
pub fn chunk_ranges(elements: usize, chunk: usize) -> impl Iterator<Item = Range<usize>> {
    (0..elements)
        .step_by(chunk)
        .map(move |start| start..elements.min(start + chunk))
}

/// One unit of parallel work: up to `JOB_ELEMENTS` values of one tensor.
#[derive(Clone, Debug)]
pub struct Job<'a> {
    pub tensor: usize,
    pub dtype: DType,
    pub bytes: &'a [u8],
}

/// Every decodable tensor's jobs as one flat list, in tensor order. A flat list
/// balances load across threads even when one tensor dwarfs the rest.
pub fn jobs<'a>(tensors: &[TensorView<'a>]) -> Vec<Job<'a>> {
    let mut jobs = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let Some(dtype) = tensor.dtype else { continue };
        for range in chunk_ranges(element_count(dtype, tensor.bytes), JOB_ELEMENTS) {
            jobs.push(Job {
                tensor: index,
                dtype,
                bytes: &tensor.bytes[dtype.byte_range(range.start, range.end)],
            });
        }
    }
    jobs
}

/// Visit a byte range's values sequentially, at most `CHUNK_ELEMENTS` at a time.
/// Used inside one job (or one row), so the visiting order is fixed.
pub fn for_each_chunk(
    dtype: DType,
    bytes: &[u8],
    buffer: &mut Vec<f32>,
    backend: Backend,
    mut visit: impl FnMut(&[f32]),
) {
    for range in chunk_ranges(element_count(dtype, bytes), CHUNK_ELEMENTS) {
        decode_into(
            dtype,
            &bytes[dtype.byte_range(range.start, range.end)],
            buffer,
            backend,
        );
        visit(buffer);
    }
}

/// Like `for_each_chunk`, but walks two equally long tensors in lockstep (their
/// dtypes may differ), handing the visitor matching runs of values.
pub fn for_each_chunk_pair(
    a: (DType, &[u8]),
    b: (DType, &[u8]),
    buffers: &mut (Vec<f32>, Vec<f32>),
    backend: Backend,
    mut visit: impl FnMut(&[f32], &[f32]),
) {
    let elements = element_count(a.0, a.1);
    debug_assert_eq!(elements, element_count(b.0, b.1));
    for range in chunk_ranges(elements, CHUNK_ELEMENTS) {
        let (start, end) = (range.start, range.end);
        decode_into(
            a.0,
            &a.1[a.0.byte_range(start, end)],
            &mut buffers.0,
            backend,
        );
        decode_into(
            b.0,
            &b.1[b.0.byte_range(start, end)],
            &mut buffers.1,
            backend,
        );
        visit(&buffers.0, &buffers.1);
    }
}
