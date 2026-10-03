//! A format-independent view of one stored tensor. Safetensors and GGUF headers
//! both lower to this, so stats, health and diff never need to know the file type.

use std::fmt;

/// Encodings biopsy can turn into F32 values. Everything else is still inspected
/// (sizes are validated) but reported as "not decoded" by the numeric commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// GGML block of 32 values: one F16 scale, then 32 signed bytes.
    Q8_0,
    /// GGML block of 32 values: one F16 scale, then 16 bytes of 4-bit codes.
    Q4_0,
}

impl DType {
    /// Values stored together in one block. Plain floats are blocks of one.
    pub fn block_elements(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q8_0 | DType::Q4_0 => crate::quant::BLOCK_ELEMENTS,
        }
    }

    /// Bytes occupied by one block.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q8_0 => crate::quant::Q8_0_BLOCK_BYTES,
            DType::Q4_0 => crate::quant::Q4_0_BLOCK_BYTES,
        }
    }

    /// Byte range holding elements `start..end`. Both ends must be block-aligned.
    pub fn byte_range(self, start: usize, end: usize) -> std::ops::Range<usize> {
        let per_block = self.block_elements();
        debug_assert!(start.is_multiple_of(per_block) && end.is_multiple_of(per_block));
        start / per_block * self.block_bytes()..end / per_block * self.block_bytes()
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::Q8_0 => "Q8_0",
            DType::Q4_0 => "Q4_0",
        })
    }
}

/// One tensor's metadata plus a zero-copy borrow of its stored bytes.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub name: String,
    /// The type name exactly as the file spells it, e.g. "BF16" or "Q4_K".
    pub type_name: String,
    /// `None` when biopsy does not decode this encoding.
    pub dtype: Option<DType>,
    /// Row-major shape, outermost dimension first (PyTorch order). GGUF stores
    /// dimensions innermost first, so its reader reverses them into this order.
    pub shape: Vec<u64>,
    pub elements: u64,
    pub bytes: &'a [u8],
}

impl TensorView<'_> {
    /// Elements per row, where a row is one index along the first (outermost) axis.
    /// For `[out, in]` linear weights that is one output unit's input weights; for
    /// `[vocab, dim]` embeddings it is one token. Fewer than two axes: no rows.
    /// `None` too if the product overflows, which only an empty tensor such as
    /// `[0, u64::MAX, 2]` can reach (the parsers validate every nonempty shape).
    pub fn row_len(&self) -> Option<u64> {
        match self.shape.as_slice() {
            [] | [_] => None,
            [_, rest @ ..] => rest.iter().try_fold(1u64, |n, &d| n.checked_mul(d)),
        }
    }

    pub fn rows(&self) -> u64 {
        self.shape.first().copied().unwrap_or(1)
    }
}
