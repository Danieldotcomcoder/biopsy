//! Hand-written GGUF reader (the llama.cpp file format). Everything is little-endian:
//!
//! ```text
//! "GGUF" | version: u32 | tensor_count: u64 | metadata_count: u64
//! metadata_count × [key: string][value_type: u32][value]
//! tensor_count   × [name: string][n_dims: u32][dims: n_dims × u64][ggml_type: u32][offset: u64]
//! zero padding up to a multiple of `general.alignment` (default 32)
//! tensor data: each tensor at data_start + offset, padded to the alignment
//! ```
//!
//! A string is `[len: u64][len bytes of UTF-8]` with no terminator. Dimensions
//! are listed innermost first (ggml's `ne[0]` is the contiguous axis), the
//! reverse of PyTorch/safetensors order: a `[out, in]` weight appears as `[in, out]`.

use std::collections::HashSet;

use anyhow::{Context, Result, bail, ensure};

use crate::tensor::{DType, TensorView};

pub const MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml's `GGML_MAX_DIMS`; llama.cpp rejects tensors with more.
const MAX_DIMS: u32 = 4;
/// Arrays may contain arrays. A depth limit keeps a hostile file from
/// recursing until the stack overflows.
const MAX_ARRAY_DEPTH: usize = 8;
/// Array elements kept for display. Every element is still parsed and validated.
const ARRAY_PREVIEW: usize = 8;

/// A ggml storage type. Quantized types pack `block_elements` values into
/// `block_bytes` bytes; plain types are blocks of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlType {
    pub id: u32,
    pub name: &'static str,
    pub block_elements: u64,
    pub block_bytes: u64,
}

const fn ty(id: u32, name: &'static str, block_elements: u64, block_bytes: u64) -> GgmlType {
    GgmlType {
        id,
        name,
        block_elements,
        block_bytes,
    }
}

/// Block geometry from ggml's `type_traits` and the block structs in
/// `ggml-common.h`. Ids 4, 5 and 31-33, 36-38 were removed from the format.
const GGML_TYPES: &[GgmlType] = &[
    ty(0, "F32", 1, 4),
    ty(1, "F16", 1, 2),
    ty(2, "Q4_0", 32, 18),
    ty(3, "Q4_1", 32, 20),
    ty(6, "Q5_0", 32, 22),
    ty(7, "Q5_1", 32, 24),
    ty(8, "Q8_0", 32, 34),
    ty(9, "Q8_1", 32, 36),
    ty(10, "Q2_K", 256, 84),
    ty(11, "Q3_K", 256, 110),
    ty(12, "Q4_K", 256, 144),
    ty(13, "Q5_K", 256, 176),
    ty(14, "Q6_K", 256, 210),
    ty(15, "Q8_K", 256, 292),
    ty(16, "IQ2_XXS", 256, 66),
    ty(17, "IQ2_XS", 256, 74),
    ty(18, "IQ3_XXS", 256, 98),
    ty(19, "IQ1_S", 256, 50),
    ty(20, "IQ4_NL", 32, 18),
    ty(21, "IQ3_S", 256, 110),
    ty(22, "IQ2_S", 256, 82),
    ty(23, "IQ4_XS", 256, 136),
    ty(24, "I8", 1, 1),
    ty(25, "I16", 1, 2),
    ty(26, "I32", 1, 4),
    ty(27, "I64", 1, 8),
    ty(28, "F64", 1, 8),
    ty(29, "IQ1_M", 256, 56),
    ty(30, "BF16", 1, 2),
    ty(34, "TQ1_0", 256, 54),
    ty(35, "TQ2_0", 256, 66),
    ty(39, "MXFP4", 32, 17),
];

/// Ids that once existed and were dropped from the format. Naming them turns a
/// puzzling "unknown type" into an actionable message.
const REMOVED_GGML_TYPES: &[(u32, &str)] = &[
    (4, "Q4_2"),
    (5, "Q4_3"),
    (31, "Q4_0_4_4"),
    (32, "Q4_0_4_8"),
    (33, "Q4_0_8_8"),
    (36, "IQ4_NL_4_4"),
    (37, "IQ4_NL_4_8"),
    (38, "IQ4_NL_8_8"),
];

impl GgmlType {
    pub fn from_id(id: u32) -> Option<GgmlType> {
        GGML_TYPES.iter().copied().find(|t| t.id == id)
    }

    pub fn from_name(name: &str) -> Option<GgmlType> {
        GGML_TYPES.iter().copied().find(|t| t.name == name)
    }

    /// The decoder for this type, if biopsy has one.
    pub fn dtype(self) -> Option<DType> {
        match self.name {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "Q8_0" => Some(DType::Q8_0),
            "Q4_0" => Some(DType::Q4_0),
            _ => None,
        }
    }
}

/// A metadata value. Arrays keep their length and a short preview only:
/// tokenizer vocabularies alone can hold hundreds of thousands of strings.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        len: u64,
        preview: Vec<Value>,
    },
}

/// Metadata value type ids, in the order the GGUF specification numbers them.
const VALUE_TYPE_NAMES: [&str; 13] = [
    "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64",
];

fn value_type_name(id: u32) -> &'static str {
    VALUE_TYPE_NAMES.get(id as usize).copied().unwrap_or("?")
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array { .. } => "array",
        }
    }

    /// Short human-readable rendering: long strings and arrays are abbreviated.
    pub fn describe(&self) -> String {
        match self {
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            Value::String(s) => {
                let shown: String = s.chars().take(60).collect();
                if shown.len() < s.len() {
                    format!("{shown:?}... ({} chars)", s.chars().count())
                } else {
                    format!("{s:?}")
                }
            }
            Value::Array {
                element_type,
                len,
                preview,
            } => {
                let items: Vec<String> = preview.iter().map(Value::describe).collect();
                let more = if (preview.len() as u64) < *len {
                    ", ..."
                } else {
                    ""
                };
                format!(
                    "[{} x {len}] [{}{more}]",
                    value_type_name(*element_type),
                    items.join(", ")
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub ggml_type: GgmlType,
    /// As stored: innermost (contiguous) dimension first.
    pub dims: Vec<u64>,
    /// Relative to the start of the tensor data section.
    pub data_offset: u64,
    /// Absolute position in the file.
    pub file_offset: usize,
    pub byte_len: usize,
    pub elements: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    /// In file order. Keys are unique.
    pub metadata: Vec<(String, Value)>,
    /// In file order, which is also data order.
    pub tensors: Vec<GgufTensor>,
    pub alignment: u64,
    pub data_start: usize,
}

impl Gguf {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// A cursor over the file's bytes. Every read is bounds-checked, so a lying
/// length becomes a "truncated" error rather than a panic or a huge allocation.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .with_context(|| {
                format!(
                    "truncated: {what} needs {len} bytes at offset {}, {} remain",
                    self.pos,
                    self.remaining()
                )
            })?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        Ok(self
            .take(N, what)?
            .try_into()
            .expect("take returned N bytes"))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn str(&mut self, what: &str) -> Result<&'a str> {
        let len = self.u64(what)?;
        let len = usize::try_from(len).with_context(|| format!("{what}: length overflow"))?;
        let bytes = self.take(len, what)?;
        std::str::from_utf8(bytes).with_context(|| format!("{what} is not valid UTF-8"))
    }

    /// Refuse a count that cannot possibly fit before decoding any element.
    fn check_count(&self, count: u64, min_item_bytes: usize, what: &str) -> Result<()> {
        ensure!(
            count <= (self.remaining() / min_item_bytes) as u64,
            "{what} count {count} cannot fit in the remaining {} bytes",
            self.remaining()
        );
        Ok(())
    }

    fn value(&mut self, value_type: u32, depth: usize) -> Result<Value> {
        Ok(match value_type {
            0 => Value::U8(self.array::<1>("u8")?[0]),
            1 => Value::I8(i8::from_le_bytes(self.array("i8")?)),
            2 => Value::U16(u16::from_le_bytes(self.array("u16")?)),
            3 => Value::I16(i16::from_le_bytes(self.array("i16")?)),
            4 => Value::U32(self.u32("u32")?),
            5 => Value::I32(i32::from_le_bytes(self.array("i32")?)),
            6 => Value::F32(f32::from_le_bytes(self.array("f32")?)),
            7 => match self.array::<1>("bool")?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                other => bail!("bool must be 0 or 1, found {other}"),
            },
            8 => Value::String(self.str("string value")?.to_owned()),
            9 => {
                ensure!(
                    depth < MAX_ARRAY_DEPTH,
                    "arrays nested more than {MAX_ARRAY_DEPTH} deep"
                );
                let element_type = self.u32("array element type")?;
                let len = self.u64("array length")?;
                self.check_count(len, min_value_bytes(element_type)?, "array element")?;
                let mut preview = Vec::new();
                for index in 0..len {
                    let value = self
                        .value(element_type, depth + 1)
                        .with_context(|| format!("array element {index}"))?;
                    if preview.len() < ARRAY_PREVIEW {
                        preview.push(value);
                    }
                }
                Value::Array {
                    element_type,
                    len,
                    preview,
                }
            }
            10 => Value::U64(self.u64("u64")?),
            11 => Value::I64(i64::from_le_bytes(self.array("i64")?)),
            12 => Value::F64(f64::from_le_bytes(self.array("f64")?)),
            other => bail!("unknown metadata value type {other}"),
        })
    }
}

/// Smallest encoding of a value of this type, used to reject impossible counts.
fn min_value_bytes(value_type: u32) -> Result<usize> {
    Ok(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10..=12 => 8, // a string is at least its 8-byte length prefix
        9 => 12,          // nested array: element type + length
        other => bail!("unknown metadata value type {other}"),
    })
}

fn align_up(position: u64, alignment: u64) -> Option<u64> {
    position.checked_next_multiple_of(alignment)
}

/// Parse and validate a whole GGUF file. Like the safetensors parser, this
/// touches only the header; tensor bytes stay in the caller's mapping.
pub fn parse(bytes: &[u8]) -> Result<Gguf> {
    let mut r = Reader { bytes, pos: 0 };
    ensure!(
        r.array::<4>("magic")? == MAGIC,
        "not a GGUF file (bad magic)"
    );
    let version = r.u32("version")?;
    match version {
        2 | 3 => {}
        1 => bail!("GGUF v1 (2023) used 32-bit counts and is not supported; re-convert the model"),
        // Read little-endian, a big-endian 3 looks like 0x0300_0000.
        v if matches!(v.swap_bytes(), 1..=3) => bail!("big-endian GGUF files are not supported"),
        v => bail!("unsupported GGUF version {v}"),
    }
    let tensor_count = r.u64("tensor count")?;
    let metadata_count = r.u64("metadata count")?;
    // Minimum sizes: a key/value pair is 8 (key length) + 4 (type) + 1 byte; a
    // tensor info is 8 (name length) + 4 (n_dims) + 4 (type) + 8 (offset).
    r.check_count(metadata_count, 13, "metadata")?;
    r.check_count(tensor_count, 24, "tensor")?;

    let mut metadata = Vec::new();
    let mut keys = HashSet::new();
    for _ in 0..metadata_count {
        let key = r.str("metadata key")?.to_owned();
        let value_type = r.u32("metadata value type")?;
        let value = r
            .value(value_type, 0)
            .with_context(|| format!("metadata key {key:?}"))?;
        ensure!(keys.insert(key.clone()), "duplicate metadata key {key:?}");
        metadata.push((key, value));
    }

    let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some((_, Value::U32(a))) if a.is_power_of_two() => u64::from(*a),
        Some((_, Value::U32(a))) => {
            bail!("general.alignment must be a nonzero power of two, found {a}")
        }
        Some((_, other)) => bail!(
            "general.alignment must be a u32, found {}",
            other.type_name()
        ),
    };

    struct Info {
        name: String,
        dims: Vec<u64>,
        ggml_type: GgmlType,
        offset: u64,
    }
    let mut infos = Vec::new();
    let mut names = HashSet::new();
    for _ in 0..tensor_count {
        let name = r.str("tensor name")?.to_owned();
        ensure!(names.insert(name.clone()), "duplicate tensor name {name:?}");
        let n_dims = r.u32("dimension count")?;
        ensure!(
            n_dims <= MAX_DIMS,
            "tensor {name:?}: {n_dims} dimensions; ggml supports at most {MAX_DIMS}"
        );
        let dims = (0..n_dims)
            .map(|_| r.u64("tensor dimension"))
            .collect::<Result<Vec<_>>>()?;
        let type_id = r.u32("tensor type")?;
        let Some(ggml_type) = GgmlType::from_id(type_id) else {
            if let Some((_, old)) = REMOVED_GGML_TYPES.iter().find(|(id, _)| *id == type_id) {
                bail!(
                    "tensor {name:?}: ggml type {type_id} ({old}) was removed from GGUF; \
                     current llama.cpp repacks plain types at load time instead"
                );
            }
            bail!("tensor {name:?}: unknown ggml type id {type_id} (block size unknown)");
        };
        let offset = r.u64("tensor offset")?;
        infos.push(Info {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    let data_start = align_up(r.pos as u64, alignment)
        .and_then(|p| usize::try_from(p).ok())
        .context("tensor data start overflows")?;
    let mut tensors = Vec::with_capacity(infos.len());
    // llama.cpp's loader requires each tensor to start exactly where the previous
    // one's padded data ends, in header order. We enforce the same rule, which
    // also rules out overlaps and holes.
    let mut expected_offset = 0u64;
    for Info {
        name,
        dims,
        ggml_type,
        offset,
    } in infos
    {
        let elements = if dims.contains(&0) {
            0
        } else {
            dims.iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d))
                .with_context(|| format!("tensor {name:?}: element count overflow"))?
        };
        let row = dims.first().copied().unwrap_or(1);
        ensure!(
            row.is_multiple_of(ggml_type.block_elements),
            "tensor {name:?}: innermost dimension {row} is not a multiple of the {} block size {}",
            ggml_type.name,
            ggml_type.block_elements
        );
        let byte_len = (elements / ggml_type.block_elements)
            .checked_mul(ggml_type.block_bytes)
            .with_context(|| format!("tensor {name:?}: byte size overflow"))?;
        ensure!(
            offset == expected_offset,
            "tensor {name:?}: data offset {offset}, expected {expected_offset} \
             (tensors follow header order, each padded to {alignment} bytes)"
        );
        expected_offset = offset
            .checked_add(byte_len)
            .and_then(|end| align_up(end, alignment))
            .with_context(|| format!("tensor {name:?}: data offset overflow"))?;
        let file_offset = (data_start as u64)
            .checked_add(offset)
            .and_then(|o| usize::try_from(o).ok())
            .with_context(|| format!("tensor {name:?}: offset exceeds address space"))?;
        let byte_len = usize::try_from(byte_len).context("tensor size exceeds address space")?;
        ensure!(
            file_offset
                .checked_add(byte_len)
                .is_some_and(|end| end <= bytes.len()),
            "tensor {name:?}: data extends beyond end of file"
        );
        tensors.push(GgufTensor {
            name,
            ggml_type,
            dims,
            data_offset: offset,
            file_offset,
            byte_len,
            elements,
        });
    }
    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
    })
}

/// Lower a parsed file to format-independent views borrowing `bytes`. Shapes are
/// reversed into row-major order, so a GGUF row is a contiguous run of `dims[0]`
/// values, exactly like a safetensors row.
pub fn views<'a>(gguf: &Gguf, bytes: &'a [u8]) -> Vec<TensorView<'a>> {
    gguf.tensors
        .iter()
        .map(|t| TensorView {
            name: t.name.clone(),
            type_name: t.ggml_type.name.to_owned(),
            dtype: t.ggml_type.dtype(),
            shape: t.dims.iter().rev().copied().collect(),
            elements: t.elements,
            bytes: &bytes[t.file_offset..t.file_offset + t.byte_len],
        })
        .collect()
}
