//! Hand-written safetensors framing and validation; serde only handles JSON syntax.
use std::{collections::BTreeMap, fmt, marker::PhantomData};

use anyhow::{Context, Result, bail, ensure};
use serde::{
    Deserialize, Deserializer,
    de::{Error, MapAccess, Visitor},
};

use crate::tensor::{DType, TensorView};

// A policy limit prevents a corrupt length prefix from requesting enormous JSON work.
const MAX_HEADER_BYTES: usize = 100_000_000;

#[derive(Debug)]
pub struct Header {
    pub header_len: usize,
    pub data_start: usize,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<TensorInfo>,
    pub total_elements: u64,
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offset: usize,
    pub file_offset: usize,
    pub byte_len: usize,
    pub elements: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

// Untagged means JSON has no explicit enum tag: its fields identify the variant.
#[derive(Deserialize)]
#[serde(untagged)]
enum Entry {
    Tensor(RawTensor),
    Metadata(UniqueMap<String>),
}

// serde_json's ordinary maps silently overwrite duplicate keys. The format forbids
// them, so this small visitor rejects duplicates before any value is lost.
#[derive(Debug)]
struct UniqueMap<T>(BTreeMap<String, T>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct MapVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for MapVisitor<T> {
            type Value = UniqueMap<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object with unique keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    if values.contains_key(&key) {
                        return Err(M::Error::custom(format!("duplicate key {key:?}")));
                    }
                    values.insert(key, value);
                }
                Ok(UniqueMap(values))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

fn bytes_per_element(dtype: &str) -> Result<u64> {
    // Inspection validates the sizes of every byte-sized dtype, including integer
    // buffers and FP8. Numeric commands decode only F32, F16 and BF16 (see `views`).
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Ok(1),
        "U16" | "I16" | "F16" | "BF16" => Ok(2),
        "U32" | "I32" | "F32" => Ok(4),
        "U64" | "I64" | "F64" => Ok(8),
        _ => bail!("unsupported dtype {dtype:?} (sub-byte types are not handled)"),
    }
}

/// Lower a parsed header to format-independent views borrowing `bytes`, the same
/// slice `parse_header` validated. Tensors stay in the header's name order.
pub fn views<'a>(header: &Header, bytes: &'a [u8]) -> Vec<TensorView<'a>> {
    header
        .tensors
        .iter()
        .map(|t| TensorView {
            name: t.name.clone(),
            type_name: t.dtype.clone(),
            dtype: match t.dtype.as_str() {
                "F32" => Some(DType::F32),
                "F16" => Some(DType::F16),
                "BF16" => Some(DType::BF16),
                _ => None,
            },
            shape: t.shape.clone(),
            elements: t.elements,
            bytes: &bytes[t.file_offset..t.file_offset + t.byte_len],
        })
        .collect()
}

/// Parse an entire file's byte slice, touching only its length prefix and JSON.
/// Tensor data stays borrowed by the caller's mapping; this function owns metadata only.
pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .context("file is shorter than the 8-byte length prefix")?
        .try_into()
        .context("invalid length prefix")?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .context("header length exceeds address space")?;
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "header exceeds the 100 MB safety limit"
    );
    let data_start = 8usize
        .checked_add(header_len)
        .context("header length overflow")?;
    let json = bytes.get(8..data_start).context("truncated JSON header")?;
    ensure!(json.first() == Some(&b'{'), "header must begin with '{{'");
    let entries: UniqueMap<Entry> =
        serde_json::from_slice(json).context("invalid header JSON or tensor fields")?;
    let data_len = bytes.len() - data_start;
    let mut header = Header {
        header_len,
        data_start,
        metadata: BTreeMap::new(),
        tensors: Vec::new(),
        total_elements: 0,
    };
    for (name, entry) in entries.0 {
        if name == "__metadata__" {
            match entry {
                Entry::Metadata(metadata) => header.metadata = metadata.0,
                _ => bail!("__metadata__ must be a string-to-string object"),
            }
            continue;
        }
        let Entry::Tensor(raw) = entry else {
            bail!("tensor {name:?}: missing dtype, shape, or data_offsets")
        };
        let width = bytes_per_element(&raw.dtype).with_context(|| format!("tensor {name:?}"))?;
        // [] is a scalar (one element); any zero dimension makes an empty tensor.
        let elements = if raw.shape.contains(&0) {
            0
        } else {
            raw.shape
                .iter()
                .try_fold(1u64, |n, &dim| n.checked_mul(dim))
                .with_context(|| format!("tensor {name:?}: shape product overflow"))?
        };
        let expected = elements
            .checked_mul(width)
            .with_context(|| format!("tensor {name:?}: byte size overflow"))?;
        let [begin, end] = raw.data_offsets;
        ensure!(begin <= end, "tensor {name:?}: reversed offsets");
        ensure!(
            end - begin == expected,
            "tensor {name:?}: shape/dtype require {expected} bytes, offsets describe {}",
            end - begin
        );
        let begin = usize::try_from(begin).context("tensor offset exceeds address space")?;
        let end = usize::try_from(end).context("tensor end exceeds address space")?;
        ensure!(
            end <= data_len,
            "tensor {name:?}: data extends beyond end of file"
        );
        header.total_elements = header
            .total_elements
            .checked_add(elements)
            .context("total element count overflow")?;
        header.tensors.push(TensorInfo {
            name,
            dtype: raw.dtype,
            shape: raw.shape,
            data_offset: begin,
            file_offset: data_start + begin,
            byte_len: end - begin,
            elements,
        });
    }
    // JSON key order is irrelevant. Sort by physical range, placing empty tensors
    // before a nonempty tensor at the same offset, then check full data coverage.
    let mut ranges: Vec<_> = header.tensors.iter().collect();
    ranges.sort_by_key(|t| (t.data_offset, t.byte_len));
    let mut cursor = 0;
    for tensor in ranges {
        ensure!(
            tensor.data_offset == cursor,
            "tensor {:?}: gap or overlap at data offset {} (expected {cursor})",
            tensor.name,
            tensor.data_offset
        );
        cursor += tensor.byte_len;
    }
    ensure!(
        cursor == data_len,
        "unclaimed trailing tensor data: {} bytes",
        data_len - cursor
    );
    Ok(header)
}
