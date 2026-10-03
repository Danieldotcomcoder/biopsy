//! Writers for small synthetic files. They exist so tests, examples and
//! `cargo verify` never need to download a model; they are not general-purpose
//! serializers. Writing a format is also the quickest way to check you understand it.

use std::collections::BTreeMap;

use crate::{
    gguf::{self, GgmlType, Value},
    half, quant,
};

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f32_to_f16(v).to_le_bytes())
        .collect()
}

pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f32_to_bf16(v).to_le_bytes())
        .collect()
}

/// A tensor to write: name, file-level dtype name, row-major shape, raw bytes.
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

impl Tensor {
    pub fn new(name: &str, dtype: &str, shape: &[u64], bytes: Vec<u8>) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            bytes,
        }
    }

    /// Encode F32 values as F32, F16 or BF16 for safetensors, or as any decodable
    /// ggml type (including Q8_0 and Q4_0) for GGUF.
    pub fn encode(name: &str, dtype: &str, shape: &[u64], values: &[f32]) -> Tensor {
        let bytes = match dtype {
            "F32" => f32_bytes(values),
            "F16" => f16_bytes(values),
            "BF16" => bf16_bytes(values),
            "Q8_0" => quant::quantize_q8_0(values),
            "Q4_0" => quant::quantize_q4_0(values),
            other => panic!("fixtures cannot encode {other}"),
        };
        Tensor::new(name, dtype, shape, bytes)
    }
}

/// Lay tensors out back to back in the given order. The JSON header is padded
/// with spaces so the payload starts on an 8-byte boundary.
pub fn safetensors(metadata: &[(&str, &str)], tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let map: BTreeMap<_, _> = metadata.iter().copied().collect();
        header.insert("__metadata__".into(), serde_json::json!(map));
    }
    let mut offset = 0;
    for t in tensors {
        let end = offset + t.bytes.len();
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut json = serde_json::Value::Object(header).to_string();
    while !(8 + json.len()).is_multiple_of(8) {
        json.push(' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    for t in tensors {
        bytes.extend_from_slice(&t.bytes);
    }
    bytes
}

fn gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn value_type_id(value: &Value) -> u32 {
    match value {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array { .. } => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// Write a value's payload (no type id). Arrays are written from `preview`,
/// which for fixtures holds every element, so `len` must equal its length.
fn gguf_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U8(v) => out.push(*v),
        Value::I8(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => out.push(u8::from(*v)),
        Value::String(s) => gguf_string(out, s),
        Value::Array {
            element_type,
            len,
            preview,
        } => {
            assert_eq!(
                *len,
                preview.len() as u64,
                "fixture arrays must be complete"
            );
            out.extend_from_slice(&element_type.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
            for item in preview {
                gguf_value(out, item);
            }
        }
    }
}

/// A complete array value for fixtures.
pub fn array(element_type: u32, items: Vec<Value>) -> Value {
    Value::Array {
        element_type,
        len: items.len() as u64,
        preview: items,
    }
}

/// Write a GGUF v3 file. `tensors` use row-major shapes; they are reversed into
/// ggml's innermost-first order on the way out. Alignment comes from a valid
/// `general.alignment` entry in `metadata`, else the default 32. An invalid entry
/// is still written, so tests can check that the reader rejects it.
pub fn gguf(metadata: &[(&str, Value)], tensors: &[Tensor]) -> Vec<u8> {
    let alignment = metadata
        .iter()
        .find_map(|(k, v)| match (k, v) {
            (&"general.alignment", Value::U32(a)) if a.is_power_of_two() => Some(u64::from(*a)),
            _ => None,
        })
        .unwrap_or(gguf::DEFAULT_ALIGNMENT) as usize;
    let mut out = gguf::MAGIC.to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        gguf_string(&mut out, key);
        out.extend_from_slice(&value_type_id(value).to_le_bytes());
        gguf_value(&mut out, value);
    }
    let mut offset = 0;
    for t in tensors {
        let ty = GgmlType::from_name(&t.dtype).expect("known ggml type name");
        gguf_string(&mut out, &t.name);
        out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for dim in t.shape.iter().rev() {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&ty.id.to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        offset = (offset + t.bytes.len()).next_multiple_of(alignment);
    }
    out.resize(out.len().next_multiple_of(alignment), 0);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
        out.resize(out.len().next_multiple_of(alignment), 0);
    }
    out
}

/// Deterministic pseudo-random numbers (SplitMix64), so fixtures are identical
/// on every machine without a `rand` dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately normal (Box-Muller), scaled by `std`.
    pub fn normal(&mut self, std: f32) -> f32 {
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32 * std
    }

    pub fn normals(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal(std)).collect()
    }
}

/// The demo files `cargo run --example make_tiny` writes and `cargo verify`
/// exercises: (file name, bytes).
pub fn demo_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("tiny.safetensors", tiny()),
        ("encoder.safetensors", encoder(false)),
        ("encoder_finetuned.safetensors", encoder(true)),
        ("sick.safetensors", sick()),
        ("tiny_q8_0.gguf", tiny_gguf("Q8_0")),
        ("tiny_q4_0.gguf", tiny_gguf("Q4_0")),
    ]
}

/// The milestone 1 fixture: two F32 tensors, eight elements.
pub fn tiny() -> Vec<u8> {
    safetensors(
        &[("description", "milestone 1 fixture")],
        &[
            Tensor::encode("weight", "F32", &[2, 3], &[1.0, -2.0, 0.0, 4.0, 5.0, 6.0]),
            Tensor::encode("bias", "F32", &[2], &[0.1, -0.1]),
        ],
    )
}

/// A small encoder in three dtypes. The fine-tuned variant perturbs one layer,
/// keeps the embeddings bit-identical, reshapes the pooler and adds a classifier,
/// so a diff shows every category: identical, changed, mismatched, unmatched.
pub fn encoder(finetuned: bool) -> Vec<u8> {
    let mut rng = Rng::new(7);
    let embeddings = rng.normals(16 * 32, 0.02);
    let mut dense = rng.normals(32 * 32, 0.05);
    let mut dense_bias = rng.normals(32, 0.01);
    if finetuned {
        let mut tweak = Rng::new(99);
        for v in dense.iter_mut().chain(&mut dense_bias) {
            *v += tweak.normal(0.002);
        }
    }
    let mut tensors = vec![
        Tensor::encode("embeddings.weight", "F32", &[16, 32], &embeddings),
        Tensor::encode("layer.0.dense.weight", "BF16", &[32, 32], &dense),
        Tensor::encode("layer.0.dense.bias", "F16", &[32], &dense_bias),
    ];
    if finetuned {
        tensors.push(Tensor::encode(
            "pooler.weight",
            "F32",
            &[16, 32],
            &rng.normals(16 * 32, 0.05),
        ));
        tensors.push(Tensor::encode(
            "classifier.weight",
            "F32",
            &[2, 32],
            &rng.normals(64, 0.05),
        ));
    } else {
        tensors.push(Tensor::encode(
            "pooler.weight",
            "F32",
            &[32, 32],
            &rng.normals(32 * 32, 0.05),
        ));
        tensors.push(Tensor::encode(
            "lm_head.weight",
            "F32",
            &[16, 32],
            &rng.normals(16 * 32, 0.05),
        ));
    }
    let purpose = if finetuned { "fine-tuned" } else { "base" };
    safetensors(&[("purpose", purpose)], &tensors)
}

/// One tensor per health problem, plus one healthy tensor that must stay quiet.
pub fn sick() -> Vec<u8> {
    let mut rng = Rng::new(3);
    let mut non_finite = rng.normals(4 * 8, 0.1);
    non_finite[5] = f32::NAN;
    non_finite[9] = f32::INFINITY;
    let mut dead_rows = rng.normals(8 * 16, 0.1);
    for row in [2, 5] {
        dead_rows[row * 16..(row + 1) * 16].fill(0.0);
    }
    let mut outlier_rows = rng.normals(16 * 16, 0.1);
    for v in &mut outlier_rows[3 * 16..4 * 16] {
        *v *= 100.0;
    }
    let mut sparse = rng.normals(8 * 16, 0.1);
    for (i, v) in sparse.iter_mut().enumerate() {
        if i % 4 != 0 {
            *v = 0.0;
        }
    }
    let mut huge = rng.normals(16, 1.0);
    huge[0] = 1.0e5;
    safetensors(
        &[("purpose", "health check demo")],
        &[
            Tensor::encode("healthy.weight", "F32", &[16, 16], &rng.normals(256, 0.1)),
            Tensor::encode("non_finite.weight", "F32", &[4, 8], &non_finite),
            Tensor::encode("dead_rows.weight", "F32", &[8, 16], &dead_rows),
            Tensor::encode("outlier_rows.weight", "F32", &[16, 16], &outlier_rows),
            Tensor::encode("zeros.bias", "F32", &[8], &[0.0; 8]),
            Tensor::encode("constant.weight", "BF16", &[8], &[1.0; 8]),
            Tensor::encode("sparse.weight", "F16", &[8, 16], &sparse),
            Tensor::encode("huge.weight", "F32", &[4, 4], &huge),
        ],
    )
}

/// The same small weights quantized two ways. Norms stay F32, as llama.cpp keeps
/// them; one Q4_K tensor (all-zero blocks) shows how undecoded types are reported.
pub fn tiny_gguf(quant_type: &str) -> Vec<u8> {
    let mut rng = Rng::new(11);
    let tokens = ["<pad>", "<s>", "</s>", "the", "a", "model", "file", "bytes"];
    let mut metadata = vec![
        ("general.architecture", Value::String("tiny".into())),
        (
            "general.name",
            Value::String(format!("biopsy demo {quant_type}")),
        ),
        ("tiny.embedding_length", Value::U32(64)),
        ("tiny.rope.freq_base", Value::F32(10000.0)),
        ("tiny.use_parallel_residual", Value::Bool(false)),
        (
            "tokenizer.ggml.tokens",
            array(
                8,
                tokens.iter().map(|t| Value::String((*t).into())).collect(),
            ),
        ),
        (
            "tokenizer.ggml.scores",
            array(6, (0..8).map(|i| Value::F32(0.0 - i as f32)).collect()),
        ),
    ];
    if quant_type == "Q4_0" {
        // Show that alignment is configurable: every tensor starts on 64 bytes.
        metadata.push(("general.alignment", Value::U32(64)));
    }
    let q4_k = GgmlType::from_name("Q4_K").expect("Q4_K is in the table");
    gguf(
        &metadata,
        &[
            Tensor::encode(
                "token_embd.weight",
                quant_type,
                &[8, 64],
                &rng.normals(8 * 64, 0.02),
            ),
            Tensor::encode(
                "blk.0.attn_v.weight",
                "F16",
                &[64, 64],
                &rng.normals(64 * 64, 0.05),
            ),
            Tensor::encode(
                "blk.0.ffn_up.weight",
                quant_type,
                &[128, 64],
                &rng.normals(128 * 64, 0.05),
            ),
            Tensor::new(
                "blk.0.ffn_down.weight",
                "Q4_K",
                &[64, 256],
                vec![0; 64 * q4_k.block_bytes as usize],
            ),
            Tensor::encode(
                "output_norm.weight",
                "F32",
                &[64],
                &rng.normals(64, 0.05)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect::<Vec<_>>(),
            ),
        ],
    )
}
