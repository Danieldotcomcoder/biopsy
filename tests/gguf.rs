use biopsy::{
    decode,
    fixtures::{self, Tensor, array},
    gguf::{self, Value},
    kernels::Backend,
    tensor::DType,
};

// ---- raw builders for malformed files the fixture writer refuses to produce ----

fn header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&tensors.to_le_bytes());
    out.extend_from_slice(&kvs.to_le_bytes());
    out
}

fn string(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}

/// A file whose only metadata entry is `key` of `value_type` with raw `payload`.
fn one_kv(value_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = header(3, 0, 1);
    string(&mut out, b"key");
    out.extend_from_slice(&value_type.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn tensor_info(out: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64) {
    string(out, name.as_bytes());
    out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
}

/// A file with one tensor info and enough zero bytes after it for any small tensor.
fn one_tensor(dims: &[u64], ty: u32, offset: u64) -> Vec<u8> {
    let mut out = header(3, 1, 0);
    tensor_info(&mut out, "t", dims, ty, offset);
    out.resize(out.len().next_multiple_of(32) + 4096, 0);
    out
}

fn error(bytes: &[u8]) -> String {
    format!("{:#}", gguf::parse(bytes).unwrap_err())
}

fn assert_error(bytes: &[u8], expected: &str) {
    let message = error(bytes);
    assert!(
        message.contains(expected),
        "expected {expected:?} in {message:?}"
    );
}

// ---- valid files ----

fn sample() -> Vec<u8> {
    let nested = array(9, vec![array(4, vec![Value::U32(1)]), array(4, vec![])]);
    fixtures::gguf(
        &[
            ("u8", Value::U8(200)),
            ("i8", Value::I8(-5)),
            ("u16", Value::U16(60000)),
            ("i16", Value::I16(-30000)),
            ("u32", Value::U32(7)),
            ("i32", Value::I32(-7)),
            ("u64", Value::U64(u64::MAX)),
            ("i64", Value::I64(i64::MIN)),
            ("f32", Value::F32(0.5)),
            ("f64", Value::F64(-0.25)),
            ("bool", Value::Bool(true)),
            ("text", Value::String("héllo".into())),
            ("strings", array(8, vec![Value::String("a".into()); 20])),
            ("nested", nested),
        ],
        &[
            Tensor::encode("matrix", "F32", &[2, 32], &[1.5; 64]),
            Tensor::encode("q8", "Q8_0", &[3, 64], &[0.9921875; 192]), // 127 x 2^-7
            Tensor::encode("q4", "Q4_0", &[32], &[-1.0; 32]),
        ],
    )
}

#[test]
fn parses_every_value_type_and_lays_out_tensors() {
    let bytes = sample();
    let file = gguf::parse(&bytes).unwrap();
    assert_eq!(file.version, 3);
    assert_eq!(file.alignment, 32);
    assert_eq!(file.metadata.len(), 14);
    assert_eq!(file.get("u8"), Some(&Value::U8(200)));
    assert_eq!(file.get("i64"), Some(&Value::I64(i64::MIN)));
    assert_eq!(file.get("text"), Some(&Value::String("héllo".into())));
    let Some(Value::Array { len, preview, .. }) = file.get("strings") else {
        panic!("strings is not an array")
    };
    assert_eq!(*len, 20);
    assert_eq!(preview.len(), 8); // only a preview is kept
    assert_eq!(
        file.get("strings").unwrap().describe(),
        r#"[string x 20] ["a", "a", "a", "a", "a", "a", "a", "a", ...]"#
    );
    assert_eq!(
        file.get("nested").unwrap().describe(),
        "[array x 2] [[u32 x 1] [1], [u32 x 0] []]"
    );

    assert_eq!(file.data_start % 32, 0);
    let names: Vec<_> = file.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["matrix", "q8", "q4"]); // file order is kept
    let matrix = &file.tensors[0];
    assert_eq!(matrix.dims, [32, 2]); // innermost first on disk
    assert_eq!((matrix.data_offset, matrix.byte_len), (0, 256));
    let q8 = &file.tensors[1];
    assert_eq!(q8.dims, [64, 3]);
    assert_eq!((q8.data_offset, q8.byte_len), (256, 6 * 34));
    let q4 = &file.tensors[2];
    assert_eq!(q4.data_offset, (256 + 204usize).next_multiple_of(32) as u64);
    assert_eq!(q4.byte_len, 18);

    let views = gguf::views(&file, &bytes);
    assert_eq!(views[0].shape, [2, 32]); // row-major again
    assert_eq!(views[1].dtype, Some(DType::Q8_0));
    let mut buffer = Vec::new();
    decode::decode_into(DType::Q8_0, views[1].bytes, &mut buffer, Backend::Scalar);
    assert!(buffer.iter().all(|&v| v == 0.9921875));
    decode::decode_into(DType::Q4_0, views[2].bytes, &mut buffer, Backend::Scalar);
    assert!(buffer.iter().all(|&v| v == -1.0));
}

#[test]
fn custom_alignment_moves_every_offset() {
    let bytes = fixtures::gguf(
        &[("general.alignment", Value::U32(64))],
        &[
            Tensor::encode("a", "F32", &[3], &[1.0; 3]),
            Tensor::encode("b", "F16", &[5], &[2.0; 5]),
        ],
    );
    let file = gguf::parse(&bytes).unwrap();
    assert_eq!(file.alignment, 64);
    assert_eq!(file.data_start % 64, 0);
    assert_eq!(file.tensors[1].data_offset, 64);
}

#[test]
fn metadata_only_files_parse() {
    let bytes = fixtures::gguf(&[("general.name", Value::String("vocab".into()))], &[]);
    let file = gguf::parse(&bytes).unwrap();
    assert!(file.tensors.is_empty());
}

#[test]
fn every_truncation_inside_the_tensor_data_is_rejected() {
    let bytes = sample();
    let file = gguf::parse(&bytes).unwrap();
    let last = file.tensors.last().unwrap();
    let data_end = last.file_offset + last.byte_len;
    for end in 0..data_end {
        assert!(gguf::parse(&bytes[..end]).is_err(), "accepted length {end}");
    }
    // Padding after the last tensor is optional, as in llama.cpp.
    for end in data_end..=bytes.len() {
        assert!(gguf::parse(&bytes[..end]).is_ok(), "rejected length {end}");
    }
}

// ---- framing ----

#[test]
fn rejects_wrong_magic_and_versions() {
    let mut bytes = sample();
    bytes[0] = b'g';
    assert_error(&bytes, "bad magic");
    assert_error(&header(1, 0, 0), "v1");
    let mut big_endian = header(3, 0, 0);
    big_endian[4..8].copy_from_slice(&3u32.to_be_bytes());
    assert_error(&big_endian, "big-endian");
    assert_error(&header(4, 0, 0), "unsupported GGUF version 4");
    assert_error(&b"GGUF\x03\x00"[..], "truncated");
}

#[test]
fn rejects_counts_that_cannot_fit() {
    assert_error(&header(3, 0, u64::MAX), "metadata count");
    assert_error(&header(3, u64::MAX, 0), "tensor count");
    let mut huge_array = 4u32.to_le_bytes().to_vec();
    huge_array.extend_from_slice(&u64::MAX.to_le_bytes());
    assert_error(&one_kv(9, &huge_array), "cannot fit");
}

// ---- metadata ----

#[test]
fn rejects_malformed_metadata_values() {
    assert_error(&one_kv(7, &[2]), "bool must be 0 or 1");
    assert_error(&one_kv(13, &[0; 8]), "unknown metadata value type 13");
    let mut bad_utf8 = Vec::new();
    string(&mut bad_utf8, &[0x66, 0xff]);
    assert_error(&one_kv(8, &bad_utf8), "UTF-8");
    let mut long_string = Vec::new();
    long_string.extend_from_slice(&1000u64.to_le_bytes());
    long_string.extend_from_slice(b"short");
    assert_error(&one_kv(8, &long_string), "truncated");
}

#[test]
fn rejects_arrays_nested_too_deep() {
    // Eight levels of array-of-array, then an empty u8 array: nine arrays deep.
    let mut payload = Vec::new();
    for _ in 0..8 {
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
    }
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    // `one_kv` adds the outermost array type; its payload starts with the element type.
    assert_error(&one_kv(9, &payload), "nested more than 8 deep");
}

#[test]
fn rejects_duplicate_keys_and_names() {
    let duplicate_keys = fixtures::gguf(&[("k", Value::U8(1)), ("k", Value::U8(2))], &[]);
    assert_error(&duplicate_keys, "duplicate metadata key \"k\"");
    let duplicate_names = fixtures::gguf(
        &[],
        &[
            Tensor::encode("t", "F32", &[1], &[1.0]),
            Tensor::encode("t", "F32", &[1], &[1.0]),
        ],
    );
    assert_error(&duplicate_names, "duplicate tensor name \"t\"");
}

#[test]
fn rejects_bad_alignment_values() {
    for (value, expected) in [
        (Value::U32(0), "power of two"),
        (Value::U32(48), "power of two"),
        (Value::U64(32), "must be a u32"),
    ] {
        let bytes = fixtures::gguf(&[("general.alignment", value)], &[]);
        assert_error(&bytes, expected);
    }
}

// ---- tensor infos ----

#[test]
fn rejects_bad_tensor_infos() {
    assert_error(&one_tensor(&[1, 1, 1, 1, 1], 0, 0), "at most 4");
    assert_error(&one_tensor(&[32], 31, 0), "Q4_0_4_4) was removed");
    assert_error(&one_tensor(&[32], 99, 0), "unknown ggml type id 99");
    assert_error(
        &one_tensor(&[33, 2], 8, 0),
        "not a multiple of the Q8_0 block size 32",
    );
    assert_error(&one_tensor(&[u64::MAX, 2], 0, 0), "element count overflow");
    assert_error(&one_tensor(&[4], 0, 32), "data offset 32, expected 0");
    assert_error(&one_tensor(&[1 << 40], 0, 0), "beyond end of file");
    assert!(gguf::parse(&one_tensor(&[0, 7], 0, 0)).is_ok()); // empty tensors are fine
}

#[test]
fn rejects_holes_between_tensors() {
    let mut bytes = header(3, 2, 0);
    tensor_info(&mut bytes, "a", &[4], 0, 0);
    tensor_info(&mut bytes, "b", &[4], 0, 64); // should be 32: 16 bytes padded to 32
    bytes.resize(bytes.len().next_multiple_of(32) + 256, 0);
    assert_error(&bytes, "\"b\": data offset 64, expected 32");
}

#[test]
fn ggml_type_table_matches_known_block_sizes() {
    for (name, elements, bytes) in [
        ("Q4_0", 32, 18),
        ("Q8_0", 32, 34),
        ("Q4_K", 256, 144),
        ("Q6_K", 256, 210),
        ("BF16", 1, 2),
    ] {
        let ty = gguf::GgmlType::from_name(name).unwrap();
        assert_eq!(
            (ty.block_elements, ty.block_bytes),
            (elements, bytes),
            "{name}"
        );
    }
    assert_eq!(
        gguf::GgmlType::from_id(8).unwrap().dtype(),
        Some(DType::Q8_0)
    );
    assert_eq!(gguf::GgmlType::from_id(12).unwrap().dtype(), None);
}
