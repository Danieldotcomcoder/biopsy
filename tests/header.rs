use biopsy::safetensors::parse_header;

fn file(json: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn parses_three_float_formats_and_metadata() {
    let json = r#"{"__metadata__":{"purpose":"test"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[2],"data_offsets":[16,20]},"c":{"dtype":"BF16","shape":[1],"data_offsets":[20,22]}}   "#;
    let header = parse_header(&file(json, &[0; 22])).unwrap();
    assert_eq!(header.metadata["purpose"], "test");
    assert_eq!(header.total_elements, 7);
    assert_eq!(header.tensors.len(), 3);
    assert_eq!(header.tensors[0].shape, [2, 2]);
    assert_eq!(header.tensors[0].byte_len, 16);
    assert_eq!(header.tensors[2].data_offset, 20);
    assert_eq!(header.tensors[2].file_offset, 8 + json.len() + 20);
}

#[test]
fn scalars_empty_tensors_and_unsorted_ranges() {
    let json = r#"{"z":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[0,18446744073709551615,2],"data_offsets":[4,4]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}"#;
    let header = parse_header(&file(json, &[0; 6])).unwrap();
    assert_eq!(header.total_elements, 2);
    assert_eq!(header.tensors[1].elements, 0);
    assert_eq!(header.tensors[2].elements, 1);
    assert_eq!(parse_header(&file("{}", &[])).unwrap().total_elements, 0);
}

#[test]
fn rejects_bad_framing() {
    for len in 0..8 {
        assert!(parse_header(&vec![0; len]).is_err());
    }
    assert!(
        parse_header(&100_000_001u64.to_le_bytes())
            .unwrap_err()
            .to_string()
            .contains("safety limit")
    );
    assert!(
        parse_header(&100u64.to_le_bytes())
            .unwrap_err()
            .to_string()
            .contains("truncated")
    );
    assert!(parse_header(&file(" {}", &[])).is_err());
    assert!(parse_header(&file("{broken}", &[])).is_err());
    let mut invalid_utf8 = file("{}", &[]);
    invalid_utf8[9] = 0xff;
    assert!(parse_header(&invalid_utf8).is_err());
    let mut big_endian = file("{}", &[]);
    big_endian[..8].copy_from_slice(&2u64.to_be_bytes());
    assert!(parse_header(&big_endian).is_err());
}

#[test]
fn rejects_duplicate_keys_and_invalid_fields() {
    for json in [
        r#"{"x":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"F32","dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
        r#"{"__metadata__":{"a":"one","a":"two"}}"#,
        r#"{"__metadata__":{"a":42}}"#,
        r#"{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1.5],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"F32","shape":[0],"data_offsets":[0]}}"#,
        r#"{"x":{"shape":[0],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"Q4_0","shape":[0],"data_offsets":[0,0]}}"#,
    ] {
        assert!(parse_header(&file(json, &[])).is_err(), "accepted {json}");
    }
}

#[test]
fn rejects_bad_ranges_and_sizes() {
    for (json, payload_len) in [
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            4,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            4,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            4,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            8,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            4,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            5,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
            0,
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}"#,
            0,
        ),
    ] {
        assert!(
            parse_header(&file(json, &vec![0; payload_len])).is_err(),
            "accepted {json}"
        );
    }
}

#[test]
fn every_truncation_of_a_valid_file_is_rejected() {
    let bytes = file(
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0; 8],
    );
    for end in 0..bytes.len() {
        assert!(
            parse_header(&bytes[..end]).is_err(),
            "accepted length {end}"
        );
    }
    assert!(parse_header(&bytes).is_ok());
}

#[test]
fn cli_maps_file_and_reports_errors_without_panicking() {
    use std::{io::Write, process::Command};
    let mut fixture = tempfile::NamedTempFile::new().unwrap();
    fixture
        .write_all(&file(
            r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0; 8],
        ))
        .unwrap();
    fixture.flush().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_biopsy"))
        .arg("inspect")
        .arg(fixture.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("dtype=F32 shape=[2]"));
    assert!(stdout.contains("Total parameters (stored tensor elements): 2"));
    fixture.as_file().set_len(0).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_biopsy"))
        .arg("inspect")
        .arg(fixture.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("too short"));
}
