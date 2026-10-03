//! `biopsy inspect`: describe a file's header without reading any weight values.

use std::{collections::BTreeMap, fmt::Write};

use anyhow::{Context, Result};

use crate::{
    gguf::{self, Gguf},
    model::Format,
    safetensors::{self, Header},
};

/// Parse `bytes` in whichever format they are and describe the header.
pub fn render(path: &str, bytes: &[u8]) -> Result<String> {
    Ok(match Format::detect(bytes) {
        Format::Safetensors => render_safetensors(
            path,
            &safetensors::parse_header(bytes)
                .with_context(|| format!("cannot inspect safetensors file {path}"))?,
        )?,
        Format::Gguf => render_gguf(
            path,
            &gguf::parse(bytes).with_context(|| format!("cannot inspect GGUF file {path}"))?,
        ),
    })
}

/// Per-type tensor count and element total, ordered by type name.
fn type_totals<'a>(entries: impl Iterator<Item = (&'a str, u64)>) -> String {
    let mut totals: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for (ty, elements) in entries {
        let total = totals.entry(ty).or_default();
        total.0 += 1;
        total.1 += elements;
    }
    let parts: Vec<String> = totals
        .iter()
        .map(|(ty, (n, e))| format!("{ty} x{n} ({e} elements)"))
        .collect();
    parts.join(", ")
}

pub fn render_safetensors(path: &str, header: &Header) -> Result<String> {
    let mut out = String::new();
    writeln!(out, "File: {path}")?;
    writeln!(out, "Format: safetensors")?;
    writeln!(
        out,
        "Header: {} bytes; tensor data begins at byte {}",
        header.header_len, header.data_start
    )?;
    writeln!(
        out,
        "Metadata: {}",
        serde_json::to_string(&header.metadata)?
    )?;
    for tensor in &header.tensors {
        writeln!(
            out,
            "{:?}: dtype={} shape={:?} data_offset={} file_offset={} size={} bytes elements={}",
            tensor.name,
            tensor.dtype,
            tensor.shape,
            tensor.data_offset,
            tensor.file_offset,
            tensor.byte_len,
            tensor.elements
        )?;
    }
    writeln!(out, "Tensors: {}", header.tensors.len())?;
    if !header.tensors.is_empty() {
        let types = header
            .tensors
            .iter()
            .map(|t| (t.dtype.as_str(), t.elements));
        writeln!(out, "Types: {}", type_totals(types))?;
    }
    writeln!(
        out,
        "Total parameters (stored tensor elements): {}",
        header.total_elements
    )?;
    Ok(out)
}

pub fn render_gguf(path: &str, gguf: &Gguf) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "File: {path}");
    let _ = writeln!(
        out,
        "Format: GGUF v{} (little-endian); alignment {} bytes; tensor data begins at byte {}",
        gguf.version, gguf.alignment, gguf.data_start
    );
    let _ = writeln!(out, "Metadata ({} entries):", gguf.metadata.len());
    for (key, value) in &gguf.metadata {
        let _ = writeln!(out, "  {key}: {} = {}", value.type_name(), value.describe());
    }
    // `dims` is how the file stores them (innermost first); `shape` is the
    // row-major order every other biopsy command and PyTorch use.
    for t in &gguf.tensors {
        let shape: Vec<u64> = t.dims.iter().rev().copied().collect();
        let _ = writeln!(
            out,
            "{:?}: type={} dims={:?} shape={:?} data_offset={} file_offset={} size={} bytes elements={}",
            t.name,
            t.ggml_type.name,
            t.dims,
            shape,
            t.data_offset,
            t.file_offset,
            t.byte_len,
            t.elements
        );
    }
    let _ = writeln!(out, "Tensors: {}", gguf.tensors.len());
    if !gguf.tensors.is_empty() {
        let types = gguf.tensors.iter().map(|t| (t.ggml_type.name, t.elements));
        let _ = writeln!(out, "Types: {}", type_totals(types));
    }
    let total: u64 = gguf.tensors.iter().map(|t| t.elements).sum();
    let _ = writeln!(out, "Total parameters (stored tensor elements): {total}");
    out
}
