//! Dump every tensor biopsy decodes, for checking it against an independent reader.
//!
//! `cargo run --release --example dump -- <model> <out_dir> [scalar|auto]`
//!
//! Writes `<index>.f32` (raw little-endian F32 values) per decodable tensor, plus
//! `manifest.json` with each tensor's name, type, shape, byte range and summary.
//! `tools/crosscheck.py` compares these with llama.cpp's gguf-py and with numpy.
use std::{fs::File, io::BufWriter, io::Write};

use anyhow::{Context, Result, ensure};
use biopsy::{decode, kernels::Backend, model, stats};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: dump <model> <out_dir> [scalar|auto]"
    );
    let backend = match args.get(2).map(String::as_str) {
        Some("scalar") => Backend::Scalar,
        None | Some("auto") => Backend::detect(),
        Some(other) => anyhow::bail!("unknown backend {other:?}"),
    };
    let file = File::open(&args[0]).with_context(|| format!("cannot open {}", args[0]))?;
    // SAFETY: as in the CLI, the file must stay unchanged while it is mapped.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let model = model::parse(&map)?;
    let out = std::path::Path::new(&args[1]);
    std::fs::create_dir_all(out)?;
    let summaries = stats::summarize_all(&model.tensors, backend);
    let base = map.as_ptr() as usize;
    let mut manifest = Vec::new();
    let mut buffer = Vec::new();
    for (i, (t, summary)) in model.tensors.iter().zip(&summaries).enumerate() {
        let mut entry = serde_json::json!({
            "index": i,
            "name": t.name,
            "type": t.type_name,
            "shape": t.shape,
            "elements": t.elements,
            "file_offset": t.bytes.as_ptr() as usize - base,
            "byte_len": t.bytes.len(),
            "decoded": t.dtype.is_some(),
        });
        if let (Some(dtype), Some(s)) = (t.dtype, summary) {
            let name = format!("{i}.f32");
            let mut writer = BufWriter::new(File::create(out.join(&name))?);
            let mut result = Ok(());
            decode::for_each_chunk(dtype, t.bytes, &mut buffer, backend, |values| {
                for v in values {
                    if result.is_ok() {
                        result = writer.write_all(&v.to_le_bytes());
                    }
                }
            });
            result?;
            writer.flush()?;
            entry["file"] = name.into();
            entry["summary"] = serde_json::json!({
                "count": s.count, "finite": s.finite, "nan": s.nan, "inf": s.inf,
                "zeros": s.zeros, "min": s.min, "max": s.max, "mean": s.mean, "std": s.std(),
            });
        }
        manifest.push(entry);
    }
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!(
        "dumped {} tensors from {} with {} kernels",
        model.tensors.len(),
        args[0],
        backend.name()
    );
    Ok(())
}
