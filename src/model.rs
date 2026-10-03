//! Format detection: one entry point that turns file bytes into tensor views.

use anyhow::Result;

use crate::{gguf, safetensors, tensor::TensorView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
}

impl Format {
    /// GGUF announces itself with a magic number. Safetensors has none: its first
    /// eight bytes are a length, so anything that is not GGUF is tried as safetensors.
    pub fn detect(bytes: &[u8]) -> Format {
        if bytes.starts_with(&gguf::MAGIC) {
            Format::Gguf
        } else {
            Format::Safetensors
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Safetensors => "safetensors",
            Format::Gguf => "GGUF",
        }
    }
}

/// A parsed file's tensors, borrowing the caller's bytes (normally a mapping).
pub struct Model<'a> {
    pub format: Format,
    pub tensors: Vec<TensorView<'a>>,
}

pub fn parse(bytes: &[u8]) -> Result<Model<'_>> {
    let format = Format::detect(bytes);
    let tensors = match format {
        Format::Safetensors => safetensors::views(&safetensors::parse_header(bytes)?, bytes),
        Format::Gguf => gguf::views(&gguf::parse(bytes)?, bytes),
    };
    Ok(Model { format, tensors })
}
