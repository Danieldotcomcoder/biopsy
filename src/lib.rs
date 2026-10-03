//! biopsy: learn model file formats by inspecting their raw bytes.
//!
//! Reading order: `safetensors` and `gguf` (framing), `half` and `quant`
//! (encodings), `decode` and `kernels` (bytes to F32), then `stats`, `health`
//! and `diff` (what you can learn from the numbers).

pub mod decode;
pub mod diff;
pub mod fixtures;
pub mod gguf;
pub mod half;
pub mod health;
pub mod inspect;
pub mod kernels;
pub mod model;
pub mod quant;
pub mod safetensors;
pub mod stats;
pub mod tensor;
