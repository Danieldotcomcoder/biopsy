# biopsy

**Inspect neural network weight files from their raw bytes.**

biopsy is a learning-first Rust command-line tool for safetensors and GGUF
files. It parses both formats with hand-written code (no safetensors or GGUF
crate), decodes F32, F16, BF16, Q8_0 and Q4_0 weights, and reports statistics,
health problems and differences between two models.

It is also a tutorial. The code was built in six steps, from reading an 8-byte
length prefix to AVX2 kernels, and [How it works](#how-it-works) walks through
them in order.

```text
$ biopsy health models/sick.safetensors
File: models/sick.safetensors (safetensors, 8 tensors)
Thresholds: max-abs=1e4 max-zero-fraction=0.5 max-dead-row-fraction=0.01 row-norm-ratio=10
Rows are indices along the first axis of the row-major (PyTorch-order) shape.
WARN  "constant.weight" [constant] all 8 values equal 1 (an initial value never trained?)
WARN  "dead_rows.weight" [dead-rows] 2 of 8 rows are all zero (25.0%; WARN above 1.0%): rows 2, 5
WARN  "huge.weight" [large-magnitude] max |x| = 1e5 exceeds 1e4
INFO  "huge.weight" [f16-range] max |x| = 1e5 exceeds the F16 maximum 65504; casting to F16 would overflow
WARN  "huge.weight" [row-norm] 1 of 4 rows have an L2 norm outside [median/10, median*10] (median 1.3374e0): rows 0 (x74772.339)
ERROR "non_finite.weight" [non-finite] 1 NaN and 1 infinite values (none allowed)
WARN  "outlier_rows.weight" [row-norm] 1 of 16 rows have an L2 norm outside [median/10, median*10] (median 4.1227e-1): rows 3 (x77.638)
WARN  "sparse.weight" [sparse] 75.0% of values are exactly zero (WARN above 50.0%)
WARN  "zeros.bias" [all-zero] all 8 values are zero (untrained or a placeholder?)
Checked 8 tensors, skipped 0: 1 errors, 7 warnings, 1 info
```

## Contents

- [Features](#features)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Testing and verification](#testing-and-verification)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Validation against real models](#validation-against-real-models)
- [Models to try](#models-to-try)
- [Limitations](#limitations)
- [Project layout](#project-layout)
- [Contributing](#contributing)

## Features

- **Two formats, parsed by hand:** safetensors and GGUF (v2/v3), with defensive
  parsing that turns malformed or hostile files into error messages rather than
  crashes, tested against thousands of randomly corrupted files.
- **Five decoders:** F32, F16, BF16, and llama.cpp's Q8_0 and Q4_0 block
  quantization. Every other ggml type is still size-checked by `inspect`.
- **Four commands:** `inspect` (header only), `stats` (per-tensor statistics and
  histograms), `health` (NaN/Inf, dead and outlier rows, and more) and `diff`
  (compare two models, including two quantizations of the same model).
- **Fast and deterministic:** memory-mapped files, streaming decoding, Rayon
  parallelism with identical results for any thread count, and AVX2 kernels
  chosen at run time with a portable fallback.
- **Verified:** decoding is bit-identical to llama.cpp's reference implementation
  on real models, and one command, `cargo verify`, checks the whole project.

## Getting started

Requires [Rust](https://rustup.rs) 1.88 or newer.

```sh
git clone https://github.com/Danieldotcomcoder/biopsy.git
cd biopsy
cargo install --path .        # puts `biopsy` on your PATH
```

To run without installing, replace `biopsy` with `cargo run --release --` in the
commands below.

No model download is needed to try it. The `make_tiny` example writes six small
demo files into `models/`:

```sh
cargo run --example make_tiny
biopsy inspect models/tiny_q4_0.gguf
biopsy stats models/tiny_q8_0.gguf --histogram
biopsy health models/sick.safetensors
biopsy diff models/encoder.safetensors models/encoder_finetuned.safetensors
biopsy diff models/tiny_q8_0.gguf models/tiny_q4_0.gguf
```

| Demo file | Purpose |
|---|---|
| `tiny.safetensors` | The smallest example: `weight` `[2, 3]` and `bias` `[2]`, 8 elements. |
| `encoder.safetensors`, `encoder_finetuned.safetensors` | A base/fine-tuned pair in F32, BF16 and F16: identical, changed, reshaped and unmatched tensors. |
| `sick.safetensors` | One tensor per health problem, plus one healthy tensor. |
| `tiny_q8_0.gguf`, `tiny_q4_0.gguf` | The same weights quantized two ways; F16 and F32 tensors, a 64-byte alignment, and a Q4_K tensor that is inspected but not decoded. |

## Usage

| Command | What it does |
|---|---|
| `biopsy inspect FILE` | Header only: metadata, and every tensor's type, shape, offsets and size. Reads no weights. |
| `biopsy stats FILE [--filter TEXT] [--histogram] [--bins N]` | Count, mean, std, min, max and zero/NaN/Inf counts per tensor; optional text histograms. |
| `biopsy health FILE [--filter TEXT] [--strict] [threshold flags]` | Looks for NaN/Inf, all-zero, constant, sparse and huge tensors, dead rows and outlier rows. |
| `biopsy diff A B [--strip-prefix P]... [--top N]` | Unmatched names, shape mismatches, and numeric change per shared tensor. |

- **Global flags:** `--threads N` sets the worker count (results are identical for
  any N); `--scalar` uses the portable kernels instead of SIMD.
- **Format detection:** the file type comes from its first bytes, so every command
  accepts either format.
- **Exit status:** 0 success; 1 the input could not be read or parsed; 2 `health`
  found an ERROR (or a WARN with `--strict`), which makes it usable in CI.
- **Output conventions:** offsets and sizes are in bytes. Safetensors tensors are
  listed alphabetically, GGUF tensors in file order. `data_offset` is relative to
  the start of the tensor data; `file_offset` is absolute in the file.

Example `diff` between a base model and a fine-tune:

```text
$ biopsy diff models/encoder.safetensors models/encoder_finetuned.safetensors
A: models/encoder.safetensors (safetensors, 5 tensors)
B: models/encoder_finetuned.safetensors (safetensors, 5 tensors)
Matched by name: 4; compared: 3; shape mismatches: 1; not decoded: 0; only in A: 1; only in B: 1
Only in A: "lm_head.weight"
Only in B: "classifier.weight"
Shape mismatch: "pooler.weight": [32, 32] vs [16, 32]
Identical: 1; changed: 2

Changed tensors, largest relative L2 change first (showing 2 of 2):
tensor                       types   elements  changed     max|d|       rmse     rel_l2     cosine
"layer.0.dense.bias"           F16         32  100.00%   5.165e-3   2.293e-3   1.911e-1   0.981810
"layer.0.dense.weight"        BF16       1024   95.31%   7.812e-3   2.049e-3   4.094e-2   0.999162

Overall (1568 compared elements): changed 64.29%, rel_l2 4.010e-2, cosine 0.999196
```

## Testing and verification

```sh
cargo verify
```

`cargo verify` is an alias (defined in `.cargo/config.toml`) for
`examples/verify.rs`. It prints a PASS/FAIL checklist and exits non-zero if
anything fails:

1. `cargo fmt --check` and `cargo clippy --all-targets -D warnings`
2. every test, in debug (integer overflow checks on) and in release (SIMD as shipped)
3. the release binary on freshly generated demo files: every command, exit code and key output
4. the kernel benchmark in a short run, which refuses to time a SIMD kernel whose
   output differs from the scalar reference
5. every real model placed in `models/`: `inspect` and `health` must not fail

It needs no network access. The last line reads `VERIFY PASSED: N checks in Ns`.

Other entry points:

- `cargo test` runs the test suites: exhaustive checks of all 65,536 F16 bit patterns,
  GGUF parser rejection cases, SIMD-vs-scalar agreement, random file corruption,
  and the CLI end to end. All test files are generated in memory; nothing is
  downloaded.
- `cargo run --release --example bench` runs the kernel benchmarks (see [Performance](#performance)).
- `tools/crosscheck.py` compares biopsy with independent readers (see
  [Validation](#validation-against-real-models)).

## How it works

The code was written in six stages; each section below explains one and names
the files to read. The doc comment at the top of `src/lib.rs` suggests a reading
order for the whole library.

### 1. Safetensors framing and memory mapping

The [safetensors specification](https://github.com/safetensors/safetensors#format)
describes this layout:

```text
file offset 0       8                      8 + N
            [N: u64][N bytes of JSON header][raw tensor payload]
```

1. **Bytes and endianness.** A file is a sequence of bytes, not Rust values.
   The first eight are read as a `u64` using `from_le_bytes`. Little-endian
   puts the least significant byte first: `08 00 00 00 00 00 00 00` means 8.
   The host CPU's native byte order does not change this rule.
2. **Framing.** That length says exactly where the JSON ends and the weights start.
   A tensor with offsets `[begin, end]` is `end - begin` bytes long and starts at
   `8 + N + begin` in the file. The end is exclusive.
3. **Shape and dtype.** A `[2, 3]` F32 tensor has six elements of four bytes each:
   24 bytes. A scalar has shape `[]` and one element. Any zero dimension means
   no elements.
4. **Memory mapping.** `memmap2` gives a byte slice backed by the operating
   system's file mapping. Mapping a multi-GB file reserves virtual address space
   instead of copying the file into a `Vec`, and pages are read from disk only when
   touched. `inspect` only touches the prefix and the JSON. Use a 64-bit build
   for large models.
5. **Alignment.** A byte slice makes no promise that a tensor address is aligned
   for `f32`, so it is never cast to `&[f32]`. Decoding goes through fixed-size
   byte arrays and `from_le_bytes`, and the SIMD code uses unaligned loads.
6. **Defensive parsing.** Checked arithmetic and slice bounds turn malformed
   lengths into errors. Shape sizes must match byte spans, and spans must cover the
   payload without holes or overlaps. Duplicate JSON keys are rejected by a small
   Serde visitor. Serde parses the JSON syntax; biopsy implements the file format.

Read `src/safetensors.rs` starting at `parse_header`, then the `UniqueMap` helper.
The parser takes a `&[u8]`, so it works the same on a file mapping or an in-memory
test vector.

The `unsafe` block in `src/main.rs` exists because a file mapping depends on
something Rust cannot enforce: no other process may modify or truncate the file
while it is mapped. Inspect stable files, not a download or checkpoint still being
written. See the [memmap2 safety documentation](https://docs.rs/memmap2/latest/memmap2/struct.MmapOptions.html#safety).

### 2. Decoding, streaming statistics, Rayon

Read `src/half.rs`, `src/decode.rs`, then `src/stats.rs`.

1. **Three float layouts.** F32 is `[sign 1][exponent 8][mantissa 23]`. BF16 keeps
   F32's 8-bit exponent and cuts the mantissa to 7 bits: it is the top half of an
   F32, so decoding is a 16-bit shift. F16 has a 5-bit exponent (bias 15), so
   decoding re-biases the exponent and handles subnormals, infinity and NaN.
   Every F16 value is exactly representable as F32. The tests check all 65,536
   F16 patterns against an independent formula, and every rounding midpoint of
   the F32-to-F16 encoder.
2. **Streaming.** A tensor is never decoded whole. Values are decoded 4,096 at a
   time into a reused buffer, so memory use depends on the thread count, not on
   the model size.
3. **Mergeable statistics.** Each buffer gets a summary: count, mean, M2 (sum of
   squared deviations), min and max. Two summaries combine exactly as if their
   values had been seen together (Chan et al.):
   `mean = mean_a + delta*n_b/n` and `M2 = M2_a + M2_b + delta^2*n_a*n_b/n`.
   Inside a buffer there are two passes (mean, then deviations) rather than
   Welford's one-pass update, which needs a division per value and cannot be
   vectorized. The tests use data with a mean far from zero, which would break
   the tempting shortcut `sum(x^2) - n*mean^2`.
4. **Non-finite values.** NaN and infinities are counted, never averaged, so one
   bad value cannot hide the statistics of all the others.
5. **Rayon and determinism.** Work is a flat list of jobs of 64 Ki values
   across all tensors, so one huge embedding does not leave threads idle.
   `map_init` gives each worker its own scratch buffer. Results are collected
   in job order and merged sequentially, so floating-point additions happen in
   the same order for any thread count: `--threads 1` and `--threads 16` print
   identical numbers (tested).
6. **Histograms.** Binning needs the range first, so a histogram is a second pass
   over the mapped bytes using each tensor's min and max from the first pass.
   Bin counts are integers, so merging them is exact.

### 3. Health checks

Read `src/health.rs`. Every finding names its check and states its threshold,
and every threshold is a flag.

| Check | Severity | Default threshold (flag) |
|---|---|---|
| `non-finite`: any NaN or Inf | ERROR | none allowed |
| `all-zero`: every value is 0 | WARN | |
| `constant`: every value equal, e.g. an untrained LayerNorm weight of 1.0 | WARN | |
| `sparse`: share of exact zeros | WARN | above 50% (`--max-zero-fraction`) |
| `large-magnitude`: largest absolute value | WARN | above 1e4 (`--max-abs`) |
| `f16-range`: F32/BF16 values beyond 65504 would overflow if cast to F16 | INFO | |
| `dead-rows`: rows that are entirely zero | INFO; WARN above 1% of rows | `--max-dead-row-fraction` |
| `row-norm`: a row's L2 norm vs the median row norm | WARN | beyond x10 or /10 (`--row-norm-ratio`) |

**Row semantics.** A row is one index along the *first* axis of the row-major
(PyTorch-order) shape: one contiguous run of `product(shape[1..])` values. For an
`nn.Linear` weight `[out, in]`, a row is one output unit; for an embedding
`[vocab, dim]`, one token; for a convolution `[out, in, kh, kw]`, one filter.
GGUF lists dimensions innermost first, so biopsy reverses them before defining
rows; a test writes the same matrix to both formats and checks that both report
the same row numbers. Tensors with fewer than two axes have no rows.

A finding is a prompt to look closer, not a verdict: a zero bias can be
intentional, and a single zero row in an embedding is often the padding token.
Dead or outlier rows in an embedding are still worth knowing about, because
tokens that were never trained can break fine-tuning.

### 4. Diff

Read `src/diff.rs`. Tensors are matched by name (`--strip-prefix model.` helps
when one checkpoint wraps another), then by shape. Only same-name, same-shape
pairs are compared numerically. Their dtypes may differ, because both sides are
decoded to F32, so `diff model-q8_0.gguf model-q4_0.gguf` measures quantization
error directly.

For each pair, with `d = a - b` over the positions where both values are finite:

- `changed`: share of positions where `a != b`
- `max|d|`, and `rmse = sqrt(mean(d^2))`
- `rel_l2 = ||a - b|| / ||a||`: the change relative to the tensor's own size, so it
  is comparable across tensors; the table is sorted by it
- `cosine = a.b / (||a|| ||b||)`: 1.0 means the same direction
- NaN or Inf on only one side is counted separately, never averaged

A real example: Google's BERT-Tiny compared with a
[SQuAD v2 fine-tune](https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2).
`diff` reports the pretraining heads (`cls.*`, 7 tensors) only in A and the QA
head (`qa_outputs.*`) only in B. `position_embeddings` changed in exactly 75.00%
of its values; checking with numpy showed that rows 384 to 511 are bit-identical.
The fine-tune used SQuAD's usual 384-token maximum, so later positions never
received a gradient. 8,310 of 30,522 word-embedding rows are also bit-identical,
most likely tokens that never appeared in the fine-tuning data. The pooler is
untouched because the QA head does not use it.

### 5. GGUF and block quantization

Read `src/gguf.rs`, then `src/quant.rs`.

```text
"GGUF" | version: u32 | tensor_count: u64 | metadata_count: u64
metadata_count x [key: string][value_type: u32][value]
tensor_count   x [name: string][n_dims: u32][dims: n_dims x u64][ggml_type: u32][offset: u64]
zero padding to a multiple of general.alignment (default 32)
tensor data: each tensor at data_start + offset, padded to the alignment
```

1. **Strings** are `[len: u64][bytes]` with no terminator. Metadata values have
   13 types, including arrays, which may nest. The reader keeps an 8-element
   preview of each array but parses and validates every element; tokenizer
   vocabularies hold tens of thousands of strings.
2. **Dimension order.** GGUF lists dimensions innermost first (ggml's `ne[0]` is
   the contiguous axis), so a PyTorch `[out, in]` weight appears as `dims=[in, out]`.
   `inspect` prints both `dims` (as stored) and `shape` (row-major).
3. **Alignment.** `general.alignment` must be a power of two. Tensor offsets
   follow header order, each tensor padded to the alignment, the same rule
   llama.cpp's loader enforces, which also rules out holes and overlaps.
4. **Block sizes.** A quantized type packs `block_elements` values into
   `block_bytes` bytes, and the innermost dimension must be a multiple of the
   block size. `inspect` knows the block size of every ggml type up to MXFP4
   (id 39), including K-quants and IQ types, so it validates their sizes and
   offsets even though it does not decode them. Types that were removed from the
   format (`Q4_0_4_4` and others) get an explicit message.
5. **Defensive parsing.** A hostile count cannot trigger a huge allocation: each
   count is checked against the remaining bytes before anything is read. Array
   nesting has a depth limit, so the parser cannot overflow the stack.
   Big-endian and v1 files are recognized and refused with a clear message.
   Every truncation of a valid file is tested, and `tests/corruption.rs`
   corrupts the demo files at random 2,400 times per run: each must be rejected
   or handled, never panic.
6. **Q8_0** blocks are 34 bytes: an F16 scale `d` and 32 signed bytes,
   `value = d * q`. **Q4_0** blocks are 18 bytes: `d` and 16 bytes holding 32
   four-bit codes, `value = d * (code - 8)`. Byte `j`'s low nibble is element `j`
   and its high nibble element `j + 16`; neighbours are not interleaved.
   Q4_0 sets `d = max / -8`, so the largest-magnitude value maps exactly onto
   code 0.

### 6. SIMD, measured before and after

Read `src/kernels.rs`, then run `cargo run --release --example bench`.

1. **Measure first.** The scalar kernels were benchmarked before any SIMD code
   existed. The time went to the two summary passes and the diff sums (branches
   plus f64 accumulation), F16 decoding (bit manipulation), and Q8_0/Q4_0
   decoding at about one value per cycle.
2. **Masks instead of branches.** An AVX2 register holds eight f32. A comparison
   returns a lane mask (all ones or all zeros), which `and`/`blendv` use to zero
   or replace non-finite lanes without branching. `movemask` packs the mask into
   an integer and `count_ones` counts the matching lanes. f64 registers hold
   four lanes, so each eight-float vector is widened in two halves.
3. **Soundness.** Running AVX2 instructions on a CPU without AVX2 is undefined
   behaviour. `Backend::Avx2` carries a token with a private field that only
   `Backend::detect()` can create, after checking AVX2, F16C, FMA and POPCNT at
   run time, so safe code cannot reach the `unsafe` kernels on the wrong CPU.
   Other CPUs use the scalar kernels.
4. **Correctness.** Decoders are bit-identical between backends, tested on all
   65,536 F16 patterns and random Q8_0/Q4_0 blocks, including NaN/Inf scales.
   Sums add eight lanes separately, a different order of floating-point
   additions, so they agree within 1e-10 relative rather than bit for bit. Each
   backend is still deterministic.
5. **Keep only what pays.** A hand-written BF16 kernel measured 1.06x to 1.34x
   across runs, while a control row running identical code under both labels
   showed run-to-run noise of up to about 10%. The compiler had already
   vectorized the scalar shift, so the BF16 kernel was deleted.
6. **Cache size matters as much as instructions.** The first version decoded
   64 Ki values (256 KiB) per buffer. With a 256 KiB L2 cache shared by two
   hyperthreads, the 16-thread SIMD pipeline took 7.2 ms. Buffers of 4 Ki values
   (16 KiB, inside the 32 KiB L1) cut it to 1.6 ms. Parallel jobs stayed at
   64 Ki values and the timing did not change, which ruled out load balancing
   as the cause and kept job lists small.

## Performance

Measured on an Intel Core i7-10870H (8 cores, 16 threads) with
`cargo run --release --example bench`: 16 Mi values, median of 9 runs.
Expect different absolute numbers on other machines.

| Kernel (single thread) | Scalar | AVX2 | Speedup |
|---|---:|---:|---:|
| decode F16 (F16C `vcvtph2ps`) | 9.79 ms | 1.86 ms | 5.3x |
| decode Q8_0 | 3.22 ms | 1.91 ms | 1.7x |
| decode Q4_0 | 4.17 ms | 2.03 ms | 2.1x |
| decode BF16 (no SIMD kernel) | 2.23 ms | same code | 1.0x |
| summary, two passes | 46.20 ms | 9.88 ms | 4.7x |
| diff sums | 48.35 ms | 12.27 ms | 3.9x |

| `stats` pipeline on F16 (decode + summary) | Scalar | AVX2 |
|---|---:|---:|
| 1 thread | 54.7 ms | 10.4 ms |
| 16 threads | 7.2 ms | 1.6 ms (35x the 1-thread scalar time) |

End to end, `biopsy stats` on the 269 MB SmolLM2-135M-Instruct BF16 safetensors
file (process start, mapping and all) took 527 ms with `--scalar --threads 1`
and 103 ms by default.

## Validation against real models

On SmolLM2-135M-Instruct's
[BF16 safetensors](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
and its [Q8_0 and Q4_0 GGUFs](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF),
every value biopsy decodes (131.9 to 134.5 million per file) is bit-identical to
llama.cpp's own reference dequantizer (`gguf-py`) and to numpy, with both the
scalar and the AVX2 kernels. Tensor names, types, shapes, offsets, metadata
count, alignment and per-tensor statistics match as well.

To reproduce this on any model, `examples/dump.rs` writes every decoded tensor
plus a manifest, and `tools/crosscheck.py` compares them with `gguf-py` (GGUF)
or plain numpy (safetensors). It needs Python, so `cargo verify` does not run it.

```sh
pip install numpy gguf
cargo build --release
cargo run --release --example dump -- models/model.gguf target/dump auto
python tools/crosscheck.py models/model.gguf target/dump target/release/biopsy
```

Use `scalar` instead of `auto` to check the portable kernels.

## Models to try

All of these are public, with no Hugging Face account or token needed. Save each
file in `models/` under a distinct name (most repos call their file
`model.safetensors`); `cargo verify` then checks it too. `models/` is
git-ignored.

| Model | Size | Why it is interesting |
|---|---:|---|
| [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) + [SQuAD v2 fine-tune](https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2) | 18 MB each | F32; the smallest download, and the `diff` example in [Diff](#4-diff) |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) + [base model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 269 MB each | BF16; `diff` shows what instruction tuning changed |
| [SmolLM2-135M-Instruct GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) (`f16`, `Q8_0`, `Q4_0`) | 271 / 145 / 92 MB | Same names in every file, so `diff` measures each quantization's error. Q8_0 to Q4_0 on this model: rel_l2 8.2%, cosine 0.9967 |
| [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) | 166 MB | F16 safetensors, plus U8 attention masks that are reported as not decoded |
| [Qwen2.5-0.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (`q4_0`) | 429 MB | A larger file that is fully decodable (Q4_0, Q8_0, F32) |
| [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) + [SST-2 fine-tune](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 268 MB each | F32; different task heads show up as unmatched tensors |
| [SmolLM2-135M-Instruct GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) (`Q4_K_M`) | 106 MB | Shows the limits: `inspect` works, but most tensors are not decoded |

Download a file with its `resolve/main` URL, for example:

```sh
curl -L -o models/bert-tiny.safetensors https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors
```

On Windows PowerShell, type `curl.exe` rather than `curl`, which is an alias for
`Invoke-WebRequest` there.

## Limitations

- Decodes F32, F16, BF16, Q8_0 and Q4_0 only. Other types (integers, FP8, Q4_1,
  K-quants, IQ types) are size-checked by `inspect` and reported as "not decoded"
  by `stats`, `health` and `diff`.
- GGUF v2 and v3, little-endian, at most 4 dimensions (as in ggml).
- Safetensors sub-byte dtypes are rejected with an explicit error, and JSON
  headers are capped at 100 MB.
- SIMD kernels need x86-64 with AVX2, F16C and FMA; other CPUs use the scalar
  kernels.
- `diff` matches names exactly (after `--strip-prefix`). GGUF and Hugging Face
  name the same tensor differently (`blk.0.attn_q.weight` vs
  `model.layers.0.self_attn.q_proj.weight`), and llama.cpp permutes some
  attention weights, so a GGUF-vs-safetensors diff of the same model reports
  most tensors as unmatched.
- The parameter count is the number of stored tensor elements, including
  buffers such as attention masks. The file does not say which values are
  trainable or which omitted weights are tied.
- Health thresholds are heuristics with explicit defaults, not proof of a bug.

## Project layout

```text
src/main.rs        CLI: argument parsing, memory mapping, exit codes
src/safetensors.rs safetensors header parser           (stage 1)
src/half.rs        F16 and BF16 by hand                (stage 2)
src/decode.rs      bytes to F32 in chunks; job lists   (stage 2)
src/stats.rs       summaries, merging, histograms      (stage 2)
src/health.rs      health checks and row semantics     (stage 3)
src/diff.rs        name/shape matching and metrics     (stage 4)
src/gguf.rs        GGUF parser and ggml type table     (stage 5)
src/quant.rs       Q8_0 and Q4_0 blocks                (stage 5)
src/kernels.rs     scalar and AVX2 hot loops           (stage 6)
src/tensor.rs      format-independent tensor view
src/model.rs       format detection
src/inspect.rs     the inspect report
src/fixtures.rs    writers for synthetic test files
examples/          make_tiny (demo files), bench, verify, dump
tests/             one suite per module, the CLI, random corruption
tools/             crosscheck.py (optional, Python)
```

## Contributing

Issues and pull requests are welcome. Please run `cargo verify` before opening a
pull request; it runs the same formatting, lint, test and end-to-end checks
described above.
