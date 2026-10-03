# biopsy

A learning-first Rust CLI for inspecting neural network files from raw bytes.
It reads safetensors and GGUF files with hand-written parsers (no safetensors
or GGUF crate), decodes F32, F16, BF16, Q8_0 and Q4_0 weights, and reports
statistics, health problems and differences between two models.

All six milestones are complete. Their code is explained milestone by milestone in
"What you are learning" below.

## Verify everything with one command

Open a new PowerShell terminal after installing Rust (so PATH includes Cargo):

```powershell
cd C:\Users\danie\Desktop\System\biopsy
cargo verify
```

`cargo verify` is an alias (`.cargo/config.toml`) for `examples/verify.rs`. It
prints a PASS/FAIL checklist and exits non-zero if anything fails:

1. `cargo fmt --check` and `cargo clippy --all-targets -D warnings`
2. every test, in debug (integer overflow checks on) and in release (SIMD as shipped)
3. the release binary on freshly generated demo files: every command, exit code and key output
4. the kernel benchmark in a short run, which refuses to time a SIMD kernel whose
   output differs from the scalar reference
5. every real model you have put in `models/`: `inspect` and `health` must not fail

It needs no network and no downloads. Expected last line: `VERIFY PASSED: N checks in Ns`.

If an already-open terminal cannot find Cargo, run:

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
```

## Try it

```powershell
cargo run --example make_tiny              # writes six small demo files into models/
cargo run --release -- inspect models/tiny.safetensors
cargo run --release -- inspect models/tiny_q4_0.gguf
cargo run --release -- stats models/tiny_q8_0.gguf --histogram
cargo run --release -- health models/sick.safetensors
cargo run --release -- diff models/encoder.safetensors models/encoder_finetuned.safetensors
cargo run --release -- diff models/tiny_q8_0.gguf models/tiny_q4_0.gguf
cargo run -- --help
```

| Command | What it does |
|---|---|
| `inspect FILE` | Header only: metadata, every tensor's type, shape, offsets and size. Reads no weights. |
| `stats FILE [--filter TEXT] [--histogram] [--bins N]` | Count, mean, std, min, max, zero/NaN/Inf counts per tensor; optional text histograms. |
| `health FILE [--filter TEXT] [--strict] [threshold flags]` | NaN/Inf, all-zero, constant, sparse, huge values, dead rows, outlier rows. |
| `diff A B [--strip-prefix P]... [--top N]` | Unmatched names, shape mismatches, and numeric change per shared tensor. |

Global flags: `--threads N` (results are identical for any N) and `--scalar`
(use the portable kernels instead of SIMD). The file type is detected from its
first bytes, so every command takes either format.

Exit status: 0 success; 1 the input could not be read or parsed; 2 `health`
found an ERROR (or a WARN with `--strict`).

The demo files `make_tiny` writes:

| File | Purpose |
|---|---|
| `tiny.safetensors` | The milestone 1 fixture: `weight` `[2, 3]` and `bias` `[2]`, 8 elements. |
| `encoder.safetensors`, `encoder_finetuned.safetensors` | A base/fine-tuned pair in F32, BF16 and F16: identical, changed, reshaped and unmatched tensors. |
| `sick.safetensors` | One tensor per health problem, plus one healthy tensor. |
| `tiny_q8_0.gguf`, `tiny_q4_0.gguf` | The same weights quantized two ways; F16 and F32 tensors, a 64-byte alignment, and a Q4_K tensor that is inspected but not decoded. |

All offsets and sizes are bytes. Safetensors tensors are listed alphabetically,
GGUF tensors in file order. `data_offset` is relative to the tensor data;
`file_offset` is absolute in the file.

## What you are learning

### Milestone 1: safetensors framing and memory mapping

The [safetensors specification](https://github.com/safetensors/safetensors#format)
describes this layout:

```text
file offset 0       8                      8 + N
            [N: u64][N bytes of JSON header][raw tensor payload]
```

1. **Bytes and endianness.** A file is a sequence of bytes, not Rust values.
   We interpret the first eight as a `u64` using `from_le_bytes`. Little-endian
   puts the least significant byte first: `08 00 00 00 00 00 00 00` means 8.
   The host CPU's native byte order does not change this rule.
2. **Framing.** That length tells us exactly where JSON ends and weights start.
   For a tensor with offsets `[begin, end]`, its length is `end - begin`;
   its absolute start is `8 + N + begin`. The end is exclusive.
3. **Shape and dtype.** A `[2, 3]` F32 tensor has six elements, four bytes each:
   24 bytes. A scalar has shape `[]` and one element. Any zero dimension means
   no elements.
4. **Memory mapping.** `memmap2` gives a byte slice backed by the OS's file mapping.
   Mapping a multi-GB file reserves virtual address space; it does not copy the
   entire file into a `Vec`. Pages are brought into physical memory as accessed.
   `inspect` only touches the prefix and JSON. Metadata is allocated normally;
   tensor data is zero-copy. Use a 64-bit process for large models.
5. **Alignment.** A byte slice has no promise that a tensor address is correctly
   aligned for `f32`. We never cast it to `&[f32]`; decoding goes through fixed
   byte arrays and `from_le_bytes`, and SIMD code uses unaligned loads.
6. **Defensive parsing.** Checked arithmetic and slice bounds turn malformed
   lengths into errors. Shape sizes must match spans; spans must cover the
   payload without holes or overlap. JSON duplicate keys are rejected with a
   small Serde visitor. Serde parses JSON syntax; we implement the file format.

Read `src/safetensors.rs` starting at `parse_header`, then the `UniqueMap` helper.
`&[u8]` borrows bytes, so the same parser works with an OS mapping or an in-memory
test vector. The structs own only metadata, making them independent of the
mapping's lifetime.

The `unsafe` block in `src/main.rs` is needed because a file mapping depends on
something Rust cannot enforce: another process must not modify or truncate the
file while it is mapped. Inspect stable files, not a download or checkpoint being
written. See [memmap2 safety documentation](https://docs.rs/memmap2/latest/memmap2/struct.MmapOptions.html#safety).

### Milestone 2: decoding, streaming statistics, Rayon

Read `src/half.rs`, `src/decode.rs`, then `src/stats.rs`.

1. **Three float layouts.** F32 is `[sign 1][exponent 8][mantissa 23]`. BF16 keeps
   F32's 8-bit exponent and cuts the mantissa to 7 bits, so it is the top half of
   an F32 and decoding is a 16-bit shift. F16 has a 5-bit exponent (bias 15), so
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
   Inside a buffer we take two passes (mean, then deviations) instead of
   Welford's one-pass update, which needs a division per value and cannot be
   vectorized. A tensor with a mean far from zero would break the shortcut
   `sum(x^2) - n*mean^2`; the tests use exactly such data.
4. **Non-finite values.** NaN and infinities are counted, never averaged, so one
   bad value cannot hide the statistics of the other values.
5. **Rayon and determinism.** Work is a flat list of jobs of 64 Ki values
   across all tensors, so one huge embedding does not leave threads idle.
   `map_init` gives each worker its own scratch buffer. Results are collected
   in job order and merged sequentially, so floating-point additions happen in
   the same order whatever the thread count: `--threads 1` and `--threads 16`
   print identical numbers (tested).
6. **Histograms.** A histogram needs the range before binning, so it is a second
   pass over the mapped bytes, using each tensor's min and max from the first.
   Bin counts are integers, so merging them is exact.

### Milestone 3: health checks

Read `src/health.rs`. Every finding names its check and states its threshold;
every threshold is a flag.

| Check | Severity | Default threshold (flag) |
|---|---|---|
| `non-finite`: any NaN or Inf | ERROR | none allowed |
| `all-zero`: every value is 0 | WARN | |
| `constant`: every value equal, e.g. an untrained LayerNorm weight of 1.0 | WARN | |
| `sparse`: share of exact zeros | WARN | above 50% (`--max-zero-fraction`) |
| `large-magnitude`: max abs value | WARN | above 1e4 (`--max-abs`) |
| `f16-range`: F32/BF16 values beyond 65504 would overflow if cast to F16 | INFO | |
| `dead-rows`: rows that are entirely zero | INFO, WARN above 1% of rows (`--max-dead-row-fraction`) | |
| `row-norm`: a row's L2 norm vs the median row norm | WARN | beyond x10 or /10 (`--row-norm-ratio`) |

**Row semantics.** A row is one index along the *first* axis of the row-major
(PyTorch-order) shape: one contiguous run of `product(shape[1..])` values. For an
`nn.Linear` weight `[out, in]`, a row is one output unit; for an embedding
`[vocab, dim]`, one token; for a convolution `[out, in, kh, kw]`, one filter.
GGUF lists dimensions innermost first, so biopsy reverses them before defining
rows. A test writes the same matrix to both formats and checks that both report
the same row numbers. Tensors with fewer than two axes have no rows.

A finding is a prompt to look closer, not a verdict: a zero bias can be
intentional, and one zero row in an embedding is often the padding token. Dead
and outlier rows in embeddings are worth knowing about, because tokens that
were never trained can break fine-tuning.

### Milestone 4: diff

Read `src/diff.rs`. Tensors are matched by name (`--strip-prefix model.` helps
when one checkpoint wraps another), then by shape. Only same-name, same-shape
pairs are compared numerically. Dtypes may differ, because both sides are
decoded to F32. So `diff q8_0.gguf q4_0.gguf` measures quantization error directly.

For each pair, with `d = a - b` over the positions where both values are finite:

- `changed`: share of positions where `a != b`
- `max|d|`, `rmse = sqrt(mean(d^2))`
- `rel_l2 = ||a - b|| / ||a||`: change relative to the tensor's own size, so it is
  comparable across tensors; the table is sorted by it
- `cosine = a.b / (||a|| ||b||)`: 1.0 means same direction
- NaN/Inf on only one side are counted separately, never averaged

A real example: Google's BERT-Tiny vs the
[SQuAD v2 fine-tune](https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2)
reports the pretraining heads (`cls.*`, 7 tensors) only in A and the QA head
(`qa_outputs.*`) only in B. `position_embeddings` changed in exactly 75.00% of
its values. Checking with numpy confirmed that rows 384 to 511 are bit-identical:
the fine-tune used SQuAD's usual 384-token maximum, so later positions never
received a gradient. 8,310 of 30,522 word-embedding rows are also bit-identical,
most likely tokens that never appeared in the fine-tuning data and so never got a
gradient. The pooler is untouched because the QA head does not use it.

### Milestone 5: GGUF and block quantization

Read `src/gguf.rs`, then `src/quant.rs`.

```text
"GGUF" | version: u32 | tensor_count: u64 | metadata_count: u64
metadata_count x [key: string][value_type: u32][value]
tensor_count   x [name: string][n_dims: u32][dims: n_dims x u64][ggml_type: u32][offset: u64]
zero padding to a multiple of general.alignment (default 32)
tensor data: each tensor at data_start + offset, padded to the alignment
```

1. **Strings** are `[len: u64][bytes]` with no terminator. Values have 13 types,
   including arrays, which may nest. The reader keeps only an 8-element preview of
   each array, but parses and validates every element; tokenizer vocabularies hold
   tens of thousands of strings.
2. **Dimension order.** GGUF lists dimensions innermost first (ggml's `ne[0]` is
   the contiguous axis). A PyTorch `[out, in]` weight appears as `dims=[in, out]`.
   `inspect` prints both `dims` (as stored) and `shape` (row-major).
3. **Alignment.** `general.alignment` must be a power of two. Tensor offsets
   follow header order, each tensor padded to the alignment, the same rule
   llama.cpp's loader enforces. That rule also rules out holes and overlaps.
4. **Block sizes.** A quantized type packs `block_elements` values into
   `block_bytes`. The innermost dimension must be a multiple of the block size.
   `inspect` knows the block size of every ggml type up to MXFP4 (id 39),
   including K-quants and IQ types, so it validates their sizes and offsets. Removed types (`Q4_0_4_4` and
   others) get an explicit message.
5. **Defensive parsing again.** A hostile count cannot trigger a huge allocation:
   each count is checked against the bytes that remain before anything is read.
   Array nesting has a depth limit, so the parser cannot overflow the stack. Big-endian and
   v1 files are recognized and refused with a clear message. Every truncation of
   a valid file is tested, and `tests/corruption.rs` corrupts the demo files at
   random 2,400 times per run: each must be rejected or handled, never panic.
6. **Q8_0** blocks are 34 bytes: an F16 scale `d` and 32 signed bytes;
   `value = d * q`. **Q4_0** blocks are 18 bytes: `d` and 16 bytes holding 32
   four-bit codes, `value = d * (code - 8)`. Byte `j`'s low nibble is element `j`
   and its high nibble element `j + 16`, so neighbours are not interleaved.
   Q4_0 sets `d = max / -8`, so the largest-magnitude value maps exactly onto
   code 0.

Verified on real files: on SmolLM2-135M-Instruct's Q8_0 and Q4_0 GGUFs and its
BF16 safetensors, every decoded value (131.9 to 134.5 million per file) is
bit-identical to llama.cpp's own reference dequantizer (`gguf-py`) and to
numpy, with both the scalar and the AVX2 kernels. Names, types, shapes, offsets, metadata count, alignment and per-tensor
statistics match too. Reproduce with `tools/crosscheck.py` (see "Cross-checking").

### Milestone 6: SIMD, measured before and after

Read `src/kernels.rs`, then run `cargo run --release --example bench`.

1. **Measure first.** The benchmark ran the scalar kernels before any SIMD code
   existed. It showed where the time went: the two summary passes and the diff
   sums (branches plus f64 accumulation), F16 decoding (bit manipulation), and
   Q8_0/Q4_0 decoding at about one value per cycle.
2. **Masks instead of branches.** An AVX2 register holds eight f32. A comparison
   returns a lane mask (all ones or all zeros); `and`/`blendv` use it to zero or
   replace non-finite lanes without branching. `movemask` packs the mask into
   an integer, and `count_ones` counts the matching lanes. f64 accumulators hold four lanes,
   so each eight-float vector is widened in two halves.
3. **Soundness.** AVX2 instructions on a CPU without AVX2 are undefined behaviour.
   `Backend::Avx2` carries a token with a private field that only
   `Backend::detect()` can create, after checking AVX2, F16C, FMA and POPCNT at
   run time. So safe code cannot reach the `unsafe` kernels on the wrong CPU.
   Other CPUs use the scalar kernels.
4. **Correctness.** Decoders are bit-identical between backends: all 65,536 F16
   patterns and random Q8_0/Q4_0 blocks, including NaN/Inf scales. Sums add
   eight lanes separately, a different order of floating-point additions, so
   they agree within 1e-10 relative (the test tolerance), not bit for bit. Each backend is
   still deterministic.
5. **Keep only what pays.** A hand-written BF16 kernel measured 1.06x to 1.34x
   across runs. The F32 row, where both labels run identical code, showed
   run-to-run noise of up to about 10%. The compiler had already vectorized the scalar shift,
   so the BF16 kernel was deleted.
6. **Cache size matters as much as instructions.** The first version decoded
   64 Ki values (256 KiB) per buffer. On this CPU's 256 KiB L2, shared by two
   hyperthreads, the 16-thread SIMD pipeline took 7.2 ms. Buffers of 4 Ki
   values (16 KiB, inside the 32 KiB L1) cut it to 1.6 ms. Parallel jobs stayed
   at 64 Ki values, and the timing did not change, which ruled out load
   balancing as the cause and kept job lists small.

Results on an Intel i7-10870H (8 cores, 16 threads), 16 Mi values, median of 9 runs:

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

End to end, `biopsy stats` on the 269 MB SmolLM2 BF16 safetensors (process start,
mapping and all) takes 527 ms with `--scalar --threads 1` and 103 ms by default.

## Cross-checking against independent readers

Optional, and not part of `cargo verify` because it needs Python packages.
`examples/dump.rs` writes every decoded tensor and a manifest;
`tools/crosscheck.py` compares them with llama.cpp's `gguf-py` (GGUF) or plain
numpy (safetensors):

```powershell
pip install numpy gguf
cargo build --release
cargo run --release --example dump -- models/smollm2-135m-instruct-q4_0.gguf target/dump auto
python tools/crosscheck.py models/smollm2-135m-instruct-q4_0.gguf target/dump target/release/biopsy
```

Use `scalar` instead of `auto` to check the portable kernels.

## Scope and limitations

- Decodes F32, F16, BF16, Q8_0 and Q4_0. Other types (integers, FP8, Q4_1,
  K-quants, IQ types) are size-checked by `inspect` and reported as "not decoded"
  by `stats`, `health` and `diff`.
- GGUF v2 and v3, little-endian. At most 4 dimensions, as in ggml.
- Safetensors sub-byte dtypes are rejected with an explicit error. JSON headers
  are capped at 100 MB.
- SIMD kernels need x86-64 with AVX2, F16C and FMA. Other CPUs use the scalar kernels.
- `diff` matches names exactly (after `--strip-prefix`). GGUF and Hugging Face
  name the same tensor differently (`blk.0.attn_q.weight` vs
  `model.layers.0.self_attn.q_proj.weight`), and llama.cpp permutes some attention
  weights, so cross-format diffs report most tensors as unmatched.
- Parameter count means stored tensor elements, including buffers. The file does
  not say which values are trainable or which omitted weights are tied.
- Health thresholds are heuristics with explicit defaults, not proofs of a bug.
- Tests create synthetic files locally; no model downloads are required.

## Real testing models

All are public downloads; no account or token is needed. Put them in `models/`
with distinct names. `cargo verify` then checks them too. `models/` is
git-ignored. Sizes verified on Hugging Face on 2026-10-03.

- **BERT-Tiny (smallest download):** [Google base, 17.7 MB](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/main)
  and [SQuAD v2 fine-tune, 17.5 MB](https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2/tree/main),
  F32. The diff example above uses this pair.
- **SmolLM2-135M-Instruct:** [BF16 safetensors, 269 MB](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/main),
  plus GGUF quantizations from [bartowski](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/main):
  `Q8_0` (145 MB) and `Q4_0` (92 MB). `diff` between the two GGUFs shows real
  quantization error: rel_l2 8.2%, cosine 0.9967 over 132 million values. The
  [base model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main) pairs
  with Instruct for a fine-tuning diff.
- **DistilBERT:** [base uncased](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/model.safetensors)
  and [SST-2 sentiment fine-tune](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/main/model.safetensors),
  about 268 MB each. Their task heads differ, so `diff` reports them as unmatched.

Download with `curl -L -o models/<name> <url>`, using a `/resolve/main/<file>`
URL. For example:
`curl -L -o models/bert-tiny.safetensors https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors`

## Project layout

```text
src/main.rs        CLI: argument parsing, memory mapping, exit codes
src/safetensors.rs safetensors header parser          (milestone 1)
src/half.rs        F16 and BF16 by hand               (milestone 2)
src/decode.rs      bytes to F32, chunked; job lists   (milestone 2)
src/stats.rs       summaries, merging, histograms     (milestone 2)
src/health.rs      health checks and row semantics    (milestone 3)
src/diff.rs        name/shape matching and metrics    (milestone 4)
src/gguf.rs        GGUF parser and ggml type table    (milestone 5)
src/quant.rs       Q8_0 and Q4_0 blocks               (milestone 5)
src/kernels.rs     scalar and AVX2 hot loops          (milestone 6)
src/tensor.rs      format-independent tensor view
src/model.rs       format detection
src/inspect.rs     the inspect report
src/fixtures.rs    writers for synthetic test files
examples/          make_tiny (demo files), bench, verify, dump
tests/             one integration suite per module, the CLI, random corruption
tools/             crosscheck.py (optional, Python)
```
