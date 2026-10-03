//! Before/after benchmarks for the kernels: `cargo run --release --example bench`.
//!
//! Each kernel runs single-threaded over 16 Mi synthetic values in the same 64 Ki
//! chunks the real commands use, once per backend. Before timing anything, the
//! SIMD output is checked against the scalar output. A final section times the
//! whole `stats` pipeline on 1 thread and on all threads to show Rayon's share.
//!
//! Options: `--elements N` (default 16777216), `--repeats R` (default 7).
use std::{hint::black_box, time::Instant};

use biopsy::{
    decode::{self, CHUNK_ELEMENTS},
    fixtures::{self, Rng},
    kernels::{self, Backend},
    quant, stats,
    tensor::{DType, TensorView},
};

struct Config {
    elements: usize,
    repeats: usize,
}

fn config() -> Config {
    let mut c = Config {
        elements: 1 << 24,
        repeats: 7,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    for pair in args.chunks(2) {
        let value = pair.get(1).and_then(|v| v.parse().ok());
        match (pair[0].as_str(), value) {
            ("--elements", Some(v)) => c.elements = v,
            ("--repeats", Some(v)) => c.repeats = v,
            _ => panic!("usage: bench [--elements N] [--repeats R]"),
        }
    }
    c.elements = c.elements.next_multiple_of(CHUNK_ELEMENTS);
    c
}

/// Median wall time of `repeats` runs after one warm-up run, in seconds.
fn time(repeats: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut samples: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn backends() -> Vec<Backend> {
    let mut list = vec![Backend::Scalar];
    if Backend::detect() != Backend::Scalar {
        list.push(Backend::detect());
    }
    list
}

fn report(name: &str, bytes: usize, results: &[(Backend, f64)]) {
    report_with_note(name, bytes, results, "");
}

/// `note` is printed after non-scalar rows, e.g. to flag a noise-control row.
fn report_with_note(name: &str, bytes: usize, results: &[(Backend, f64)], note: &str) {
    let base = results[0].1;
    for &(backend, seconds) in results {
        println!(
            "{name:<22} {:<7} {:>9.2} ms {:>8.2} GB/s {:>7.2}x{}",
            backend.name(),
            seconds * 1e3,
            bytes as f64 / seconds / 1e9,
            base / seconds,
            if backend == Backend::Scalar { "" } else { note }
        );
    }
}

fn decode_all(dtype: DType, bytes: &[u8], buffer: &mut Vec<f32>, backend: Backend) -> f32 {
    let mut checksum = 0.0;
    for range in decode::chunk_ranges(decode::element_count(dtype, bytes), CHUNK_ELEMENTS) {
        decode::decode_into(
            dtype,
            &bytes[dtype.byte_range(range.start, range.end)],
            buffer,
            backend,
        );
        checksum += buffer[0];
    }
    checksum
}

/// SIMD decoding must reproduce scalar decoding bit for bit.
fn assert_same_decode(dtype: DType, bytes: &[u8], backend: Backend) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    decode::for_each_chunk(dtype, bytes, &mut a, Backend::Scalar, |_| {});
    for range in decode::chunk_ranges(decode::element_count(dtype, bytes), CHUNK_ELEMENTS) {
        let chunk = &bytes[dtype.byte_range(range.start, range.end)];
        decode::decode_into(dtype, chunk, &mut a, Backend::Scalar);
        decode::decode_into(dtype, chunk, &mut b, backend);
        let same = a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        assert!(
            same,
            "{dtype} decode differs between scalar and {}",
            backend.name()
        );
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1e-300)
}

fn main() {
    let c = config();
    let n = c.elements;
    println!(
        "biopsy kernel benchmark: {n} values, median of {} runs, single thread unless noted",
        c.repeats
    );
    println!("detected backend: {}\n", Backend::detect().name());
    let values = Rng::new(42).normals(n, 0.05);
    let encoded = [
        (DType::F16, fixtures::f16_bytes(&values)),
        (DType::BF16, fixtures::bf16_bytes(&values)),
        (DType::Q8_0, quant::quantize_q8_0(&values)),
        (DType::Q4_0, quant::quantize_q4_0(&values)),
        (DType::F32, fixtures::f32_bytes(&values)),
    ];
    let mut buffer = Vec::with_capacity(CHUNK_ELEMENTS);

    println!(
        "{:<22} {:<7} {:>12} {:>13} {:>8}",
        "kernel", "backend", "time", "input rate", "speedup"
    );
    for (dtype, bytes) in &encoded {
        let mut results = Vec::new();
        for backend in backends() {
            assert_same_decode(*dtype, bytes, backend);
            let seconds = time(c.repeats, || {
                black_box(decode_all(*dtype, black_box(bytes), &mut buffer, backend));
            });
            results.push((backend, seconds));
        }
        // F32 and BF16 have no SIMD kernel: both rows run the same code, so their
        // ratio shows the benchmark's noise.
        let note = match dtype {
            DType::F32 | DType::BF16 => "  same code: noise check",
            _ => "",
        };
        report_with_note(&format!("decode {dtype}"), bytes.len(), &results, note);
    }

    let chunks: Vec<&[f32]> = values.chunks(CHUNK_ELEMENTS).collect();
    let mut results = Vec::new();
    for backend in backends() {
        let (a, b) = (
            stats::Summary::of(chunks[0], Backend::Scalar),
            stats::Summary::of(chunks[0], backend),
        );
        assert!(a.min == b.min && a.max == b.max && a.finite == b.finite && a.zeros == b.zeros);
        assert!(
            close(a.mean, b.mean) && close(a.m2, b.m2),
            "summary differs"
        );
        let seconds = time(c.repeats, || {
            for chunk in &chunks {
                black_box(stats::Summary::of(black_box(chunk), backend));
            }
        });
        results.push((backend, seconds));
    }
    report("summary (2 passes)", n * 4, &results);

    let other: Vec<f32> = values.iter().map(|v| v * 1.01 + 1e-4).collect();
    let other_chunks: Vec<&[f32]> = other.chunks(CHUNK_ELEMENTS).collect();
    let mut results = Vec::new();
    for backend in backends() {
        let (a, b) = (
            kernels::diff_sums(Backend::Scalar, chunks[0], other_chunks[0]),
            kernels::diff_sums(backend, chunks[0], other_chunks[0]),
        );
        assert!(
            a.finite_pairs == b.finite_pairs
                && a.equal == b.equal
                && a.max_abs_diff == b.max_abs_diff
        );
        assert!(
            close(a.sum_sq_diff, b.sum_sq_diff) && close(a.sum_ab, b.sum_ab),
            "diff sums differ"
        );
        let seconds = time(c.repeats, || {
            for (x, y) in chunks.iter().zip(&other_chunks) {
                black_box(kernels::diff_sums(backend, black_box(x), black_box(y)));
            }
        });
        results.push((backend, seconds));
    }
    report("diff sums", n * 8, &results);

    // End to end: the `stats` pipeline on an F16 tensor, the common checkpoint dtype.
    let f16 = &encoded[0].1;
    let tensor = [TensorView {
        name: "bench".into(),
        type_name: "F16".into(),
        dtype: Some(DType::F16),
        shape: vec![n as u64],
        elements: n as u64,
        bytes: f16,
    }];
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    println!("\nstats pipeline on F16 (decode + summary, chunks merged in order):");
    let mut base = None;
    for threads in [1, cores] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for backend in backends() {
            let seconds = time(c.repeats, || {
                pool.install(|| black_box(stats::summarize_all(black_box(&tensor), backend)));
            });
            let base = *base.get_or_insert(seconds);
            println!(
                "{:>2} threads {:<7} {:>9.2} ms {:>8.2} GB/s {:>7.2}x",
                threads,
                backend.name(),
                seconds * 1e3,
                f16.len() as f64 / seconds / 1e9,
                base / seconds
            );
        }
    }
}
