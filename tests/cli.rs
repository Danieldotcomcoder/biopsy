//! Run the real binary on the demo files, the way a user would.
use std::{path::PathBuf, process::Command};

use biopsy::fixtures;
use tempfile::TempDir;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn biopsy(args: &[&str]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_biopsy"))
        .args(args)
        .output()
        .unwrap();
    Run {
        code: output.status.code().unwrap(),
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

/// Write every demo file into a fresh directory.
fn demo() -> (TempDir, impl Fn(&str) -> String) {
    let dir = TempDir::new().unwrap();
    for (name, bytes) in fixtures::demo_files() {
        std::fs::write(dir.path().join(name), bytes).unwrap();
    }
    let root: PathBuf = dir.path().to_owned();
    (dir, move |name: &str| root.join(name).display().to_string())
}

#[test]
fn inspect_reads_both_formats() {
    let (_dir, path) = demo();
    for (name, _) in fixtures::demo_files() {
        let run = biopsy(&["inspect", &path(name)]);
        assert_eq!(run.code, 0, "{name}: {}", run.stderr);
    }
    let st = biopsy(&["inspect", &path("tiny.safetensors")]).stdout;
    assert!(st.contains("\"weight\": dtype=F32 shape=[2, 3] data_offset=0 file_offset=184"));
    assert!(st.contains("Total parameters (stored tensor elements): 8"));
    let gg = biopsy(&["inspect", &path("tiny_q4_0.gguf")]).stdout;
    assert!(
        gg.contains("Format: GGUF v3 (little-endian); alignment 64 bytes"),
        "{gg}"
    );
    assert!(gg.contains("\"blk.0.ffn_up.weight\": type=Q4_0 dims=[64, 128] shape=[128, 64]"));
    assert!(gg.contains("tokenizer.ggml.tokens: array = [string x 8]"));
}

#[test]
fn stats_decodes_and_reports_undecoded_types() {
    let (_dir, path) = demo();
    let run = biopsy(&[
        "stats",
        &path("tiny_q8_0.gguf"),
        "--histogram",
        "--bins",
        "4",
    ]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(
        run.stdout.contains("\"blk.0.ffn_down.weight\"   Q4_K"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains("not decoded"));
    assert!(run.stdout.contains("Histogram of \"token_embd.weight\""));
    let filtered = biopsy(&["stats", &path("encoder.safetensors"), "--filter", "dense"]);
    assert!(
        filtered.stdout.contains("(safetensors, 2 tensors)"),
        "{}",
        filtered.stdout
    );
}

#[test]
fn stats_output_is_identical_for_any_thread_count() {
    let (_dir, path) = demo();
    let body = |threads: &str| {
        let run = biopsy(&["--threads", threads, "stats", &path("sick.safetensors")]);
        assert_eq!(run.code, 0, "{}", run.stderr);
        // The first line names the thread count; the numbers follow.
        run.stdout.lines().skip(1).collect::<Vec<_>>().join("\n")
    };
    assert_eq!(body("1"), body("5"));
}

#[test]
fn health_exit_status_reflects_findings() {
    let (_dir, path) = demo();
    let sick = biopsy(&["health", &path("sick.safetensors")]);
    assert_eq!(sick.code, 2);
    assert!(
        sick.stdout
            .contains("ERROR \"non_finite.weight\" [non-finite]")
    );
    let healthy = biopsy(&["health", &path("encoder.safetensors")]);
    assert_eq!(healthy.code, 0, "{}", healthy.stdout);
    // Warnings alone pass, unless --strict.
    let warn_only = ["health", &path("sick.safetensors"), "--filter", "dead_rows"];
    assert_eq!(biopsy(&warn_only).code, 0);
    assert_eq!(biopsy(&[&warn_only[..], &["--strict"]].concat()).code, 2);
    let gguf = biopsy(&["health", &path("tiny_q4_0.gguf")]);
    assert_eq!(gguf.code, 0, "{}", gguf.stdout);
    assert!(gguf.stdout.contains("SKIP  \"blk.0.ffn_down.weight\""));
}

#[test]
fn diff_compares_across_files_and_quantizations() {
    let (_dir, path) = demo();
    let run = biopsy(&[
        "diff",
        &path("encoder.safetensors"),
        &path("encoder_finetuned.safetensors"),
    ]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(run.stdout.contains("Only in A: \"lm_head.weight\""));
    assert!(run.stdout.contains("Only in B: \"classifier.weight\""));
    let quant = biopsy(&["diff", &path("tiny_q8_0.gguf"), &path("tiny_q4_0.gguf")]);
    assert!(quant.stdout.contains("Q8_0>Q4_0"), "{}", quant.stdout);
}

#[test]
fn scalar_flag_selects_the_portable_kernels() {
    let (_dir, path) = demo();
    let run = biopsy(&["--scalar", "stats", &path("tiny.safetensors")]);
    assert!(run.stdout.contains("kernels: scalar"), "{}", run.stdout);
}

#[test]
fn bad_input_fails_cleanly() {
    let (dir, path) = demo();
    let missing = biopsy(&["inspect", &path("missing.safetensors")]);
    assert_eq!(missing.code, 1);
    assert!(missing.stderr.contains("cannot open"));
    let garbage = dir.path().join("garbage.bin");
    std::fs::write(&garbage, b"GGUF but not really, just text").unwrap();
    let run = biopsy(&["stats", &garbage.display().to_string()]);
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains("cannot parse"), "{}", run.stderr);
    assert!(!run.stderr.contains("panicked"));
    let zero_threads = biopsy(&["--threads", "0", "inspect", &path("tiny.safetensors")]);
    assert_eq!(zero_threads.code, 1);
}
