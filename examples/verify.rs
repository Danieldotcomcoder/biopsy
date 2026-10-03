//! `cargo verify`: the one command that checks the whole project.
//!
//! The alias lives in `.cargo/config.toml`. Steps, in order:
//!  1. formatting (`cargo fmt --check`) and lints (`cargo clippy -D warnings`)
//!  2. every test, in debug (overflow checks on) and release (SIMD as shipped)
//!  3. the release binary on freshly generated demo files, every command
//!  4. the kernel benchmark in a short run: it refuses to time SIMD kernels whose
//!     output differs from the scalar reference, so it doubles as a correctness check
//!  5. any real models you downloaded into `models/`: inspect + health must not fail
//!
//! Exit status 0 only if every step passed.
use std::{
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Instant,
};

const DEMO_DIR: &str = "target/verify-demo";

struct Verifier {
    root: PathBuf,
    cargo: String,
    passed: usize,
    failed: Vec<String>,
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Verifier {
    fn run(&self, program: &str, args: &[&str]) -> Output {
        match Command::new(program)
            .args(args)
            .current_dir(&self.root)
            .output()
        {
            Ok(o) => Output {
                code: o.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
            },
            Err(e) => Output {
                code: -1,
                stdout: String::new(),
                stderr: format!("cannot start {program}: {e}"),
            },
        }
    }

    fn record(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            self.passed += 1;
            println!("  PASS  {name}{detail}");
        } else {
            self.failed.push(name.to_owned());
            println!("  FAIL  {name}{detail}");
        }
    }

    /// Run a cargo subcommand; on failure show the tail of its output.
    fn cargo_step(&mut self, name: &str, args: &[&str]) -> Output {
        let start = Instant::now();
        let out = self.run(&self.cargo.clone(), args);
        let ok = out.code == 0;
        let detail = format!("  ({:.1}s)", start.elapsed().as_secs_f64());
        self.record(name, ok, &detail);
        if !ok {
            let text = format!("{}{}", out.stdout, out.stderr);
            let lines: Vec<&str> = text.lines().collect();
            for line in &lines[lines.len().saturating_sub(30)..] {
                println!("        | {line}");
            }
        }
        out
    }

    /// Run the release binary and check its exit code and output.
    fn biopsy(&mut self, name: &str, args: &[&str], code: i32, expect: &[&str]) {
        let exe = self.root.join(format!(
            "target/release/biopsy{}",
            std::env::consts::EXE_SUFFIX
        ));
        let out = self.run(&exe.display().to_string(), args);
        let missing: Vec<&&str> = expect
            .iter()
            .filter(|e| !out.stdout.contains(**e))
            .collect();
        let ok = out.code == code && missing.is_empty();
        self.record(name, ok, "");
        if !ok {
            println!(
                "        | exit {} (expected {code}); missing {missing:?}",
                out.code
            );
            for line in out.stderr.lines().chain(out.stdout.lines()).take(15) {
                println!("        | {line}");
            }
        }
    }
}

fn count_tests(output: &Output) -> usize {
    // "test result: ok. 13 passed; ..." once per test binary.
    output
        .stdout
        .lines()
        .filter_map(|l| l.strip_prefix("test result: ok. "))
        .filter_map(|l| l.split(' ').next()?.parse::<usize>().ok())
        .sum()
}

fn demo(name: &str) -> String {
    format!("{DEMO_DIR}/{name}")
}

fn main() -> ExitCode {
    let started = Instant::now();
    let mut v = Verifier {
        root: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        cargo: std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()),
        passed: 0,
        failed: Vec::new(),
    };
    let rustc = v.run("rustc", &["--version"]).stdout;
    println!("biopsy verify  ({})", rustc.trim());
    println!(
        "kernels on this CPU: {}\n",
        biopsy::kernels::Backend::detect().name()
    );

    println!("1. Static checks");
    v.cargo_step("cargo fmt --check", &["fmt", "--check"]);
    v.cargo_step(
        "cargo clippy --all-targets -D warnings",
        &["clippy", "--all-targets", "--quiet", "--", "-D", "warnings"],
    );

    println!("2. Tests");
    let debug = v.cargo_step("cargo test (debug, overflow checks)", &["test", "--quiet"]);
    let release = v.cargo_step("cargo test --release", &["test", "--release", "--quiet"]);
    println!(
        "        {} tests in debug, {} in release",
        count_tests(&debug),
        count_tests(&release)
    );

    println!("3. The CLI on generated demo files");
    v.cargo_step("cargo build --release", &["build", "--release", "--quiet"]);
    let dir = v.root.join(DEMO_DIR);
    let wrote = std::fs::create_dir_all(&dir).is_ok()
        && biopsy::fixtures::demo_files()
            .into_iter()
            .all(|(name, bytes)| std::fs::write(dir.join(name), bytes).is_ok());
    v.record("write demo files", wrote, &format!("  ({DEMO_DIR})"));
    v.biopsy(
        "inspect safetensors",
        &["inspect", &demo("tiny.safetensors")],
        0,
        &[
            "shape=[2, 3]",
            "Total parameters (stored tensor elements): 8",
        ],
    );
    v.biopsy(
        "inspect GGUF",
        &["inspect", &demo("tiny_q4_0.gguf")],
        0,
        &[
            "GGUF v3",
            "alignment 64 bytes",
            "type=Q4_0 dims=[64, 128] shape=[128, 64]",
        ],
    );
    v.biopsy(
        "stats with histogram (Q8_0, F16, F32; Q4_K skipped)",
        &["stats", &demo("tiny_q8_0.gguf"), "--histogram"],
        0,
        &["Q8_0", "not decoded", "Histogram of"],
    );
    v.biopsy(
        "stats --scalar --threads 1",
        &[
            "--scalar",
            "--threads",
            "1",
            "stats",
            &demo("encoder.safetensors"),
        ],
        0,
        &["kernels: scalar; threads: 1", "BF16"],
    );
    v.biopsy(
        "health finds every planted problem (exit 2)",
        &["health", &demo("sick.safetensors")],
        2,
        &[
            "[non-finite]",
            "[dead-rows]",
            "[row-norm]",
            "[all-zero]",
            "[constant]",
            "[sparse]",
            "[large-magnitude]",
            "1 errors, 7 warnings, 1 info",
        ],
    );
    v.biopsy(
        "health passes a healthy model (exit 0)",
        &["health", &demo("encoder.safetensors")],
        0,
        &["0 errors, 0 warnings"],
    );
    v.biopsy(
        "diff: unmatched, mismatched, identical, changed",
        &[
            "diff",
            &demo("encoder.safetensors"),
            &demo("encoder_finetuned.safetensors"),
        ],
        0,
        &[
            "Only in A",
            "Only in B",
            "Shape mismatch",
            "Identical: 1; changed: 2",
        ],
    );
    v.biopsy(
        "diff across quantizations (Q8_0 vs Q4_0)",
        &["diff", &demo("tiny_q8_0.gguf"), &demo("tiny_q4_0.gguf")],
        0,
        &["Q8_0>Q4_0", "Overall"],
    );
    v.biopsy(
        "malformed input fails cleanly (exit 1)",
        &["inspect", "Cargo.toml"],
        1,
        &[],
    );

    println!("4. Kernel benchmark (short run, SIMD checked against scalar)");
    let bench = v.cargo_step(
        "cargo run --release --example bench",
        &[
            "run",
            "--release",
            "--quiet",
            "--example",
            "bench",
            "--",
            "--elements",
            "2097152",
            "--repeats",
            "3",
        ],
    );
    for line in bench
        .stdout
        .lines()
        .filter(|l| l.contains("summary") || l.contains("threads "))
    {
        println!("        {line}");
    }

    println!("5. Real models in models/ (optional)");
    let demo_names: Vec<&str> = biopsy::fixtures::demo_files()
        .iter()
        .map(|(n, _)| *n)
        .collect();
    let mut real: Vec<PathBuf> = std::fs::read_dir(v.root.join("models"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("safetensors" | "gguf")
            )
        })
        .filter(|p| !demo_names.contains(&p.file_name().and_then(|n| n.to_str()).unwrap_or("")))
        .collect();
    real.sort();
    if real.is_empty() {
        println!("        none found; see README \"Real testing models\" to add some");
    }
    for path in &real {
        let shown = Path::new("models").join(path.file_name().unwrap_or_default());
        let shown = shown.display().to_string();
        v.biopsy(
            &format!("inspect {shown}"),
            &["inspect", &shown],
            0,
            &["Tensors:"],
        );
        // Health may legitimately warn (exit 2); only a crash or parse failure fails.
        let exe = v.root.join(format!(
            "target/release/biopsy{}",
            std::env::consts::EXE_SUFFIX
        ));
        let out = v.run(&exe.display().to_string(), &["health", &shown]);
        let summary = out.stdout.lines().last().unwrap_or("").to_owned();
        v.record(
            &format!("health {shown}"),
            out.code == 0 || out.code == 2,
            &format!("  ({summary})"),
        );
    }

    let seconds = started.elapsed().as_secs_f64();
    println!();
    if v.failed.is_empty() {
        println!("VERIFY PASSED: {} checks in {seconds:.0}s", v.passed);
        ExitCode::SUCCESS
    } else {
        println!(
            "VERIFY FAILED: {} of {} checks failed: {}",
            v.failed.len(),
            v.passed + v.failed.len(),
            v.failed.join(", ")
        );
        ExitCode::FAILURE
    }
}
