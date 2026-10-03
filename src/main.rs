use std::{
    fs::File,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, ensure};
use biopsy::{
    diff, health, inspect,
    kernels::Backend,
    model::{self, Model},
    stats,
    tensor::TensorView,
};
use clap::{Parser, Subcommand};
use memmap2::{Mmap, MmapOptions};

/// Learn model file formats by inspecting their raw bytes.
///
/// Reads safetensors and GGUF files. Every command memory-maps its inputs;
/// keep them unchanged while biopsy is running.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Worker threads for decoding (default: one per logical CPU). Results are
    /// identical for any thread count.
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Use the portable scalar kernels even if the CPU supports SIMD ones.
    #[arg(long, global = true)]
    scalar: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Describe a file's header without reading weight values.
    Inspect { file: PathBuf },
    /// Per-tensor statistics of decoded values (F32, F16, BF16, Q8_0, Q4_0).
    Stats {
        file: PathBuf,
        /// Only tensors whose name contains this text.
        #[arg(long)]
        filter: Option<String>,
        /// Also draw a histogram of each tensor's values.
        #[arg(long)]
        histogram: bool,
        /// Histogram bins.
        #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u16).range(1..=1000))]
        bins: u16,
    },
    /// Look for NaN/Inf, dead rows, outlier rows and other symptoms.
    /// Exit status 2 if any ERROR is found (or any WARN with --strict).
    Health {
        file: PathBuf,
        /// Only tensors whose name contains this text.
        #[arg(long)]
        filter: Option<String>,
        /// WARN when any |value| exceeds this.
        #[arg(long, default_value_t = health::Thresholds::default().max_abs)]
        max_abs: f32,
        /// WARN when more than this fraction of values are exactly zero.
        #[arg(long, default_value_t = health::Thresholds::default().max_zero_fraction)]
        max_zero_fraction: f64,
        /// All-zero rows are INFO; above this fraction of rows they are WARN.
        #[arg(long, default_value_t = health::Thresholds::default().max_dead_row_fraction)]
        max_dead_row_fraction: f64,
        /// WARN when a row's L2 norm is this many times above or below the median.
        #[arg(long, default_value_t = health::Thresholds::default().row_norm_ratio)]
        row_norm_ratio: f64,
        /// Treat warnings as failures for the exit status.
        #[arg(long)]
        strict: bool,
    },
    /// Compare two models: unmatched names, shape mismatches, numeric change.
    Diff {
        a: PathBuf,
        b: PathBuf,
        /// Strip this prefix from names on both sides before matching (repeatable).
        #[arg(long = "strip-prefix")]
        strip_prefixes: Vec<String>,
        /// Changed tensors to list, largest change first (0 = all).
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
}

/// Memory-map a file for reading.
fn map(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    ensure!(
        file.metadata()?.len() >= 8,
        "{}: file is too short to be safetensors or GGUF",
        path.display()
    );
    // SAFETY: the CLI requires the input to remain unchanged for this mapping's
    // lifetime. The mapping is read-only and never escapes its caller's scope.
    // Read-only mapping alone does NOT prevent another process modifying a file.
    unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("cannot memory-map {}", path.display()))
}

fn parse<'a>(path: &Path, bytes: &'a [u8]) -> Result<Model<'a>> {
    model::parse(bytes).with_context(|| format!("cannot parse {}", path.display()))
}

fn filtered<'a>(tensors: Vec<TensorView<'a>>, filter: &Option<String>) -> Vec<TensorView<'a>> {
    match filter {
        Some(text) => tensors
            .into_iter()
            .filter(|t| t.name.contains(text.as_str()))
            .collect(),
        None => tensors,
    }
}

fn run(cli: Cli, out: &mut impl Write) -> Result<ExitCode> {
    if let Some(threads) = cli.threads {
        ensure!(threads > 0, "--threads must be at least 1");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("cannot configure the thread pool")?;
    }
    let backend = if cli.scalar {
        Backend::Scalar
    } else {
        Backend::detect()
    };
    match cli.command {
        Command::Inspect { file } => {
            let map = map(&file)?;
            write!(
                out,
                "{}",
                inspect::render(&file.display().to_string(), &map)?
            )?;
        }
        Command::Stats {
            file,
            filter,
            histogram,
            bins,
        } => {
            let map = map(&file)?;
            let model = parse(&file, &map)?;
            let tensors = filtered(model.tensors, &filter);
            writeln!(
                out,
                "File: {} ({}, {} tensors); kernels: {}; threads: {}",
                file.display(),
                model.format.name(),
                tensors.len(),
                backend.name(),
                rayon::current_num_threads()
            )?;
            let summaries = stats::summarize_all(&tensors, backend);
            write!(out, "{}", stats::render_table(&tensors, &summaries))?;
            if histogram {
                let histograms = stats::histograms(&tensors, &summaries, bins.into(), backend);
                for (tensor, histogram) in tensors.iter().zip(&histograms) {
                    if let Some(histogram) = histogram {
                        writeln!(out, "\nHistogram of {:?} (finite values):", tensor.name)?;
                        write!(out, "{}", stats::render_histogram(histogram, 40))?;
                    }
                }
            }
        }
        Command::Health {
            file,
            filter,
            max_abs,
            max_zero_fraction,
            max_dead_row_fraction,
            row_norm_ratio,
            strict,
        } => {
            let map = map(&file)?;
            let model = parse(&file, &map)?;
            let tensors = filtered(model.tensors, &filter);
            let thresholds = health::Thresholds {
                max_abs,
                max_zero_fraction,
                max_dead_row_fraction,
                row_norm_ratio,
            };
            writeln!(
                out,
                "File: {} ({}, {} tensors)",
                file.display(),
                model.format.name(),
                tensors.len()
            )?;
            let report = health::check(&tensors, &thresholds, backend);
            write!(out, "{}", report.render())?;
            let failing = report.count(health::Severity::Error)
                + if strict {
                    report.count(health::Severity::Warn)
                } else {
                    0
                };
            if failing > 0 {
                return Ok(ExitCode::from(2));
            }
        }
        Command::Diff {
            a,
            b,
            strip_prefixes,
            top,
        } => {
            let (map_a, map_b) = (map(&a)?, map(&b)?);
            let (model_a, model_b) = (parse(&a, &map_a)?, parse(&b, &map_b)?);
            writeln!(
                out,
                "A: {} ({}, {} tensors)",
                a.display(),
                model_a.format.name(),
                model_a.tensors.len()
            )?;
            writeln!(
                out,
                "B: {} ({}, {} tensors)",
                b.display(),
                model_b.format.name(),
                model_b.tensors.len()
            )?;
            let options = diff::Options { strip_prefixes };
            let report = diff::diff(&model_a.tensors, &model_b.tensors, &options, backend)?;
            write!(out, "{}", report.render(top))?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut out = io::BufWriter::new(io::stdout().lock());
    let result = run(cli, &mut out).and_then(|code| {
        out.flush()?;
        Ok(code)
    });
    match result {
        Ok(code) => code,
        Err(error) => {
            // Show any partial report before the error message.
            let _ = out.flush();
            if error
                .downcast_ref::<io::Error>()
                .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
            {
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
