//! Health checks: symptoms that a checkpoint is broken, untrained or unusual.
//!
//! Every check states its threshold in its message, and every threshold is a CLI
//! flag. A finding is a prompt to look closer, not a verdict: an all-zero bias
//! can be intentional, and one zero row in an embedding is often a padding token.
//!
//! Row semantics: a row is one index along the FIRST axis of the row-major shape,
//! i.e. one contiguous run of `row_len` values. For a PyTorch `[out, in]` linear
//! weight, a row is one output unit; for a `[vocab, dim]` embedding, one token;
//! for a `[out, in, kh, kw]` convolution, one filter. GGUF shapes are reversed into
//! this order first, so the same tensor gives the same rows in either format.
//! Tensors with fewer than two axes have no rows and skip the row checks.

use std::fmt::{self, Write};

use rayon::prelude::*;

use crate::{
    decode::{self, CHUNK_ELEMENTS},
    kernels::Backend,
    stats::{self, Summary},
    tensor::{DType, TensorView},
};

/// The largest finite F16 value. Larger values become infinity when cast to F16.
pub const F16_MAX: f32 = 65504.0;
/// Row indices listed per finding before eliding the rest.
const LISTED_ROWS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// WARN when any |value| exceeds this.
    pub max_abs: f32,
    /// WARN when more than this fraction of a tensor's values are exactly zero.
    pub max_zero_fraction: f64,
    /// Any all-zero row is reported (INFO); above this fraction of rows it is a WARN.
    pub max_dead_row_fraction: f64,
    /// WARN when a nonzero row's L2 norm is more than this factor above or below
    /// the tensor's median row norm.
    pub row_norm_ratio: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            max_abs: 1.0e4,
            max_zero_fraction: 0.5,
            max_dead_row_fraction: 0.01,
            row_norm_ratio: 10.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(match self {
            Severity::Info => "INFO",
            Severity::Warn => "WARN",
            Severity::Error => "ERROR",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub tensor: String,
    pub severity: Severity,
    /// Stable short identifier, e.g. "non-finite" or "dead-rows".
    pub check: &'static str,
    pub message: String,
}

#[derive(Debug)]
pub struct Report {
    pub thresholds: Thresholds,
    pub checked: usize,
    /// Tensors whose encoding biopsy does not decode: (name, type).
    pub skipped: Vec<(String, String)>,
    /// In tensor order; within a tensor, in check order.
    pub findings: Vec<Finding>,
}

impl Report {
    pub fn count(&self, severity: Severity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == severity)
            .count()
    }

    pub fn render(&self) -> String {
        let t = &self.thresholds;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Thresholds: max-abs={:e} max-zero-fraction={} max-dead-row-fraction={} row-norm-ratio={}",
            t.max_abs, t.max_zero_fraction, t.max_dead_row_fraction, t.row_norm_ratio
        );
        let _ = writeln!(
            out,
            "Rows are indices along the first axis of the row-major (PyTorch-order) shape."
        );
        for f in &self.findings {
            let _ = writeln!(
                out,
                "{:<5} {:?} [{}] {}",
                f.severity, f.tensor, f.check, f.message
            );
        }
        for (name, ty) in &self.skipped {
            let _ = writeln!(
                out,
                "SKIP  {name:?} [not-decoded] dtype {ty} is not decoded"
            );
        }
        let _ = writeln!(
            out,
            "Checked {} tensors, skipped {}: {} errors, {} warnings, {} info",
            self.checked,
            self.skipped.len(),
            self.count(Severity::Error),
            self.count(Severity::Warn),
            self.count(Severity::Info)
        );
        out
    }
}

fn percent(fraction: f64) -> String {
    format!("{:.1}%", fraction * 100.0)
}

fn list_rows(rows: &[usize]) -> String {
    let mut text: Vec<String> = rows
        .iter()
        .take(LISTED_ROWS)
        .map(usize::to_string)
        .collect();
    if rows.len() > LISTED_ROWS {
        text.push(format!("... {} more", rows.len() - LISTED_ROWS));
    }
    text.join(", ")
}

/// Per-row facts the row checks need.
struct RowFacts {
    dead: bool,
    norm: f64,
}

fn row_facts(tensor: &TensorView, dtype: DType, row_len: usize, backend: Backend) -> Vec<RowFacts> {
    let row_bytes = dtype.byte_range(0, row_len).end;
    tensor
        .bytes
        .par_chunks(row_bytes)
        .map_init(
            || Vec::with_capacity(CHUNK_ELEMENTS.min(row_len)),
            |buffer, row| {
                let mut summary = Summary::default();
                decode::for_each_chunk(dtype, row, buffer, backend, |values| {
                    summary.merge(&Summary::of(values, backend));
                });
                RowFacts {
                    dead: summary.zeros == summary.count,
                    norm: summary.l2_norm(),
                }
            },
        )
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn check_rows(
    tensor: &TensorView,
    dtype: DType,
    t: &Thresholds,
    backend: Backend,
    findings: &mut Vec<Finding>,
) {
    let Some(row_len) = tensor.row_len().and_then(|n| usize::try_from(n).ok()) else {
        return;
    };
    if row_len == 0 || tensor.bytes.is_empty() {
        return;
    }
    let facts = row_facts(tensor, dtype, row_len, backend);
    let mut finding = |severity, check, message| {
        findings.push(Finding {
            tensor: tensor.name.clone(),
            severity,
            check,
            message,
        })
    };

    let dead: Vec<usize> = (0..facts.len()).filter(|&i| facts[i].dead).collect();
    if !dead.is_empty() && dead.len() < facts.len() {
        let fraction = dead.len() as f64 / facts.len() as f64;
        let severity = if fraction > t.max_dead_row_fraction {
            Severity::Warn
        } else {
            Severity::Info
        };
        finding(
            severity,
            "dead-rows",
            format!(
                "{} of {} rows are all zero ({}; WARN above {}): rows {}",
                dead.len(),
                facts.len(),
                percent(fraction),
                percent(t.max_dead_row_fraction),
                list_rows(&dead)
            ),
        );
    }

    let mut norms: Vec<f64> = facts.iter().filter(|f| !f.dead).map(|f| f.norm).collect();
    if norms.len() < 2 {
        return;
    }
    let typical = median(&mut norms);
    if typical <= 0.0 {
        return;
    }
    let (low, high) = (typical / t.row_norm_ratio, typical * t.row_norm_ratio);
    let outliers: Vec<usize> = (0..facts.len())
        .filter(|&i| !facts[i].dead && (facts[i].norm < low || facts[i].norm > high))
        .collect();
    if !outliers.is_empty() {
        let detail: Vec<String> = outliers
            .iter()
            .take(LISTED_ROWS)
            .map(|&i| format!("{i} (x{:.3})", facts[i].norm / typical))
            .collect();
        finding(
            Severity::Warn,
            "row-norm",
            format!(
                "{} of {} rows have an L2 norm outside [median/{r}, median*{r}] (median {typical:.4e}): rows {}{}",
                outliers.len(),
                facts.len(),
                detail.join(", "),
                if outliers.len() > LISTED_ROWS {
                    ", ..."
                } else {
                    ""
                },
                r = t.row_norm_ratio,
            ),
        );
    }
}

/// Run every check over every decodable tensor.
pub fn check(tensors: &[TensorView], thresholds: &Thresholds, backend: Backend) -> Report {
    let summaries = stats::summarize_all(tensors, backend);
    let mut report = Report {
        thresholds: *thresholds,
        checked: 0,
        skipped: Vec::new(),
        findings: Vec::new(),
    };
    let t = thresholds;
    for (tensor, summary) in tensors.iter().zip(&summaries) {
        let (Some(dtype), Some(s)) = (tensor.dtype, summary) else {
            report
                .skipped
                .push((tensor.name.clone(), tensor.type_name.clone()));
            continue;
        };
        report.checked += 1;
        if s.count == 0 {
            continue;
        }
        let mut finding = |severity, check, message| {
            report.findings.push(Finding {
                tensor: tensor.name.clone(),
                severity,
                check,
                message,
            })
        };
        if s.nan + s.inf > 0 {
            finding(
                Severity::Error,
                "non-finite",
                format!("{} NaN and {} infinite values (none allowed)", s.nan, s.inf),
            );
        }
        if s.zeros == s.count {
            finding(
                Severity::Warn,
                "all-zero",
                format!(
                    "all {} values are zero (untrained or a placeholder?)",
                    s.count
                ),
            );
        } else if s.finite == s.count && s.count > 1 && s.min == s.max {
            finding(
                Severity::Warn,
                "constant",
                format!(
                    "all {} values equal {} (an initial value never trained?)",
                    s.count, s.min
                ),
            );
        } else if s.zero_fraction() > t.max_zero_fraction {
            finding(
                Severity::Warn,
                "sparse",
                format!(
                    "{} of values are exactly zero (WARN above {})",
                    percent(s.zero_fraction()),
                    percent(t.max_zero_fraction)
                ),
            );
        }
        if s.abs_max() > t.max_abs {
            finding(
                Severity::Warn,
                "large-magnitude",
                format!("max |x| = {:e} exceeds {:e}", s.abs_max(), t.max_abs),
            );
        }
        if matches!(dtype, DType::F32 | DType::BF16) && s.abs_max() > F16_MAX {
            finding(
                Severity::Info,
                "f16-range",
                format!(
                    "max |x| = {:e} exceeds the F16 maximum {F16_MAX}; casting to F16 would overflow",
                    s.abs_max()
                ),
            );
        }
        check_rows(tensor, dtype, t, backend, &mut report.findings);
    }
    report
}
