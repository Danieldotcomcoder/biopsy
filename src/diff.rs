//! Compare two models tensor by tensor.
//!
//! Matching is by name (after optional prefix stripping), then by shape. Only
//! same-name, same-shape pairs are compared numerically; dtypes may differ, since
//! both sides are decoded to F32. That makes "Q8_0 vs Q4_0 of the same model"
//! a direct measurement of quantization error.

use std::{collections::HashMap, fmt::Write};

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::{
    decode::{self, CHUNK_ELEMENTS, JOB_ELEMENTS},
    kernels::{self, Backend, DiffSums},
    tensor::{DType, TensorView},
};

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Removed from the start of names on both sides before matching, e.g.
    /// "model." when one checkpoint wraps the other. The first match wins.
    pub strip_prefixes: Vec<String>,
}

/// Accumulated comparison of one tensor pair (or of every pair together).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub elements: u64,
    pub sums: DiffSums,
}

impl Metrics {
    pub fn merge(&mut self, other: &Metrics) {
        let (a, b) = (&mut self.sums, &other.sums);
        self.elements += other.elements;
        a.finite_pairs += b.finite_pairs;
        a.equal += b.equal;
        a.nonfinite_mismatch += b.nonfinite_mismatch;
        a.sum_sq_diff += b.sum_sq_diff;
        a.sum_abs_diff += b.sum_abs_diff;
        a.max_abs_diff = a.max_abs_diff.max(b.max_abs_diff);
        a.sum_aa += b.sum_aa;
        a.sum_bb += b.sum_bb;
        a.sum_ab += b.sum_ab;
    }

    pub fn identical(&self) -> bool {
        self.sums.equal == self.elements
    }

    /// Fraction of elements whose values differ.
    pub fn changed_fraction(&self) -> f64 {
        if self.elements == 0 {
            0.0
        } else {
            1.0 - self.sums.equal as f64 / self.elements as f64
        }
    }

    pub fn mean_abs(&self) -> f64 {
        self.sums.sum_abs_diff / (self.sums.finite_pairs.max(1)) as f64
    }

    /// Root-mean-square difference.
    pub fn rmse(&self) -> f64 {
        (self.sums.sum_sq_diff / (self.sums.finite_pairs.max(1)) as f64).sqrt()
    }

    /// ||a - b|| / ||a||: the change relative to A's own size. Comparable across
    /// tensors of different scales, so diff sorts by it.
    pub fn rel_l2(&self) -> f64 {
        let s = &self.sums;
        if s.sum_aa > 0.0 {
            (s.sum_sq_diff / s.sum_aa).sqrt()
        } else if s.sum_sq_diff == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    }

    /// Cosine similarity a·b / (||a|| ||b||); `None` when either side is all zero.
    pub fn cosine(&self) -> Option<f64> {
        let s = &self.sums;
        let denominator = (s.sum_aa * s.sum_bb).sqrt();
        (denominator > 0.0).then(|| s.sum_ab / denominator)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Compared(Metrics),
    ShapeMismatch { a: Vec<u64>, b: Vec<u64> },
    NotDecoded,
}

#[derive(Clone, Debug)]
pub struct Pair {
    /// The matched (normalized) name.
    pub name: String,
    pub a_type: String,
    pub b_type: String,
    pub outcome: Outcome,
}

#[derive(Debug)]
pub struct Report {
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
    /// In A's tensor order.
    pub pairs: Vec<Pair>,
}

fn normalize<'n>(name: &'n str, options: &Options) -> &'n str {
    options
        .strip_prefixes
        .iter()
        .find_map(|p| name.strip_prefix(p.as_str()))
        .unwrap_or(name)
}

fn index_by_name<'t>(
    tensors: &'t [TensorView],
    options: &Options,
    side: &str,
) -> Result<HashMap<&'t str, usize>> {
    let mut index = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        let key = normalize(&t.name, options);
        if let Some(previous) = index.insert(key, i) {
            bail!(
                "model {side}: {:?} and {:?} both become {key:?} after prefix stripping",
                tensors[previous].name,
                t.name
            );
        }
    }
    Ok(index)
}

struct Job<'a> {
    pair: usize,
    a: (DType, &'a [u8]),
    b: (DType, &'a [u8]),
}

pub fn diff(
    a: &[TensorView],
    b: &[TensorView],
    options: &Options,
    backend: Backend,
) -> Result<Report> {
    let a_index = index_by_name(a, options, "A")?;
    let b_index = index_by_name(b, options, "B")?;
    let mut report = Report {
        only_a: Vec::new(),
        only_b: Vec::new(),
        pairs: Vec::new(),
    };
    let mut jobs = Vec::new();
    for ta in a {
        let name = normalize(&ta.name, options);
        let Some(&j) = b_index.get(name) else {
            report.only_a.push(ta.name.clone());
            continue;
        };
        let tb = &b[j];
        let outcome = match (ta.dtype, tb.dtype) {
            _ if ta.shape != tb.shape => Outcome::ShapeMismatch {
                a: ta.shape.clone(),
                b: tb.shape.clone(),
            },
            (Some(da), Some(db)) => {
                // Same shape means the same element count; both sides are split
                // at identical element boundaries (a multiple of every block size).
                let elements = decode::element_count(da, ta.bytes);
                for range in decode::chunk_ranges(elements, JOB_ELEMENTS) {
                    jobs.push(Job {
                        pair: report.pairs.len(),
                        a: (da, &ta.bytes[da.byte_range(range.start, range.end)]),
                        b: (db, &tb.bytes[db.byte_range(range.start, range.end)]),
                    });
                }
                Outcome::Compared(Metrics::default())
            }
            _ => Outcome::NotDecoded,
        };
        report.pairs.push(Pair {
            name: name.to_owned(),
            a_type: ta.type_name.clone(),
            b_type: tb.type_name.clone(),
            outcome,
        });
    }
    for tb in b {
        if !a_index.contains_key(normalize(&tb.name, options)) {
            report.only_b.push(tb.name.clone());
        }
    }

    let partials: Vec<Metrics> = jobs
        .par_iter()
        .map_init(
            || {
                (
                    Vec::with_capacity(CHUNK_ELEMENTS),
                    Vec::with_capacity(CHUNK_ELEMENTS),
                )
            },
            |buffers, job| {
                let mut metrics = Metrics::default();
                decode::for_each_chunk_pair(job.a, job.b, buffers, backend, |a, b| {
                    metrics.merge(&Metrics {
                        elements: a.len() as u64,
                        sums: kernels::diff_sums(backend, a, b),
                    });
                });
                metrics
            },
        )
        .collect();
    for (job, partial) in jobs.iter().zip(&partials) {
        if let Outcome::Compared(metrics) = &mut report.pairs[job.pair].outcome {
            metrics.merge(partial);
        }
    }
    Ok(report)
}

impl Report {
    /// Every compared pair merged: the whole-model change.
    pub fn overall(&self) -> Metrics {
        let mut total = Metrics::default();
        for pair in &self.pairs {
            if let Outcome::Compared(m) = &pair.outcome {
                total.merge(m);
            }
        }
        total
    }

    /// Render the report. `top` limits the changed-tensor table (0 = all rows).
    pub fn render(&self, top: usize) -> String {
        let mut out = String::new();
        let compared: Vec<(&Pair, &Metrics)> = self
            .pairs
            .iter()
            .filter_map(|p| match &p.outcome {
                Outcome::Compared(m) => Some((p, m)),
                _ => None,
            })
            .collect();
        let mismatched: Vec<&Pair> = self
            .pairs
            .iter()
            .filter(|p| matches!(p.outcome, Outcome::ShapeMismatch { .. }))
            .collect();
        let undecoded: Vec<&Pair> = self
            .pairs
            .iter()
            .filter(|p| p.outcome == Outcome::NotDecoded)
            .collect();
        let _ = writeln!(
            out,
            "Matched by name: {}; compared: {}; shape mismatches: {}; not decoded: {}; only in A: {}; only in B: {}",
            self.pairs.len(),
            compared.len(),
            mismatched.len(),
            undecoded.len(),
            self.only_a.len(),
            self.only_b.len()
        );
        for name in &self.only_a {
            let _ = writeln!(out, "Only in A: {name:?}");
        }
        for name in &self.only_b {
            let _ = writeln!(out, "Only in B: {name:?}");
        }
        for p in &mismatched {
            if let Outcome::ShapeMismatch { a, b } = &p.outcome {
                let _ = writeln!(out, "Shape mismatch: {:?}: {a:?} vs {b:?}", p.name);
            }
        }
        for p in &undecoded {
            let _ = writeln!(
                out,
                "Not decoded: {:?}: {} vs {}",
                p.name, p.a_type, p.b_type
            );
        }

        let identical = compared.iter().filter(|(_, m)| m.identical()).count();
        let mut changed: Vec<_> = compared.iter().filter(|(_, m)| !m.identical()).collect();
        // Largest relative change first; names break ties so output is stable.
        changed.sort_by(|(pa, ma), (pb, mb)| {
            mb.rel_l2()
                .total_cmp(&ma.rel_l2())
                .then_with(|| pa.name.cmp(&pb.name))
        });
        let _ = writeln!(out, "Identical: {identical}; changed: {}", changed.len());
        if !changed.is_empty() {
            let shown = if top == 0 {
                changed.len()
            } else {
                top.min(changed.len())
            };
            let width = changed
                .iter()
                .take(shown)
                .map(|(p, _)| p.name.len() + 2)
                .max()
                .unwrap_or(4)
                .max(6);
            let _ = writeln!(
                out,
                "\nChanged tensors, largest relative L2 change first (showing {shown} of {}):",
                changed.len()
            );
            let _ = writeln!(
                out,
                "{:<width$} {:>11} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10}",
                "tensor", "types", "elements", "changed", "max|d|", "rmse", "rel_l2", "cosine"
            );
            for (p, m) in changed.iter().take(shown) {
                let types = if p.a_type == p.b_type {
                    p.a_type.clone()
                } else {
                    format!("{}>{}", p.a_type, p.b_type)
                };
                let _ = writeln!(
                    out,
                    "{:<width$} {:>11} {:>10} {:>7.2}% {:>10.3e} {:>10.3e} {:>10.3e} {:>10}",
                    format!("{:?}", p.name),
                    types,
                    m.elements,
                    m.changed_fraction() * 100.0,
                    m.sums.max_abs_diff,
                    m.rmse(),
                    m.rel_l2(),
                    m.cosine().map_or("-".into(), |c| format!("{c:.6}"))
                );
                if m.sums.nonfinite_mismatch > 0 {
                    let _ = writeln!(
                        out,
                        "  ^ {} positions are NaN/Inf on one side only",
                        m.sums.nonfinite_mismatch
                    );
                }
            }
        }
        if !compared.is_empty() {
            let all = self.overall();
            let _ = writeln!(
                out,
                "\nOverall ({} compared elements): changed {:.2}%, rel_l2 {:.3e}, cosine {}",
                all.elements,
                all.changed_fraction() * 100.0,
                all.rel_l2(),
                all.cosine().map_or("-".into(), |c| format!("{c:.6}"))
            );
        }
        out
    }
}
