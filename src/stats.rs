//! Streaming statistics over decoded tensors, parallelized with Rayon.
//!
//! Each decoded buffer of 4 Ki values is summarized independently (count, mean,
//! M2 = sum of squared deviations, min, max) and the summaries are merged with
//! Chan et al.'s pairwise formula. Within a buffer we take two passes over the
//! cache-resident values (mean first, then deviations) instead of Welford's
//! one-pass update, which needs a division per element and cannot vectorize.

use rayon::prelude::*;

use crate::{
    decode::{self, CHUNK_ELEMENTS},
    kernels::{self, Backend},
    tensor::TensorView,
};

/// Mergeable summary of a set of values. Moments cover finite values only;
/// NaN and infinities are counted separately so one bad value cannot hide the rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// Every value seen, finite or not.
    pub count: u64,
    pub finite: u64,
    pub nan: u64,
    pub inf: u64,
    /// Values exactly equal to zero (+0 or -0).
    pub zeros: u64,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    /// Sum of squared deviations from the mean.
    pub m2: f64,
}

impl Default for Summary {
    fn default() -> Self {
        Summary {
            count: 0,
            finite: 0,
            nan: 0,
            inf: 0,
            zeros: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            m2: 0.0,
        }
    }
}

impl Summary {
    /// Summarize one chunk with two passes over the buffer.
    pub fn of(values: &[f32], backend: Backend) -> Summary {
        let scan = kernels::scan(backend, values);
        let mean = if scan.finite > 0 {
            scan.sum / scan.finite as f64
        } else {
            0.0
        };
        Summary {
            count: values.len() as u64,
            finite: scan.finite,
            nan: scan.nan,
            inf: values.len() as u64 - scan.finite - scan.nan,
            zeros: scan.zeros,
            min: scan.min,
            max: scan.max,
            mean,
            m2: kernels::sum_sq_dev(backend, values, mean),
        }
    }

    /// Combine two disjoint summaries as if their values had been seen together:
    /// `mean = mean_a + delta·n_b/n` and `M2 = M2_a + M2_b + delta²·n_a·n_b/n`.
    pub fn merge(&mut self, other: &Summary) {
        let (na, nb) = (self.finite as f64, other.finite as f64);
        let n = na + nb;
        if nb > 0.0 {
            let delta = other.mean - self.mean;
            self.mean += delta * nb / n;
            self.m2 += other.m2 + delta * delta * na * nb / n;
        }
        self.count += other.count;
        self.finite += other.finite;
        self.nan += other.nan;
        self.inf += other.inf;
        self.zeros += other.zeros;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Population standard deviation of the finite values.
    pub fn std(&self) -> f64 {
        if self.finite == 0 {
            0.0
        } else {
            (self.m2 / self.finite as f64).sqrt()
        }
    }

    /// Root mean square, sqrt(mean of x²), recovered as sqrt(M2/n + mean²).
    pub fn rms(&self) -> f64 {
        if self.finite == 0 {
            0.0
        } else {
            (self.m2 / self.finite as f64 + self.mean * self.mean).sqrt()
        }
    }

    /// Euclidean (L2) norm of the finite values.
    pub fn l2_norm(&self) -> f64 {
        self.rms() * (self.finite as f64).sqrt()
    }

    pub fn abs_max(&self) -> f32 {
        if self.finite == 0 {
            0.0
        } else {
            self.min.abs().max(self.max.abs())
        }
    }

    pub fn zero_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.zeros as f64 / self.count as f64
        }
    }
}

/// Summaries for every tensor, `None` for encodings biopsy does not decode.
pub fn summarize_all(tensors: &[TensorView], backend: Backend) -> Vec<Option<Summary>> {
    let jobs = decode::jobs(tensors);
    // `map_init` hands each Rayon worker its own scratch buffer, reused across the
    // jobs that worker processes. `collect` keeps job order, and buffers within a
    // job are merged in order, so the result is identical for 1 or 64 threads.
    let partials: Vec<Summary> = jobs
        .par_iter()
        .map_init(
            || Vec::with_capacity(CHUNK_ELEMENTS),
            |buffer, job| {
                let mut summary = Summary::default();
                decode::for_each_chunk(job.dtype, job.bytes, buffer, backend, |values| {
                    summary.merge(&Summary::of(values, backend));
                });
                summary
            },
        )
        .collect();
    let mut out: Vec<Option<Summary>> = tensors
        .iter()
        .map(|t| t.dtype.map(|_| Summary::default()))
        .collect();
    for (job, partial) in jobs.iter().zip(&partials) {
        if let Some(summary) = &mut out[job.tensor] {
            summary.merge(partial);
        }
    }
    out
}

/// Fixed-width bins over `[low, high]`; the last bin includes `high`.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub low: f32,
    pub high: f32,
    pub counts: Vec<u64>,
}

impl Histogram {
    pub fn new(low: f32, high: f32, bins: usize) -> Histogram {
        Histogram {
            low,
            high,
            counts: vec![0; bins.max(1)],
        }
    }

    /// Bin index for a finite value inside the range. Computed in f64 so the
    /// arithmetic cannot overflow when the range spans most of f32.
    fn bin(&self, x: f32) -> usize {
        let width = f64::from(self.high) - f64::from(self.low);
        if width <= 0.0 {
            return 0;
        }
        let position = (f64::from(x) - f64::from(self.low)) / width;
        ((position * self.counts.len() as f64) as usize).min(self.counts.len() - 1)
    }

    pub fn add_all(&mut self, values: &[f32]) {
        for &x in values {
            if x.is_finite() {
                let bin = self.bin(x);
                self.counts[bin] += 1;
            }
        }
    }

    pub fn merge(&mut self, other: &Histogram) {
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    /// Lower and upper edge of bin `i`.
    pub fn edges(&self, i: usize) -> (f64, f64) {
        let width = (f64::from(self.high) - f64::from(self.low)) / self.counts.len() as f64;
        let low = f64::from(self.low);
        (low + width * i as f64, low + width * (i + 1) as f64)
    }
}

/// Histograms of each tensor's finite values over that tensor's own [min, max].
/// This is a second pass over the mapped bytes: the range is unknown until the
/// summary pass has finished. Bin counts are integers, so merging is exact.
pub fn histograms(
    tensors: &[TensorView],
    summaries: &[Option<Summary>],
    bins: usize,
    backend: Backend,
) -> Vec<Option<Histogram>> {
    let empty: Vec<Option<Histogram>> = summaries
        .iter()
        .map(|s| {
            s.filter(|s| s.finite > 0)
                .map(|s| Histogram::new(s.min, s.max, bins))
        })
        .collect();
    let jobs: Vec<_> = decode::jobs(tensors)
        .into_iter()
        .filter(|job| empty[job.tensor].is_some())
        .collect();
    let partials: Vec<Histogram> = jobs
        .par_iter()
        .map_init(
            || Vec::with_capacity(CHUNK_ELEMENTS),
            |buffer, job| {
                let mut histogram = empty[job.tensor].clone().expect("filtered above");
                decode::for_each_chunk(job.dtype, job.bytes, buffer, backend, |values| {
                    histogram.add_all(values);
                });
                histogram
            },
        )
        .collect();
    let mut out = empty;
    for (job, partial) in jobs.iter().zip(&partials) {
        if let Some(histogram) = &mut out[job.tensor] {
            histogram.merge(partial);
        }
    }
    out
}

/// One row per tensor plus an "(all decoded)" row merging every summary.
pub fn render_table(tensors: &[TensorView], summaries: &[Option<Summary>]) -> String {
    let width = tensors
        .iter()
        .map(|t| t.name.len() + 2)
        .max()
        .unwrap_or(0)
        .max(16);
    let mut text = format!(
        "{:<width$} {:>6} {:>11} {:>11} {:>11} {:>11} {:>11} {:>7} {:>5} {:>5}\n",
        "tensor", "dtype", "elements", "mean", "std", "min", "max", "zeros", "nan", "inf"
    );
    let mut total = Summary::default();
    let row = |text: &mut String, name: &str, dtype: &str, s: &Summary| {
        let (min, max) = if s.finite > 0 {
            (format!("{:.4e}", s.min), format!("{:.4e}", s.max))
        } else {
            ("-".into(), "-".into())
        };
        text.push_str(&format!(
            "{name:<width$} {dtype:>6} {:>11} {:>11.4e} {:>11.4e} {min:>11} {max:>11} {:>6.2}% {:>5} {:>5}\n",
            s.count,
            s.mean,
            s.std(),
            s.zero_fraction() * 100.0,
            s.nan,
            s.inf
        ));
    };
    for (tensor, summary) in tensors.iter().zip(summaries) {
        let name = format!("{:?}", tensor.name);
        match summary {
            Some(s) => {
                row(&mut text, &name, &tensor.type_name, s);
                total.merge(s);
            }
            None => text.push_str(&format!(
                "{name:<width$} {:>6} {:>11}  not decoded\n",
                tensor.type_name, tensor.elements
            )),
        }
    }
    row(&mut text, "(all decoded)", "", &total);
    text
}

/// Draw a histogram as text: one line per bin, bar length proportional to count.
pub fn render_histogram(histogram: &Histogram, width: usize) -> String {
    let peak = histogram.counts.iter().copied().max().unwrap_or(0).max(1);
    let mut text = String::new();
    for (i, &count) in histogram.counts.iter().enumerate() {
        let (low, high) = histogram.edges(i);
        let close = if i + 1 == histogram.counts.len() {
            ']'
        } else {
            ')'
        };
        // Round up so any nonzero bin shows at least one mark.
        let bar = (count as u128 * width as u128).div_ceil(peak as u128) as usize;
        text.push_str(&format!(
            "  [{low:>11.4e}, {high:>11.4e}{close} {:<width$} {count}\n",
            "#".repeat(bar)
        ));
    }
    text
}
