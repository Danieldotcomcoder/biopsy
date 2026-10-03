use biopsy::{
    decode::JOB_ELEMENTS,
    fixtures::{self, Rng, Tensor},
    kernels::Backend,
    model,
    stats::{self, Histogram, Summary},
    tensor::{DType, TensorView},
};

fn view(values: &[f32]) -> (Vec<u8>, Vec<u64>) {
    (fixtures::f32_bytes(values), vec![values.len() as u64])
}

fn f32_view<'a>(bytes: &'a [u8], shape: &[u64]) -> TensorView<'a> {
    TensorView {
        name: "t".into(),
        type_name: "F32".into(),
        dtype: Some(DType::F32),
        shape: shape.to_vec(),
        elements: shape.iter().product(),
        bytes,
    }
}

/// Plain two-pass statistics in f64 over the finite values: the oracle.
fn naive(values: &[f32]) -> (f64, f64, f32, f32) {
    let finite: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| f64::from(v))
        .collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    (mean, var.sqrt(), min, max)
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1e-30)
}

#[test]
fn chunked_parallel_summary_matches_the_naive_oracle() {
    // Several chunks plus a ragged tail, with an offset mean that would expose a
    // naive sum-of-squares variance formula.
    let mut values: Vec<f32> = Rng::new(5)
        .normals(3 * JOB_ELEMENTS + 1234, 0.5)
        .iter()
        .map(|v| v + 1000.0)
        .collect();
    values[10] = f32::NAN;
    values[70_000] = f32::NEG_INFINITY;
    values[100] = 0.0;
    let (bytes, shape) = view(&values);
    let summary = stats::summarize_all(&[f32_view(&bytes, &shape)], Backend::Scalar)[0].unwrap();
    let (mean, std, min, max) = naive(&values);
    assert_eq!(summary.count, values.len() as u64);
    assert_eq!((summary.nan, summary.inf, summary.zeros), (1, 1, 1));
    assert_eq!(summary.finite, values.len() as u64 - 2);
    assert_eq!((summary.min, summary.max), (min, max));
    assert!(
        close(summary.mean, mean, 1e-12),
        "{} vs {mean}",
        summary.mean
    );
    assert!(
        close(summary.std(), std, 1e-9),
        "{} vs {std}",
        summary.std()
    );
}

#[test]
fn merging_equals_summarizing_the_concatenation() {
    let mut rng = Rng::new(9);
    let (a, b) = (rng.normals(1000, 2.0), rng.normals(37, 0.1));
    let mut merged = Summary::of(&a, Backend::Scalar);
    merged.merge(&Summary::of(&b, Backend::Scalar));
    let whole = Summary::of(&[a, b].concat(), Backend::Scalar);
    assert_eq!(merged.count, whole.count);
    assert_eq!((merged.min, merged.max), (whole.min, whole.max));
    assert!(close(merged.mean, whole.mean, 1e-12));
    assert!(close(merged.m2, whole.m2, 1e-12));
    // Merging an empty summary changes nothing.
    let mut same = whole;
    same.merge(&Summary::default());
    assert_eq!(same, whole);
}

#[test]
fn results_are_bit_identical_for_any_thread_count() {
    let values = Rng::new(4).normals(5 * JOB_ELEMENTS + 77, 1.0);
    let (bytes, shape) = view(&values);
    let tensors = [f32_view(&bytes, &shape)];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| stats::summarize_all(&tensors, Backend::Scalar))
    };
    let one = run(1);
    for threads in [2, 3, 8] {
        assert_eq!(run(threads), one, "{threads} threads");
    }
}

#[test]
fn every_decodable_dtype_summarizes_from_a_real_file() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
    let safetensors = fixtures::safetensors(
        &[],
        &[
            Tensor::encode("f32", "F32", &[64], &values),
            Tensor::encode("f16", "F16", &[64], &values),
            Tensor::encode("bf16", "BF16", &[64], &values),
            Tensor::new("ints", "I32", &[2], vec![0; 8]),
        ],
    );
    let gguf = fixtures::gguf(
        &[],
        &[
            Tensor::encode("q8", "Q8_0", &[64], &values),
            Tensor::encode("q4", "Q4_0", &[64], &values),
        ],
    );
    // Every value is a multiple of 1/8 within [-4, 3.875]: exact in F16 and BF16.
    let (mean, std, min, max) = naive(&values);
    let model = model::parse(&safetensors).unwrap();
    let summaries = stats::summarize_all(&model.tensors, Backend::Scalar);
    for (tensor, summary) in model.tensors.iter().zip(&summaries) {
        if tensor.name == "ints" {
            assert!(summary.is_none(), "I32 is not decoded");
            continue;
        }
        let s = summary.unwrap();
        assert_eq!((s.min, s.max, s.count), (min, max, 64), "{}", tensor.name);
        assert!(close(s.mean, mean, 1e-12) && close(s.std(), std, 1e-12));
    }
    let model = model::parse(&gguf).unwrap();
    for (tensor, s) in model
        .tensors
        .iter()
        .zip(stats::summarize_all(&model.tensors, Backend::Scalar))
    {
        let s = s.unwrap();
        assert_eq!(s.count, 64);
        // Quantized: close to the original range, not exact.
        assert!(
            (s.min - min).abs() < 0.3 && (s.max - max).abs() < 0.3,
            "{}",
            tensor.name
        );
    }
}

#[test]
fn histogram_counts_every_finite_value_once() {
    let mut values = Rng::new(8).normals(JOB_ELEMENTS + 500, 1.0);
    values[3] = f32::NAN;
    let (bytes, shape) = view(&values);
    let tensors = [f32_view(&bytes, &shape)];
    let summaries = stats::summarize_all(&tensors, Backend::Scalar);
    let histogram = stats::histograms(&tensors, &summaries, 10, Backend::Scalar)[0]
        .clone()
        .unwrap();
    assert_eq!(histogram.counts.len(), 10);
    assert_eq!(
        histogram.counts.iter().sum::<u64>(),
        summaries[0].unwrap().finite
    );

    let mut edges = Histogram::new(0.0, 1.0, 4);
    edges.add_all(&[0.0, 0.25, 0.5, 0.999, 1.0, f32::INFINITY]);
    assert_eq!(edges.counts, [1, 1, 1, 2]); // the maximum lands in the last bin

    let mut constant = Histogram::new(2.0, 2.0, 3);
    constant.add_all(&[2.0; 5]);
    assert_eq!(constant.counts, [5, 0, 0]);
    let text = stats::render_histogram(&constant, 10);
    assert!(text.contains("########## 5"), "{text}");
}

#[test]
fn derived_measures() {
    let s = Summary::of(&[3.0, -4.0, 0.0, 0.0], Backend::Scalar);
    assert_eq!(s.l2_norm(), 5.0);
    assert_eq!(s.abs_max(), 4.0);
    assert_eq!(s.zero_fraction(), 0.5);
    let empty = Summary::of(&[], Backend::Scalar);
    assert_eq!((empty.std(), empty.rms(), empty.abs_max()), (0.0, 0.0, 0.0));
    let only_nan = Summary::of(&[f32::NAN], Backend::Scalar);
    assert_eq!((only_nan.finite, only_nan.nan, only_nan.std()), (0, 1, 0.0));
}
