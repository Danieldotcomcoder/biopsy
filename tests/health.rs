use biopsy::{
    fixtures::{self, Rng, Tensor},
    health::{self, Severity, Thresholds},
    kernels::Backend,
    model,
};

fn findings(bytes: &[u8], thresholds: &Thresholds) -> Vec<(String, &'static str, Severity)> {
    let model = model::parse(bytes).unwrap();
    health::check(&model.tensors, thresholds, Backend::Scalar)
        .findings
        .into_iter()
        .map(|f| (f.tensor, f.check, f.severity))
        .collect()
}

fn owned(expected: &[(&str, &'static str, Severity)]) -> Vec<(String, &'static str, Severity)> {
    expected
        .iter()
        .map(|&(t, c, s)| (t.to_owned(), c, s))
        .collect()
}

#[test]
fn every_planted_problem_is_found_and_nothing_else() {
    use Severity::*;
    let got = findings(&fixtures::sick(), &Thresholds::default());
    // Safetensors tensors come out in name order, so findings do too.
    let expected = owned(&[
        ("constant.weight", "constant", Warn),
        ("dead_rows.weight", "dead-rows", Warn),
        ("huge.weight", "large-magnitude", Warn),
        ("huge.weight", "f16-range", Info),
        ("huge.weight", "row-norm", Warn),
        ("non_finite.weight", "non-finite", Error),
        ("outlier_rows.weight", "row-norm", Warn),
        ("sparse.weight", "sparse", Warn),
        ("zeros.bias", "all-zero", Warn),
    ]);
    assert_eq!(got, expected);
}

#[test]
fn healthy_demo_models_produce_no_findings() {
    for bytes in [
        fixtures::encoder(false),
        fixtures::encoder(true),
        fixtures::tiny_gguf("Q8_0"),
        fixtures::tiny_gguf("Q4_0"),
    ] {
        assert_eq!(findings(&bytes, &Thresholds::default()), []);
    }
}

#[test]
fn thresholds_change_the_verdict() {
    let relaxed = Thresholds {
        max_abs: 1.0e6,
        max_zero_fraction: 0.9,
        max_dead_row_fraction: 0.5,
        row_norm_ratio: 1.0e6,
    };
    let got = findings(&fixtures::sick(), &relaxed);
    // Dead rows drop to INFO; sparsity, magnitude and row-norm warnings vanish.
    assert!(got.contains(&("dead_rows.weight".into(), "dead-rows", Severity::Info)));
    for check in ["sparse", "large-magnitude", "row-norm"] {
        assert!(got.iter().all(|f| f.1 != check), "{check} still reported");
    }
}

#[test]
fn messages_state_thresholds_and_rows() {
    let bytes = fixtures::sick();
    let model = model::parse(&bytes).unwrap();
    let report = health::check(&model.tensors, &Thresholds::default(), Backend::Scalar);
    let text = report.render();
    assert!(text.contains("rows 2, 5"), "{text}");
    assert!(text.contains("WARN above 1.0%"), "{text}");
    assert!(text.contains("rows 3 (x"), "{text}");
    assert!(text.contains("1 errors, 7 warnings, 1 info"), "{text}");
}

/// Row semantics must agree across formats: GGUF stores dims innermost-first,
/// and biopsy reverses them, so row r is the same values in both files.
#[test]
fn rows_mean_the_same_thing_in_safetensors_and_gguf() {
    let mut values = Rng::new(1).normals(6 * 64, 0.1);
    values[4 * 64..5 * 64].fill(0.0); // row 4 is dead
    for v in &mut values[64..128] {
        *v *= 50.0; // row 1 is an outlier
    }
    let st = fixtures::safetensors(&[], &[Tensor::encode("w", "F32", &[6, 64], &values)]);
    let gg = fixtures::gguf(&[], &[Tensor::encode("w", "F32", &[6, 64], &values)]);
    let render = |bytes: &[u8]| {
        let model = model::parse(bytes).unwrap();
        let report = health::check(&model.tensors, &Thresholds::default(), Backend::Scalar);
        report
            .findings
            .iter()
            .map(|f| f.message.clone())
            .collect::<Vec<_>>()
    };
    let (a, b) = (render(&st), render(&gg));
    assert_eq!(a, b);
    assert!(
        a[0].contains("rows 4") && a[1].contains("rows 1 (x"),
        "{a:?}"
    );
}

#[test]
fn undecoded_tensors_are_skipped_not_judged() {
    let bytes = fixtures::tiny_gguf("Q8_0");
    let model = model::parse(&bytes).unwrap();
    let report = health::check(&model.tensors, &Thresholds::default(), Backend::Scalar);
    assert_eq!(report.checked, 4);
    assert_eq!(
        report.skipped,
        [("blk.0.ffn_down.weight".to_owned(), "Q4_K".to_owned())]
    );
}
