use biopsy::{
    diff::{self, Metrics, Options, Outcome},
    fixtures::{self, Tensor},
    kernels::Backend,
    model,
};

fn run(a: &[u8], b: &[u8], options: &Options) -> diff::Report {
    let (a, b) = (model::parse(a).unwrap(), model::parse(b).unwrap());
    diff::diff(&a.tensors, &b.tensors, options, Backend::Scalar).unwrap()
}

fn metrics(report: &diff::Report, name: &str) -> Metrics {
    match report
        .pairs
        .iter()
        .find(|p| p.name == name)
        .map(|p| &p.outcome)
    {
        Some(Outcome::Compared(m)) => *m,
        other => panic!("{name}: {other:?}"),
    }
}

fn file(tensors: &[Tensor]) -> Vec<u8> {
    fixtures::safetensors(&[], tensors)
}

#[test]
fn reports_every_matching_category() {
    let report = run(
        &fixtures::encoder(false),
        &fixtures::encoder(true),
        &Options::default(),
    );
    assert_eq!(report.only_a, ["lm_head.weight"]);
    assert_eq!(report.only_b, ["classifier.weight"]);
    let pooler = report
        .pairs
        .iter()
        .find(|p| p.name == "pooler.weight")
        .unwrap();
    assert_eq!(
        pooler.outcome,
        Outcome::ShapeMismatch {
            a: vec![32, 32],
            b: vec![16, 32]
        }
    );
    assert!(metrics(&report, "embeddings.weight").identical());
    let dense = metrics(&report, "layer.0.dense.weight");
    assert!(!dense.identical() && dense.cosine().unwrap() > 0.99);
    let text = report.render(0);
    assert!(text.contains("Identical: 1; changed: 2"), "{text}");
    assert!(text.contains("Shape mismatch: \"pooler.weight\": [32, 32] vs [16, 32]"));
}

#[test]
fn metrics_match_hand_computation() {
    let a = file(&[Tensor::encode("t", "F32", &[4], &[1.0, 2.0, 3.0, 4.0])]);
    let b = file(&[Tensor::encode("t", "F32", &[4], &[1.0, 2.0, 3.0, 5.0])]);
    let m = metrics(&run(&a, &b, &Options::default()), "t");
    assert_eq!(m.elements, 4);
    assert_eq!(m.sums.equal, 3);
    assert_eq!(m.changed_fraction(), 0.25);
    assert_eq!(m.sums.max_abs_diff, 1.0);
    assert_eq!(m.mean_abs(), 0.25);
    assert_eq!(m.rmse(), 0.5);
    assert_eq!(m.rel_l2(), (1.0f64 / 30.0).sqrt());
    assert_eq!(m.cosine().unwrap(), 34.0 / (30.0f64 * 39.0).sqrt());
}

#[test]
fn self_diff_is_identical_everywhere() {
    let bytes = fixtures::tiny_gguf("Q4_0");
    let report = run(&bytes, &bytes, &Options::default());
    let overall = report.overall();
    assert!(overall.identical());
    assert_eq!((overall.rel_l2(), overall.sums.max_abs_diff), (0.0, 0.0));
    assert!((overall.cosine().unwrap() - 1.0).abs() < 1e-12);
    // The Q4_K tensor is matched but cannot be compared.
    let undecoded = report
        .pairs
        .iter()
        .filter(|p| p.outcome == Outcome::NotDecoded);
    assert_eq!(undecoded.count(), 1);
}

#[test]
fn different_dtypes_compare_after_decoding() {
    let q8 = fixtures::tiny_gguf("Q8_0");
    let q4 = fixtures::tiny_gguf("Q4_0");
    let report = run(&q8, &q4, &Options::default());
    let embed = metrics(&report, "token_embd.weight");
    // 4-bit codes lose roughly 10% of the signal energy; 8-bit lose far less.
    assert!(
        embed.rel_l2() > 0.03 && embed.rel_l2() < 0.2,
        "{}",
        embed.rel_l2()
    );
    assert!(metrics(&report, "blk.0.attn_v.weight").identical()); // F16 in both
}

#[test]
fn prefix_stripping_and_collisions() {
    let a = file(&[Tensor::encode("model.w", "F32", &[1], &[1.0])]);
    let b = file(&[Tensor::encode("w", "F32", &[1], &[1.0])]);
    let plain = run(&a, &b, &Options::default());
    assert_eq!((plain.only_a.len(), plain.only_b.len()), (1, 1));
    let stripped = Options {
        strip_prefixes: vec!["model.".into()],
    };
    let report = run(&a, &b, &stripped);
    assert!(report.only_a.is_empty() && metrics(&report, "w").identical());

    let colliding = file(&[
        Tensor::encode("model.w", "F32", &[1], &[1.0]),
        Tensor::encode("w", "F32", &[1], &[1.0]),
    ]);
    let (a, b) = (
        model::parse(&colliding).unwrap(),
        model::parse(&colliding).unwrap(),
    );
    let error = diff::diff(&a.tensors, &b.tensors, &stripped, Backend::Scalar).unwrap_err();
    assert!(error.to_string().contains("both become \"w\""), "{error}");
}

#[test]
fn non_finite_values_are_counted_not_averaged() {
    let a = file(&[Tensor::encode(
        "t",
        "F32",
        &[3],
        &[f32::NAN, 1.0, f32::INFINITY],
    )]);
    let b = file(&[Tensor::encode("t", "F32", &[3], &[f32::NAN, 1.0, 2.0])]);
    let m = metrics(&run(&a, &b, &Options::default()), "t");
    assert_eq!(m.sums.equal, 2); // NaN vs NaN, 1 vs 1
    assert_eq!(m.sums.nonfinite_mismatch, 1);
    assert_eq!(m.sums.finite_pairs, 1);
    assert_eq!(m.rel_l2(), 0.0); // the finite part is unchanged
}
