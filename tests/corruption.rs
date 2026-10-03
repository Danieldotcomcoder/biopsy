//! Defensive parsing, tested by brute force: corrupt valid files at random and
//! check that every command path either rejects them or handles them, never panics.
use biopsy::{
    diff, fixtures, fixtures::Rng, health, inspect, kernels::Backend, model, stats,
    tensor::TensorView,
};

fn exercise(tensors: &[TensorView]) {
    let backend = Backend::detect();
    let summaries = stats::summarize_all(tensors, backend);
    let _ = stats::render_table(tensors, &summaries);
    let _ = stats::histograms(tensors, &summaries, 8, backend);
    let _ = health::check(tensors, &health::Thresholds::default(), backend).render();
    let _ = diff::diff(tensors, tensors, &diff::Options::default(), backend).map(|r| r.render(0));
}

#[test]
fn random_corruption_is_rejected_or_handled_never_panics() {
    let mut rng = Rng::new(2026);
    let mut accepted = 0;
    let mut rejected = 0;
    for (name, original) in fixtures::demo_files() {
        for round in 0..400u32 {
            let mut bytes = original.clone();
            // Most flips land in the header, where the structure is.
            let header = bytes.len().min(1024) as u64;
            for _ in 0..1 + rng.next_u64() % 6 {
                let span = if rng.next_u64().is_multiple_of(4) {
                    bytes.len() as u64
                } else {
                    header
                };
                let i = (rng.next_u64() % span) as usize;
                bytes[i] = rng.next_u64() as u8;
            }
            if round.is_multiple_of(5) {
                bytes.truncate((rng.next_u64() % bytes.len() as u64) as usize);
            }
            let report = inspect::render(name, &bytes);
            match model::parse(&bytes) {
                Ok(model) => {
                    accepted += 1;
                    assert!(report.is_ok(), "{name}: parse ok but inspect failed");
                    exercise(&model.tensors);
                }
                Err(_) => {
                    rejected += 1;
                    assert!(
                        report.is_err(),
                        "{name}: inspect accepted what parse rejected"
                    );
                }
            }
        }
    }
    // Both outcomes must actually occur, or the test is not testing anything.
    assert!(
        accepted > 50 && rejected > 50,
        "accepted {accepted}, rejected {rejected}"
    );
}
