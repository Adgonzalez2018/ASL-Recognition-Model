# Phase 4: Evaluation Layer

Status: Complete
Archived: 2026-08-10

Also unblocked: metrics, calibration, and selective prediction are testable on synthetic logits, and the runner on a tiny model.

## Phase 4A: Core Classification Metrics

- [x] top-1 and top-5 accuracy
- [x] macro F1 and weighted F1
- [x] mean per-class accuracy
- [x] per-class precision and recall
- [x] per-signer accuracy and worst-signer reporting
- [x] confusion analysis
- [x] per-example prediction and logit export
- [x] metric correctness tests

## Phase 4B: Calibration

- [x] validation logit export
- [x] raw maximum-softmax confidence
- [x] negative log-likelihood
- [x] expected calibration error
- [x] reliability data
- [x] temperature scaling fit on validation logits
- [x] pre- and post-calibration comparison
- [x] persisted calibration parameters

## Phase 4C: Selective Prediction

- [x] threshold sweep
- [x] coverage and selective accuracy
- [x] accuracy-versus-coverage curve
- [x] threshold selection on validation
- [x] thresholds applied once to test

## Acceptance Criteria

- [x] Metrics match an independent reference (sklearn cross-checks).
- [x] Per-class and per-signer aggregation is reproducible.
- [x] Evaluation preserves sample IDs, signer IDs, and true labels.
- [x] Results are generated from a saved checkpoint.
- [x] Temperature fitting never uses test labels; the script refuses.
- [x] Calibration does not change top-1 ranking, verified rather than assumed.
- [x] Pre- and post-calibration metrics are both reported.
- [x] Calibration parameters apply to later test predictions.
- [x] Validation and test operating points are reported side by side.

## Implementation

```text
src/asl_training/evaluation/
├── metrics.py      classification metrics, per-class and per-signer
├── calibration.py  temperature scaling, NLL, ECE, reliability
├── selective.py    coverage, selective accuracy, threshold selection
└── runner.py       execution, per-example and logit export

scripts/evaluate.py
```

## Design notes

**Test-set isolation is structural, not procedural.** `--split test` refuses to run without `--calibration` from a validation run and without `--reason`. The reason and date are written into the report, so the number of test reads is recoverable — repeated reads erode the split even without formal tuning.

**Metrics are implemented directly and cross-checked against sklearn.** The risk is not that a formula is wrong but that the project's aggregation or label alignment differs from what a reader assumes, so the reference comparison targets exactly that.

**Policy choices travel with the numbers.** Macro F1 reports how many classes it averaged over; ECE reports its bin count and scheme. A macro average over 2731 classes and one over 40 are not comparable, and nothing in the bare number says which you have.

**Monotonicity is verified, not assumed.** Temperature scaling must not change any top-1 prediction. `fit_temperature` checks and raises if any moved, because reported accuracy would otherwise stop describing the model.

**Top-5 is unavailable, not 1.0, below five classes.** Reporting a defined-looking value for an undefined metric is worse than reporting nothing.

**Worst-signer accuracy respects a support floor.** Computed over a signer with three samples it is noise; those signers are reported separately rather than being allowed to define the worst case.

**Rejected-but-correct is reported.** Selective accuracy alone shows only what rejection buys, never what it costs.

**Raw logits are exported, never probabilities.** Temperature scaling cannot be recovered from normalized values, so `load_logits` refuses anything not marked raw.

## Validation

659 tests total. Evaluation coverage: metrics against hand-computed fixtures and sklearn, calibration invariants, selective-prediction boundaries, runner weight-immutability and sample accounting, and an end-to-end test that trains a checkpoint and evaluates it through the real script on both splits.

## Remaining

* Robustness evaluation transforms belong to Phase 6 and are not built here.
* Cross-dataset (WLASL) reporting boundaries are defined in the contract but unimplemented until Phase 8.

---

## Note on data-loader workers

`--num-workers` defaults to 4 and is configurable, because video decoding is CPU-bound and worker count is commonly the training bottleneck rather than the GPU. Colab's CPU allocation varies, so this will want tuning against real hardware; preflight should report whether the loader or the GPU is limiting.

Integration tests run with `--num-workers 0`. Spawned workers deadlock under pytest on macOS, and worker behaviour is not what those tests exercise.
