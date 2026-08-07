# Evaluation Contract

## Purpose

This document defines the required behavior of the evaluation system in:

```text
ASL PROJECT/ASL_training/
```

The evaluation layer measures the quality of a trained isolated-sign classifier without modifying model weights.

It covers:

* clean classification metrics
* per-class and per-signer analysis
* confusion analysis
* prediction and logit export
* confidence calibration
* selective prediction
* robustness reporting boundaries
* cross-dataset reporting boundaries

The evaluation layer must preserve experimental validity. Its most important responsibility is not computing metrics, but refusing to compute them in ways that invalidate the experiment.

## Scope

The evaluation layer owns:

* top-1 and top-5 accuracy
* macro F1 and weighted F1
* mean per-class accuracy
* per-class precision, recall, and support
* per-signer accuracy and worst-signer reporting
* confusion analysis
* negative log-likelihood
* per-example prediction export
* validation logit export
* temperature scaling
* expected calibration error
* reliability data
* threshold analysis
* coverage and selective accuracy
* robustness result aggregation
* evaluation report generation

The evaluation layer does not own:

* gradient updates
* optimizer or scheduler state
* checkpoint selection policy definition
* dataset split creation
* manifest or label-map generation
* training augmentation
* robustness transform definitions
* experiment interpretation
* production monitoring
* serving thresholds

The training layer calls the evaluation layer for validation. The evaluation layer must not call the training layer.

## Layer Dependencies

The evaluation layer may depend on:

* model interfaces
* dataset and data-loader interfaces
* label-map utilities
* shared configuration utilities
* logging utilities

The evaluation layer must not import:

* training orchestration
* experiment definitions
* serving code

## Weight Immutability

Evaluation must never modify model weights.

Every evaluation path must:

* place the model in evaluation mode
* disable gradient computation
* leave optimizer and scheduler state untouched
* restore the caller's prior mode when evaluation is invoked mid-training

Temperature scaling fits one scalar calibration parameter. That parameter is a property of the evaluation artifact, not of the model. Fitting it must not alter any model weight.

## Evaluation Modes

The system must distinguish four modes. They must not share ambiguous behavior.

### In-Training Validation

Runs on the validation manifest during training to drive checkpoint selection.

* deterministic evaluation preprocessing
* a restricted metric set is acceptable for speed
* must never touch the test manifest

### Full Validation Evaluation

Runs on the validation manifest from a saved checkpoint.

Produces the artifacts calibration and threshold selection consume.

### Final Test Evaluation

Runs once on the test manifest, after model selection, calibration, and threshold selection are fixed.

### Robustness Evaluation

Runs on perturbed copies of validation or test data. Reported separately from clean results.

## Test-Set Isolation

The test split must not influence:

* checkpoint selection
* early stopping
* hyperparameter choices
* architecture choices
* augmentation choices
* temperature fitting
* confidence-threshold selection
* preprocessing changes

The default validation and calibration commands must not require or load the test manifest.

Final test evaluation must be an explicit, separately invoked operation.

A run that reports test metrics before model and threshold selection are fixed is not a valid experiment.

### Repeated Test Evaluation

Repeatedly evaluating the test split while iterating on the model erodes its value even without formal tuning against it.

Every test evaluation should be recorded with:

* the checkpoint evaluated
* the calibration parameters applied
* the threshold applied
* the date
* the reason

The project should be able to state how many times the test split has been read.

## Required Inputs

A full evaluation run must receive:

* checkpoint path
* architecture and model configuration
* label map and label-map identity
* evaluation manifest
* manifest identity
* preprocessing configuration and identity
* batch size and device
* precision mode
* output directory
* evaluation mode

Evaluation must fail before computing metrics if:

* the checkpoint class count differs from the label-map size
* the label-map identity differs from the one recorded in the checkpoint
* the preprocessing identity differs from the one recorded in the checkpoint without an explicit override
* the manifest contains class IDs outside the label map
* the manifest is empty

A preprocessing override must be recorded in the evaluation report and disqualifies the result from direct comparison against runs using the training-time preprocessing.

## Deterministic Evaluation

Evaluation preprocessing must be deterministic, as defined in `docs/DATA_CONTRACT.md`.

Given the same checkpoint, manifest, preprocessing configuration, and software version, evaluation must produce identical predictions across runs.

Evaluation must not use:

* unseeded random temporal sampling
* random spatial crops
* training augmentation
* shuffled data loaders
* drop-last batching

Dropping the final partial batch would silently change the evaluated sample count. Evaluation must process every sample in the manifest.

### Sample Accounting

Every evaluation report must state:

* manifest sample count
* samples successfully evaluated
* samples skipped
* skip reasons

Evaluated count must equal manifest count unless an explicit, recorded exclusion policy applies.

Silent sample loss during evaluation is prohibited.

## Per-Example Prediction Export

Evaluation must persist per-example records, not only aggregate metrics.

Each record should contain:

* `sample_id`
* `signer_id`
* `gloss`
* `true_class_id`
* `predicted_class_id`
* `top_k_class_ids`
* `top_k_scores`
* `max_softmax_confidence`
* `correct`
* `split`
* `dataset_name`

Logits should be exported separately when needed for calibration.

Per-example export is required because:

* per-class and per-signer metrics must be recomputable without rerunning the model
* calibration must be refittable without rerunning the model
* threshold analysis must be re-runnable at zero cost
* error analysis requires tracing individual failures

Aggregate-only evaluation is insufficient for a real experiment.

## Logit Export

Validation logits must be exportable for calibration.

The export must preserve:

* raw pre-softmax logits
* true class IDs
* sample IDs
* label-map identity
* checkpoint identity
* preprocessing identity

Logits must be exported before any temperature is applied. Calibration operates on raw logits.

Storing softmax probabilities instead of logits is prohibited, because temperature scaling cannot be recovered from normalized probabilities.

## Core Classification Metrics

### Top-1 Accuracy

Fraction of samples whose highest-logit class equals the true class.

### Top-5 Accuracy

Fraction of samples whose true class appears among the five highest-logit classes.

When the class count is fewer than five, top-5 accuracy is undefined and must be reported as unavailable rather than as 1.0.

### Macro F1

Unweighted mean of per-class F1 scores.

Classes with no support in the evaluated split must be handled by an explicit, documented rule. The chosen rule must be identical across all compared experiments.

### Weighted F1

Per-class F1 weighted by class support.

### Mean Per-Class Accuracy

Unweighted mean of per-class recall. This is the balanced-accuracy view and must be reported alongside top-1, because ASL Citizen class support is uneven.

### Per-Class Metrics

For every class: precision, recall, F1, support, and predicted count.

### Negative Log-Likelihood

Mean negative log probability assigned to the true class.

NLL must be computed from logits with numerically stable log-softmax, never by taking the logarithm of a rounded probability.

NLL must be reported both before and after calibration.

## Tie-Breaking

Argmax ties must resolve deterministically, by lowest class ID.

Ties are rare in floating point but must not introduce run-to-run variation.

## Per-Signer Evaluation

Signer-independent generalization is a primary project goal, so per-signer reporting is required, not optional.

The evaluation layer must report:

* accuracy per signer
* sample count per signer
* macro F1 per signer where support permits
* worst-signer accuracy
* best-signer accuracy
* mean and standard deviation of per-signer accuracy
* the distribution of per-signer accuracy

Worst-signer accuracy computed over a signer with very few samples is noise. Reports must state the minimum signer support used, and signers below a configured support floor must be reported separately rather than silently included in worst-signer selection.

A model with strong overall accuracy and a weak worst-signer accuracy has a generalization problem that headline accuracy conceals.

## Confusion Analysis

Full confusion matrices over a large vocabulary are not directly interpretable, so the evaluation layer should produce:

* the full matrix in a machine-readable form
* the most frequent confused class pairs
* the classes with lowest recall
* the classes with lowest precision
* classes never predicted
* classes always confused with a single other class

Confusion output should retain glosses, not only class IDs, so that linguistically plausible confusions can be recognized during review.

## Calibration

### Motivation

Raw maximum softmax score is not a reliable probability. Modern classifiers are typically overconfident. Confidence is only useful if it supports a rejection decision.

### Temperature Scaling

A single scalar temperature `T` is fit on validation logits by minimizing validation negative log-likelihood.

Calibrated logits are:

```text
calibrated_logits = logits / T
```

Requirements:

* `T` must be fit on validation data only
* `T` must never be fit on test data
* `T > 0`
* the optimization method, initialization, and convergence criterion must be recorded
* the fitted `T` must be persisted with the checkpoint and label-map identity

Temperature scaling is monotonic, so it must not change top-1 predictions. The implementation must verify this invariant and fail if predicted classes change after calibration. This is a correctness check, not an assumption.

### Expected Calibration Error

ECE partitions predictions into confidence bins and measures the weighted mean absolute gap between mean confidence and accuracy within each bin.

The report must state:

* number of bins
* binning scheme, equal-width or equal-mass
* whether empty bins are excluded

ECE is sensitive to the binning scheme, so ECE values may only be compared across experiments that used identical binning.

### Reliability Data

Reliability output should include per-bin:

* bin boundaries
* sample count
* mean confidence
* accuracy
* gap

Both pre- and post-calibration reliability data must be produced.

### Required Calibration Reporting

Every calibration report must state, before and after calibration:

* NLL
* ECE
* mean confidence
* accuracy
* top-1 accuracy, which must be unchanged

## Selective Prediction

### Motivation

A deployed system should be able to decline to answer rather than return a confident wrong sign.

### Definitions

For a confidence threshold `t`:

```text
accepted        = predictions with confidence >= t
coverage        = accepted / total
selective accuracy = correct among accepted / accepted
```

Confidence should be the calibrated maximum softmax score. The report must state which confidence was used.

### Required Output

An accuracy-versus-coverage curve computed over a range of thresholds, plus, at each evaluated threshold:

* threshold
* coverage
* selective accuracy
* accepted count
* rejected count
* rejected-but-correct count

Rejected-but-correct count is required because it measures the cost of the rejection policy.

### Threshold Selection

Thresholds must be selected on validation data.

The selection rule must be explicit, for example:

```text
select the lowest threshold achieving at least 90% selective accuracy on validation
```

The selected threshold is then applied once to test predictions and reported.

Selecting a threshold by inspecting test results is prohibited.

### Threshold Transfer

Validation-selected thresholds will not reproduce validation behavior exactly on test data. Reports must present validation and test operating points side by side and must not present the validation operating point as the expected test behavior.

## Robustness Reporting Boundary

Robustness evaluation reuses the metric implementations but must remain separately reported.

Clean and perturbed results must never be averaged into one headline number.

Each robustness result must record:

* perturbation name
* parameters
* seed where applicable
* the clean baseline it is compared against
* absolute and relative performance drop
* calibration change under perturbation

Robustness perturbations are defined by the data layer, per `docs/DATA_CONTRACT.md`. The evaluation layer consumes them but does not define them.

Calibration parameters fit on clean validation data may be applied to perturbed data, but the report must state that the temperature was not refit, since calibration typically degrades under distribution shift.

## Cross-Dataset Reporting Boundary

WLASL results must never be merged into ASL Citizen metrics.

Cross-dataset evaluation requires the reviewed label-harmonization artifact described in `docs/DATA_CONTRACT.md`. Reports must state:

* number of classes in the verified overlap
* number of classes excluded and why
* that accuracy over the overlap subset is not comparable to full-vocabulary accuracy

## Evaluation Output Structure

An evaluation run should write to a unique directory:

```text
evaluation/
├── resolved_config
├── evaluation_metadata
├── metrics_summary
├── per_class_metrics
├── per_signer_metrics
├── confusion
├── predictions
├── logits
├── calibration
└── selective_prediction
```

Evaluation metadata must record:

* checkpoint path and identity
* architecture
* label-map identity
* manifest identity
* preprocessing identity
* dataset identity
* evaluation mode
* device, precision, and batch size
* Git commit
* dependency versions
* sample accounting
* date

Existing evaluation outputs must not be silently overwritten.

## Metric Correctness

Metric implementations must be tested against independently computed reference values on small fixtures with known answers.

Tests must cover:

* perfect prediction
* completely wrong prediction
* the single-class edge case
* classes with zero support
* classes never predicted
* top-5 when class count is below five
* per-class aggregation
* per-signer aggregation
* deterministic tie-breaking
* NLL numerical stability with extreme logits
* temperature scaling preserving top-1
* ECE on a hand-computed binning example
* coverage and selective accuracy at boundary thresholds
* empty accepted set at a threshold above every confidence
* gradient absence during evaluation

Where an established library is used for a metric, at least one test must verify the project's aggregation and label alignment rather than trusting the library's defaults.

## Failure Conditions

Evaluation must fail clearly when:

* checkpoint and label-map sizes disagree
* label-map identity mismatches
* preprocessing identity mismatches without explicit override
* the manifest contains unknown class IDs
* the manifest is empty
* evaluated sample count differs from manifest count without a recorded policy
* logits contain non-finite values
* temperature fitting fails to converge
* fitted temperature is non-positive
* calibration changes top-1 predictions
* test data reaches a calibration or threshold-selection path
* evaluation is invoked with gradients enabled
* required per-example metadata is missing

Warnings are appropriate for:

* signers below the support floor
* classes with very low support
* very high or very low fitted temperature
* large calibration change
* empty confidence bins

Warnings must not conceal integrity failures.

## Completion Criteria

The evaluation layer is complete for its phase when it can:

* evaluate a saved checkpoint on a manifest without modifying weights
* process every manifest sample with full accounting
* compute the core classification metric set correctly
* produce per-class and per-signer breakdowns
* produce confusion analysis retaining glosses
* export per-example predictions and raw logits
* fit temperature on validation logits only
* verify that calibration preserves top-1
* report pre- and post-calibration NLL and ECE
* produce accuracy-versus-coverage results
* select a threshold on validation and apply it once to test
* keep clean, robustness, and cross-dataset results separate
* record complete evaluation metadata
* fail clearly on the conditions above
* pass focused metric-correctness tests

## Initial Implementation Priority

1. Deterministic evaluation loop with full sample accounting.
2. Per-example prediction and logit export.
3. Core classification metrics with correctness tests.
4. Per-class and per-signer aggregation.
5. Confusion analysis.
6. Temperature scaling on validation logits.
7. NLL, ECE, and reliability data.
8. Selective prediction and threshold selection.
9. Evaluation metadata and report structure.
10. Robustness result aggregation.

Do not prioritize initially:

* out-of-distribution detection
* ensemble evaluation
* test-time augmentation
* multi-clip or multi-crop inference
* conformal prediction
* interactive dashboards
* production threshold enforcement

Multi-clip inference in particular changes the preprocessing contract and must be a separate documented experiment, not an evaluation-layer default.

## Relationship to Other Documents

This document depends on:

* `docs/PROJECT.md`
* `docs/ARCHITECTURE.md`
* `docs/ROADMAP.md`
* `docs/MODEL_CONTRACT.md`
* `docs/DATA_CONTRACT.md`
* `docs/TRAINING_CONTRACT.md`

Experiment naming and interpretation are defined in:

```text
docs/experiments/
```

Serving-time confidence behavior belongs to the future sibling project:

```text
ASL PROJECT/ASL_serving/
```

The serving project should consume the exported calibration parameters and threshold policy rather than re-deriving them.
