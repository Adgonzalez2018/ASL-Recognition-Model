# Training Roadmap

## Project Location

The parent workspace is:

```text
ASL PROJECT/
├── ASL_training/
└── ASL_serving/
```

All work in this roadmap belongs to:

```text
ASL PROJECT/ASL_training/
```

`ASL_serving` is outside the current implementation scope.

## Roadmap Goal

Build a reproducible isolated ASL video-classification training system that:

* supports VideoMAE-Base and Video Swin-Tiny
* trains on ASL Citizen using official signer-independent splits
* compares both architectures fairly
* measures clean accuracy, class balance, signer generalization, and confidence calibration
* stress-tests model robustness
* supports targeted augmentation experiments
* exports a selected checkpoint and preprocessing contract for future serving work

## Implementation Order

The intended sequence is:

```text
Phase 1: Basic model layer
Phase 2: Data audit and data layer
Phase 3: Training layer
Phase 4: Evaluation layer
Phase 5: Baseline experiments
Phase 6: Robustness evaluation
Phase 7: Targeted robustness training
Phase 8: Secondary and external benchmarking
Phase 9: Training handoff package
```

The evaluation and experiment layers may be developed in parallel once the first full training run is stable.

---

# Phase 0: Repository Foundation

## Objective

Establish the repository, documentation, dependency boundaries, and development commands.

## Tasks

* Initialize `ASL_training` as its own Git repository.
* Add `CLAUDE.md`.
* Add `PROJECT.md`.
* Add `docs/ARCHITECTURE.md`.
* Add `docs/ROADMAP.md`.
* Define the initial Python package.
* Add dependency management.
* Add formatting, linting, and test commands.
* Add `.gitignore` rules for datasets, checkpoints, caches, secrets, and notebook outputs.
* Add a minimal local development README.
* Add empty or placeholder configuration directories.
* Add small synthetic test fixtures only.

## Acceptance Criteria

* The repository installs in a clean environment.
* The test command runs successfully.
* No raw data or checkpoints are tracked.
* Package imports use the intended `src` layout.
* `ASL_training` does not depend on or modify `ASL_serving`.
* Claude can identify the current phase and authoritative documents.

## Non-Goals

* Model implementation
* Dataset download
* Training
* Evaluation
* Colab or Kaggle optimization

---

# Phase 1: Basic Model Layer

## Objective

Establish a shared video-classification contract and verify both initial pretrained architectures.

## Scope

Models:

* VideoMAE-Base
* Video Swin-Tiny

Input:

```text
fixed-length RGB video batch
```

Output:

```text
multiclass logits
```

## Tasks

* Define the canonical logical input shape.
* Define the shared classification output.
* Add a model factory.
* Implement the VideoMAE adapter.
* Implement the Video Swin adapter.
* Load pretrained weights.
* Replace each original classification head with a configurable ASL head.
* Support a configurable number of classes.
* Add model-specific tensor adaptation internally.
* Report model parameter counts.
* Add dummy-forward smoke tests.
* Add checkpoint state-loading tests where practical.

## Acceptance Criteria

For both models:

* Pretrained construction succeeds.
* A dummy video batch is accepted.
* Output logits have shape:

```text
[batch_size, number_of_classes]
```

* The classification head has the configured output size.
* Architecture-specific objects do not leak into the shared training boundary.
* Tests verify input and output behavior.
* No real dataset implementation is required.

## Non-Goals

* Dataset parsing
* Video decoding
* Training loops
* Hyperparameter optimization
* Real model accuracy
* Serving

## Completion Artifact

A stable model-layer interface that the data and training layers can target.

---

# Phase 2: ASL Citizen Audit and Data Layer

## Objective

Understand the actual ASL Citizen distribution and build a reproducible path from raw videos to standardized model batches.

## Phase 2A: Dataset Access and Audit

### Tasks

* Confirm the dataset source and license requirements.
* Resolve expected Kaggle and Colab paths.
* Inspect the annotation and split files.
* Confirm the official train, validation, and test structure.
* Identify signer metadata.
* Count classes, videos, and signers by split.
* Audit class-frequency distribution.
* Audit video durations, frame rates, and resolutions.
* Detect missing and corrupted videos.
* Determine whether mirroring or handedness metadata exists.
* Produce a versioned dataset audit report.

### Acceptance Criteria

The audit identifies:

* exact usable sample counts
* exact class count
* exact signer counts
* split assignments
* missing or corrupted samples
* class imbalance
* duration and frame-rate ranges
* unresolved metadata concerns
* any difference between the hosted Kaggle copy and official metadata

No training implementation should rely on unverified assumptions about the dataset structure.

## Phase 2B: Manifest and Label Contracts

### Tasks

* Define the manifest schema.
* Build stable gloss-to-ID and ID-to-gloss mappings.
* Generate split-specific manifests.
* Preserve signer identifiers.
* Include audited video metadata where useful.
* Add duplicate detection.
* Add signer-leakage validation.
* Add manifest versioning or identity checks.

### Acceptance Criteria

* Every usable training sample maps to one stable class ID.
* Both model families can consume the same manifests.
* Train, validation, and test splits remain signer-independent.
* Duplicate videos are detected.
* Label maps are stable and tested.
* Missing or corrupted records are reported explicitly.

## Phase 2C: Video Input Pipeline

### Tasks

* Implement video decoding.
* Convert decoded frames to RGB.
* Implement fixed-frame temporal sampling.
* Define short-video behavior.
* Define long-video sampling behavior.
* Implement deterministic evaluation sampling.
* Implement minimal baseline training transforms.
* Ensure spatial transforms are consistent across all frames.
* Implement batch collation.
* Add data-loader configuration.
* Add focused data-pipeline tests.

## Baseline Transform Policy

The initial baseline should use only restrained preprocessing and augmentation.

Allowed baseline behavior may include:

* required resizing
* fixed crop size
* mild random crop during training
* model-compatible normalization
* controlled temporal sampling

Disabled initially:

* global horizontal flipping
* speed jitter
* frame dropping
* artificial compression
* blur
* strong lighting changes
* aggressive rotation
* heavy random erasing

## Acceptance Criteria

* A real dataset sample can be decoded.
* Exactly the configured number of frames is returned.
* The output matches the model-layer contract.
* Evaluation preprocessing is deterministic.
* Train transforms remain temporally consistent.
* Short and malformed videos are handled according to documented policy.
* A batch from ASL Citizen passes through both model adapters.

## Non-Goals

* Full training
* Robustness augmentation
* WLASL integration
* Dataset merging
* Serving-time video formats

## Completion Artifact

A reproducible ASL Citizen data pipeline shared by both model families.

---

# Phase 3: Training Layer

## Objective

Connect the model and data layers into a complete supervised fine-tuning system.

## Tasks

* Implement multiclass cross-entropy training.
* Construct the optimizer through configuration.
* Add learning-rate scheduling.
* Add mixed precision where supported.
* Add gradient accumulation.
* Add gradient clipping.
* Add configurable full-model fine-tuning.
* Add epoch and step logging.
* Add validation scheduling.
* Add best-checkpoint selection using validation metrics.
* Add periodic checkpointing.
* Add checkpoint resume.
* Capture configuration and environment metadata.
* Capture Git commit and dataset identities.
* Add one-batch and short smoke-run tests.
* Add a normal command-line training entry point.

## Initial Training Policy

The initial default should be:

* pretrained backbone
* replaced ASL classification head
* full-model fine-tuning
* cross-entropy loss
* baseline data transformations
* official ASL Citizen splits
* no test-set access during tuning

## Preflight Mode

Before a full run, the system should support a short preflight that measures:

* GPU type
* peak memory
* batch size
* videos per second
* estimated epoch duration
* data-loader behavior
* checkpoint size
* checkpoint-resume success

A preflight is not a real experiment and must be labeled clearly.

## Acceptance Criteria

* One complete epoch can run on a controlled subset.
* Loss decreases during a smoke run.
* Validation executes without updating weights.
* Best-checkpoint tracking works.
* Interrupted training can resume.
* The run captures enough metadata for reproducibility.
* No silent dataset reduction occurs.
* The same training orchestration supports both model adapters.

## Non-Goals

* Final hyperparameter tuning
* Robustness training
* Confidence calibration
* Model serving

## Completion Artifact

A reusable, resumable full-dataset fine-tuning command.

---

# Phase 4: Evaluation Layer

## Objective

Build reliable clean evaluation and confidence-analysis tooling.

This phase may proceed while the first full baseline trains, provided it does not alter the active training path.

## Phase 4A: Core Classification Metrics

### Tasks

* Implement top-1 accuracy.
* Implement top-5 accuracy.
* Implement macro F1.
* Implement weighted F1.
* Implement mean per-class accuracy.
* Implement per-class precision and recall.
* Implement per-signer accuracy.
* Implement worst-signer reporting.
* Implement confusion analysis.
* Persist per-example predictions and logits.
* Add metric correctness tests.

### Acceptance Criteria

* Metrics match trusted reference calculations.
* Per-class and per-signer aggregation is reproducible.
* Evaluation preserves video IDs, signer IDs, and true labels.
* Results can be generated from a saved checkpoint.

## Phase 4B: Calibration

### Tasks

* Export validation logits.
* Compute raw maximum-softmax confidence.
* Implement negative log-likelihood.
* Implement expected calibration error.
* Generate reliability data.
* Fit temperature scaling on validation logits.
* Compare pre- and post-calibration behavior.
* Persist calibration parameters.

### Acceptance Criteria

* Temperature fitting never uses test labels.
* Calibration does not change top-1 class ranking.
* Pre- and post-calibration metrics are reported.
* Calibration parameters can be applied to later test predictions.

## Phase 4C: Selective Prediction

### Tasks

* Evaluate confidence thresholds.
* Compute coverage.
* Compute selective accuracy.
* Produce accuracy-versus-coverage results.
* Select candidate thresholds on validation data.
* Apply selected thresholds once to test results.

### Initial Candidate Target

The initial serving-candidate benchmark is:

```text
accepted prediction accuracy: at least 90%
coverage: at least 50%
```

This is a project target, not a universal guarantee.

## Non-Goals

* Out-of-distribution training
* Production threshold enforcement
* API responses
* Drift monitoring

## Completion Artifact

A checkpoint-to-evaluation pipeline that produces classification, signer, calibration, and selective-prediction reports.

---

# Phase 5: Baseline Experiments

## Objective

Train and compare clean baselines for VideoMAE-Base and Video Swin-Tiny.

## Experiment 1: VideoMAE Baseline

### Required Properties

* ASL Citizen full usable training split
* official validation and test splits
* full vocabulary
* baseline transforms
* cross-entropy
* full-model fine-tuning
* reproducible configuration
* checkpoint resume
* clean evaluation
* calibration analysis

## Experiment 2: Video Swin Baseline

Use the same protocol where practical.

Architecture-specific optimization differences are allowed only when documented.

## Comparison Criteria

Compare:

* top-1 accuracy
* top-5 accuracy
* macro F1
* mean class accuracy
* per-signer performance
* worst-signer accuracy
* negative log-likelihood
* calibrated ECE
* selective accuracy and coverage
* training throughput
* peak GPU memory
* checkpoint size
* inference throughput during evaluation

## Fairness Requirements

The two baseline experiments should share:

* dataset manifests
* class map
* split definitions
* evaluation transforms
* primary metrics
* seed policy
* reporting format

Do not claim one architecture is better based on incomparable preprocessing or test conditions.

## Acceptance Criteria

* Both models complete a real baseline run or have a clearly documented resource failure.
* Both are evaluated using the same clean protocol.
* Results include uncertainty or seed variability where feasible.
* The comparison identifies a leading architecture or states that the result is inconclusive.
* Baseline checkpoints remain preserved.

## Completion Artifact

Two clean baseline experiment records and a model-comparison report.

---

# Phase 6: Baseline Robustness Evaluation

## Objective

Measure where each clean baseline fails before adding robustness augmentation to training.

No model retraining occurs in this phase.

## Robustness Conditions

Candidate controlled conditions include:

* horizontally mirrored clips
* mild speed changes
* temporal start shifts
* temporal end truncation
* lower resolution
* mild blur
* video compression
* brightness changes
* contrast changes

Each perturbation must be:

* deterministic or reproducibly seeded
* separately reported
* realistically bounded
* label-preserving to the best of current knowledge

## Handedness and Mirroring Audit

Before interpreting mirrored results:

* determine whether source videos are already mirrored
* determine whether stored pixel data matches displayed previews
* identify any handedness metadata
* avoid assuming every class is mirror-invariant
* mark signs requiring review

Global horizontal flipping must not enter training solely because mirrored test performance is weak.

## Tasks

* Add robustness evaluation transforms.
* Run each baseline under each condition.
* Measure clean-to-perturbed performance drops.
* Measure calibration changes.
* Identify the most affected classes.
* Identify the most affected signers.
* Identify likely invalid or semantically unsafe perturbations.
* Rank robustness weaknesses by practical importance.

## Acceptance Criteria

* Clean baseline weights remain unchanged.
* Every robustness result is reported separately.
* The project identifies measured weaknesses rather than assumed weaknesses.
* Proposed training augmentations are justified by the results.
* Mirroring conclusions remain appropriately qualified.

## Completion Artifact

A robustness profile for each baseline and a prioritized augmentation plan.

---

# Phase 7: Targeted Robustness Fine-Tuning

## Objective

Train separate model variants using augmentations selected from Phase 6 findings.

## Experimental Rule

Every robustness run starts from the original pretrained checkpoint unless an experiment explicitly studies continued fine-tuning.

Do not overwrite baseline checkpoints.

## Candidate Ablations

Depending on Phase 6 findings:

* baseline
* baseline plus temporal augmentation
* baseline plus mild visual degradation
* baseline plus controlled spatial variation
* baseline plus reviewed safe-class flipping
* baseline plus all selected augmentations

Do not enable all possible augmentations without ablation evidence.

## Evaluation

Each robustness-trained model must be evaluated on:

* clean ASL Citizen test data
* the robustness conditions it targets
* calibration
* per-class metrics
* per-signer metrics
* selective accuracy and coverage

## Selection Criteria

Model selection should consider:

* clean accuracy
* macro F1
* worst-signer performance
* robustness gains
* clean-to-robust tradeoffs
* calibration
* compute requirements

A small clean-accuracy decrease may be acceptable when realistic robustness improves materially, but the tradeoff must be explicit.

## Acceptance Criteria

* Each augmentation run has a clear hypothesis.
* Results compare against the untouched baseline.
* Clean and robustness results are both reported.
* Harmful or ineffective augmentations are rejected.
* The project selects a preferred training strategy or states that the baseline remains better.

## Completion Artifact

A controlled augmentation ablation report and selected robustness policy.

---

# Phase 8: Secondary and External Benchmarking

## Objective

Determine whether the selected architecture and preprocessing generalize beyond ASL Citizen.

## Phase 8A: WLASL Audit

### Tasks

* Inspect WLASL annotations and source availability.
* Audit missing videos.
* Preserve WLASL’s own split definitions.
* Build a separate WLASL label map.
* Document collection and preprocessing differences.
* Avoid contaminating ASL Citizen artifacts.

## Phase 8B: Independent WLASL Benchmark

Train and test selected architectures using WLASL’s own vocabulary and splits.

This answers:

```text
Does the architecture ranking hold on another isolated-sign dataset?
```

## Phase 8C: Cross-Dataset Evaluation

Build a reviewed label-overlap table between ASL Citizen and WLASL.

Mapping statuses should distinguish:

* exact
* reviewed match
* ambiguous
* variant
* no match

Only verified mappings enter external evaluation.

This answers:

```text
How well does an ASL Citizen-trained model transfer to WLASL recording conditions?
```

## Acceptance Criteria

* WLASL results remain separate from ASL Citizen metrics.
* Label mappings are explicit and reviewable.
* Ambiguous mappings are excluded.
* Cross-dataset performance is not represented as ordinary in-dataset test accuracy.
* Dataset-shift limitations are documented.

## Completion Artifact

A secondary benchmark and external-generalization report.

---

# Phase 9: Training Handoff Package

## Objective

Package the selected training result for future consumption by `ASL_serving`.

This phase does not implement serving.

## Required Handoff Contents

The logical export should include:

* selected model weights
* model architecture and configuration
* stable label map
* frame-count requirement
* temporal sampling policy
* resolution and crop policy
* RGB and normalization requirements
* mirroring policy
* calibration parameters
* selected confidence threshold
* clean evaluation summary
* robustness evaluation summary
* dependency versions
* training Git commit
* dataset and manifest identities

## Preprocessing Contract

The handoff must state exactly how a valid clip becomes a model tensor.

The future serving project should not need to infer:

* frame order
* number of frames
* channel order
* resolution
* crop behavior
* normalization
* tensor dimensions
* confidence calibration

## Acceptance Criteria

* A fresh evaluation environment can load the exported bundle.
* The exported model reproduces expected test metrics within normal numerical tolerance.
* The label map matches model output dimensions.
* Calibration parameters apply correctly.
* The bundle contains no raw training data.
* No serving code is introduced into `ASL_training`.

## Completion Artifact

A versioned training-to-serving model bundle and handoff document.

---

# Deferred Serving Roadmap

The following work belongs to the future sibling project:

```text
ASL PROJECT/ASL_serving/
```

Deferred topics include:

* webcam recording
* browser video encoding
* clip boundary capture
* inference preprocessing
* API design
* frontend interface
* low-confidence rejection behavior
* model loading
* CPU or GPU serving
* ONNX or quantization
* Docker deployment
* canary releases
* production logging
* data drift
* concept drift
* retraining triggers

These concerns should not alter the initial training implementation unless a specific training artifact is required for reproducible inference.

---

# Immediate Next Task

Begin Phase 0 if the repository foundation does not yet exist.

Otherwise begin Phase 1:

```text
Implement the basic shared model layer for VideoMAE-Base and Video Swin-Tiny using dummy video batches only.
```

The first phase should establish model contracts and smoke tests without implementing the full ASL Citizen data pipeline.
