# Current Phase

## Active Phase

Phase 2: ASL Citizen Audit and Data Layer

## Status

Phase 2: complete. The audit ran clean against the Kaggle mirror on 2026-08-07.
Phase 3: training layer complete except preflight.
Phase 4: evaluation layer complete. Both started in parallel, since neither needs the dataset.

## Objective

Understand the actual ASL Citizen distribution, then build a reproducible path from raw videos to standardized model batches shared by both architectures.

## Current Task

Commit the audit artifacts, then Phase 5 baseline runs. Phases 2, 3, and 4 are complete apart from training preflight.

## Blockers

None. The audit is complete and reported no integrity failures.

## Environment Decisions

Settled for this phase:

* **Dataset source**: a Kaggle mirror of ASL Citizen. Per `docs/DATA_CONTRACT.md`, the mirror must be audited against official metadata rather than assumed identical. Any divergence is recorded in the audit, not silently accepted.
* **No local dataset copy.** Development and testing run on synthetic fixtures. The dataset is only ever materialized inside a Colab or Kaggle runtime.
* **Storage split**: dataset on ephemeral runtime disk, checkpoints and run outputs on Google Drive. See D-007.
* **Expected class count**: 2731. This is the value the audit verifies against, not an assumption the code may rely on. The label map is built from the data.

Git remote: `https://github.com/Adgonzalez2018/ASL-Recognition-Model.git`. The Colab and Kaggle notebooks clone from it.

---

## Phase 2A: Dataset Access and Audit

Complete. Tooling built ahead of the data; the audit ran clean on the first attempt.

### Tooling

- [x] Annotation parser that discovers structure rather than assuming it.
- [x] Layout resolution: split files and video directory, searched recursively.
- [x] Column resolution across official and mirror naming.
- [x] Video probing: frame count, fps, duration, resolution, codec, rotation.
- [x] Audit report generation with a versioned schema.
- [x] `scripts/audit_dataset.py` entry point.
- [x] `notebooks/kaggle/01_dataset_audit.ipynb` launcher.

### Execution

- [x] Attach the Kaggle mirror and record its identity: `abd0kamel/asl-citizen`.
- [x] Run the layout check and confirm the resolved columns.
- [x] Run the full audit. Complete: all 83,399 videos probed.
- [x] Compare counts against the official ASL Citizen publication.
- [x] Resolve every problem the audit reports.
- [x] Commit the audit report and label map to the repository.

### Audit Results

Run 2026-08-07 on Kaggle against mirror `abd0kamel/asl-citizen`.

```text
label map identity  asl_citizen:2731:sha256:3a0b873befec998c
manifest identity   asl_citizen:83399:sha256:b864a6d5d84c5531
```

| Split | Samples | Share | Per class | Signers |
|---|---|---|---|---|
| train | 40,154 | 48.1% | 14.7 | 35 |
| validation | 10,304 | 12.4% | 3.8 | 6 |
| test | 32,941 | 39.5% | 12.1 | 11 |
| **total** | **83,399** | | **30.5** | **52** |

**Integrity: clean.** No signer leakage, no duplicate sample IDs or paths, no video in more than one split. The 52 signers partition exactly across the three splits. Annotation rows equal manifest records, so no row was dropped during parsing.

**Media: every one of the 83,399 videos decoded.** Zero missing, zero unreadable, zero zero-frame.

| | |
|---|---|
| Frame count | min 3, p05 42, median 75, p95 158, max 680 |
| Duration | min 0.064s, median 2.57s, p95 5.27s, max 22.6s |
| Frame rate | min 11.3, median 30.0, max 120.0 |
| Resolutions | 640x480 (80,184), 960x540 (3,211), 480x640 (4, portrait) |
| Codecs | h264 (79,873), mpeg4 (3,526) |
| Rotation metadata | none |

**Class balance is unusually even**: 21 to 45 samples per class, median 31, imbalance ratio 2.14. No class has a single sample. This is far better balanced than most large-vocabulary video datasets.

**Metadata absent, as expected**: no handedness, no mirroring status. No flip policy can be justified from this dataset alone, per `docs/DATA_CONTRACT.md`.

### Findings

**1. The single reported problem is already handled.** Eighteen clips fall below the configured 16 frames, the shortest being 3. The short-video policy repeats indices through uniform sampling, preserving chronology without inventing motion. Verified end to end: a 3-frame clip produces a valid `[16, 3, 224, 224]` tensor through both the evaluation and training paths. Locked in by `tests/data/test_real_dataset_characteristics.py`.

**2. Total count differs from the commonly cited figure.** The mirror's split files list 83,399 videos; ASL Citizen is usually described as 83,912, a difference of 513. Every listed video is present and decodable, so this is a difference in the *annotation files*, not missing media. Worth confirming against the official release before treating a run as a reproduction of published results; it does not block training.

**3. Training data is thinner than the totals suggest.** The official split allocates only 48% to train, giving roughly 14.7 training samples per class across 2,731 classes. The success targets in `docs/PROJECT.md` should be read with that in mind — this is a large-vocabulary, low-shot problem, not a data-rich one.

**4. Validation is thin per class but macro F1 remains usable.** Validation averages 3.8 samples per class, so any individual class F1 is very noisy. Macro F1 averages 2,731 such estimates, which reduces the standard error by roughly 52x, so the aggregate is stable enough to select on. D-008 stands. Per-class validation figures should not be read individually.

**5. Four portrait videos crop vertically.** The 480x640 clips resize on the short side and centre-crop, removing top and bottom, which could clip raised hands. Documented rather than corrected: 4 of 83,399 does not justify a resolution-dependent code path, and a per-aspect-ratio rule would introduce an unreviewed preprocessing difference between samples.

**6. No rotation metadata anywhere**, which removes the orientation risk between Kaggle and Colab entirely.

**7. 27% of the vocabulary is numbered variants.** 309 base words carry more than one variant — `ABOUT1`/`ABOUT2`, `DROWN1` through `DROWN5` — covering 739 classes. These are distinct signs for the same English word. Two consequences:

* Confusion analysis will likely show variant pairs confused with one another. That is linguistically meaningful rather than a defect, and should be read as such.
* Phase 8 cross-dataset work will hit these hardest. Mapping `ABOUT1` and `ABOUT2` onto a single WLASL `about` is exactly the ambiguous case `docs/DATA_CONTRACT.md` requires be excluded rather than guessed.

The label map keeps every variant distinct. The normalized-collision check would have refused to merge them.

**8. Recorded `source_id` is uninformative in this run.** The notebook passed the Kaggle attachment folder name, `datasets`, rather than the mirror slug. The true identity survives in `dataset_root` (`abd0kamel/asl-citizen`), so nothing is lost, but the notebook has been corrected to derive the owner/dataset path for future runs.

### Pipeline verification against real characteristics

Every property the audit found was exercised against the actual pipeline: 3 and 680 frame counts, all three resolutions including portrait, both codecs, and frame rates from 11 to 120. All produce valid tensors. See `tests/data/test_real_dataset_characteristics.py`.

### Design notes

The parser refuses to guess. An unidentifiable label column, an ambiguous split file, or an unrecognized layout raises rather than falling back, because a wrong guess here means training against the wrong target and would not surface until the results looked strange.

The audit reconciles annotation rows against manifest records, so a row that silently fails to become a record is caught rather than absorbed.

Short-video reporting follows the configured frame count. Clips shorter than a frame count nobody is training at yet are recorded as information, not flagged as problems.

Verified end to end against a synthetic dataset with real encoded video: a deliberately damaged copy surfaced signer leakage, a duplicate sample ID, a duplicate path, one video spanning two splits, a missing file, and a dropped row, and exited non-zero.

### Acceptance Criteria

The audit establishes exact usable sample counts, class count, signer counts, split assignments, missing and corrupted samples, class imbalance, duration and frame-rate ranges, unresolved metadata concerns, and any difference between the Kaggle mirror and official metadata.

No training implementation may rely on unverified assumptions about dataset structure.

---

## Phase 2B: Manifest and Label Contracts

Unblocked. Pure logic, tested on synthetic fixtures.

### Tasks

- [x] Define the manifest schema.
- [x] Build stable gloss-to-ID and ID-to-gloss mappings.
- [x] Preserve signer identifiers.
- [x] Include audited video metadata where useful.
- [x] Add duplicate detection.
- [x] Add signer-leakage validation.
- [x] Add manifest versioning and identity checks.
- [x] Generate split-specific manifests from real annotations.

### Acceptance Criteria

- [x] Every usable training sample maps to one stable class ID.
- [x] Both model families consume the same manifests, since the manifest carries
      no architecture-specific content.
- [x] Signer leakage across splits is detected and blocks validation.
- [x] Duplicate sample IDs, duplicate paths, and one video spanning splits are detected.
- [x] Label maps are stable, identity-checked, and tested.
- [x] Missing or corrupted records are reported explicitly.

### Implementation

```text
src/asl_training/data/
├── label_map.py   vocabulary, deterministic construction, identity
└── manifest.py    record schema, split integrity, leakage detection
```

92 tests, all offline on synthetic fixtures.

Design notes worth carrying forward:

* The label map refuses to merge distinct glosses that normalize identically, rather than fusing them. Two source glosses differing only by case or whitespace may be one sign recorded inconsistently or two different signs; only review can tell.
* Manifest identity covers sample ID, path, class ID, signer, and split — the fields whose change alters what an experiment means. Audited metadata such as resolution and codec is excluded, so re-auditing video properties does not invalidate existing manifests.
* Signer overlap can be permitted, but only explicitly, and it still surfaces as a warning.
* Validation returns a report rather than raising, so a caller can inspect every problem at once. `raise_if_invalid()` enforces it. Warnings never raise and are listed separately, so they cannot conceal a hard failure.

---

## Phase 2C: Video Input Pipeline

Complete, verified against synthetic datasets with real encoded video.

### Tasks

- [x] Implement video decoding.
- [x] Convert decoded frames to RGB, verified rather than assumed.
- [x] Implement fixed-frame temporal sampling.
- [x] Define and implement the short-video policy.
- [x] Define and implement long-video sampling.
- [x] Implement deterministic evaluation sampling.
- [x] Implement minimal baseline training transforms.
- [x] Ensure spatial transforms are temporally consistent across a clip.
- [x] Implement batch collation.
- [x] Add data-loader configuration.
- [x] Add focused data-pipeline tests.

### Baseline Transform Policy

Allowed: required resizing, fixed crop size, mild random crop during training, model-compatible normalization, controlled temporal sampling.

Disabled initially: horizontal flipping, speed jitter, frame dropping, compression simulation, blur, strong lighting changes, aggressive rotation, heavy random erasing. The disabled list is recorded in the resolved configuration rather than left implied.

### Acceptance Criteria

- [x] A real dataset sample decodes.
- [x] Exactly the configured frame count is returned, for every strategy and clip length.
- [x] Output matches the model-layer contract, verified by passing a real decoded batch through both adapters.
- [x] Evaluation preprocessing is deterministic, and non-deterministic configuration is rejected rather than merely discouraged.
- [x] Train transforms remain temporally consistent.
- [x] Short and malformed videos follow documented policy.

### Implementation

```text
src/asl_training/data/
├── sampling.py    temporal frame selection and the short-video policy
├── decode.py      video to ordered RGB frames
├── transforms.py  spatial preprocessing, one parameter set per clip
└── dataset.py     dataset, collation, loader config, preprocessing identity

configs/datasets/asl_citizen.yaml
```

### Design notes

**Determinism is enforced, not documented.** Constructing a validation or test dataset with a random sampler or a training transform raises. An unreproducible evaluation run should be impossible to configure, not merely discouraged.

**Order and colour are verified against pixels.** Decode tests encode a red ramp that increases with frame index, so chronological order is checked from decoded content rather than trusted. A constant blue channel catches RGB/BGR swaps.

**Temporal consistency is tested on identical-frame clips.** If every frame of a clip is the same image, any variation in the output means spatial parameters were drawn per frame. That is the bug the test exists to catch.

**Time reversal is unavailable.** Not disabled by default — absent from the strategy list, with a test asserting it stays absent. Reversed motion changes what a sign means.

**Runtime failures are counted.** The default policy raises, because a sample failing at runtime means the audit missed something and the run's dataset is not what was recorded. The `skip` policy substitutes a neighbour and records the failure; it never silently shrinks a batch.

### Finding: silent normalization bug

The output-range test caught a real defect. `normalize` scaled `uint8` to [0, 1] conditionally on dtype, but both transforms had already cast to float during resizing, so the 1/255 scaling was skipped. Every tensor left the pipeline roughly 255x too large.

Nothing raised. Training would have diverged immediately and looked like a bad learning rate or an unstable model.

The fix removes the dtype-conditional behaviour: `to_unit_float` converts explicitly and `normalize` only standardizes. A regression test asserts the output range directly, since that is the symptom a shape check cannot see.

---

## Non-Goals

* Full training
* Robustness augmentation
* WLASL integration
* Dataset merging
* Serving-time video formats

## Completion Artifact

A reproducible ASL Citizen data pipeline shared by both model families.

## Phase Summary

Complete. The ASL Citizen data pipeline runs end to end on the real dataset: 83,399 videos across 2,731 classes and 52 signers, signer-independent splits verified, every video decodable, and every edge case the data contains exercised against the pipeline.

---

# Phase 3: Training Layer (started in parallel)

Phase 2 is held open pending the audit, which needs the dataset. Phase 3 does not: the training layer is testable end to end on synthetic manifests and tiny models, and nothing the audit could find would change its design.

This is a deliberate deviation from the roadmap's strict phase ordering, recorded here rather than left implicit.

## Tasks

- [x] Implement multiclass cross-entropy training.
- [x] Construct the optimizer through configuration.
- [x] Add learning-rate scheduling.
- [x] Add mixed precision where supported.
- [x] Add gradient accumulation.
- [x] Add gradient clipping.
- [x] Add configurable full-model fine-tuning.
- [x] Add epoch and step logging.
- [x] Add validation scheduling.
- [x] Add best-checkpoint selection using validation metrics.
- [x] Add periodic checkpointing.
- [x] Add checkpoint resume.
- [x] Capture configuration and environment metadata.
- [x] Capture Git commit and dataset identities.
- [x] Add one-batch and short smoke-run tests.
- [x] Add a normal command-line training entry point.
- [x] Preflight mode reporting throughput and estimated epoch duration.

## Acceptance Criteria

- [x] One complete epoch runs on a controlled subset.
- [x] Loss decreases during a smoke run.
- [x] Validation executes without updating weights.
- [x] Best-checkpoint tracking works.
- [x] Interrupted training resumes, verified across separate invocations.
- [x] The run captures enough metadata for reproducibility.
- [x] No silent dataset reduction occurs; truncation forces a `subset` label.
- [x] The same orchestration supports both model adapters.

## Implementation

```text
src/asl_training/training/
├── config.py      resolved run configuration
├── optim.py       optimizer groups, per-optimizer-step scheduling
├── checkpoint.py  atomic checkpointing, resume, compatibility validation
└── loop.py        training orchestration

configs/training/baseline.yaml
scripts/train.py
```

## Design notes

**Resume is treated as the normal path, not recovery.** Per D-004, Colab sessions end without warning. Checkpoint writes are atomic — temp file, fsync, rename — and the previous checkpoint is retained, so a session killed mid-write cannot leave a truncated file as the only resume point. Loading falls back to the retained copy and says so.

**Resume is distinguished from transfer.** Resuming validates architecture, class count, label-map identity, preprocessing identity, fine-tuning strategy, optimizer type, and scheduler type. Any mismatch fails with a message pointing at model-state loading instead. Restoring optimizer momentum belonging to a different configuration would perturb a run invisibly.

**The schedule advances per optimizer step.** Under gradient accumulation, stepping per micro-batch would compress the schedule by the accumulation factor with no error at all. A test asserts `scheduler.last_epoch == optimizer_step < micro_step`.

**Accumulation arithmetic is verified by equivalence.** Batch 8 and batch 4 with accumulation 2 are trained from the same seed on the same data, and the resulting weights must agree. Scaling the loss twice, or not at all, moves them apart — that is the failure this catches, because neither shows up as an error.

**A reduced run cannot pose as a baseline.** `--limit-samples` forces `run_kind` to `subset` even when the caller explicitly passed `--run-kind full`, and the warning says so.

**The test split is never loaded.** The training command reads only train and validation manifests, and a test asserts it by spying on manifest loading.

**Non-finite losses are counted, not hidden.** The batch is skipped so one bad step cannot poison the weights, but the count appears in the epoch record.

## Validation

| Command | Result |
|---|---|
| `pytest ASL_training/tests` | passing |
| `ruff check` / `ruff format --check` | passing |

Training-layer coverage: checkpoint and resume, optimizer and scheduler, loop behaviour, and an end-to-end integration test that runs the real `scripts/train.py` from annotations through manifests, decoding, training, checkpointing, and resume across separate invocations.

## Preflight

`scripts/train_preflight.py` measures what a run will cost before one is started: throughput, peak GPU memory, estimated epoch and run duration, checkpoint size, and whether the data loader or the GPU is limiting.

Design notes:

* **Data and compute are timed separately.** The clock starts before the batch is pulled, so time spent waiting on the loader is attributed to data. Which side dominates is the actionable output: video decoding is CPU-bound and Colab's CPU allocation is modest, so the loader is a likely limiter, and the fix there is worker count rather than anything about the model.
* **Warmup steps are discarded.** The first passes allocate caching-allocator blocks, compile kernels, and fill the prefetch queue, so timing them reports a number no later step reproduces.
* **A loader restart is recorded and warned about.** If the dataset is smaller than the measured span the loop wraps, and those repeated clips decode from a warm page cache. Throughput would look better than reality, so the wrap count is reported and flagged.
* **It exercises the real path**, including a checkpoint save and load, so checkpoint size is measured rather than guessed. That figure drives the Drive budget under D-007.
* **It cannot be mistaken for an experiment.** `run_kind` is `preflight`, the report says so in its own payload, and no loss or accuracy is reported at all — only cost.

## Remaining
* ~~The default metric set is a placeholder.~~ Resolved: in-training validation now computes the real restricted metric set, and `selection_metric` is macro F1. See D-008.

---

# Phase 4: Evaluation Layer (started in parallel)

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
