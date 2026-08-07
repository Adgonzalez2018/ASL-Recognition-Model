# Current Phase

## Active Phase

Phase 2: ASL Citizen Audit and Data Layer

## Status

In progress. Phase 2B and 2C complete; 2A tooling complete, its execution blocked on dataset access.

## Objective

Understand the actual ASL Citizen distribution, then build a reproducible path from raw videos to standardized model batches shared by both architectures.

## Current Task

Running the Phase 2A audit once a Kaggle runtime has the mirror attached. Everything buildable without the dataset is done.

## Blockers

Phase 2A cannot execute until the dataset is reachable from a runtime. The audit reads real files; it cannot be simulated.

Phases 2B and 2C are complete, having been built and tested against synthetic datasets with real encoded video. The pipeline is ready for real data; only the audit itself needs the dataset.

## Environment Decisions

Settled for this phase:

* **Dataset source**: a Kaggle mirror of ASL Citizen. Per `docs/DATA_CONTRACT.md`, the mirror must be audited against official metadata rather than assumed identical. Any divergence is recorded in the audit, not silently accepted.
* **No local dataset copy.** Development and testing run on synthetic fixtures. The dataset is only ever materialized inside a Colab or Kaggle runtime.
* **Storage split**: dataset on ephemeral runtime disk, checkpoints and run outputs on Google Drive. See D-007.
* **Expected class count**: 2731. This is the value the audit verifies against, not an assumption the code may rely on. The label map is built from the data.

Git remote: `https://github.com/Adgonzalez2018/ASL-Recognition-Model.git`. The Colab and Kaggle notebooks clone from it.

---

## Phase 2A: Dataset Access and Audit

Tooling complete. Execution blocked on dataset access — the audit reads real files and cannot be simulated. Running it is now one command.

### Tooling

- [x] Annotation parser that discovers structure rather than assuming it.
- [x] Layout resolution: split files and video directory, searched recursively.
- [x] Column resolution across official and mirror naming.
- [x] Video probing: frame count, fps, duration, resolution, codec, rotation.
- [x] Audit report generation with a versioned schema.
- [x] `scripts/audit_dataset.py` entry point.
- [x] `notebooks/kaggle/01_dataset_audit.ipynb` launcher.

### Execution

- [ ] Attach the Kaggle mirror and record its identity.
- [ ] Run the layout check and confirm the resolved columns.
- [ ] Run the full audit and commit the report.
- [ ] Compare counts against the official ASL Citizen publication.
- [ ] Resolve every problem the audit reports.

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
- [ ] Generate split-specific manifests from real annotations. Blocked on 2A.

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

Not yet complete.
