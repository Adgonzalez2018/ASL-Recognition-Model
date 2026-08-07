# Current Phase

## Active Phase

Phase 2: ASL Citizen Audit and Data Layer

## Status

In progress. Phase 2A tooling and Phase 2B contracts complete; 2A execution blocked on dataset access.

## Objective

Understand the actual ASL Citizen distribution, then build a reproducible path from raw videos to standardized model batches shared by both architectures.

## Current Task

Phase 2C: video decoding, temporal sampling, and transforms. Running the 2A audit once a Kaggle runtime has the mirror attached.

## Blockers

Phase 2A cannot execute until the dataset is reachable from a runtime. The audit reads real files; it cannot be simulated.

Phase 2B and the sampler logic in 2C are unblocked, because they are pure logic testable on synthetic fixtures.

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

Sampler logic is unblocked. Decoder work needs real video fixtures, which can be small and synthetic.

### Tasks

- [ ] Implement video decoding.
- [ ] Convert decoded frames to RGB, verified rather than assumed.
- [ ] Implement fixed-frame temporal sampling.
- [ ] Define and implement the short-video policy.
- [ ] Define and implement long-video sampling.
- [ ] Implement deterministic evaluation sampling.
- [ ] Implement minimal baseline training transforms.
- [ ] Ensure spatial transforms are temporally consistent across a clip.
- [ ] Implement batch collation.
- [ ] Add data-loader configuration.
- [ ] Add focused data-pipeline tests.

### Baseline Transform Policy

Allowed: required resizing, fixed crop size, mild random crop during training, model-compatible normalization, controlled temporal sampling.

Disabled initially: horizontal flipping, speed jitter, frame dropping, compression simulation, blur, strong lighting changes, aggressive rotation, heavy random erasing.

### Acceptance Criteria

- [ ] A real dataset sample decodes.
- [ ] Exactly the configured frame count is returned.
- [ ] Output matches the model-layer contract, verified against both adapters.
- [ ] Evaluation preprocessing is deterministic.
- [ ] Train transforms remain temporally consistent.
- [ ] Short and malformed videos follow documented policy.

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
