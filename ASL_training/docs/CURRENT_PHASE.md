# Current Phase

## Active Phase

Phase 2: ASL Citizen Audit and Data Layer

## Status

In progress. Phase 2B foundations started.

## Objective

Understand the actual ASL Citizen distribution, then build a reproducible path from raw videos to standardized model batches shared by both architectures.

## Current Task

Phase 2B: label map and manifest contracts, built and tested against synthetic fixtures so they are ready the moment real data is reachable.

## Blockers

Phase 2A cannot execute until the dataset is reachable from a runtime. The audit reads real files; it cannot be simulated.

Phase 2B and the sampler logic in 2C are unblocked, because they are pure logic testable on synthetic fixtures.

## Environment Decisions

Settled for this phase:

* **Dataset source**: a Kaggle mirror of ASL Citizen. Per `docs/DATA_CONTRACT.md`, the mirror must be audited against official metadata rather than assumed identical. Any divergence is recorded in the audit, not silently accepted.
* **No local dataset copy.** Development and testing run on synthetic fixtures. The dataset is only ever materialized inside a Colab or Kaggle runtime.
* **Storage split**: dataset on ephemeral runtime disk, checkpoints and run outputs on Google Drive. See D-007.
* **Expected class count**: 2731. This is the value the audit verifies against, not an assumption the code may rely on. The label map is built from the data.

Outstanding: the Git remote URL, needed so the Colab notebook can clone rather than depend on a Drive copy.

---

## Phase 2A: Dataset Access and Audit

Blocked on dataset access. The tooling is built ahead of the data so that the audit is one command once a runtime has the files.

### Tasks

- [ ] Resolve the Kaggle mirror identity and record it.
- [ ] Compare the mirror's structure and metadata against the official ASL Citizen release.
- [ ] Inspect the annotation and split files.
- [ ] Confirm the official train, validation, and test structure.
- [ ] Identify signer metadata.
- [ ] Count classes, videos, and signers by split.
- [ ] Audit class-frequency distribution.
- [ ] Audit video durations, frame rates, and resolutions.
- [ ] Detect missing and corrupted videos.
- [ ] Determine whether mirroring or handedness metadata exists.
- [ ] Produce a versioned dataset audit report.

### Acceptance Criteria

The audit establishes exact usable sample counts, class count, signer counts, split assignments, missing and corrupted samples, class imbalance, duration and frame-rate ranges, unresolved metadata concerns, and any difference between the Kaggle mirror and official metadata.

No training implementation may rely on unverified assumptions about dataset structure.

---

## Phase 2B: Manifest and Label Contracts

Unblocked. Pure logic, tested on synthetic fixtures.

### Tasks

- [ ] Define the manifest schema.
- [ ] Build stable gloss-to-ID and ID-to-gloss mappings.
- [ ] Generate split-specific manifests.
- [ ] Preserve signer identifiers.
- [ ] Include audited video metadata where useful.
- [ ] Add duplicate detection.
- [ ] Add signer-leakage validation.
- [ ] Add manifest versioning and identity checks.

### Acceptance Criteria

- [ ] Every usable training sample maps to one stable class ID.
- [ ] Both model families consume the same manifests.
- [ ] Train, validation, and test splits remain signer-independent.
- [ ] Duplicate videos are detected.
- [ ] Label maps are stable and tested.
- [ ] Missing or corrupted records are reported explicitly.

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
