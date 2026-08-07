# Current Phase

## Active Phase

Phase 0: Repository Foundation — complete
Phase 1: Basic Model Layer — complete, pending review

## Status

Both phases are implemented and validated. Neither has been archived, and `docs/ROADMAP.md` has not been updated, because `CLAUDE.md` requires explicit approval before archiving a phase, marking it complete in the roadmap, or creating the next phase plan.

Awaiting approval to archive Phases 0 and 1 and open Phase 2.

## Current Task

None in progress. Next task is Phase 2A, the ASL Citizen dataset audit.

## Blockers

Phase 2 cannot begin until the ASL Citizen dataset is accessible and its runtime root is known. See Open Questions.

---

# Phase 0: Repository Foundation

## Objective

Establish `ASL_training` as an installable, testable Python project ready for the model-layer implementation.

## Documentation Repair

Three authoritative documents were empty or truncated, and two cross-document conflicts existed.

- [x] Write `docs/EVALUATION_CONTRACT.md`, previously empty despite being listed as authoritative in `CLAUDE.md` and cross-referenced by all three other contracts.
- [x] Write `docs/DECISIONS.md`, previously empty; added format and initial decisions.
- [x] Add `docs/ENVIRONMENTS.md` covering local, Colab, and Kaggle execution.
- [x] Resolve the `PROJECT.md` location conflict in `docs/ARCHITECTURE.md`.
- [x] Resolve the phase-numbering conflict between `docs/PROJECT.md` and `docs/ROADMAP.md`.
- [x] Complete this file, previously truncated mid-directory-tree.

## Repository Setup

- [x] Confirm the repository root. Resolved as the parent workspace; see D-001.
- [x] Initialize Git.
- [x] Add a Python `src` package layout.
- [x] Add the initial test structure.
- [x] Add dependency and project configuration.
- [x] Add a minimal `README.md`.
- [x] Add repository ignore rules.
- [x] Add placeholder directories supporting the current architecture.

## Acceptance Criteria

- [x] The package installs with `pip install -e ASL_training`.
- [x] `import asl_training` resolves through the `src` layout.
- [x] The test command runs successfully.
- [x] Lint and format commands are defined and pass.
- [x] No raw data, checkpoints, or credentials are tracked by Git.
- [x] `ASL_training` does not import from or reference `ASL_serving`.
- [x] Every authoritative document exists and is non-empty.
- [x] The active phase and authoritative documents are identifiable from the docs alone.

## Deviation

Git is rooted at the parent workspace rather than at `ASL_training/`, so `CLAUDE.md` and `.gitignore` live at the parent level rather than being duplicated into the subproject. See D-001.

---

# Phase 1: Basic Model Layer

## Objective

Establish a shared video-classification contract and verify both pretrained architectures on synthetic batches.

## Tasks

- [x] Define the canonical logical input shape: `[batch, frames, channels, height, width]`.
- [x] Define the shared classification output: `VideoClassifierOutput` carrying logits and optional loss.
- [x] Add a model factory with an explicit, stable architecture registry.
- [x] Implement the VideoMAE adapter.
- [x] Implement the Video Swin adapter.
- [x] Load pretrained weights.
- [x] Replace each original classification head with a configurable ASL head.
- [x] Support a configurable number of classes, resolved at runtime from the label map.
- [x] Add model-specific tensor adaptation internally.
- [x] Report model parameter counts.
- [x] Add dummy-forward smoke tests.
- [x] Add checkpoint state-loading tests.

## Acceptance Criteria

For both architectures:

- [x] Pretrained construction succeeds.
- [x] A dummy video batch is accepted.
- [x] Output logits have shape `[batch_size, num_classes]`.
- [x] The classification head has the configured output size.
- [x] Architecture-specific objects do not leak past the model layer.
- [x] Tests verify input and output behavior.
- [x] No real dataset implementation is required.

## Validation

| Command | Result |
|---|---|
| `pytest ASL_training/tests` | 115 passed |
| `pytest ASL_training/tests -m pretrained` | 7 passed |
| `ruff check ASL_training` | passed |
| `ruff format --check ASL_training` | passed |
| `scripts/model_preflight.py`, VideoMAE, 2731 classes | passed |
| `scripts/model_preflight.py`, Video Swin, 2731 classes | passed |

Preflight measured, at 2731 classes:

| | VideoMAE-Base | Video Swin-Tiny |
|---|---|---|
| Total parameters | 88,336,555 | 29,950,609 |
| Approx. fp32 size | 337 MB | 114 MB |
| Head parameters | 2,100,139 | 2,100,139 |
| Initial loss | 7.70 | 7.98 |

Expected initial loss for a random head over 2731 classes is ln(2731) = 7.91.

VideoMAE-Base is roughly three times the size of Video Swin-Tiny, which will shape batch size and epoch time on Colab. Both share a 768-dimensional feature width, hence identical head sizes.

## Decisions Recorded

* D-002: Video Swin weights come from torchvision rather than mmaction2.
* D-003: Both architectures run at 16 frames in the baseline.
* D-005: Model-layer tests do not require network access.
* D-006: VideoMAE legacy attention-bias names are repaired on load.

## Finding: silent VideoMAE weight-loading defect

Preflight surfaced that `transformers` 5.x does not translate VideoMAE's published `q_bias`/`v_bias` parameter names to the `query.bias`/`value.bias` it expects. Twenty-four bias tensors were left at zero while `from_pretrained` reported success. Measured on layer 0: loaded norm 0.0 against the checkpoint's 17.54.

The adapter now detects and repairs this, classifies every missing and unused key as expected or unexplained, and records the result in the load report. See D-006.

This is the reason a real training run must gate on preflight rather than on the test suite: the offline structural suite cannot see it.

---

# Open Questions for Phase 2

These need answers before or during the dataset audit. None block reviewing Phases 0 and 1.

1. **Dataset access.** Which ASL Citizen copy will be used — the official release or a Kaggle mirror? `docs/DATA_CONTRACT.md` requires a hosted copy to be compared against official metadata rather than assumed identical.
2. **Class count.** The preflight used 2731 as the full vocabulary. The audit must establish the real usable class count; it is not yet verified from the data.
3. **Colab storage.** Where will checkpoints and run outputs be written, and where will the dataset be staged from? See `docs/ENVIRONMENTS.md`.
4. **Git remote.** The repository has no remote. The Colab notebook needs one to clone, or must fall back to a Drive copy.
