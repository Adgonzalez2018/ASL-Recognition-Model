# Phase 0: Repository Foundation

Status: Complete
Archived: 2026-08-07

## Objective

Establish `ASL_training` as an installable, testable Python project ready for the model-layer implementation.

## Documentation Repair

Three authoritative documents were empty or truncated when the phase began, and two cross-document conflicts existed.

- [x] Write `docs/EVALUATION_CONTRACT.md`, previously empty despite being listed as authoritative in `CLAUDE.md` and cross-referenced by all three other contracts.
- [x] Write `docs/DECISIONS.md`, previously empty; added format and initial decisions.
- [x] Add `docs/ENVIRONMENTS.md` covering local, Colab, and Kaggle execution.
- [x] Resolve the `PROJECT.md` location conflict in `docs/ARCHITECTURE.md`.
- [x] Resolve the phase-numbering conflict between `docs/PROJECT.md` and `docs/ROADMAP.md`.
- [x] Complete `docs/CURRENT_PHASE.md`, previously truncated mid-directory-tree.

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

## Deviations

Git is rooted at the parent workspace rather than at `ASL_training/`, departing from `docs/ROADMAP.md` Phase 0 task 1. `CLAUDE.md` and `.gitignore` therefore live at the parent level rather than being duplicated into the subproject. Recorded as D-001.

## Phase Summary

The workspace is a Git repository rooted at `ASL PROJECT/`, with `ASL_training` installed as an editable `src`-layout package. All authoritative documents exist and are non-empty. Three documentation gaps and two cross-document conflicts were resolved, and foundational decisions D-001, D-004, and D-005 were recorded.

## Completion Artifact

An installable, testable repository skeleton with a complete and internally consistent document set.
