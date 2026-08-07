# Current Phase

## Active Phase

Phase 0: Repository Foundation

## Status

In progress.

## Objective

Establish `ASL_training` as an installable, testable Python project ready for the model-layer implementation.

All implementation work must remain inside:

```text
ASL PROJECT/ASL_training/
```

Do not create or modify implementation files inside `ASL_serving`.

## Current Task

Initialize the repository foundation and validate the development environment.

## Blockers

None.

## Required Work

### Documentation Repair

- [x] Write `docs/EVALUATION_CONTRACT.md`, previously empty.
- [x] Write `docs/DECISIONS.md` with format and initial decisions.
- [x] Add `docs/ENVIRONMENTS.md` covering local, Colab, and Kaggle execution.
- [x] Resolve the `PROJECT.md` location conflict in `docs/ARCHITECTURE.md`.
- [x] Resolve the phase-numbering conflict between `docs/PROJECT.md` and `docs/ROADMAP.md`.
- [x] Complete this file with acceptance criteria and workflow fields.

### Repository Setup

- [x] Confirm the intended repository root. Resolved as the parent workspace; see D-001.
- [x] Initialize Git.
- [x] Add a Python `src` package layout.
- [x] Add the initial test structure.
- [x] Add dependency and project configuration.
- [x] Add a minimal `README.md`.
- [x] Add repository ignore rules.
- [x] Add placeholder directories only where they support the current architecture.

### Package Structure

```text
ASL PROJECT/
├── CLAUDE.md
├── .gitignore
└── ASL_training/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── docs/
    ├── configs/
    │   ├── datasets/
    │   ├── models/
    │   ├── training/
    │   ├── evaluation/
    │   └── experiments/
    ├── src/
    │   └── asl_training/
    │       ├── __init__.py
    │       ├── models/
    │       ├── data/
    │       ├── training/
    │       ├── evaluation/
    │       ├── experiments/
    │       └── utils/
    ├── scripts/
    ├── tests/
    │   ├── models/
    │   ├── data/
    │   ├── training/
    │   ├── evaluation/
    │   └── integration/
    ├── notebooks/
    │   ├── colab/
    │   └── kaggle/
    ├── artifacts/
    │   ├── manifests/
    │   ├── label_maps/
    │   ├── audits/
    │   └── reports/
    ├── outputs/
    └── data/
```

Layer packages are created as empty namespaces in this phase. They are populated by their own phases.

`scripts/` contains only entry points supported by completed phases. Speculative scripts for future phases are not created.

## Acceptance Criteria

- [x] The package installs in a clean environment with `pip install -e .`.
- [x] `import asl_training` resolves through the `src` layout.
- [x] The test command runs successfully.
- [x] Linting and formatting commands are defined and pass.
- [x] No raw data, checkpoints, or credentials are tracked by Git.
- [x] `ASL_training` does not import from or reference `ASL_serving`.
- [x] Every authoritative document listed in `CLAUDE.md` exists and is non-empty.
- [x] The active phase and authoritative documents are identifiable from the docs alone.

## Non-Goals

* Model implementation
* Dataset download or audit
* Training
* Evaluation
* Colab or Kaggle runtime optimization

## Phase Summary

Completed. The workspace is a Git repository rooted at `ASL PROJECT/`, with `ASL_training` installed as an editable `src`-layout package. All nine authoritative documents plus `ENVIRONMENTS.md` exist and are non-empty. Three documentation gaps and two cross-document conflicts identified at the start of the phase were resolved, and five foundational decisions were recorded in `docs/DECISIONS.md`.

Deviations from the roadmap text: Git is rooted at the parent workspace rather than at `ASL_training/` (D-001), so `CLAUDE.md` and `.gitignore` live at the parent level and are not duplicated into the subproject.
