# Execution Environments

## Purpose

This document defines how `ASL_training` runs across its expected environments and how environment-specific concerns are kept out of reusable source modules.

It is a supporting document. It does not override `docs/PROJECT.md` or the layer contracts. Where it appears to conflict with a contract, the contract governs.

Expected environments:

```text
Local development
Google Colab
Kaggle notebooks
```

Kaggle is the primary environment for both the dataset audit and training under free-tier compute, because the dataset is attached there with no transfer. Google Colab remains supported and is preferable on a paid tier. See D-007 and D-009 in `docs/DECISIONS.md`.

## Core Principle

The same code must run in every environment. Only configuration changes.

Environment differences are resolved at exactly two boundaries:

1. **Dataset root resolution** — where the raw videos are
2. **Output root resolution** — where checkpoints, logs, and reports are written

Everything downstream receives resolved absolute paths and knows nothing about how they were obtained.

Reusable modules under `src/asl_training/` must not:

* hardcode `/content`, `/kaggle/input`, or any local path
* call the Kaggle API
* mount Google Drive
* detect the environment to change training behavior
* download datasets

Notebooks and setup scripts own those concerns, per the notebook boundary in `docs/ARCHITECTURE.md`.

## Path Resolution

Dataset and output roots are supplied, in order of precedence:

1. command-line override
2. environment variable
3. configuration file value

Recommended environment variables:

```text
ASL_DATASET_ROOT
ASL_OUTPUT_ROOT
```

A run must fail clearly when a required root is unset or does not exist. It must not silently fall back to a default that happens to exist in one environment.

Per `docs/DATA_CONTRACT.md`, manifests store dataset-root-relative paths. Changing the root between environments must not change any `sample_id` or invalidate a manifest.

## Local Development

Purpose: writing code, running the test suite, model preflight on CPU.

Not intended for full training runs.

The default test suite must run offline on CPU with no dataset present. Tests requiring pretrained weight downloads are marked `pretrained` and excluded by default, per D-005 in `docs/DECISIONS.md`.

Apple Silicon note: the `mps` backend may be used for small smoke runs, but numerics differ from CUDA. Results from `mps` must not be reported as experiment results.

## Google Colab

### Session Model

Colab sessions terminate on idle timeout, total-usage limits, and backend reclamation. None of these are under project control. A full ASL Citizen fine-tuning run will not complete in one session.

Per D-004 in `docs/DECISIONS.md`, interruption is the expected execution path, not a failure mode.

### Storage Layout

Colab presents three storage tiers with different lifetimes:

| Tier | Lifetime | Speed | Use |
|---|---|---|---|
| Local disk (`/content`) | session | fast | dataset staging, active decode |
| Google Drive (mounted) | persistent | slow, rate-limited | checkpoints, logs, reports |
| Repository | persistent | n/a | code and configuration |

The required division:

* **Raw video data stays on local disk.** Decoding thousands of videos per epoch across a mounted Drive is prohibitively slow and will hit Drive rate limits.
* **Checkpoints, logs, and run metadata go to Drive.** They must survive session termination.

Writing every checkpoint directly to Drive is slow and can stall training. The recommended pattern is to write to local disk first, then copy to Drive, so a slow or failing Drive write cannot corrupt the active checkpoint.

### Dataset Staging

Dataset staging cost is paid once per session. Options, in preference order:

1. **Archive on Drive, extract to local disk.** One large sequential read, then fast local access. Preferred.
2. **Download from source each session.** Simple but repeats a large transfer every session.
3. **Read directly from mounted Drive.** Simplest to set up and the slowest to train against. Acceptable only for smoke runs.

Staging belongs in a setup script or notebook cell, never in the training modules, per the dataset download boundary in `docs/DATA_CONTRACT.md`.

Staging must verify what it produced — expected file count and a checksum or size check — before training begins. A partial extraction that silently yields fewer videos would violate the no-silent-reduction rule in `docs/TRAINING_CONTRACT.md`. Dataset validation before training already requires that discovered sample counts match the audited counts, which catches this.

### Checkpoint Cadence

Checkpoint frequency must be configured in wall-clock terms in addition to per-epoch, so that an interruption loses a bounded amount of work rather than up to a full epoch.

The latest checkpoint must be written atomically — to a temporary path, then renamed — so that a session terminated mid-write does not leave a truncated file as the only resume point.

At least one prior checkpoint should be retained. A single checkpoint slot is one bad write away from losing the run.

### Resume Across Sessions

A resumed run is one experiment. Its run directory, run ID, and experiment record are continuous across sessions.

Per `docs/TRAINING_CONTRACT.md`, resume restores model, optimizer, scheduler, scaler, counters, best-metric state, and RNG state where feasible.

Colab may assign a different GPU on resume — T4, L4, or A100 depending on availability and tier. Consequences:

* run metadata records hardware per session, not once per run
* a batch size chosen for A100 memory will fail on T4; the resume path must fail clearly on out-of-memory rather than silently reducing batch size, since that would change effective batch size mid-run
* changing GPU type may perturb numerics; the experiment record must note it

### Practical Constraints

* **Idle disconnection.** Colab disconnects idle sessions. Browser-side keepalive tricks are unreliable and against usage policy; correct resume is the real mitigation.
* **Data loader workers.** Colab CPU allocation is limited. Video decoding is CPU-bound, so worker count is commonly the training bottleneck, not the GPU. Preflight should report whether the loader or the GPU is limiting.
* **`/content` capacity.** Local disk is finite. Staging an extracted video dataset plus checkpoints can exhaust it. Preflight should check free space against the expected dataset size.

## Kaggle

Kaggle attaches datasets read-only at `/kaggle/input/<dataset-name>` with no download step, which removes the staging problem entirely.

Constraints:

* fixed session time limits, typically shorter than a full run
* `/kaggle/working` is the writable output location, with a size cap
* attached datasets are immutable, which is good for dataset identity

Per `docs/DATA_CONTRACT.md`, a third-party Kaggle copy of ASL Citizen must not be assumed identical to the official release. The Phase 2A audit must compare its structure and metadata against the official source and record the hosted-copy identity.

## Reproducibility Across Environments

Run metadata must capture, per `docs/TRAINING_CONTRACT.md`:

* environment name
* GPU type, per session for resumed runs
* CUDA version
* Python, torch, torchvision, and transformers versions
* dataset root used at runtime
* dataset and manifest identities

Results from different environments are comparable only when dataset identity, manifest identity, preprocessing identity, and effective batch size match. Differing GPU type alone does not invalidate a comparison, but must be recorded.

## Dependency Management

`pyproject.toml` is the authoritative dependency specification.

`requirements.txt` exists for notebook environments where a single pip install is more convenient, and must stay consistent with `pyproject.toml`.

Colab and Kaggle preinstall their own torch builds matched to their CUDA drivers. Notebook setup should install the project without forcing a torch reinstall, and should verify the resulting versions rather than assume them.

## Relationship to Other Documents

Depends on:

* `docs/PROJECT.md`
* `docs/ARCHITECTURE.md`
* `docs/DATA_CONTRACT.md`
* `docs/TRAINING_CONTRACT.md`
* `docs/DECISIONS.md`, entries D-004 and D-005

This document does not define dataset semantics, training behavior, or evaluation behavior.
