# Decisions

This document records lasting technical and experimental decisions for `ASL_training`.

A decision belongs here when reversing it would invalidate experiments, break checkpoint compatibility, or change the meaning of a reported metric. Routine implementation choices do not belong here.

Decisions are append-only. A superseded decision is marked `Superseded` and references the decision that replaced it. Entries are not deleted, because past experiment records depend on the conditions in force when they ran.

## Format

```text
## D-<number>: <title>

Date:
Status:      Accepted | Superseded | Rejected
Phase:
Supersedes:
Superseded by:

### Context
### Decision
### Consequences
### Alternatives considered
```

---

## D-001: Git repository root is the parent workspace

Date: 2026-08-07
Status: Accepted
Phase: 0

### Context

`docs/ROADMAP.md` (Phase 0) states that `ASL_training` should be initialized as its own Git repository. However, `CLAUDE.md` — which defines the workspace boundary, the authoritative document order, and the phase workflow — lives at the parent `ASL PROJECT/` level and describes both `ASL_training` and the future `ASL_serving`.

Initializing at `ASL_training/` would leave the governing instructions untracked, or would require duplicating `CLAUDE.md` into the subproject.

### Decision

Git is initialized at the parent workspace root:

```text
ASL PROJECT/
```

`ASL_training/` and the future `ASL_serving/` are directories within one repository.

### Consequences

* `CLAUDE.md` is version-controlled alongside the code it governs.
* `.gitignore` lives at the workspace root and covers both subprojects.
* The workspace-boundary rules in `docs/ARCHITECTURE.md` are enforced by convention and review, not by repository separation. Code review must confirm that `ASL_training` introduces no dependency on `ASL_serving`.
* If the two projects later need independent release cycles, `ASL_serving` can be split out with `git subtree` or a fresh repository. Training history remains intact.
* This is a documented deviation from `docs/ROADMAP.md` Phase 0, task 1. The roadmap text has been left unchanged; this decision governs.

### Alternatives considered

**Separate repository per project.** Matches the roadmap literally and enforces the boundary mechanically. Rejected because it orphans the parent `CLAUDE.md` and adds coordination cost well before any serving work exists.

---

## D-002: Video Swin-Tiny weights come from torchvision

Date: 2026-08-07
Status: Accepted
Phase: 1

### Context

`docs/MODEL_CONTRACT.md` requires a Video Swin-Tiny adapter but names no weight source.

VideoMAE-Base has an unambiguous source: `transformers`, checkpoint `MCG-NJU/videomae-base`. Video Swin has no `transformers` implementation, so the adapter must target an external source. The realistic options are `torchvision.models.video.swin3d_t` and the original `mmaction2` implementation.

The project targets Google Colab as a primary training environment, where `mmcv` installation is fragile: it pins CUDA and torch versions, frequently requires source builds, and breaks when the Colab runtime is upgraded. That cost recurs on every session.

### Decision

The Video Swin-Tiny adapter targets:

```text
torchvision.models.video.swin3d_t
weights: Swin3D_T_Weights.KINETICS400_V1
```

### Consequences

* No dependency beyond torchvision, which the project already requires.
* Colab and Kaggle sessions install cleanly with pip alone.
* torchvision's Swin3D expects `[batch, channels, frames, height, width]`. The adapter permutes from the project canonical `[batch, frames, channels, height, width]` internally, per `docs/MODEL_CONTRACT.md`.
* Reported Video Swin numbers are not directly comparable to published Video Swin papers, which use the mmaction2 implementation and training recipe. Experiment records must not present this project's Swin results as reproductions of published results.
* Kinetics-400 pretraining differs from VideoMAE-Base's Kinetics-400 self-supervised pretraining plus supervised fine-tuning. The two backbones do not start from equivalent representations. See D-003.

### Alternatives considered

**mmaction2 Video Swin.** Original implementation and weights, closest to published numbers. Rejected on Colab install fragility.

**Defer the Swin adapter.** Would reach a first training run sooner but postpone the Phase 5 architecture comparison, which is a stated project goal.

---

## D-003: Both architectures run at 16 frames in the baseline

Date: 2026-08-07
Status: Accepted
Phase: 1

### Context

`docs/MODEL_CONTRACT.md` sets the initial input to 16 frames at 224x224 and requires both values to remain configurable.

The two backbones were pretrained at different temporal resolutions:

* VideoMAE-Base: 16 frames, tubelet size 2
* torchvision `swin3d_t`: 32 frames

`docs/TRAINING_CONTRACT.md` requires that both architectures share frame count and spatial resolution where practical, and that any architecture-specific deviation be reported rather than concealed.

### Decision

The baseline runs both architectures at 16 frames, 224x224.

Frame count remains a configuration value and part of the preprocessing identity. A 32-frame comparison is deferred to an explicit later experiment.

### Consequences

* The comparison holds the data protocol constant, which is the fairness requirement that matters most for ranking architectures on this dataset.
* Video Swin operates below its pretraining temporal resolution. Its patch-embedding stride handles 16 frames without architectural change, but the result may understate Video Swin relative to a 32-frame configuration.
* Every experiment record comparing the two architectures must state this deviation explicitly.
* If Video Swin underperforms in Phase 5, frame count is a confound that must be ruled out with a 32-frame run before concluding that the architecture is weaker for this task.
* VideoMAE-Base requires frame count divisible by its tubelet size and consistent with its position embeddings. The adapter validates this and fails clearly rather than silently interpolating.

### Alternatives considered

**Each model at its native frame count.** Would respect each backbone's pretraining but give the models different information, violating the fairness requirement in `docs/TRAINING_CONTRACT.md`.

**Both at 32 frames.** Doubles activation memory and epoch time. Rejected as a baseline given Colab GPU and session constraints; retained as a follow-up experiment.

---

## D-004: Colab is treated as an interruptible environment

Date: 2026-08-07
Status: Accepted
Phase: 0

### Context

`docs/TRAINING_CONTRACT.md` requires checkpoint resume, but treats it as one requirement among many. Google Colab sessions terminate on idle timeout, usage limits, and backend reclamation, on a schedule outside the project's control. ASL Citizen is large enough that a full fine-tuning run will not complete within a single session.

Under Colab, resume is not a recovery feature. It is the normal execution path.

### Decision

The training layer treats interruption as expected rather than exceptional. Checkpoint-resume correctness is a blocking Phase 3 acceptance criterion, and multi-session resume is verified before any full baseline run begins.

Run directories, checkpoints, and logs are written to a configured persistent location that survives runtime termination. Ephemeral local disk may be used for dataset staging only.

### Consequences

* Checkpoint write frequency is configured in wall-clock terms, not only per-epoch, so an interrupted session loses a bounded amount of work.
* A resumed run is one experiment, not several. Run metadata records each session's hardware separately, since Colab may assign a different GPU on resume.
* Changing GPU type mid-run may perturb numerics. Run records must note it.
* Dataset staging cost is paid per session. See `docs/ENVIRONMENTS.md`.
* Checkpoints are still excluded from Git, per `docs/DATA_CONTRACT.md`.

### Alternatives considered

**Treat interruption as failure and restart.** Not viable for a multi-hour run on a dataset of this size.

---

## D-005: Model-layer tests do not require network access

Date: 2026-08-07
Status: Accepted
Phase: 1

### Context

`docs/MODEL_CONTRACT.md` requires tests for pretrained construction and for classification-head replacement. Downloading pretrained weights on every test run is slow, requires network access, and fails in offline or sandboxed environments.

The same document states that model-layer unit tests may use synthetic video tensors and do not require real dataset videos.

### Decision

The default test suite constructs models with randomly initialized weights and verifies the structural contract: input acceptance, tensor adaptation, logits shape, head output size, parameter reporting, and failure conditions.

Tests that download pretrained weights are marked `pretrained` and excluded from the default run. They are executed deliberately, and before any real training run as part of model preflight.

### Consequences

* The default suite runs offline and fast.
* Genuine pretrained-loading regressions — a renamed checkpoint, a changed state-dict layout — are not caught by the default suite. Model preflight is the gate that catches them, and `docs/MODEL_CONTRACT.md` already requires preflight to pass before a full run.
* CI, if added later, should run the `pretrained` marker on a schedule rather than per commit.

### Alternatives considered

**Always download in tests.** Slow and network-dependent.

**Vendor small fixture checkpoints.** Adds binary files to Git for limited benefit, since a fixture checkpoint does not validate the real checkpoint's layout.

---

## D-006: VideoMAE legacy attention-bias names are repaired on load

Date: 2026-08-07
Status: Accepted
Phase: 1

### Context

Published VideoMAE checkpoints, including `MCG-NJU/videomae-base`, store attention biases under the original implementation's names `q_bias` and `v_bias`. Transformers 5.x expects the standard `query.bias` and `value.bias` and performs no translation.

The result observed during Phase 1 preflight on transformers 5.14.1:

* 24 bias tensors reported as `MISSING` and left zero-initialized
* the corresponding `q_bias` and `v_bias` tensors reported as `UNEXPECTED` and discarded
* `from_pretrained` still returning successfully

Measured on layer 0: the loaded `query.bias` had norm 0.0 against the checkpoint's 17.54.

Every VideoMAE run would therefore have begun from a partially unloaded backbone, with the loader reporting success. Fine-tuning would likely have recovered much of it, which is what makes this dangerous: the damage is invisible in the loss curve, and any VideoMAE-versus-Swin comparison would have been silently unfair.

VideoMAE genuinely defines no key bias, so a zero `key.bias` is correct and must not be "repaired".

### Decision

The VideoMAE adapter detects missing `query.bias`/`value.bias` that correspond to `q_bias`/`v_bias` in the checkpoint, reads the raw checkpoint tensors, and copies them into place. Repaired keys are recorded in the load report.

The adapter additionally classifies every missing and unused key as either expected or unexplained, and warns on anything unexplained. Expected categories are the replaced classification head, the absent key bias, the self-supervised reconstruction decoder, and the legacy bias names consumed by the repair.

### Consequences

* VideoMAE starts from fully loaded pretrained weights.
* The repair is version-agnostic. On a transformers version that translates the names correctly, no key is missing, the repair does nothing, and `repaired_keys` is empty.
* If the raw checkpoint cannot be read, the adapter warns and proceeds with the biases unrepaired rather than failing. This is visible in the load report and in preflight output.
* The load report is the mechanism that makes a partial load visible. Run metadata should retain it, and a run whose report contains unexplained missing keys should not be treated as a valid pretrained baseline.
* Any VideoMAE run performed before this fix is not comparable to runs after it.

### Alternatives considered

**Pin transformers below 5.0.** Would likely avoid the rename, but pins the project to an aging release, conflicts with Colab's preinstalled versions, and leaves the failure mode undetected rather than fixed.

**Fail on the mismatch.** Safer in isolation, but a future transformers release could rename other keys cosmetically and block all training for no real reason. Detecting, repairing, and reporting is more robust than refusing to run.
