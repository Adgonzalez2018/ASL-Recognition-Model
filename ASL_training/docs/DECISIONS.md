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

---

## D-007: Dataset on ephemeral disk, checkpoints on Drive; audit runs on Kaggle

Date: 2026-08-07
Status: Accepted
Phase: 2

### Context

The project trains on Google Colab against a Kaggle mirror of ASL Citizen, and no local dataset copy is wanted.

The governing constraint is size. ASL Citizen is roughly 84,000 videos, tens of gigabytes. Google Drive's free tier is 15 GB, so the dataset does not fit on Drive at all on that tier. Meanwhile Colab's runtime disk is ample but disappears when the session ends, and Drive is persistent but slow and rate-limited.

Checkpoints have the opposite profile. A resumable VideoMAE checkpoint is roughly 1 GB — 337 MB of weights plus AdamW's two moment tensors — and a model-only export is about 337 MB. Those fit on Drive comfortably.

Separately, Kaggle attaches its datasets read-only at `/kaggle/input` with no download step at all, while Colab would have to transfer the whole dataset every session.

### Decision

**Storage split.** Raw video data lives only on ephemeral runtime disk. Checkpoints, logs, run metadata, and reports live on Google Drive.

**Dataset staging on Colab.** Download from the Kaggle mirror to `/content` at the start of each session. Do not stage the dataset through Drive, and do not decode video off a mounted Drive.

**Checkpoint retention.** Keep one resumable `latest` checkpoint and one model-only `best` export on Drive, roughly 1.4 GB combined. Retain one prior `latest` during writes so an interrupted write cannot destroy the only resume point. Periodic checkpoints are opt-in, because they multiply Drive usage quickly.

**Run the Phase 2A audit on Kaggle, not Colab.** The dataset is already mounted there, so the audit needs no transfer. The audit is read-only and produces small artifacts, which is exactly what Kaggle's short sessions suit. Training remains on Colab.

### Consequences

* Per-session dataset transfer is the standing cost of training on Colab. Preflight should measure it so epoch-time estimates include it.
* Free-tier Drive is sufficient for checkpoints under this retention policy, but not for the dataset. This must not be worked around by staging the dataset to Drive.
* Auditing on Kaggle and training on Colab means two environments touch the data. Dataset identity, recorded per `docs/DATA_CONTRACT.md`, is what proves they saw the same bytes. The audit records the mirror's identity; training validates against it.
* Writing checkpoints locally and then copying to Drive keeps a slow or failing Drive write from stalling or corrupting training.
* If the runtime disk cannot hold the dataset plus working files, the fallback is a documented subset run, explicitly labeled as such, never a silent reduction.

### Alternatives considered

**Dataset archive cached on Drive, extracted per session.** Faster than re-downloading and the pattern `docs/ENVIRONMENTS.md` prefers in general. Rejected as the default here because the archive does not fit in free-tier Drive. Worth revisiting on a paid tier, where it would be the better option.

**Train on Kaggle instead of Colab.** Removes staging entirely, since the data is already attached. Rejected because Kaggle's session limits are tighter than Colab's and its GPU allocation is less generous for a multi-hour fine-tuning run. Kaggle is used for the audit, where its strengths apply.

**Read the dataset directly from mounted Drive.** Simplest to set up, far too slow to decode thousands of videos per epoch, and prone to Drive rate limiting.

---

## D-008: Best-checkpoint selection uses macro F1

Date: 2026-08-07
Status: Accepted
Phase: 3

### Context

`docs/TRAINING_CONTRACT.md` requires an explicit checkpoint-selection metric and direction. The Phase 3 baseline configuration used top-1 accuracy, chosen when the training loop had only a placeholder metric set and macro F1 was not yet computable. Phase 4 made the full metric set available.

ASL Citizen has an uneven class distribution across a large vocabulary. Top-1 accuracy is dominated by well-supported classes, so a checkpoint can improve on frequent signs while regressing on rare ones and still score higher. Selecting on top-1 would systematically prefer such checkpoints.

`docs/PROJECT.md` lists balanced performance across classes as a primary goal, and macro F1 as a success target. The selection metric should match the goal.

Changing this after baseline runs would invalidate comparisons between them, so it had to be settled before the first real run.

### Decision

Best-checkpoint selection maximizes validation macro F1.

Top-1 accuracy remains the headline reported metric and is still recorded every epoch. It is no longer what decides which weights are kept.

In-training validation computes a restricted metric set — top-1, top-5, macro F1, mean per-class accuracy, and NLL — which `docs/EVALUATION_CONTRACT.md` permits. Per-class, per-signer, and confusion breakdowns belong to full evaluation from a checkpoint.

### Consequences

* Selection favours checkpoints that serve rare signs, which is what the project's stated goals ask for.
* Macro F1 is noisier than top-1 on a validation split where many classes have few samples, so the selected epoch may vary more between seeds. Seed variability should be reported in Phase 5.
* Macro F1 over the validation split uses the `support` averaging policy: classes absent from validation are excluded and the count is reported alongside the value.
* Validation now computes per-class counts over the full vocabulary each epoch. This is done with `bincount` rather than a per-class scan, so the cost is negligible.
* A `selection_metric` the metric function does not produce now raises rather than silently selecting no checkpoint. Previously a typo would have produced a run that finished successfully with no best checkpoint.
* Runs selected on top-1 are not directly comparable to runs selected on macro F1. No baseline runs had been performed when this changed, so nothing was invalidated.

### Alternatives considered

**Keep top-1 accuracy.** Simplest and matches the headline success target, but can improve while the long tail degrades, which is the failure mode this project most needs to avoid.

**Decide after a smoke run.** Would have shown how the two diverge on real data, at the cost of an extra run and a decision made under time pressure once training was already underway.

**Validation loss.** Stable and cheap, but only loosely coupled to the balanced-performance goal, and harder to reason about across vocabulary sizes.

---

## D-009: Training runs on Kaggle rather than Colab under free-tier compute

Date: 2026-08-10
Status: Accepted
Phase: 5
Amends: D-007

### Context

D-007 sent the audit to Kaggle and training to Colab, reasoning that Colab's longer sessions outweighed the cost of staging the dataset there. Two things have changed since.

The audit measured the dataset: 83,399 videos. Staging that into a Colab runtime is a large transfer, and it repeats every session because `/content` is ephemeral. Kaggle attaches the same mirror read-only at `/kaggle/input` with no transfer at all.

The project is also running on free compute. Free Colab gives shorter and less predictable sessions than the paid tier D-007 implicitly assumed. When a session is a few hours rather than twelve, an hour of dataset transfer is a large fraction of it, and that fraction is paid again on every reconnect.

Kaggle's free tier grants roughly 30 GPU hours per week with sessions up to about 9-12 hours.

### Decision

Under free-tier compute, both the audit and training run on Kaggle. The Colab training notebook is retained for use on a paid tier or where Drive-backed persistence is preferred.

Manifests are regenerated in each Kaggle session with `--probe-limit 0`, which skips video probing and completes in about a minute. This is verified to produce byte-identical manifests and label map to a full audit, and the session asserts that the regenerated manifest identity matches the one recorded in the committed audit report.

Resume across sessions works by attaching the previous notebook version's output as an input and copying its checkpoints back into `/kaggle/working`.

### Consequences

* No per-session dataset transfer, which is the dominant fixed cost on free Colab.
* Enabling a GPU restarts a Kaggle session and clears `/kaggle/working`, so an audit session cannot be continued into a training session. Regenerating manifests is the intended path rather than a workaround.
* Persistence requires an explicit *Save Version*. An interactive session that is simply closed loses its checkpoints, which is a sharper failure mode than Colab's mounted Drive.
* *Save & Run All* re-executes the notebook from the top. Training resumes rather than restarting, but the session clock restarts, so a run must fit the remaining session rather than the full one.
* The weekly GPU quota is a real constraint on run length. Epoch count should be chosen from measured preflight numbers, not from the configuration default.
* Kaggle and Colab may assign different GPUs. Run metadata records hardware per session, and D-004's guidance on mid-run hardware changes continues to apply.

### Alternatives considered

**Keep training on Colab.** Retained as an option, and preferable on a paid tier where sessions are long enough to amortize staging and Drive gives smoother persistence. Rejected as the default under free compute purely on transfer cost.

**Cache the dataset archive on Drive and extract per Colab session.** Faster than re-downloading, but the archive does not fit in free-tier Drive, which is what D-007 already established.

**Carry manifests between sessions as a Kaggle Dataset.** Works, and avoids regeneration entirely. Rejected as the default because regeneration takes about a minute, and re-deriving from the dataset with an identity check is a stronger guarantee than trusting a copied artifact.

---

## D-010: Both architectures train at batch 8 with 4 accumulation steps

Date: 2026-08-11
Status: Accepted
Phase: 5

### Context

Phase 5 planning recorded that VideoMAE-Base runs out of memory at batch 8 on a T4 and therefore requires batch 4 with 8 accumulation steps, while Video Swin-Tiny runs batch 8 with 4. Effective batch stayed 32 for both, but the physical batch differed, and `docs/TRAINING_CONTRACT.md` requires such a difference to be reported in every comparison.

That constraint was measured while `resolve_precision` was selecting emulated bf16 on a pre-Ampere GPU, the fault corrected in the preceding commit. Emulated bf16 does not use the tensor cores and carries fp32 working copies, so it inflated memory as well as time.

Re-measured on 2026-08-11 with fp16 active, VideoMAE-Base at batch 8 peaks at 6.09 GB of the T4's 14.56 GB — well within budget — and is faster than 4x8: 6.163 s per optimizer step against 6.688 s. Video Swin-Tiny at the same batch peaks at 10.47 GB.

### Decision

Both architectures train at batch 8 with 4 gradient accumulation steps, effective batch 32.

### Consequences

* The physical-batch asymmetry disappears. Both models share batch size, accumulation, worker count, and precision, so nothing about the optimization differs between the two Phase 5 baselines.
* `docs/TRAINING_CONTRACT.md`'s reporting requirement is satisfied trivially rather than by disclosure.
* Video Swin-Tiny uses 72% more memory than VideoMAE-Base despite having a third the parameters, because window attention holds more activations than VideoMAE's flat token stream. Swin, not VideoMAE, is the memory-constrained architecture on this hardware.
* Memory headroom is 28% for Swin and 58% for VideoMAE. Neither is close to failing, but Swin is the one to watch if frame count or resolution is raised later.
* Any measurement taken before the precision correction is suspect on both time and memory, and should be re-measured rather than reasoned from.

### Alternatives considered

**Keep VideoMAE at 4x8.** Preserves the recorded constraint and is known to fit. Rejected because it is slower and because carrying a documented protocol difference that no longer exists would misdescribe the experiment.

**Raise the batch further now that headroom exists.** Would change the effective batch and break comparability with the configuration Phase 5 was planned around, for a gain that is not the bottleneck. The bottleneck is CPU video decode, not GPU throughput.

---

## D-011: Training reads a re-encoded mirror of ASL Citizen at short side 256

Date: 2026-08-11
Status: Accepted
Phase: 5

### Context

After the precision fix (D-010), preflight found both architectures spending ~52% of every optimizer step waiting on CPU video decode, with the GPU idle for roughly half of each step. Compute is ~2.95 s per step; data loading ~3.2 s.

Workers are not the lever. Kaggle grants 4 cores, and 8 workers measured slower than 4 on both the step time and the data component. The floor is about 97 ms to decode one clip: the source is 640x480 and a median 75 frames are decoded to keep the 16 the sampler wants.

Options assessed were a downscaled mirror, a validation cache, GPU decoding via DALI, decord, storing fewer frames per clip, and renting a machine with more cores. Of these, only the mirror is both free and independent of the Kaggle environment. GPU decoding requires moving the spatial transform pipeline onto the GPU, replacing the layer that guarantees temporally consistent transforms. Storing fewer frames was rejected outright: it would shrink the pool `random_segment` draws from, reducing temporal augmentation on a dataset already at 14.7 training samples per class, and would pre-corrupt the Phase 6 temporal robustness work.

Calibration over 300 clips on 2026-08-11 measured, through the real loader path:

```text
decode          134.3 ms -> 51.5 ms per clip     2.61x
size ratio      0.153 of source, 7.2 GB projected
encode          5.5 h at 4 jobs
frame drift     0 of 300
failures        0 of 300
```

### Decision

Phase 5 and later phases train against a mirror of ASL Citizen re-encoded at short side 256, h264, CRF 20, frame rate passed through.

Short side 256 is not arbitrary: the spatial transform resizes to 256 and then crops 224, so encoding below it would leave the random crop nothing to move within, removing the augmentation rather than merely shrinking the file. The scale filter is orientation-aware so the dataset's portrait clips are scaled by their short side like everything else.

The mirror preserves relative paths, file names, frame counts, and the split files. Switching to it is a change of `--dataset-root` and nothing else. `docs/DATA_CONTRACT.md` defines the substrate requirements.

Encoding settings live in `asl_training/data/mirror.py`, shared by the calibration that measured this and the build that produces it, so the two cannot drift.

### Consequences

* Decoding is expected to stop being the bottleneck; the step should become compute-bound and the epoch fall from ~2.1 h toward ~1.4 h for both architectures. **This is arithmetic from an isolated measurement and must be confirmed by preflight against the built mirror.**
* The re-encode is lossy. Every split must use the mirror, and no result from the mirror may be compared against a result from the source. This binds the Phase 5 baselines, the Phase 6 robustness evaluation, the Phase 7 targeted retraining, and any Phase 8 cross-dataset work.
* Run metadata must record the substrate. A run whose substrate is unknown is not a usable control.
* The mirror reproduces the source's manifest and label-map identities exactly, because those hash sample IDs, paths, labels, signers, and splits, and deliberately exclude resolution and codec. That equality is the acceptance check, not a sanity check.
* Every clip is verified on write: frame count against the source, and short side 256. Wrong geometry is the dangerous failure, because it decodes without error and trains at the wrong scale.
* Calibration could not verify the sample's resolution mix, because manifests regenerated with `--probe-limit 0` carry no width or height. The portrait branch of the scale filter is therefore untested against real video, and per-file verification during the build is what covers it.
* The mirror is roughly 7 GB and lives as a Kaggle Dataset. It is not committed, per the repository data boundary.
* If the mirror is ever rebuilt at different settings, it is a different substrate and prior results do not carry over.

### Alternatives considered

**Cache the validation set.** Validation sampling is deterministic, so the tensors are identical every epoch and caching could not change a metric — the safest option of the set. Deferred because the exact cache is ~25 GB against Kaggle's ~20 GB working directory, and because the mirror reduces validation cost anyway. Worth re-measuring after the mirror.

**All-intra encoding, seeking to the 16 wanted frames.** Would decode 16 frames instead of 75. Rejected because `decode_clip` deliberately does not seek: these are variable-frame-rate webcam recordings, 11 to 120 fps, where seeking can change which frames are selected, and an optimization must not do that. It would also inflate storage past the working-directory limit.

**GPU decoding via DALI.** The highest ceiling, and the T4's decode ASIC is idle. Rejected for now because it requires adopting a GPU-side transform pipeline in place of the current one, and because Kaggle rebuilds the environment every session. Revisit if the project leaves Kaggle.

**decord.** Worth benchmarking after the mirror rather than before, since its advantage is keyframe seeking and the decoder already stops early at the highest requested index.

**Rent a machine with more cores.** Effective and costs money rather than engineering. Rejected because Kaggle's value is the attached dataset; staging tens of gigabytes elsewhere each session costs more than the speedup returns.

---

## D-012: Phase 5 baselines run 12 epochs, validating every 4

Date: 2026-08-11
Status: Accepted
Phase: 5

### Context

`configs/training/baseline.yaml` defaulted to 20 epochs, chosen before any measurement. The cosine schedule spans the configured epoch count, so the value must be fixed before the first run; changing it later changes the learning-rate trajectory and makes runs non-comparable.

Validation evaluates all 10,304 clips. Before the mirror that cost ~78 min for Swin and ~54 min for VideoMAE against ~126 and ~129 min training epochs — about a third of every epoch.

Free Kaggle grants roughly 30 GPU hours per week, and Phase 5 needs two baselines.

### Decision

Baselines run **12 epochs** with **`validate_every_epochs: 4`**.

Validation cadence is reduced rather than the validation set subsetted. Validation averages 3.8 samples per class across 2,731 classes; a subset would push that toward 1 and turn macro F1 — the selection metric under D-008 — into noise.

### Consequences

* 15,060 optimizer steps at effective batch 32, which is a reasonable fine-tuning budget for a pretrained video backbone on 40,154 samples.
* Projected against the mirror: roughly 19 h per architecture, about 37 h for both. That spans two quota weeks with room for interruption, resume, and evaluation.
* Validation runs at epochs 4, 8, and 12, so best-checkpoint selection chooses among **three** candidates. That is thin. Post-mirror, validation is cheap enough that `validate_every_epochs: 2` would give six candidates for roughly 3 h more per architecture, and is worth reconsidering once preflight confirms the mirror's effect.
* Both figures are projections from calibration, not measurements of a training run. Preflight against the built mirror should confirm them before the first baseline starts.
* Both architectures use the same epoch count and cadence, so neither is advantaged.
* A run stopped early by quota is not a 12-epoch run and must not be reported as one.

### Alternatives considered

**Keep 20 epochs.** Roughly 62 h for both architectures against the mirror, more than two weeks of quota with nothing left for evaluation or a seed repeat. Rejected on budget, not on principle; the checkpoint selected by macro F1 is unlikely to be the last epoch anyway.

**8 epochs.** Fits both baselines in about one week. Rejected as the default because it risks undertraining a 2,731-class problem, and because the saving buys less than it costs in confidence.

**A fixed validation subset.** Cheaper per validation and allows a tighter cadence. Rejected because it degrades the selection metric, which is the thing validation exists to produce.
