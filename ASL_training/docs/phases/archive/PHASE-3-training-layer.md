# Phase 3: Training Layer

Status: Complete
Archived: 2026-08-10

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
