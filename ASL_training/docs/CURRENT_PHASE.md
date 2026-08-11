# Current Phase

## Active Phase

Phase 5: Baseline Experiments

## Status

Not started. All supporting layers are complete; what remains needs GPU time.

Preflight has run once and produced a blocking finding: the run as configured does not fit free Kaggle quota. The precision bug behind it is fixed but the improvement is unmeasured.

## Objective

Train and compare clean baselines for Video Swin-Tiny and VideoMAE-Base on ASL Citizen, under one shared protocol, and report which architecture leads or that the result is inconclusive.

## Current Task

Re-run preflight with the bf16 fix. Every scoping decision below depends on the new numbers.

## Blockers

None technical. The constraint is free-tier GPU quota, roughly 30 hours per week on Kaggle.

---

## Where the project stands

| Phase | State |
|---|---|
| 0 Repository foundation | Complete, archived |
| 1 Model layer | Complete, archived |
| 2 Data layer and audit | Complete, archived |
| 3 Training layer | Complete, archived |
| 4 Evaluation layer | Complete, archived |
| **5 Baseline experiments** | **Active** |
| 6 Robustness evaluation | Not started |
| 7 Targeted robustness training | Not started |
| 8 WLASL benchmarking | Not started |
| 9 Handoff package | Not started |

715 tests pass offline; 7 more are marked `pretrained` and download real weights.

### Verified facts

Dataset, from the full audit committed at `artifacts/audits/asl_citizen_audit.json`:

```text
83,399 videos, 2,731 classes, 52 signers, every file decodable
train 40,154 (35 signers) / validation 10,304 (6) / test 32,941 (11)
signer-independent, no duplicates, no row dropped in parsing
label map  asl_citizen:2731:sha256:3a0b873befec998c
manifest   asl_citizen:83399:sha256:b864a6d5d84c5531
```

Roughly 14.7 training samples per class. This is a large-vocabulary, low-shot problem, and the success targets in `docs/PROJECT.md` should be read against that.

Environment: Kaggle, Tesla T4 (14.56 GB), free tier. See D-009.

---

## Immediate work, in order

### 1. Re-run preflight

`notebooks/kaggle/02_train_kaggle.ipynb`, section 5. Everything below depends on the result.

The previous run measured 4.5 videos/s and ~2.5 h per epoch under **emulated bf16**, which bypasses the T4's tensor cores. That is fixed; the gain is unverified.

### 2. Re-examine the bottleneck

The previous run reported data loading at 0% of step time. That was only true because compute was pathologically slow — four workers kept up trivially. If compute speeds up several times, decoding may become the limit, and `--num-workers` becomes the lever.

### 3. Cut validation cost

Validation added ~66 min per epoch against a ~149 min epoch, because it evaluates all 10,304 clips every time.

Options: `validate_every_epochs: 2` or higher, or a fixed validation subset.

**Decide before the first baseline.** Changing the validation protocol mid-project makes checkpoint selection non-comparable between runs.

### 4. Choose the epoch count from measurement

`configs/training/baseline.yaml` defaults to 20, which was a guess. The run must fit the weekly quota with room for the VideoMAE baseline as well.

### 5. Settle what the Swin run is

Is Video Swin-Tiny the baseline VideoMAE is compared against, or a cheaper warm-up that proves the pipeline?

This changes what the Phase 5 comparison means and should be decided **before results exist**. Deciding afterwards is how architecture comparisons get rationalized after the fact.

### 6. Then run the baselines

Per architecture:

1. Train, resuming across sessions.
2. Evaluate on validation. Fit temperature, select the confidence threshold.
3. Evaluate on test **once**, applying what validation chose. `scripts/evaluate.py` refuses to do this without the validation artifacts and an explicit `--reason`.
4. Record the run.

Then compare on top-1, top-5, macro F1, mean per-class accuracy, worst-signer accuracy, NLL, calibrated ECE, selective accuracy at coverage, throughput, peak memory, and checkpoint size.

---

## Constraints that carry into this phase

**Physical batch differs between architectures.** Video Swin-Tiny runs at batch 8 x 4 accumulation; VideoMAE-Base runs out of memory at batch 8 on a T4 and needs batch 4 x 8. Effective batch stays 32 for both. `docs/TRAINING_CONTRACT.md` requires the difference be reported, not concealed.

**Both architectures run at 16 frames**, though Swin was pretrained at 32 (D-003). If Swin underperforms, rule out frame count with a 32-frame run before concluding the architecture is weaker.

**Checkpoint selection is macro F1** (D-008), not top-1. Validation averages 3.8 samples per class, so individual per-class figures are noise; the macro average over 2,731 classes is stable enough to select on.

**Video Swin weights come from torchvision, not mmaction2** (D-002). Results are not comparable to published Video Swin numbers and must not be presented as reproductions.

**The test split has been read zero times.** Keep it that way until model and threshold selection are fixed.

---

## Acceptance criteria

- [ ] Both models complete a real baseline run, or a resource failure is documented.
- [ ] Both are evaluated under the same clean protocol.
- [ ] Results include seed variability where feasible.
- [ ] The comparison names a leading architecture or states the result is inconclusive.
- [ ] Baseline checkpoints are preserved as the control for Phase 6 and 7.

## Non-goals

Robustness augmentation, WLASL, hyperparameter search, serving.

## Completion artifact

Two clean baseline experiment records and a model-comparison report.

## Phase Summary

Not yet complete.

---

# Appendix: first preflight measurement (2026-08-10)

Measured 2026-08-10 on Kaggle, Tesla T4 (14.56 GB), Video Swin-Tiny, batch 8 x 4 accumulation, 4 workers.

```text
throughput          4.5 videos/s
per optimizer step  7.123 s   (data 0.032s / 0%, compute 7.092s / 100%)
bottleneck          compute
peak memory         10.48 GB of 14.56 GB   (28% headroom)
checkpoint          357 MB
steps               1,255 per epoch, 25,100 over 20 epochs
epoch               ~2.5 h
full run            ~49.7 h, plus ~22 h validation
```

**This does not fit free Kaggle quota** (~30 GPU hours per week). Roughly 72 hours as configured.

## Finding: bf16 was being emulated on a pre-Ampere GPU

Preflight reported `precision bfloat16` on a T4. A T4 is compute capability 7.5; native bf16 needs 8.0.

`torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`, so on pre-Ampere hardware it returns True and falls through to an emulation check. Emulated bf16 bypasses the tensor cores entirely and runs near fp32 speed. The T4's fp16 tensor-core path is several times faster.

`resolve_precision` trusted that call, so it selected the slow path and logged it as a success. Fixed: bf16 now requires compute capability 8.0 or higher, and anything older falls back to fp16 with a message explaining why torch claims otherwise.

The 4.5 videos/s figure above was measured under emulated bf16 and is expected to improve substantially. **The improvement is unverified — re-run preflight to measure it.**

Superseded by the Immediate work section above; kept as the measurement record.
