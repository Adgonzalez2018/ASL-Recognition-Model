# Current Phase

## Active Phase

Phase 5: Baseline Experiments

## Status

Not started. All supporting layers are complete; what remains needs GPU time.

Preflight has now run four times with the precision fix in place. The bug is confirmed fixed and two constraints changed as a result, but the run still does not fit free Kaggle quota comfortably: both architectures spend about half of every step waiting on CPU video decode.

## Objective

Train and compare clean baselines for Video Swin-Tiny and VideoMAE-Base on ASL Citizen, under one shared protocol, and report which architecture leads or that the result is inconclusive.

## Current Task

Build the re-encoded mirror (D-011), then re-run preflight against it. Calibration measured 2.61x cheaper decoding; the build is a CPU-only Kaggle session, `notebooks/kaggle/03_mirror_kaggle.ipynb`.

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

### 1. Preflight, done

Four runs on 2026-08-11, recorded in the appendix. Compute fell 2.4x with fp16 active. Both architectures now cost the same and both are limited by CPU video decode, not the GPU.

Two constraints changed: VideoMAE fits batch 8 (D-010), and raising `--num-workers` past 4 makes things *worse*, because Kaggle gives 4 cores.

### 2. Decide the decode bottleneck

Data loading is ~52% of every step for both models. Workers are not the lever — 8 workers measured slower than 4. The floor is ~97 ms to decode one 640x480 clip, of which 16 frames are kept out of a median 75.

The remaining options, in the order they were assessed:

* **Downscaled mirror** of the dataset at short side 256. One-time CPU-only cost, no GPU quota, ~2-2.5x estimated. Gated on `scripts/calibrate_video_mirror.py`; abandon if measured under ~1.6x.
* **Validation cache.** Validation sampling is deterministic, so the tensors are identical every epoch and caching cannot change a metric. Deferred: the exact cache is ~25 GB against Kaggle's ~20 GB working directory, and the mirror may make it unnecessary. Re-measure after.
* **GPU decode (NVDEC/DALI).** Highest ceiling, but requires moving the transform pipeline onto the GPU and fights Kaggle's per-session environment. Not while on Kaggle.
* **decord.** Worth benchmarking after the mirror, not before: its advantage is keyframe seeking, and `decode_clip` already stops early at the highest requested index.
* **Renting CPU cores.** Works, but Kaggle's value is the attached dataset; staging it elsewhere costs more than the speedup returns.

Storing fewer frames per clip was rejected outright. It would shrink the pool `random_segment` draws from, reducing temporal augmentation on a dataset already at 14.7 samples per class, and would pre-corrupt the Phase 6 temporal robustness work.

### 3. Validation cadence, settled

`validate_every_epochs: 4`, committed to `configs/training/baseline.yaml`. Cadence reduced rather than the split subsetted, because validation averages 3.8 samples per class and a subset would turn macro F1 into noise. See D-012.

Worth revisiting once preflight confirms the mirror: post-mirror, validation should be cheap enough that cadence 2 costs ~3 h more per architecture and doubles the selection candidates from 3 to 6. Decide before the first baseline, not after.

### 4. Epoch count, settled

12 epochs, committed. 15,060 optimizer steps at effective batch 32, roughly 19 h per architecture against the mirror. See D-012.

The cosine schedule spans this count, so it cannot change once a run starts.

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

**Both architectures run batch 8 x 4 accumulation**, effective batch 32 (D-010). The earlier constraint that VideoMAE needed 4 x 8 was an artifact of emulated bf16 and no longer holds: measured at 6.09 GB of 14.56. Swin is the memory-heavier of the two at 10.47 GB, despite a third the parameters.

**Training reads the re-encoded mirror, not the source** (D-011). The re-encode is lossy, so every split must use it and no result from the mirror may be compared against one from the source. This binds Phases 6, 7, and 8 as well. Run metadata must record the substrate.

**12 epochs, validating every 4** (D-012). The cosine schedule spans the epoch count, so it is fixed before the first run.

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

The 4.5 videos/s figure above was measured under emulated bf16. Superseded by the measurements below; kept as the record of what the bug cost.

---

# Appendix: preflight after the precision fix (2026-08-11)

Kaggle, Tesla T4 (14.56 GB), fp16 active (`precision float16 (requested bf16)`), 1,255 optimizer steps per epoch. Reports in `outputs/preflight/`.

| Model | Batch | Workers | Step | Data | Compute | Peak mem | Epoch | Val |
|---|---|---|---|---|---|---|---|---|
| Swin | 8 x 4 | 4 | 6.040 s | 51% | 2.943 s | 10.47 GB | 2.1 h | 78 min |
| Swin | 8 x 4 | 8 | 6.495 s | 53% | 3.034 s | 10.47 GB | 2.3 h | 89 min |
| VideoMAE | 4 x 8 | 4 | 6.688 s | 54% | 3.106 s | 3.79 GB | 2.3 h | 59 min |
| VideoMAE | 8 x 4 | 4 | 6.163 s | 52% | 2.955 s | 6.09 GB | 2.1 h | 54 min |

## Findings

**The precision fix worked.** Compute fell from 7.092 s to 2.943 s per step for Swin, 2.4x. Total step time improved far less, 7.123 s to 6.040 s, because decoding absorbed the gain.

**The bottleneck moved to CPU video decode.** Both architectures now spend ~52% of each step waiting on data, so the GPU idles roughly half the time. The floor is ~97 ms per clip: 640x480, median 75 frames decoded to keep 16, on Kaggle's 4 cores.

**More workers made it worse.** 8 workers measured 6.495 s against 4 workers' 6.040 s, and data loading rose from 3.097 s to 3.461 s. Four workers already saturate four cores. Preflight's own warning recommends raising `--num-workers` whenever data loading dominates, with no knowledge of core count; it gave bad advice twice here and should be made core-aware.

**The two architectures cost the same.** Compute is within 0.4% between them at the same effective batch, despite VideoMAE having three times the parameters. Video Swin-Tiny at 16 frames does more work than its parameter count suggests.

**VideoMAE fits batch 8.** See D-010. The 4 x 8 requirement was an artifact of emulated bf16 holding fp32 working copies.

## Budget

At batch 8 x 4, 4 workers, `validate_every_epochs: 4`, 10 epochs each: Swin ~25 h, VideoMAE ~24 h. About **49 h, two weeks of free quota**. A successful mirror would bring both into roughly one week.
