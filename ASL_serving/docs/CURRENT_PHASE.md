# Current Phase

## Active Phase

None. `ASL_serving` is planned but not started.

## Status

Documentation only. No source code, dependencies, or deployment configuration exist in this project, and none should be created until the entry conditions below are met.

The parent `CLAUDE.md` currently instructs that only `ASL_training/` may be modified. That rule is still in force for code. This documentation set was created under explicit instruction and does not by itself open the project for implementation.

## Entry Conditions

Phase S0 may begin when **either** is true:

1. `ASL_training` Phase 5 has produced at least one completed baseline run, so the architecture and its rough cost are known; or
2. The project owner explicitly decides to start the serving skeleton in parallel, accepting that it will be built against a stub model.

Option 2 is viable and low-risk. S0 through S4 do not need a real checkpoint. Building them against a stub model surfaces contract and shape bugs early, while training is still consuming GPU quota.

## Blockers

| Blocked work | Blocked on | Can start now instead |
|---|---|---|
| S1 bundle ingest, implementation | Training Phase 9 fixes the bundle format | Specify the format; see `BUNDLE_CONTRACT.md` |
| S2 parity tests | Reference tensors from `ASL_training` | Build preprocessing against the data contract |
| S3 threshold and temperature values | Training Phase 4B/4C outputs on a real run | Build the policy with values read from the bundle |
| S7 resource sizing | Selected architecture and measured latency | Containerize with placeholder limits |
| S8 drift baselines | A deployed service and an evaluation baseline | Define what will be logged |

Nothing in the capture layer (S5) is blocked. It has no training-side dependency at all, and it carries the project's most unresolved design question.

---

## Where the project stands

| Phase | State |
|---|---|
| S0 Serving foundation | Not started |
| S1 Bundle ingest | Not started, specified |
| S2 Inference core and parity | Not started, specified |
| S3 Confidence and abstention | Not started, specified |
| S4 Prediction API | Not started, specified |
| S5 Capture and clip boundary | Not started, partially specified |
| S6 Frontend | Not started |
| S7 Packaging and deployment | Not started |
| S8 Observability and drift | Not started |

## Inherited facts

From `ASL_training`, as of 2026-08-11:

```text
vocabulary        2,731 classes
frames per clip   16 (D-003)
spatial input     224 x 224 expected, value owned by configuration
color             RGB, explicit conversion required
frame order       chronological, ordered sampling indices
architectures     VideoMAE-Base and Video Swin-Tiny, not yet compared
selection metric  macro F1 (D-008)
```

No accuracy, temperature, threshold, or latency figure exists yet. Phase 5 has not run. Any such number appearing in serving work before then is a placeholder and must be labeled as one.

## Assumed but unverified

These shape the design and are not yet confirmed:

* clips will be user-bounded rather than automatically segmented
* one sign per clip
* browser-recorded clips decode equivalently to dataset clips
* the mirroring convention between preview and encoded stream

The last is the highest-risk item on this list. See `OPEN_QUESTIONS.md` Q-002.

---

## First tasks when S0 opens

1. Create `ASL_serving/src/asl_serving/` with the package skeleton.
2. Pin dependencies independently of `ASL_training`.
3. Add lint, format, and test commands matching the training project's tooling.
4. Add `.gitignore` entries for bundles, weights, clips, and logs.
5. Add a stub-model fixture with the correct output dimension, so S2 through S4 can proceed without a checkpoint.
6. Add a CI entry point.

Item 5 is the one that unblocks the most downstream work and is easy to overlook.

## Phase Summary

Not applicable. No serving phase has been completed.
