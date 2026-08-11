# Open Questions

Unresolved items that affect serving design. Each states what is unknown, what it blocks, and how it gets answered.

A question leaves this document by being answered and, where the answer is lasting, recorded in `DECISIONS.md`.

---

## Q-001: Which architecture is served?

**Unknown.** VideoMAE-Base and Video Swin-Tiny have not been compared. Training Phase 5 has not run.

**Blocks:** resource sizing, latency budget, whether quantization is needed, container base image, CPU versus GPU default.

**Does not block:** anything in S0 through S6. Every layer takes the architecture from the bundle.

**Answered by:** training Phase 5 baseline comparison, selected on macro F1 (training D-008).

---

## Q-002: What is the mirroring convention, end to end?

**Unknown, and highest-risk.** Three separate facts are missing:

1. Whether a browser's encoded stream is mirrored relative to its preview.
2. Whether ASL Citizen clips carry a consistent handedness convention.
3. Whether the two agree.

**Why it matters:** if serving delivers frames in the opposite convention from training, every prediction is made on inputs the model never saw. The degradation would be uneven across classes, since handedness is not uniformly meaning-preserving in ASL, and no serving-side metric would reveal the cause.

**Blocks:** any accuracy claim about serving. Does not block building the pipeline.

**Answered by:** training Phase 6's handedness and mirroring audit, plus a serving-side physical asymmetry test per `CAPTURE_CONTRACT.md`. Both are required; neither alone is sufficient.

**Trap to avoid:** concluding from a preview that frames look correctly oriented. The preview is the component under suspicion.

---

## Q-003: Does the 90% / 50% target survive webcam clips?

**Unknown.** The target — at least 90% accuracy on accepted predictions at at least 50% coverage — was defined on ASL Citizen test data: pre-segmented clips, dataset recording conditions, in-vocabulary signs.

Webcam clips differ in framing, lighting, compression, boundaries, and vocabulary coverage. The threshold was fitted on validation logits from the former and applied to the latter.

**Blocks:** any claim about serving quality. Does not block implementation.

**Answered by:** it cannot be answered by serving alone, because serving has no labels. Requires either a deliberately collected labeled evaluation set recorded through the real capture path, or acceptance that the figure remains a training-set property that serving does not verify.

**Recommendation:** collect a small labeled set through the actual capture path before making any quality claim. Perhaps a few hundred clips across a vocabulary subset. Without it, the number in the interface is inherited, not measured.

---

## Q-004: Can reference clips ship in the bundle?

**Unknown.** Parity verification needs reference clips, tensors, and logits. ASL Citizen licensing may not permit redistributing clips inside a deployable artifact.

**Blocks:** the exact form of the parity fixture set, not the approach.

**Answered by:** reading the dataset license terms recorded during training Phase 2A.

**Fallback if clips cannot ship:** ship tensors and logits keyed by sample identity. Parity remains verifiable wherever the dataset is available, including CI if the dataset can be mounted there. Weaker, since it cannot verify the decode stage itself — which is exactly the stage most likely to differ.

**Second fallback:** synthetic reference clips generated and shipped by training. These carry no licensing burden and verify decode, sampling, ordering, and normalization. They cannot verify behavior on real video characteristics, so they complement rather than replace real references.

---

## Q-005: Should abstentions show alternatives?

**Undecided.** Showing top-k on an abstention helps a user recognize a near-miss and re-record. It also invites treating a rejected prediction as an answer, which defeats the purpose of abstaining.

**Blocks:** frontend design, minor API response detail.

**Answered by:** an interface decision at Phase S6. Leaning toward showing them, clearly marked, on the grounds that a user practicing a sign benefits from knowing the model was close — but the marking has to carry real weight, not a small grey caption.

---

## Q-006: What are the clip validity bounds?

**Unknown.** Duration floor and ceiling, resolution floor, and the static-clip criterion all need concrete values.

**Blocks:** capture validation, API limits. Currently placeholders.

**Answered by:** training Phase 6 robustness evaluation, which measures degradation under lower resolution, blur, and compression. That measurement converts a guess into a defensible floor.

**Interim:** placeholder values, labeled as such in code and configuration, chosen conservatively.

---

## Q-007: How is out-of-vocabulary signing handled?

**Partially answered, unsatisfactorily.** The only defense is the confidence threshold, fitted on in-distribution validation data and never evaluated against out-of-vocabulary input. It will sometimes assign high confidence to a wrong class.

**Blocks:** nothing structurally. It is a known limitation, not a gap in the plan.

**Answered by:** out-of-distribution detection, which `ASL_training/docs/ROADMAP.md` places outside the current phases.

**Interim handling:** make the vocabulary discoverable, present abstention as normal, and never describe the system as recognizing arbitrary signing.

---

## Q-008: Where does clip normalization happen?

**Undecided.** Browser recordings vary by browser and platform in container, codec, frame rate stability, and duration metadata. Normalization can happen client-side before upload, server-side before decoding, or be avoided by constraining recording parameters.

**Blocks:** capture implementation detail, upload size limits.

**Answered by:** measurement at Phase S5 — record in each supported browser, attempt decode with the serving decoder, and see what actually fails.

**Leaning:** server-side, because it is the only place with a single, testable decoder version. Client-side normalization means as many behaviors as there are browser versions.

---

## Q-009: Does the parent `CLAUDE.md` need updating?

**Yes, eventually, and it is the project owner's decision.**

The parent `CLAUDE.md` currently instructs that only `ASL_training/` may be modified and that no files should be created in `ASL_serving/`. This documentation set was created under explicit instruction, but the rule as written still forbids the code that these documents describe.

**Blocks:** starting Phase S0 implementation.

**Answered by:** an explicit decision to open `ASL_serving` for implementation, with `CLAUDE.md` amended to describe the two-project working state — which project is active, which documents are authoritative for each, and that the bundle remains the only interface.

**Not to be done silently.** Amending the governing instruction file to permit work it currently forbids is a change to the project's rules, and belongs to the owner.
