# Capture Contract

## Purpose

Defines how a clip is acquired and bounded before it reaches inference.

This is the layer with no training-side counterpart. ASL Citizen ships pre-segmented clips, each containing one isolated sign, recorded under conditions the dataset audit characterized. A webcam produces an unbounded stream with no segmentation, unknown framing, and browser-specific encoding. Bridging that gap is this project's genuinely new work, and its largest source of unmeasured risk.

## Design Position

Clip boundaries are determined by the user, explicitly.

The initial implementation records between an explicit start and an explicit stop. It does not attempt to detect sign onset or offset.

### Why

Automatic segmentation is a research problem adjacent to continuous sign recognition, which `ASL_training/docs/PROJECT.md` places outside project scope. Building a segmenter here would introduce an unevaluated model in front of an evaluated one, and every accuracy figure downstream would become uninterpretable.

Explicit boundaries also match the assumed user: someone deliberately checking one sign.

### Consequence

The user is responsible for recording one sign, cleanly bounded. The interface must make that expectation obvious, and the validation layer must catch the common violations.

Automatic boundary detection is deferred. It must not enter the initial implementation.

## Clip Requirements

A clip reaching the API must satisfy:

| Property | Requirement | Source |
|---|---|---|
| Sign count | Exactly one | Model contract, closed-set isolated recognition |
| Duration floor | Long enough to yield the required frame count | Bundle frame count and short-clip policy |
| Duration ceiling | Bounded, to keep sampling meaningful | Long-clip policy, plus request limits |
| Resolution floor | At or above the documented minimum | Training Phase 6 robustness findings |
| Frame rate | Sufficient to sample the required frames | Bundle |
| Orientation | Matching the training convention | `INFERENCE_CONTRACT.md` |
| Color | Convertible to RGB | Data contract |
| Container and codec | Decodable by the serving decoder | Deployment |

Concrete numeric bounds cannot be fixed yet. The resolution floor in particular should come from training Phase 6, which measures how performance degrades with lower resolution, blur, and compression. Guessing it now would either reject usable clips or accept degraded ones.

Until Phase 6 reports, bounds are placeholders and must be labeled as such in code and configuration.

## Mirroring

**This is the highest-risk item in the capture layer.**

Browser webcam previews are conventionally mirrored so the user sees themselves as in a mirror. The encoded stream is typically not mirrored. Whether the preview transform, the encoded frames, or neither carries the flip depends on the implementation and the platform.

If serving delivers frames in the opposite handedness convention from training, every prediction is made on inputs the model never saw, and accuracy degrades for a reason no evaluation metric would reveal. The degradation would be uneven across classes, because handedness is not uniformly meaning-preserving in ASL.

### Required resolution

Before any accuracy claim, the project must determine, by test rather than by inspection:

1. Whether the encoded stream is mirrored relative to the preview.
2. Whether the encoded stream is mirrored relative to the ASL Citizen convention.
3. What training Phase 6's handedness audit concluded about the dataset's own convention.

### Test method

Record a clip containing a physical asymmetry that survives compression and is unambiguous in a single frame — printed text held up to the camera is the clearest available option. Decode the stored clip with the serving decoder and inspect the asymmetry directly.

Visual impression of a preview is not evidence. The preview is the thing under suspicion.

### Rule

Once determined, the correction — if any — is applied in the capture layer and recorded as a decision. Serving preprocessing must not apply a compensating flip; see `INFERENCE_CONTRACT.md`.

## Container Normalization

Browser recordings vary by browser and platform in container, codec, frame rate stability, and whether duration metadata is present at all.

The capture layer must produce something the serving decoder handles reliably. Where the browser output is unreliable, normalization happens server-side before decoding, not by hoping the decoder copes.

Required verification: a clip recorded in each supported browser decodes to the same tensor shape as a dataset clip, and passes the frame-count and ordering tests.

## Validity Checks

Performed before inference cost is incurred:

| Check | Failure response |
|---|---|
| Decodable | Client error, specific |
| Non-zero frames | Client error |
| Duration within bounds | Client error, stating the bound |
| Resolution at or above floor | Client error, stating the floor |
| Not entirely static | Client error, likely a failed recording |
| Size within request limit | Client error |

A static clip check catches a common real failure: recording started and stopped without the camera ever producing frames.

### Validation is not abstention

A rejected clip means the input was unusable. An abstention means the model saw a usable clip and was not confident. These must never be conflated in the API, the interface, or the logs. Conflating them would make the abstention rate uninterpretable as a drift signal.

## Out-of-Vocabulary Signing

The vocabulary is closed at 2,731 classes. A user will sign something outside it.

The only defense available initially is the confidence threshold, which was fitted on in-distribution validation data and was never evaluated against out-of-vocabulary input. It will sometimes assign high confidence to a wrong class.

The interface must not imply otherwise. Presenting the vocabulary as discoverable, and abstention as normal, is the honest handling of a limitation that cannot be engineered away at this stage.

Out-of-distribution detection is deferred. `ASL_training/docs/ROADMAP.md` places it outside the current phases.

## Privacy

Webcam video of a person's face and hands is biometric-adjacent personal data.

Requirements:

* Recording must be explicitly user-initiated. No background capture, ever.
* Recording state must be visible while it is happening.
* Clips must not be retained by default.
* Any retention must be opt-in, bounded, and stated in plain language before the first recording.
* Clips must not be logged as a side effect of error handling.
* Prediction logs carry metadata, not video. See `MONITORING_CONTRACT.md`.

Retention for model improvement is a consent question, not a technical one. It must not be enabled by a configuration default.

## Accessibility

An ASL application whose users may be Deaf or hard of hearing must not depend on audio for any instruction, feedback, or error message. All guidance is visual.

Recording guidance, failure reasons, and remediation must be legible without sound and without color alone as the carrier of meaning.

## Deferred

* automatic sign onset and offset detection
* continuous stream segmentation
* multi-sign clips
* live prediction during recording
* mobile capture
* pose or hand-landmark preprocessing
