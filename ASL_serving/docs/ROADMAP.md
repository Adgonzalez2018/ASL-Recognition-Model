# Serving Roadmap

## Goal

Deliver a service that turns one recorded clip into an honest prediction or an honest abstention, with preprocessing provably faithful to training.

## Phase Numbering

Serving phases are numbered `S0` through `S8` to avoid collision with the training phases 0 through 9.

```text
Phase S0: Serving foundation
Phase S1: Bundle ingest
Phase S2: Inference core and preprocessing parity
Phase S3: Confidence and abstention policy
Phase S4: Prediction API
Phase S5: Capture and clip boundary
Phase S6: Frontend
Phase S7: Packaging and deployment
Phase S8: Observability and drift
```

## Dependency on Training

```text
S0  ──────────────────────────────  no training dependency
S1  ──────────────────────────────  needs the Phase 9 bundle format agreed
S2  ──────────────────────────────  needs a real bundle and reference tensors
S3  ──────────────────────────────  needs temperature and threshold values
S4, S5, S6  ──────────────────────  designable now, testable against a stub model
S7  ──────────────────────────────  needs the selected architecture for sizing
S8  ──────────────────────────────  needs a deployed service
```

S0 can start immediately. S1 can be specified now and implemented once training Phase 9 fixes the bundle format. S2 through S4 can be built against a **stub model** — a randomly initialized network with the correct output dimension — so that everything except accuracy is exercised before any real checkpoint exists.

Building against a stub is strongly preferred over waiting. It surfaces shape, ordering, and contract bugs while they are cheap.

---

# Phase S0: Serving Foundation

## Objective

Establish the project skeleton, dependency boundary, and development commands.

## Tasks

* Define the Python package under `ASL_serving/src/`.
* Add dependency management, pinned separately from `ASL_training`.
* Add formatting, linting, and test commands matching the training project's tooling.
* Add `.gitignore` rules for bundles, model weights, recorded clips, and logs.
* Add a minimal development README.
* Add synthetic test fixtures only — no real clips, no real weights.
* Add a CI entry point that runs lint and tests.

## Acceptance Criteria

* The project installs in a clean environment without the training dependency stack.
* Tests run and pass with no bundle present.
* No import from `ASL_training` exists anywhere in the source tree.
* No model weights, bundles, or video files are tracked.

## Non-Goals

Inference, API, frontend, deployment.

---

# Phase S1: Bundle Ingest

## Objective

Load, verify, and expose a training-exported bundle.

## Tasks

* Implement bundle loading from a configured location.
* Validate presence of every required field named in `docs/BUNDLE_CONTRACT.md`.
* Verify bundle integrity against its recorded checksum.
* Verify that the label map size equals the model output dimension.
* Expose preprocessing parameters as read-only values.
* Record bundle identity for logging and response metadata.
* Reject malformed, incomplete, and version-mismatched bundles with specific errors.

## Acceptance Criteria

* A valid bundle loads and reports its identity.
* A bundle missing any required field is rejected at startup, naming the field.
* A checksum mismatch is rejected.
* A label map whose size disagrees with the classifier head is rejected.
* No preprocessing parameter can be overridden by configuration.
* Every rejection path is tested with a deliberately corrupted fixture.

## Completion Artifact

A loader that either produces a fully specified inference context or refuses to start.

---

# Phase S2: Inference Core and Preprocessing Parity

## Objective

Turn a clip into logits, provably the same way training did.

This is the phase that determines whether the project is trustworthy.

## Tasks

* Implement video decoding with explicit RGB conversion.
* Implement deterministic evaluation temporal sampling per the bundle's policy.
* Implement the short-clip and long-clip policies.
* Implement spatial resize, crop, normalization, and tensor layout from bundle values.
* Handle container rotation metadata explicitly.
* Build the architecture and restore weights from the bundle.
* Implement a single-clip forward pass returning raw logits.
* Implement the parity test suite against training-produced reference tensors.

## Parity Requirement

For every reference clip shipped with the bundle:

```text
serving_tensor ≈ training_reference_tensor   within documented tolerance
```

and

```text
serving_logits ≈ training_reference_logits   within documented tolerance
```

Tensor parity catches preprocessing drift. Logit parity additionally catches weight loading, architecture construction, and precision differences.

## Acceptance Criteria

* Every reference clip passes both parity checks.
* The same clip processed twice yields identical tensors.
* Frame count matches the bundle exactly.
* Frame order is chronological, verified by a synthetic clip with ordered frame content.
* Channel order is RGB, verified by a synthetic clip with distinguishable channels.
* A clip whose rotation metadata differs is handled per the documented policy.
* Parity failures fail CI.

## Non-Goals

Thresholding, API, batching, latency optimization.

## Completion Artifact

A clip-to-logits function whose fidelity to training is demonstrated rather than asserted.

---

# Phase S3: Confidence and Abstention Policy

## Objective

Convert logits into an honest answer.

## Tasks

* Apply temperature scaling from the bundle.
* Compute maximum softmax confidence.
* Apply the bundle's threshold.
* Return accept-with-prediction or abstain.
* Return top-k alternatives with calibrated scores.
* Record the calibration and threshold identity used, per prediction.

## Acceptance Criteria

* Temperature never changes the predicted class ranking.
* Threshold behavior is tested at, just above, and just below the boundary.
* Abstention is a first-class outcome, not an error.
* Neither temperature nor threshold can be supplied by request parameters or configuration.
* The threshold in use is traceable to the validation analysis that selected it.

## Hard Rule

Thresholds and temperature come from validation-set analysis performed in `ASL_training`. They must never be tuned against serving-observed outcomes, and never adjusted to improve a demo.

## Completion Artifact

A calibrated decision function with an auditable provenance for every constant it uses.

---

# Phase S4: Prediction API

## Objective

Expose inference over HTTP.

## Tasks

* Implement the prediction endpoint per `docs/API_CONTRACT.md`.
* Implement clip validation with a specific error taxonomy.
* Implement a health endpoint reporting model readiness and bundle identity.
* Implement request size and duration limits.
* Implement structured logging hooks.
* Return bundle and calibration identity in every prediction response.

## Acceptance Criteria

* A valid clip returns a prediction or an abstention.
* An invalid clip returns a specific, actionable client error.
* Requests before model readiness fail explicitly rather than hanging.
* Health reports readiness, bundle identity, and device.
* No endpoint exposes weights, file paths, or internal traces.

## Completion Artifact

A documented HTTP surface with a stable response contract.

---

# Phase S5: Capture and Clip Boundary

## Objective

Get a valid clip out of a browser and into the API.

This is the phase with the least support from prior training work, and the most novel risk. See `docs/CAPTURE_CONTRACT.md`.

## Tasks

* Implement browser recording with an explicit start and stop.
* Normalize the recorded container and codec to a decodable form.
* Resolve the preview-mirroring question, definitively and with a test.
* Enforce the documented duration bounds during capture.
* Enforce the resolution floor during capture.
* Provide clear pre-recording guidance to the user.
* Verify a browser-recorded clip decodes to the same tensor shape as a dataset clip.

## Acceptance Criteria

* A clip recorded in the browser produces a valid prediction end to end.
* Mirroring is verified by a physical asymmetry test, not by visual impression.
* Duration and resolution violations are caught before upload.
* Clips are decodable by the serving decoder across the supported browsers.

## Open Risk

Clip boundary determination has no training-side counterpart. ASL Citizen clips are pre-segmented. The initial approach is explicit user-controlled start and stop, which sidesteps automatic segmentation entirely. Automatic boundary detection is deferred and must not enter this phase.

---

# Phase S6: Frontend

## Objective

Present the result honestly.

## Tasks

* Build the capture and result interface.
* Present abstention as a distinct, legible outcome.
* Present confidence without implying unwarranted precision.
* Show top-k alternatives when accepted.
* Communicate that the vocabulary is closed-set and finite.
* Provide recording guidance and failure remediation.

## Acceptance Criteria

* An abstention is visually distinct from a prediction.
* No interface element implies the system understands sentences or continuous signing.
* Vocabulary limits are discoverable by the user.
* Errors explain what to change about the recording.

## Design Constraint

The interface must not overstate the system. A confidence number rendered to two decimals next to a single gloss invites more trust than a large-vocabulary, low-shot classifier has earned.

---

# Phase S7: Packaging and Deployment

## Objective

Make the service reproducibly runnable.

## Tasks

* Containerize the service.
* Pin the runtime dependency set.
* Define the bundle mounting or fetching strategy.
* Configure resource requests based on measured usage.
* Add startup verification: bundle load, parity smoke check, health.
* Document CPU and GPU runtime options.
* Measure latency and throughput under realistic clip sizes.

## Acceptance Criteria

* A fresh environment runs the container with only a bundle and configuration supplied.
* Startup fails loudly on a bad bundle.
* Measured latency is recorded, and the budget in `docs/DEPLOYMENT.md` is replaced with real numbers.
* No secrets, weights, or clips are baked into the image.

## Deferred

ONNX export, quantization, and inference optimization are deferred until measured latency shows they are needed. Optimizing before measurement risks changing numerics and breaking parity for no demonstrated benefit.

---

# Phase S8: Observability and Drift

## Objective

Know when the deployed model stops being the model that was evaluated.

## Tasks

* Log per-prediction metadata per `docs/MONITORING_CONTRACT.md`.
* Track the confidence distribution over time.
* Track the abstention rate over time.
* Track input characteristics: duration, resolution, brightness, decode failures.
* Compare live distributions against the evaluation baseline.
* Define retraining trigger criteria.
* Define the escalation path when a trigger fires.

## Acceptance Criteria

* Prediction logs contain no raw video by default.
* Retention is explicit, bounded, and consented.
* Abstention-rate and confidence drift are visible without manual inspection.
* Trigger criteria are quantitative and written down before they fire.
* A fired trigger produces a report for `ASL_training`, not an automatic retrain.

## Boundary

This project detects and reports. It does not retrain. Retraining is training-project work with its own experimental controls.

---

# Deferred Beyond S8

* continuous sign recognition
* automatic clip segmentation from a live stream
* sentence-level output
* multi-model ensembles
* on-device inference
* user accounts and history
* automated retraining pipelines
* multi-region deployment

Each would change the architecture materially and requires its own decision record.
