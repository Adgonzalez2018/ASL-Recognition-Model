# Serving Architecture

## Workspace Position

```text
ASL PROJECT/
├── ASL_training/   ← produces a versioned bundle
└── ASL_serving/    ← this project, consumes it
```

One Git repository, two projects (see `ASL_training/docs/DECISIONS.md` D-001).

## Boundary Rules

`ASL_serving` must not:

* import from `ASL_training`
* modify anything under `ASL_training/`
* read training checkpoints, manifests, or dataset paths directly
* depend on the training environment (Kaggle, Colab, Drive)
* require the training dependency stack at runtime

`ASL_training` must not:

* import from `ASL_serving`
* be changed to accommodate a serving convenience

The only permitted coupling is the exported bundle described in `docs/BUNDLE_CONTRACT.md`.

### Why the boundary is strict

Training runs on a T4 under free-tier quota with a heavy dependency stack. Serving runs continuously on modest hardware and needs a small, stable dependency set. Sharing a module would force one project's constraints onto the other, and would make it impossible to answer "which version of preprocessing produced this prediction" from the bundle alone.

## Layers

```text
capture
    ↓
clip validation
    ↓
preprocessing
    ↓
inference core
    ↓
confidence policy
    ↓
API
    ↓
frontend
```

Dependency direction is downward only. The inference core must not know it is being called from HTTP. The preprocessing layer must not know a webcam exists.

### capture

Acquires a clip. Browser recording, file upload, or a supplied file path. Owns clip boundaries, mirroring correction, and container/codec normalization. See `docs/CAPTURE_CONTRACT.md`.

### clip validation

Rejects clips that cannot produce a meaningful prediction before any GPU or CPU cost is spent: too short, too long, undecodable, no motion, resolution below the documented floor.

Validation failures are a distinct outcome from abstention. A rejected clip is a client error; an abstention is a model result.

### preprocessing

Decodes, samples frames, and produces the model input tensor. This layer is a faithful reimplementation of the training evaluation path, verified by parity tests rather than by inspection. See `docs/INFERENCE_CONTRACT.md`.

### inference core

Loads the bundle, builds the architecture, restores weights, runs a forward pass, returns raw logits. Stateless per request. No thresholding, no softmax policy, no formatting.

Returning raw logits rather than a prediction keeps calibration testable in isolation and makes the core reusable for batch re-scoring.

### confidence policy

Applies temperature scaling, computes confidence, applies the threshold, and decides accept or abstain. Owns every number that came from validation-set analysis. See `docs/CONFIDENCE_CONTRACT.md`.

### API

Request validation, response shaping, error taxonomy, logging hooks. Owns no model logic.

### frontend

Capture UI, result display, abstention presentation. Owns no model logic and no thresholds.

## Runtime Shape

Initial target is a single-process service with the model resident in memory:

```text
client ──HTTP──▶ API ──▶ validation ──▶ preprocessing ──▶ inference ──▶ confidence ──▶ response
```

Model load happens once at startup, not per request. A request arriving before the model is ready must fail explicitly rather than queue indefinitely.

Batching across concurrent requests is deliberately out of scope for the first implementation. Isolated-sign practice is a low-concurrency workload, and per-request batching adds latency and failure modes that are not yet justified.

## Statelessness

Each prediction request must be independently reproducible from:

* the clip
* the bundle version
* the configuration in effect

No session state, no cross-request accumulation, no adaptive thresholds. A prediction that depends on what the service saw earlier cannot be audited.

## Configuration

Configuration must supply:

* bundle location
* device selection
* clip validity bounds
* retention policy for logged clips
* log destination

Configuration must not supply:

* frame count
* resolution
* normalization values
* label map
* temperature
* threshold

The second group belongs to the bundle. Allowing configuration to override bundle values would make it possible to silently break parity, which is precisely the failure this architecture exists to prevent.

## Failure Philosophy

Fail loudly, early, and specifically.

* bundle missing a required field → refuse to start
* bundle checksum mismatch → refuse to start
* parity test failing in CI → refuse to release
* clip undecodable → client error with a specific reason
* confidence below threshold → abstention, which is a normal result

The service must never silently substitute a default for a missing bundle value. A wrong normalization constant produces plausible-looking predictions that are quietly degraded, which is the worst possible failure mode.

## Testing Strategy

The test suite must cover:

* bundle loading, including rejection of malformed and incomplete bundles
* preprocessing parity against training-produced reference tensors
* frame count, frame order, and color channel order
* orientation and mirroring handling
* deterministic evaluation sampling — the same clip must yield the same tensor
* temperature application, including that it never changes the argmax
* threshold behavior at, above, and below the boundary
* abstention path
* clip validation rejections
* API error taxonomy
* model load failure handling

Reference tensors for parity tests must be produced by `ASL_training` and shipped in or alongside the bundle. A parity test written from reading the training code proves only that two readings agree.

## Architectural Non-Goals

* multi-model ensembles
* continuous or streaming recognition
* sentence segmentation
* user authentication and accounts
* horizontal autoscaling
* model registry infrastructure
* automated retraining execution
* on-device or mobile inference

These may become later work. None should shape the initial implementation.
