# Inference Contract

## Purpose

Defines how a clip becomes a tensor and a tensor becomes logits, and how that path is proven equivalent to the training evaluation path.

The governing source is `ASL_training/docs/DATA_CONTRACT.md`. This document does not redefine preprocessing. It states how serving reproduces it and how the reproduction is verified.

## The Parity Principle

Serving preprocessing must produce the same tensor training evaluation would produce from the same clip.

"Same" means within a documented numerical tolerance, not bit-identical. Decoder versions, resize implementations, and hardware differ. The tolerance ships in the bundle.

Parity is verified by test, against reference tensors produced by `ASL_training`. It is never established by reading both implementations and concluding they look alike. Two readings agreeing proves only that the same misunderstanding was applied twice.

## Pipeline

```text
clip bytes
    ↓  decode
frames, chronological, RGB
    ↓  rotation metadata applied
oriented frames
    ↓  temporal sampling (deterministic)
exactly N frames, ordered
    ↓  resize
    ↓  crop
    ↓  normalize
    ↓  layout
model input tensor
    ↓  forward pass
logits [1, num_classes]
```

Every stage draws its parameters from the bundle. No stage has a serving-local default.

## Decoding

Output must be RGB. A decoder returning BGR, YUV, or any other representation must be converted explicitly, and the conversion must be tested with a synthetic clip whose channels are distinguishable.

`ASL_training/docs/DATA_CONTRACT.md` is explicit that a backend must not be assumed to return RGB. Serving is more exposed to this than training: it will decode browser-produced containers that the training pipeline never saw.

### Frame order

Chronological, always. Serving must not:

* sort by generated filename
* reverse
* shuffle
* sample unordered indices

Verified by a synthetic clip whose frames carry an ordered visual marker.

### Rotation metadata

Container rotation must be handled per the bundle's policy. Browser-recorded clips and phone-recorded uploads carry orientation conventions that dataset clips do not, and decoders disagree about whether to apply them.

A clip that decodes rotated relative to training is not a subtle degradation. It is a different input distribution entirely.

## Temporal Sampling

Deterministic. The same clip must yield the same frame indices on every call, on every machine.

Serving uses the evaluation sampling policy, never the training policy. Training sampling contains randomness by design; using it in serving would make predictions irreproducible and would not match what the reported metrics measured.

### Short clips

Fewer frames than required is a real and frequent case in serving, more so than in training — a user stops recording early. The bundle's short-clip policy governs. Whatever it is, serving must apply it identically and must count occurrences for monitoring.

### Long clips

Longer clips are the normal case for user recordings. The bundle's long-clip policy governs the sampling window.

This interacts with clip boundaries: a clip containing one sign surrounded by two seconds of stillness samples very differently from a tightly bounded clip. See `CAPTURE_CONTRACT.md`.

## Spatial Preprocessing

All values from the bundle:

* resize policy and interpolation method
* aspect-ratio behavior
* crop policy and position
* final height and width
* normalization mean and standard deviation
* tensor dtype

Evaluation preprocessing is deterministic. No random crop, no random flip, no jitter.

### Mirroring

Serving applies the bundle's mirroring policy and nothing else.

Serving must not introduce a horizontal flip to compensate for a webcam preview, to match a user's handedness, or to improve apparent results. Handedness in ASL is not uniformly meaning-preserving, and training Phase 6 exists specifically to determine which classes are mirror-safe. A serving-side flip would silently override that analysis.

The capture layer is responsible for delivering frames in the same orientation convention the dataset used. That is a capture problem, not a preprocessing problem, and it is resolved in `CAPTURE_CONTRACT.md`.

## Inference

* Model built from the bundle's architecture identifier and configuration
* Weights restored with strict key matching
* Evaluation mode, gradients disabled
* Single clip per forward pass initially
* Returns raw logits, shape `[1, num_classes]`

The core returns logits, not probabilities and not predictions. Softmax, temperature, and thresholding belong to the confidence layer.

### Precision

Serving precision must be recorded and, where it differs from the precision the reference logits were produced under, the difference must be reflected in the parity tolerance.

Training measured a 2.4x compute reduction with fp16 active. If serving runs fp32 on CPU while references came from fp16 on a T4, small logit differences are expected and must be accounted for rather than treated as a parity failure.

## Determinism

The same clip and the same bundle must produce the same logits within tolerance, across repeated calls in one process and across restarts.

Sources of nondeterminism to control:

* decoder frame-seek behavior
* any randomness left in the transform path
* nondeterministic kernels
* thread-count-dependent reductions

Repeated-call determinism is a required test.

## Parity Test Suite

Required tests:

| Test | Catches |
|---|---|
| Tensor parity against reference tensors | Any preprocessing drift |
| Logit parity against reference logits | Weight loading, architecture, precision |
| Repeated-call determinism | Residual randomness |
| Frame count exactness | Sampling errors |
| Frame order, synthetic ordered clip | Ordering bugs |
| Channel order, synthetic channel clip | RGB/BGR inversion |
| Rotation handling | Orientation policy violations |
| Short-clip policy | Divergent padding or repetition |
| Long-clip policy | Divergent window selection |
| Normalization values sourced from bundle | Defaulted constants |

Parity failures must fail CI and must block release. They are not warnings.

## Failure Behavior

| Condition | Response |
|---|---|
| Undecodable clip | Client error, specific reason |
| Zero frames decoded | Client error |
| Clip shorter than the validity floor | Client error before preprocessing |
| Missing preprocessing parameter | Startup failure, never a runtime default |
| Forward pass raises | Server error, logged with bundle identity |
| Non-finite logits | Server error, never a prediction |

Non-finite logits deserve special mention: they must never be converted into a confident prediction by softmax. They indicate a real fault and must surface as one.

## Performance

Out of scope until measured. No batching, no caching of decoded frames across requests, no speculative optimization.

Optimization that changes numerics — quantization, ONNX export, fused kernels — must re-run the full parity suite and must record any tolerance change as a decision.
