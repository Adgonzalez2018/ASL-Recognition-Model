# Isolated ASL Recognition — Serving

## Project Summary

This project serves the isolated ASL sign classifier trained in `ASL_training`.

It receives one short video clip containing one isolated sign and produces:

* a predicted sign or gloss
* a calibrated confidence score
* or an explicit abstention

The training project answers *can a model recognize this sign*. This project answers *can a person get that answer, correctly and honestly, from a clip they just recorded*.

Those are different problems. A model that scores well on ASL Citizen test clips can still fail in serving for reasons no training metric predicts: the clip boundary is wrong, the webcam mirrored the frames, the browser encoded a different color range, or the user signed something outside the vocabulary. Most of this project's risk lives in that gap.

## Problem Definition

```text
user records a clip
    ↓
clip boundary and validity checks
    ↓
preprocessing that reproduces training exactly
    ↓
video classifier
    ↓
logits
    ↓
temperature scaling
    ↓
confidence threshold
    ↓
predicted gloss and confidence, or abstention
```

The classifier is closed-set over the vocabulary fixed by the training label map. Anything outside that vocabulary is, at best, rejected by the confidence threshold. This project must not present a closed-set prediction as though it were an open-world answer.

## Scope

### Included

* loading and verifying a training-exported model bundle
* inference preprocessing that provably matches training preprocessing
* single-clip inference
* temperature scaling application
* confidence thresholding and abstention
* a prediction API
* a browser capture interface
* clip boundary determination
* input validity and quality checks
* containerized deployment
* request and prediction logging
* latency and throughput measurement
* input drift and confidence drift monitoring
* retraining trigger criteria

### Not Included

* model training
* fine-tuning
* augmentation experiments
* label map changes
* dataset work
* continuous sign recognition
* sentence segmentation
* ASL-to-English translation
* language modeling
* facial grammar analysis
* user accounts or identity
* storing user video beyond an explicitly consented retention window
* automated retraining execution

Retraining *triggers* belong here. Retraining itself belongs to `ASL_training`.

## Primary Goal

Deliver a prediction path where the only source of error is the model itself.

Every other component — decoding, sampling, normalization, orientation, calibration — must be verifiably faithful to training. When accuracy in serving differs from the reported test accuracy, the project must be able to say why, and preprocessing must already be excluded as the cause.

## Success Criteria

### Correctness

Preprocessing parity is the primary correctness criterion. Given the same source clip, the serving pipeline and the training evaluation pipeline must produce equivalent tensors within documented numerical tolerance.

A serving deployment that cannot demonstrate parity is not releasable, regardless of how well it appears to perform in casual use.

### Quality

The project inherits the selective-prediction target stated in `ASL_training/docs/ROADMAP.md` Phase 4C:

```text
accepted prediction accuracy: at least 90%
coverage: at least 50%
```

That target was defined on ASL Citizen test data. Whether it survives contact with webcam clips is an open question, recorded in `docs/OPEN_QUESTIONS.md`. It must be measured, not assumed.

### Honesty

An abstention is a successful outcome, not a failure. The interface must make a low-confidence result legible as "I am not sure" rather than presenting a weak guess as an answer.

### Performance

Latency and hardware targets cannot be fixed until the architecture is selected and Phase 9 reports inference throughput. Placeholder budgets appear in `docs/DEPLOYMENT.md` and are explicitly marked as unset.

## Users and Assumptions

The initial assumed user is a person practicing or checking an isolated sign, recording one sign at a time, deliberately, in front of a laptop webcam.

This assumption drives several design choices and is worth stating plainly because it may be wrong:

* clips are short and deliberately bounded
* one sign per clip
* reasonable indoor lighting
* the signer is centered and mostly facing the camera
* the vocabulary the user wants is inside the trained vocabulary

This project is not an interpreter, not an accessibility substitute for a human interpreter, and must not be described as one.

## Relationship to `ASL_training`

The dependency is one-directional and asynchronous:

```text
ASL_training  ──[versioned bundle]──▶  ASL_serving
```

`ASL_serving` must not import from `ASL_training`, must not modify it, and must not require it at runtime. The bundle is the entire interface. If something needed for inference is missing from the bundle, that is a training-side gap to be fixed in the handoff package, not something serving should reconstruct locally.

## Non-Goals That Are Easy to Drift Into

* rebuilding preprocessing "cleanly" in the serving stack
* adjusting the confidence threshold to make the demo feel better
* adding classes or aliases to the label map
* applying a mirroring fix in serving to compensate for a training-side handedness question
* reporting serving-observed accuracy as though it were test accuracy

Each of these silently invalidates the training project's guarantees.
