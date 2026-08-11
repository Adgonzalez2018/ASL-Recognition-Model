# Confidence Contract

## Purpose

Defines how logits become a confidence score, and how that score decides between answering and abstaining.

The governing analysis lives in `ASL_training/docs/EVALUATION_CONTRACT.md`. Every constant used here was fitted there, on validation data. This project applies them; it does not derive them.

## Why This Layer Exists

A 2,731-class classifier trained on roughly 14.7 examples per class will be wrong often. The project's usable output is therefore not "the predicted class" but "the predicted class, when the model is trustworthy enough to offer one."

The target inherited from training Phase 4C:

```text
accepted prediction accuracy: at least 90%
coverage: at least 50%
```

Answering half the time and being right nine times out of ten is a more useful product than answering always and being right some unstated fraction of the time.

## Pipeline

```text
raw logits
    ↓  divide by temperature
calibrated logits
    ↓  softmax
calibrated probabilities
    ↓  max
confidence
    ↓  compare against threshold
accept or abstain
```

## Temperature Scaling

Applied by dividing logits by the bundle's temperature before softmax.

Requirements:

* Temperature comes from the bundle, fitted on validation logits in training.
* It must be positive and finite; verified at load.
* It must never be re-fitted in serving.
* It must never be supplied per request.

### Ranking invariance

Temperature scaling divides all logits by the same positive scalar and therefore cannot change their order. The predicted class before and after calibration is identical.

This is a required test, and it is also the reason calibration is safe to apply unconditionally: it changes how confident the model claims to be, never what it claims.

## Confidence Definition

Confidence is the maximum calibrated softmax probability.

It is a within-vocabulary quantity. It expresses "given that this clip contains one of the 2,731 known signs, how concentrated is the model's belief." It does not express "how likely is it that this clip contains a known sign at all."

That distinction must survive into the interface. A confidence of 0.94 on an out-of-vocabulary sign is entirely possible and means nothing reassuring.

## Threshold

Acceptance requires confidence at or above the bundle's selected threshold.

Requirements:

* The threshold comes from the bundle.
* It was selected on validation data, at a documented coverage and selective-accuracy point.
* It must never be selected or adjusted using test data.
* It must never be adjusted using serving-observed outcomes.
* It must never be a request parameter.
* It must never be lowered to make a demonstration feel more responsive.

### Why the prohibitions are absolute

A threshold tuned against observed serving behavior is a threshold fitted on unlabeled data by impression. It would drift toward whatever makes the system feel confident, which is precisely the failure the selective-prediction analysis was built to prevent.

Changing the threshold is a training-side decision, made against validation data, and it produces a new bundle version.

## Outcomes

### Accepted

Confidence at or above threshold. Response carries:

* predicted gloss
* calibrated confidence
* top-k alternatives with calibrated scores
* bundle and calibration identity

### Abstained

Confidence below threshold. Response carries:

* an explicit abstention outcome
* the confidence, or a coarse indication of it
* optionally, top-k alternatives, clearly marked as unreliable
* bundle and calibration identity

An abstention is a normal, successful result. It is not an error, not an HTTP failure, and not a degraded response.

### Whether to show alternatives on abstention

Open question. Showing them helps a user recognize a near-miss; it also invites treating a rejected prediction as an answer. Recorded in `OPEN_QUESTIONS.md` Q-005 and to be decided with the interface design.

## Top-K

Top-k alternatives come from calibrated probabilities, ranked. Ties broken by ascending class index, matching the training evaluation convention so that serving and evaluation rank identically.

## What Must Not Happen

* re-fitting temperature in serving
* selecting or adjusting a threshold in serving
* per-request temperature or threshold
* different thresholds per class, unless a training-side analysis produces them and ships them in the bundle
* adaptive thresholds that respond to traffic
* treating abstention as an error
* reporting serving-observed accuracy as test accuracy
* softmax over non-finite logits

The last is a correctness trap: non-finite logits produce a well-formed probability distribution and a confident-looking answer. They must be caught before this layer.

## Per-Class Thresholds

Not supported initially.

A single global threshold is what training Phase 4C selects. Per-class thresholds would be defensible — accuracy varies widely across 2,731 classes with uneven support — but they require their own validation-side analysis with its own overfitting controls. If that analysis is ever performed, the thresholds ship in the bundle and this document is amended by decision.

## Monitoring Interaction

The abstention rate is the single most informative drift signal this project has. It requires no labels, responds to input distribution changes, and has a known baseline from the validation selective-prediction curve.

A material rise in the abstention rate means the incoming distribution has moved away from what the model was evaluated on. A material fall, without a bundle change, is equally suspicious.

Neither is a reason to change the threshold. Both are reasons to investigate, and potentially to report to `ASL_training`. See `MONITORING_CONTRACT.md`.

## Required Tests

| Test | Verifies |
|---|---|
| Temperature preserves argmax | Ranking invariance |
| Temperature preserves full ranking | Ordering invariance |
| Confidence at threshold accepts | Boundary inclusive |
| Confidence just below threshold abstains | Boundary correctness |
| Temperature and threshold sourced from bundle only | No override path |
| Request parameters cannot alter either | No override path |
| Non-finite logits never reach softmax | Fault surfacing |
| Top-k ordering and tie-breaking | Evaluation consistency |
| Abstention is not an error response | Outcome taxonomy |
