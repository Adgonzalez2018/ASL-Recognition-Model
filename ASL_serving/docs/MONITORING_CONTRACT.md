# Monitoring Contract

## Purpose

Defines what the service records, what it watches, and what it does when the deployed model stops resembling the model that was evaluated.

## Position

This project **detects and reports**. It does not retrain.

Retraining is `ASL_training` work, under experimental controls this project does not have. A drift signal produces a report, never an automatic model update. An automated retrain triggered by unlabeled serving traffic would violate several of the training project's experimental-validity rules at once.

## The Labeling Problem

Serving has no ground truth. No user tells the system which sign they signed, and any mechanism that asked them would produce labels of unknown quality from a self-selected population.

Every signal below is therefore **unlabeled**. They detect that the input or output distribution has moved. They cannot measure accuracy.

Nothing derived from these signals may be reported as accuracy, and none of them may be used to fit a threshold, a temperature, or any model parameter.

## What Is Logged

Per prediction:

| Field | Purpose |
|---|---|
| request id | Correlation |
| timestamp | |
| outcome | prediction, abstention, or rejection |
| rejection reason | When rejected |
| predicted class index | When accepted |
| calibrated confidence | |
| bundle version | |
| label map identity | |
| calibration identity | |
| clip duration | Input characterization |
| clip resolution | |
| decoded frame count | |
| short-clip policy applied | Frequency of an edge case |
| preprocessing latency | |
| inference latency | |
| total latency | |

### What is not logged

* raw video, by default
* frames or tensors
* filenames from the upload
* anything identifying the person recording

Clips are retained only under explicit, bounded, opt-in consent. See `CAPTURE_CONTRACT.md`. Retention must never be enabled by a configuration default, and error handling must not persist a clip as a side effect of a failure.

Predicted gloss is logged as a class index rather than free text, so the log's meaning is fixed by the recorded label map identity rather than by a string that may later be re-used.

## Signals

### Abstention rate

The primary signal. No labels required, and it has a baseline: the coverage point at which the threshold was selected in training Phase 4C.

| Observation | Likely meaning |
|---|---|
| Rate rises materially | Input distribution moved away from evaluation conditions |
| Rate falls materially, no bundle change | Suspicious; investigate before celebrating |
| Rate matches the validation coverage point | Consistent with expectations |

A falling abstention rate is not good news by default. Absent a model change, it means confidence rose without evidence that correctness did.

### Confidence distribution

Compared against the validation confidence distribution shipped in the bundle's evaluation summary. Shape changes matter more than the mean.

### Input characteristics

Duration, resolution, frame count, decode failure rate, static-clip rejection rate.

These move first. Input drift precedes output drift, and it is the most actionable kind: a rise in low-resolution clips or decode failures usually points at a client, browser, or platform change rather than at the model.

### Prediction distribution

Class frequency over time. A vocabulary of 2,731 classes with real usage concentrated in a small subset is expected. A sudden concentration on a few classes, or the disappearance of previously common ones, warrants investigation.

### Latency

Preprocessing and inference tracked separately. Given that decode is expected to be a large share, a latency regression is more likely to originate in the decoder or in changed input characteristics than in the model.

### Error rates

Rejections by reason, service faults, model load failures.

Rejection reasons are a usability signal as much as a technical one. A high rate of too-short clips means the interface is not communicating the requirement.

## Baselines

Every signal needs a baseline recorded before deployment, drawn from the bundle's evaluation summary and from staging measurements.

A monitoring system whose baseline is "whatever the first week looked like" cannot detect a problem that was present in the first week.

## Retraining Triggers

Criteria must be quantitative, and written down **before** they fire. A threshold invented while looking at a concerning graph is a rationalization.

Candidate triggers, to be given concrete values once a baseline exists:

* abstention rate departs from baseline by more than a stated margin, sustained over a stated window
* confidence distribution shifts beyond a stated divergence
* input characteristics move outside the range the model was evaluated on
* decode failure rate exceeds a stated bound
* a robustness condition measured in training Phase 6 as a known weakness becomes common in live input

The last is the most valuable link between the two projects: Phase 6 measures which perturbations degrade the model, and monitoring can watch for exactly those conditions appearing in real traffic.

### When a trigger fires

1. Record the trigger, the window, and the observed values.
2. Produce a report for `ASL_training`.
3. Do not change the threshold.
4. Do not change the temperature.
5. Do not retrain.

Steps 3 and 4 are the ones under pressure when a graph looks bad. Adjusting the threshold in response to drift hides the signal without addressing the cause, and destroys the abstention rate's value as a future indicator.

## Reporting to `ASL_training`

A drift report should carry:

* the signal and its baseline
* the observed values and window
* input characteristic summaries
* bundle version in service
* which known robustness weakness, if any, the drift resembles

It must not carry raw video, and it must not carry anything identifying users.

The receiving project decides whether to retrain, what to change, and how to evaluate it. That decision is subject to the training project's experimental rules, including that new augmentation must be justified by measurement rather than by a hunch from production.

## Prohibited

* using serving data to fit thresholds, temperature, or weights
* reporting serving-derived numbers as accuracy
* automatic retraining
* automatic threshold adjustment
* retaining clips without explicit consent
* logging video by default
* treating abstentions as errors in dashboards or alerting

The last matters operationally: an alert that fires on abstentions trains the on-call response to ignore the project's most useful signal.
