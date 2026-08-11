# API Contract

## Purpose

Defines the HTTP surface: request shapes, response shapes, outcome taxonomy, and errors.

The API owns no model logic. It validates, delegates, and formats.

## Outcome Taxonomy

Three distinct outcomes, never conflated:

| Outcome | Meaning | Transport |
|---|---|---|
| Prediction | Usable clip, confidence at or above threshold | Success |
| Abstention | Usable clip, confidence below threshold | Success |
| Rejection | Clip unusable; no inference performed | Client error |

Plus:

| Outcome | Meaning | Transport |
|---|---|---|
| Service fault | Inference failed, bundle problem, internal error | Server error |

Abstention is a success. It is the model's honest answer, and encoding it as a failure would make the abstention rate impossible to distinguish from an outage in any monitoring dashboard.

## Endpoints

### Predict

Accepts one clip, returns one outcome.

Request carries:

* the clip, as an uploaded file
* nothing else that affects the result

The last point is a hard constraint. No request field may alter the threshold, temperature, preprocessing, frame count, or model. Any such parameter would let a caller invalidate the prediction contract.

An optional `top_k` is permissible, since it affects presentation only, and must be bounded.

Response, accepted:

```text
outcome        "prediction"
gloss          predicted sign
confidence     calibrated, in [0, 1]
alternatives   ranked list of {gloss, confidence}
bundle         bundle version
label_map      label map identity
calibration    calibration identity
request_id     for log correlation
```

Response, abstained:

```text
outcome        "abstention"
confidence     calibrated maximum, in [0, 1]
alternatives   optional, explicitly marked unreliable
bundle         bundle version
label_map      label map identity
calibration    calibration identity
request_id     for log correlation
```

Identity fields appear in every response. Without them, a prediction cannot be traced to the vocabulary, weights, and calibration that produced it, and a user-reported error cannot be reproduced.

### Health

Reports:

* readiness — whether the model is loaded and the parity smoke check passed
* bundle version
* label map identity
* device

Must not expose file paths, weights, internal traces, or configuration secrets.

A request arriving before readiness fails explicitly. It must not queue indefinitely; a hung request is indistinguishable from a slow model.

### Vocabulary

Optional, and recommended.

Returns the servable vocabulary, so a client can tell the user what the system can recognize. The vocabulary is closed at 2,731 classes and users will otherwise discover its limits only through confident wrong answers.

If implemented, it must serve the bundle's label map verbatim, without aliases, additions, or display-name substitutions.

## Rejection Taxonomy

Rejections must be specific and actionable. "Invalid clip" is not acceptable; the user cannot act on it.

| Reason | Message intent |
|---|---|
| Undecodable | The file could not be read as video |
| Zero frames | No video frames were found |
| Too short | State the minimum |
| Too long | State the maximum |
| Resolution below floor | State the floor |
| Static clip | The recording appears to contain no motion |
| Payload too large | State the limit |
| Unsupported container | State what is accepted |

Each maps to a stable machine-readable code, so the frontend can offer targeted remediation without parsing prose.

## Service Faults

| Condition | Behavior |
|---|---|
| Model not ready | Explicit not-ready response |
| Inference raised | Server error, logged with bundle identity and request id |
| Non-finite logits | Server error, never a prediction |
| Bundle unreadable at startup | Refuse to start |

Fault responses must not leak stack traces, file paths, or bundle internals to the client. They must log enough internally to reproduce.

## Limits

* maximum upload size
* maximum clip duration
* request timeout
* concurrent request ceiling

Concrete values depend on measured latency and the selected architecture, neither of which exists yet. They are placeholders until training Phase 5 and serving Phase S7 supply real numbers.

Limits must be enforced before decoding. Decoding an oversized clip to discover it is oversized spends the cost the limit exists to avoid.

## Versioning

The response shape is a contract. Breaking changes require a version marker.

Bundle version changes are not API version changes. The bundle version travels inside the response precisely so that the model can change without the API shape changing.

## What the API Must Not Do

* accept preprocessing, threshold, temperature, or model parameters
* return raw logits by default
* return internal paths or bundle contents
* store clips as a side effect of prediction
* retry inference silently on failure
* fall back to a different model or a default when the bundle is unavailable
* treat abstention as an error

## Security and Privacy

* Uploads are untrusted input. Enforce size and type limits before decoding.
* Never execute or shell out with user-supplied filenames.
* Clips are not retained by default. See `CAPTURE_CONTRACT.md`.
* Request logs carry metadata, not video. See `MONITORING_CONTRACT.md`.
* Errors must not echo file contents or filenames back to the client.

## Deferred

* authentication and rate limiting per user
* batch prediction endpoints
* streaming or partial-result responses
* multi-model routing
* asynchronous job submission for long clips
