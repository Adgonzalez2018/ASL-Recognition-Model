# Bundle Contract

## Purpose

Defines what `ASL_serving` requires from the bundle exported by `ASL_training` Phase 9, and how that bundle is verified before it is used.

This document states the consumer's requirements. The producer's specification lives in `ASL_training/docs/ROADMAP.md` Phase 9. If the two disagree, that is a material conflict and must be reported, not resolved silently.

## Principle

The bundle is the entire interface between the two projects.

If a value is needed to reproduce inference and it is not in the bundle, serving must not infer it, default it, or read it from the training source tree. The correct response is to treat the bundle as incomplete and fix the export.

## Required Contents

### Model

| Field | Purpose |
|---|---|
| architecture identifier | Which adapter to build |
| architecture configuration | Layer sizes, input dimensions, any construction options |
| weights | The selected checkpoint's model state |
| output dimension | Must equal label map size |
| training precision | Documents the numerics the reference logits were produced under |

Weights must be the model state alone. Optimizer state, scheduler state, and scaler state have no meaning in serving and must not ship.

### Labels

| Field | Purpose |
|---|---|
| label map | Class index to gloss, complete and ordered |
| label map identity | Versioned identity string, as recorded in training |

The label map identity in training takes the form `asl_citizen:<classes>:sha256:<digest>`. Serving must record it with each prediction so any prediction can be traced to the exact vocabulary that produced it.

### Preprocessing

Every value needed to turn a clip into a tensor:

| Field | Notes |
|---|---|
| frame count | 16 under D-003 |
| temporal sampling policy | Deterministic evaluation policy, exactly as evaluated |
| short-clip policy | What happens when a clip has fewer frames than required |
| long-clip policy | What happens when it has more |
| resize policy | Including interpolation method and aspect-ratio behavior |
| crop policy | Including crop position for evaluation |
| final height and width | |
| color space | RGB |
| channel order | |
| normalization mean and standard deviation | |
| tensor layout | Dimension order the architecture expects |
| tensor dtype | |
| mirroring policy | Whether inputs are mirrored, and the handedness conclusion reached in training Phase 6 |
| rotation-metadata policy | How container rotation is applied |

### Calibration

| Field | Purpose |
|---|---|
| temperature | Fitted on validation logits |
| calibration identity | Which run and split produced it |
| selected threshold | The confidence threshold for acceptance |
| threshold basis | The coverage and selective-accuracy point it was chosen at |

The threshold basis matters. A bare number invites future adjustment; a number attached to "chosen at 54% coverage and 91% selective accuracy on validation" resists it.

### Provenance

| Field | Purpose |
|---|---|
| training Git commit | |
| bundle version | |
| bundle checksum | Integrity verification |
| dataset identity | Which data version trained this |
| manifest identity | |
| dependency versions | Torch and any library affecting numerics |
| hardware | What it trained on |
| random seed | |

### Evaluation summary

| Field | Purpose |
|---|---|
| clean test metrics | Top-1, top-5, macro F1, mean per-class accuracy |
| per-signer summary | Including worst-signer accuracy |
| calibration metrics | Pre- and post-calibration ECE, NLL |
| selective prediction curve | Accuracy versus coverage |
| robustness summary | Per-condition results from training Phase 6 |

Serving displays none of this to end users. It exists so that observed behavior can be compared against expected behavior, and so a drift report has a baseline to reference.

### Reference fixtures

| Field | Purpose |
|---|---|
| reference clips | A small set of clips, or their identities |
| reference tensors | The exact tensors training produced from them |
| reference logits | The exact logits the selected checkpoint produced |
| tolerance | Documented numerical tolerance for parity comparison |

This is the most important section of the bundle for correctness, and the one most likely to be omitted from a first export. Without it, preprocessing parity cannot be verified and the central guarantee of this project is unavailable.

Reference clips must respect the dataset's licensing. If clips cannot ship, the tensors and logits must, keyed by sample identity, so parity can be checked wherever the dataset is available.

## Verification on Load

Before the service accepts traffic:

1. Bundle exists and is readable.
2. Checksum matches.
3. Every required field is present. Missing fields are named individually in the error.
4. Label map size equals model output dimension.
5. Architecture builds and weights load with no missing or unexpected keys.
6. Temperature is positive and finite.
7. Threshold is within `[0, 1]`.
8. Frame count, resolution, and normalization values are present and plausible.
9. Parity smoke check against at least one reference fixture.

Any failure prevents startup. The service must not start in a degraded mode.

## Prohibited Behavior

Serving must never:

* default a missing preprocessing value
* override a bundle value from configuration or a request parameter
* modify the label map, including adding aliases or display names
* re-fit temperature
* select or adjust a threshold
* accept a bundle whose checksum does not match
* start without a bundle

The first item deserves emphasis. A defaulted normalization constant does not crash. It produces confident, plausible, quietly wrong predictions — the failure mode with the longest time to detection.

## Versioning

Bundle version must change whenever weights, label map, preprocessing, or calibration change.

A prediction response must carry the bundle version. A logged prediction must carry it too. Without that, a change in observed accuracy cannot be attributed to a model change versus an input change.

## Multiple Bundles

Not supported initially. One service instance serves one bundle.

Comparing two models in production is an experiment, and experiments belong to `ASL_training` under controlled conditions. Serving two bundles behind one endpoint would produce results that look like an A/B comparison while satisfying none of the training project's fairness requirements.

## Open Items

* Concrete file format and directory layout — deferred to training Phase 9
* Whether reference clips can ship under the dataset license — see `OPEN_QUESTIONS.md` Q-004
* Whether the robustness summary ships in full or as a reference
