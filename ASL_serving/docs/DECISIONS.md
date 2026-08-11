# Decisions

Lasting technical decisions for `ASL_serving`.

A decision belongs here when reversing it would change what a prediction means, break bundle compatibility, or invalidate a guarantee inherited from `ASL_training`. Routine implementation choices do not belong here.

Decisions are append-only. A superseded decision is marked `Superseded` and references its replacement. Entries are not deleted.

Serving decisions are numbered `S-<number>` to keep them distinct from the training project's `D-<number>` series.

## Format

```text
## S-<number>: <title>

Date:
Status:      Accepted | Superseded | Rejected
Phase:
Supersedes:
Superseded by:

### Context
### Decision
### Consequences
### Alternatives considered
```

---

## S-001: The exported bundle is the only interface to `ASL_training`

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

Both projects live in one Git repository (training D-001). Nothing mechanical prevents `ASL_serving` from importing training modules, and doing so would be the fastest way to guarantee identical preprocessing.

That convenience carries a cost. Training depends on a heavy stack pinned for a T4 under free-tier quota; serving needs a small, stable runtime. Sharing modules would force each project's constraints onto the other, and would make it impossible to answer "which preprocessing produced this prediction" from a deployed artifact alone.

### Decision

`ASL_serving` consumes a versioned bundle and nothing else. No imports from `ASL_training`, no reading of training checkpoints or manifests, no runtime dependency on the training stack.

Serving reimplements preprocessing and proves equivalence by parity tests against training-produced reference tensors.

### Consequences

* Preprocessing exists twice, deliberately. The duplication is the cost of independent deployability.
* Parity tests become load-bearing. Without them, the duplication is a liability rather than a boundary.
* The bundle must be complete. A missing value is a training-side export bug, not something serving may default.
* Reference tensors and logits must ship with the bundle, or parity cannot be verified.

### Alternatives considered

**Shared preprocessing package.** Guarantees parity by construction and removes the duplication. Rejected because it couples dependency pins across two projects with different runtimes, and because a shared package version does not travel with a deployed model the way a bundle field does.

**Serving imports training directly.** Rejected for the same reasons, more strongly: it would make the training stack a production dependency.

---

## S-002: Clip boundaries are user-determined, not detected

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

ASL Citizen clips are pre-segmented, one isolated sign each. A webcam produces an unbounded stream. Something must decide where a sign starts and ends, and that decision has no counterpart anywhere in the training project.

Automatic segmentation is adjacent to continuous sign recognition, which `ASL_training/docs/PROJECT.md` places outside project scope.

### Decision

The initial implementation records between an explicit user-initiated start and stop. No onset or offset detection.

### Consequences

* The user is responsible for recording one cleanly bounded sign; the interface must make that expectation obvious.
* Accuracy figures remain attributable to the classifier alone, with no unevaluated segmenter in front of it.
* Clips will frequently contain leading and trailing stillness, which interacts with the long-clip sampling policy. This must be measured, not assumed harmless.
* Automatic boundary detection, if ever added, is a new decision and requires its own evaluation.

### Alternatives considered

**Motion-energy segmentation.** Cheap and label-free. Rejected for the initial implementation because it inserts an unevaluated component into the prediction path, making every downstream accuracy number uninterpretable.

**Fixed-duration recording.** Simple and bounded. Rejected as the sole option because signs vary in duration and a fixed window truncates or pads arbitrarily. May return as a guided default within user-controlled recording.

---

## S-003: Calibration and threshold values are applied, never derived

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

Temperature and the confidence threshold are fitted in `ASL_training` on validation data, under `EVALUATION_CONTRACT.md` rules that exist to keep the test split untouched and to prevent threshold selection from becoming an unlabeled fitting exercise.

Serving observes a continuous stream of unlabeled predictions and will face pressure to adjust these values when a demonstration abstains too often or a dashboard looks wrong.

### Decision

Temperature and threshold come from the bundle. Serving never re-fits, never adjusts, and never accepts either as configuration or as a request parameter. Changing them means producing a new bundle from a training-side analysis.

### Consequences

* The abstention rate remains a meaningful drift signal, because it is not being suppressed by threshold changes.
* A demonstration that abstains too often is evidence about the model, not a UI problem to tune away.
* Per-class thresholds are unavailable unless a training-side analysis produces them.
* Any serving component that wants a different operating point must request a new bundle.

### Alternatives considered

**Configurable threshold for operational flexibility.** Rejected. Every plausible use of that flexibility is a form of fitting on unlabeled production data, and it would make two deployments with the same bundle version behave differently.

---

## S-004: Preprocessing parity is verified against training-produced references

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

S-001 duplicates preprocessing across projects. Duplication drifts. The failure mode is silent: a wrong normalization constant or an inverted channel order produces plausible, confident, quietly degraded predictions rather than a crash.

A parity test written by reading both implementations proves only that one reading was applied twice.

### Decision

`ASL_training` ships reference clips (or their identities), the exact tensors it produced from them, the resulting logits, and a numerical tolerance. Serving verifies both tensor and logit parity against these in CI, and runs at least one reference through the pipeline at startup.

Parity failure blocks release and blocks startup respectively.

### Consequences

* Training Phase 9 must export reference fixtures. This is an added requirement on the handoff package and must be raised before Phase 9 is planned in detail.
* Numerics-affecting changes — decoder upgrade, tensor library upgrade, quantization, ONNX export — require a full parity re-run.
* Precision differences between training and serving must be reflected in the tolerance rather than treated as failures.

### Alternatives considered

**Compare accuracy on a held-out set instead.** Rejected: it needs the dataset present in serving, it is far less sensitive than tensor comparison, and it detects preprocessing bugs only after they have already cost measurable accuracy.

---

## S-005: Serving detects drift; it does not retrain

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

Serving accumulates unlabeled traffic. The obvious use is a feedback loop into training. The training project forbids several things such a loop would do casually: tuning against non-validation data, silently changing label mappings, and claiming improvement without comparable runs.

### Decision

Serving logs, computes drift signals against recorded baselines, evaluates written trigger criteria, and produces a report for `ASL_training`. It never retrains, never adjusts model parameters, and never modifies thresholds in response to observed traffic.

### Consequences

* Trigger criteria must be quantitative and recorded before they fire.
* A fired trigger is an input to a human decision, not an automated pipeline.
* Retraining remains subject to the training project's experimental controls.
* Live clips are unlabeled and cannot enter a training set without a labeling process that does not currently exist.

### Alternatives considered

**Automated retraining on drift.** Rejected. Unlabeled production data cannot supervise training, and an automatic pipeline would produce checkpoints no one had chosen to evaluate.

---

## S-006: One service instance serves exactly one bundle

Date: 2026-08-11
Status: Accepted
Phase: Planning

### Context

Serving two models behind one endpoint is a natural way to compare them in real conditions.

`ASL_training/docs/ROADMAP.md` Phase 5 defines what a fair architecture comparison requires: shared manifests, splits, transforms, metrics, and seed policy. Production traffic satisfies none of these, and yields no labels.

### Decision

One instance, one bundle. No multi-model routing, no traffic splitting, no in-service comparison.

### Consequences

* Bundle version is unambiguous for every prediction and every log line.
* Rollback means deploying a previous bundle version, which must remain retrievable.
* Model comparison stays in `ASL_training`, where it can be done fairly.

### Alternatives considered

**Shadow deployment for latency measurement.** Not rejected in principle, since it compares cost rather than quality, but deferred. It would need its own decision and must not produce anything resembling an accuracy comparison.
