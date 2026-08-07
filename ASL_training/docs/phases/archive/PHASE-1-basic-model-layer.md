# Phase 1: Basic Model Layer

Status: Complete
Archived: 2026-08-07

## Objective

Establish a shared video-classification contract and verify both pretrained architectures on synthetic batches.

## Tasks

- [x] Define the canonical logical input shape: `[batch, frames, channels, height, width]`.
- [x] Define the shared classification output: `VideoClassifierOutput` carrying logits and optional loss.
- [x] Add a model factory with an explicit, stable architecture registry.
- [x] Implement the VideoMAE adapter.
- [x] Implement the Video Swin adapter.
- [x] Load pretrained weights.
- [x] Replace each original classification head with a configurable ASL head.
- [x] Support a configurable number of classes, resolved at runtime from the label map.
- [x] Add model-specific tensor adaptation internally.
- [x] Report model parameter counts.
- [x] Add dummy-forward smoke tests.
- [x] Add checkpoint state-loading tests.

## Acceptance Criteria

For both architectures:

- [x] Pretrained construction succeeds.
- [x] A dummy video batch is accepted.
- [x] Output logits have shape `[batch_size, num_classes]`.
- [x] The classification head has the configured output size.
- [x] Architecture-specific objects do not leak past the model layer.
- [x] Tests verify input and output behavior.
- [x] No real dataset implementation is required.

## Implementation

```text
src/asl_training/models/
├── __init__.py      public surface
├── outputs.py       VideoClassifierOutput
├── config.py        ModelConfig, validation, YAML loading
├── base.py          BaseVideoClassifier: input and label validation,
│                    loss, parameter reporting, fine-tuning strategy
├── videomae.py      VideoMAE-Base adapter
├── video_swin.py    Video Swin-Tiny adapter
└── factory.py       registry, build_model, load_checkpoint_state

configs/models/videomae_base.yaml
configs/models/video_swin_tiny.yaml
scripts/model_preflight.py
notebooks/colab/00_model_preflight.ipynb
```

The base class owns validation and loss so that both architectures fail identically on identical mistakes. Adapters own only construction, head replacement, tensor adaptation, and preprocessing metadata.

## Validation

| Command | Result |
|---|---|
| `pytest ASL_training/tests` | 115 passed |
| `pytest ASL_training/tests -m pretrained` | 7 passed |
| `ruff check ASL_training` | passed |
| `ruff format --check ASL_training` | passed |
| `scripts/model_preflight.py`, VideoMAE, 2731 classes | passed |
| `scripts/model_preflight.py`, Video Swin, 2731 classes | passed |

Preflight measured, at 2731 classes:

| | VideoMAE-Base | Video Swin-Tiny |
|---|---|---|
| Total parameters | 88,336,555 | 29,950,609 |
| Approx. fp32 size | 337 MB | 114 MB |
| Head parameters | 2,100,139 | 2,100,139 |
| Initial loss | 7.70 | 7.98 |

Expected initial loss for a random head over 2731 classes is ln(2731) = 7.91.

VideoMAE-Base is roughly three times the size of Video Swin-Tiny, which will shape batch size and epoch time on Colab. Both share a 768-dimensional feature width, hence identical head sizes.

## Decisions Recorded

* D-002: Video Swin weights come from torchvision rather than mmaction2.
* D-003: Both architectures run at 16 frames in the baseline.
* D-005: Model-layer tests do not require network access.
* D-006: VideoMAE legacy attention-bias names are repaired on load.

## Finding: silent VideoMAE weight-loading defect

Preflight surfaced that `transformers` 5.x does not translate VideoMAE's published `q_bias`/`v_bias` parameter names to the `query.bias`/`value.bias` it expects. Twenty-four bias tensors were left at zero while `from_pretrained` reported success. Measured on layer 0: loaded norm 0.0 against the checkpoint's 17.54.

The adapter now detects and repairs this, classifies every missing and unused key as expected or unexplained, and records the result in the load report. See D-006.

This is the reason a real training run must gate on preflight rather than on the test suite: the offline structural suite cannot see it.

## Deviations

Video Swin-Tiny runs at 16 frames despite 32-frame pretraining, to hold the data protocol constant across architectures. Recorded as D-003 and flagged as a confound to rule out before concluding anything about relative architecture quality in Phase 5.

## Phase Summary

Both architectures load pretrained weights, accept the canonical dummy batch, expose one logical interface, and return logits at the configured class count. Preflight passes on both and blocks on failure, as `docs/MODEL_CONTRACT.md` requires.

A silent pretrained-weight defect affecting every VideoMAE run was found and fixed during this phase.

## Completion Artifact

A stable model-layer interface that the data and training layers can target.
