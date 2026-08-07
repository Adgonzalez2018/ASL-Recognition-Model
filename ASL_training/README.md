# ASL_training

Training system for isolated American Sign Language recognition.

A model receives one short video containing one isolated ASL sign and predicts a gloss label and a confidence score.

This project covers training, evaluation, calibration, and robustness only. Inference serving, frontend, and deployment belong to the future sibling project `ASL_serving` and must not be implemented here.

## Status

Phase 2 (data layer) is complete apart from running the dataset audit, which needs the dataset. Phase 3 (training layer) is complete apart from preflight.

The active granular plan is `docs/CURRENT_PHASE.md`. The stable phase sequence is `docs/ROADMAP.md`.

## Approach

| | |
|---|---|
| Task | Closed-set isolated sign classification |
| Primary dataset | ASL Citizen, official signer-independent splits |
| Secondary dataset | WLASL, as a separate benchmark and external generalization test |
| Primary model | VideoMAE-Base |
| Comparison model | Video Swin-Tiny (torchvision `swin3d_t`) |
| Training | Full-model supervised fine-tuning, multiclass cross-entropy |
| Input | 16 RGB frames at 224x224 |
| Primary training environment | Google Colab |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "ASL_training[dev]"
```

Run the tests:

```bash
pytest ASL_training/tests
```

The default suite runs offline on CPU and needs no dataset. Tests that download pretrained weights are marked `pretrained` and excluded by default:

```bash
pytest ASL_training/tests -m pretrained
```

Lint and format:

```bash
ruff check ASL_training && ruff format --check ASL_training
```

## Audit the dataset

Reads the dataset's own split files, builds the label map and manifests, probes every video, and reports what the dataset actually contains. Required before full training.

```bash
python ASL_training/scripts/audit_dataset.py --dataset-root "$ASL_DATASET_ROOT" --output-dir ASL_training/artifacts --write-manifests
```

Exits non-zero on any integrity failure. Intended to run on Kaggle, where the mirror is already attached; see `docs/DECISIONS.md` D-007.

## Train

```bash
python ASL_training/scripts/train.py \
    --model-config ASL_training/configs/models/videomae_base.yaml \
    --training-config ASL_training/configs/training/baseline.yaml \
    --experiment exp-001-videomae-baseline \
    --run-name videomae-baseline-seed42
```

Re-running the same command resumes from the last checkpoint rather than restarting, which is the normal path on Colab. The test manifest is never loaded.

## Model preflight

Verifies that an architecture constructs, loads pretrained weights, accepts the canonical input, and returns correctly shaped logits. `docs/MODEL_CONTRACT.md` requires this to pass before any real training run.

```bash
python ASL_training/scripts/model_preflight.py --config ASL_training/configs/models/videomae_base.yaml --num-classes 2731
```

Without pretrained download, for a fast structural check:

```bash
python ASL_training/scripts/model_preflight.py --config ASL_training/configs/models/video_swin_tiny.yaml --num-classes 100 --no-pretrained
```

## Layout

```text
docs/       authoritative contracts and phase plans
configs/    dataset, model, training, evaluation, and experiment configuration
src/        the implementation; layer packages under asl_training/
scripts/    command-line entry points
tests/      focused tests, mirroring the layer structure
notebooks/  Colab and Kaggle launchers only, never authoritative logic
artifacts/  generated manifests, label maps, audits, reports
outputs/    run directories; never committed
data/       dataset staging; never committed
```

Dependencies point downward: experiments to training and evaluation, then to models and data, then to utilities. Lower layers must not import higher ones.

## Environment configuration

Paths are supplied through configuration, never hardcoded in `src/`:

```bash
export ASL_DATASET_ROOT=/path/to/asl_citizen
export ASL_OUTPUT_ROOT=/path/to/outputs
```

See `docs/ENVIRONMENTS.md` for local, Colab, and Kaggle specifics, including the Colab session and checkpoint-resume strategy.

## Documentation

Read before modifying the project:

1. `docs/PROJECT.md` — scope
2. `docs/ARCHITECTURE.md` — layers and dependency direction
3. `docs/ROADMAP.md` — phase sequence
4. `docs/CURRENT_PHASE.md` — active plan
5. `docs/MODEL_CONTRACT.md`
6. `docs/DATA_CONTRACT.md`
7. `docs/TRAINING_CONTRACT.md`
8. `docs/EVALUATION_CONTRACT.md`
9. `docs/ENVIRONMENTS.md`
10. `docs/DECISIONS.md`

On conflict: `PROJECT.md` defines scope, contracts define layer behavior, `ROADMAP.md` defines order, `CURRENT_PHASE.md` defines the active plan. Report material conflicts rather than resolving them silently.

## Experimental constraints

These are enforced by review, and some by code:

- The test split is used only for final reporting. Never for checkpoint selection, temperature fitting, or threshold selection.
- Signer-independent splits are authoritative. Signer leakage blocks training.
- The label map is stable across splits, architectures, and experiments.
- Datasets are never silently reduced. A reduced run is labeled a smoke or subset run.
- The clean baseline checkpoint is preserved as the control for all robustness work.
- Clean, robustness, and cross-dataset results are reported separately.

Passing tests establishes pipeline correctness, not model quality.
