# CLAUDE.md

## Workspace

This repository is the parent repository for the ASL project.

```text
ASL PROJECT/
├── ASL_training/
└── ASL_serving/
```

`ASL_training` is the active project and contains the current model-training, evaluation, calibration, robustness, and experiment work.

`ASL_serving` is reserved for future inference, frontend, deployment, monitoring, and drift-management work.

Unless explicitly instructed otherwise:

* modify only `ASL_training/`
* read project documentation from `ASL_training/docs/`
* do not create or modify files in `ASL_serving/`
* do not introduce shared production modules between the two projects
* keep training-to-serving integration limited to future exported artifacts and contracts

## Project

The active `ASL_training` project implements an isolated American Sign Language video classifier.

The system receives one short video containing one isolated ASL sign and predicts:

* one sign or gloss label
* one confidence score

This project does not currently cover:

* continuous sign-language recognition
* sentence translation
* language modeling
* conversational interpretation
* frontend or production serving
* webcam capture
* model monitoring or drift detection

Training and serving are separate project stages. The current scope is the training system only.

## Current Objective

Build a reproducible training and evaluation pipeline for isolated ASL classification using:

* ASL Citizen as the primary dataset
* VideoMAE-Base as the primary architecture
* Video Swin-Tiny as the comparison architecture
* supervised full-model fine-tuning
* multiclass cross-entropy loss
* official signer-independent dataset splits

The initial work should proceed in this order:

1. Basic model layer
2. Data layer
3. Training layer
4. Evaluation layer
5. Experiment layer
6. Robustness experiments

A thin synthetic or dummy batch may be used while building the model layer. Do not build the complete data system before the model input and output contracts are verified.

## Authoritative Documents

Read these before modifying the project:

1. `ASL_training/docs/PROJECT.md`
2. `ASL_training/docs/ARCHITECTURE.md`
3. `ASL_training/docs/ROADMAP.md`
4. `ASL_training/docs/CURRENT_PHASE.md`
5. `ASL_training/docs/MODEL_CONTRACT.md`
6. `ASL_training/docs/DATA_CONTRACT.md`
7. `ASL_training/docs/TRAINING_CONTRACT.md`
8. `ASL_training/docs/EVALUATION_CONTRACT.md`
9. `ASL_training/docs/DECISIONS.md`

If documentation conflicts:

1. `ASL_training/docs/PROJECT.md` defines project scope.
2. Contract documents under `ASL_training/docs/` define layer behavior.
3. `ASL_training/docs/ROADMAP.md` defines the stable implementation order.
4. `ASL_training/docs/CURRENT_PHASE.md` defines the active granular plan.
5. Experiment configuration files define individual runs.

Do not silently resolve a material conflict. Report it.

## Working Principles

### Keep layers separate

The intended layers are:

* model
* data
* training
* evaluation
* experiments

Avoid circular dependencies.

Expected dependency direction:

```text
experiments
    ↓
training and evaluation
    ↓
models and datasets
```

The model layer must not know about Kaggle, Colab, Google Drive, or dataset download locations.

The dataset layer must not know which experiment is currently active.

The evaluation layer must not modify model weights.

### Keep training and serving separate

Do not add:

* FastAPI
* React
* webcam code
* ONNX serving
* deployment infrastructure
* canary releases
* production monitoring
* drift detection

The training phase should export a precise preprocessing and model contract that a future serving system can reproduce.

All current implementation belongs under `ASL_training/`. Do not create serving code or scaffolding under `ASL_serving/` unless explicitly instructed.

### Preserve experimental validity

Do not:

* mix training and test data
* place the same signer across incompatible splits
* tune hyperparameters against the test split
* select confidence thresholds using test results
* silently remove classes
* silently skip failed samples
* change label mappings between models
* compare models using different data splits
* claim experimental improvement without comparable runs

The official signer-independent ASL Citizen splits should remain authoritative unless a documented experiment explicitly studies another split.

### Baseline before robustness training

The first baseline runs should use only minimal, standard training augmentation.

Do not add aggressive robustness augmentations to the baseline by default.

The intended sequence is:

1. Train a clean baseline.
2. Evaluate clean accuracy and calibration.
3. Stress-test the baseline with controlled perturbations.
4. Identify measured weaknesses.
5. Retrain separate checkpoints with targeted augmentations.
6. Compare clean and robustness performance.

The baseline checkpoint must remain available as the control.

### Shared protocol across architectures

VideoMAE and Video Swin must share, where practical:

* dataset manifests
* label map
* train, validation, and test splits
* frame-selection policy
* spatial input resolution
* evaluation transforms
* metrics
* experiment reporting format

Model-specific tensor ordering and optimization settings may differ when required.

## Dataset Rules

Raw datasets must not be committed to Git.

Do not commit:

* videos
* dataset archives
* extracted raw data
* decoded frame caches
* large generated tensors
* model checkpoints
* optimizer states
* private tokens or credentials

Dataset paths must be supplied through configuration or environment variables.

Expected environments may include:

```text
Local development
Kaggle notebooks
Google Colab
```

Do not hardcode environment-specific absolute paths in reusable modules.

Dataset loaders must make missing or corrupted samples visible. Do not silently ignore them unless the configured policy explicitly permits this and reports the skipped count.

## Model Rules

The initial architectures are:

* VideoMAE-Base
* Video Swin-Tiny

Both must expose a common logical contract:

```text
Input:
A batch of fixed-length RGB video tensors

Output:
Multiclass logits shaped [batch_size, number_of_classes]
```

The model layer should support:

* loading pretrained weights
* replacing the original classification head
* configuring the ASL class count
* full-model fine-tuning
* dummy forward-pass validation
* model-specific input adaptation
* checkpoint loading

Do not add unnecessary abstraction before both initial models work.

## Training Rules

The default training approach is:

* pretrained video backbone
* new ASL classification head
* full supervised fine-tuning
* cross-entropy loss
* AdamW or another explicitly configured optimizer
* learning-rate scheduling
* mixed precision where supported
* checkpointing
* resume support
* reproducible seeds
* configuration capture

A smoke run is not a completed experiment.

Every real run must record:

* experiment name
* Git commit
* dataset version
* manifest version
* label-map version
* model checkpoint source
* hyperparameters
* preprocessing configuration
* augmentation configuration
* hardware
* random seed
* metrics
* checkpoint location

Never silently reduce the dataset to make training succeed. A reduced-data run must be labeled as a smoke test or subset experiment.

## Evaluation Rules

The core metrics are:

* top-1 accuracy
* top-5 accuracy
* macro F1
* mean per-class accuracy
* per-class precision and recall
* per-signer performance
* confusion analysis
* negative log-likelihood

Confidence evaluation should include:

* raw maximum softmax score
* temperature scaling on validation logits
* expected calibration error
* reliability analysis
* selective accuracy
* coverage at chosen thresholds

Temperature fitting and threshold selection must use validation data.

The untouched test split should be used only for final reporting.

Robustness results must remain separate from clean test results.

WLASL may later be used as:

* an independently trained secondary benchmark
* an external cross-dataset evaluation on verified overlapping labels

Do not merge WLASL and ASL Citizen labels without an explicit reviewed harmonization table.

## Testing Expectations

Tests should focus on software correctness and experimental safeguards.

Important tests include:

* model output shapes
* label-map stability
* manifest parsing
* split integrity
* signer leakage detection
* deterministic evaluation sampling
* consistent transforms across video frames
* short-video handling
* corrupted-video reporting
* checkpoint save and resume
* metric correctness
* configuration validation

Tests do not prove model quality. Model quality is established through controlled experiments and evaluation.

## Task Execution

For each implementation task:

1. Read the relevant project and contract documents.
2. Inspect the existing code before proposing changes.
3. Identify the smallest coherent implementation scope.
4. Implement only that scope.
5. Add or update focused tests.
6. Run the relevant validation commands.
7. Report any assumptions, unresolved questions, or deviations.

Do not rewrite unrelated modules.

Do not add speculative functionality outside the active roadmap phase.

Do not update architectural documents merely to justify an implementation change. Architecture changes require an explicit decision.

## Current Phase Workflow

`ASL_training/docs/ROADMAP.md` defines the stable, high-level project phases.

`ASL_training/docs/CURRENT_PHASE.md` defines the active phase in granular detail and is the primary execution plan for implementation work.

At the start of each task:

1. Read `ASL_training/docs/CURRENT_PHASE.md`.
2. Confirm the task belongs to the active phase.
3. Work only on the current scoped task unless explicitly instructed otherwise.

During implementation:

- Check off a task only after its implementation and focused validation are complete.
- Update `Current Task`, `Status`, and `Blockers` when they materially change.
- Do not mark the full phase complete merely because code was written.
- Do not add unrelated future-phase work.

When the phase acceptance criteria are satisfied:

1. Complete the phase summary in `ASL_training/docs/CURRENT_PHASE.md`.
2. Mark the phase complete in `ASL_training/docs/ROADMAP.md`.
3. Move the completed file to:
   `ASL_training/docs/phases/archive/PHASE-<number>-<name>.md`
4. Create a new `ASL_training/docs/CURRENT_PHASE.md` for the next roadmap phase.
5. Preserve important lasting choices in `ASL_training/docs/DECISIONS.md` only when necessary.

Do not archive or advance a phase when tests are failing, acceptance criteria remain unmet, or unresolved deviations materially affect completion.

Claude may update task-level progress autonomously. Do not archive the active phase, mark the phase complete in the roadmap, or create the next phase plan without explicit user approval.

## Completion Report

At the end of each task, report:

### Summary

What was implemented.

### Files Changed

List production, test, configuration, and documentation files separately.

### Validation

List commands run and their outcomes.

### Behavior

Describe the resulting input and output behavior.

### Deviations

State any differences from the requested task or contracts.

### Remaining Risks

List unresolved technical or experimental concerns.

### Next Step

State the most direct next roadmap task.

If a task could not be fully completed, clearly distinguish completed work from partial or unverified work.
