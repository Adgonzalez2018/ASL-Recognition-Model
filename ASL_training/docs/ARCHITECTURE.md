# Training Architecture

## Purpose

This document defines the architecture of the isolated ASL recognition training project.

The parent workspace is:

```text
ASL PROJECT/
├── ASL_training/
└── ASL_serving/
```

`ASL_training` is the active project.

It owns:

* dataset preparation
* model integration
* supervised fine-tuning
* evaluation
* confidence calibration
* robustness analysis
* experiment tracking
* checkpoint export

`ASL_serving` is reserved for future work.

It may eventually own:

* webcam capture
* video upload
* inference APIs
* frontend integration
* deployment
* canary releases
* monitoring
* drift detection

No serving implementation should be added to `ASL_training`.

## Workspace Boundary

Claude may have visibility into the full `ASL PROJECT` parent folder, but current tasks should modify only:

```text
ASL PROJECT/ASL_training/
```

Unless explicitly instructed otherwise, Claude must not:

* create production code in `ASL_serving`
* move training modules into `ASL_serving`
* create shared infrastructure across both projects
* couple training code to a future serving implementation
* design APIs or frontend behavior
* add deployment dependencies

The only connection between the two future projects should be a versioned model artifact and preprocessing contract exported by `ASL_training`.

## System Overview

The training system is divided into five primary layers:

```text
Experiment layer
        ↓
Training and evaluation layers
        ↓
Model and data layers
```

The layers are:

1. Model layer
2. Data layer
3. Training layer
4. Evaluation layer
5. Experiment layer

Supporting utilities may exist, but they must not become an unstructured shared layer that owns domain behavior.

## Dependency Direction

The expected dependency direction is:

```text
experiments
    ├── training
    └── evaluation
          ↓
      models + data
          ↓
        utilities
```

More explicitly:

```text
Experiment configurations
        ↓
Training orchestration ───── Evaluation orchestration
        ↓                            ↓
Model adapters  ←──────── Shared batch contract ───────→ Metrics
        ↑                            ↑
        └──────────── Data pipeline ─┘
```

Dependencies should point downward.

Lower layers must not import higher layers.

Examples:

* Models must not import training code.
* Data modules must not import experiment configurations directly.
* Evaluation must not update model weights.
* Training must not define dataset-specific annotation parsing.
* Experiment files should compose existing layers rather than contain core logic.

## Proposed Repository Structure

All paths below are relative to:

```text
ASL PROJECT/ASL_training/
```

```text
ASL_training/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── docs/
│   ├── PROJECT.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── CURRENT_PHASE.md
│   ├── MODEL_CONTRACT.md
│   ├── DATA_CONTRACT.md
│   ├── TRAINING_CONTRACT.md
│   ├── EVALUATION_CONTRACT.md
│   ├── ENVIRONMENTS.md
│   ├── DECISIONS.md
│   ├── phases/
│   │   └── archive/
│   └── experiments/
│       ├── README.md
│       └── templates/
│           └── EXPERIMENT_TEMPLATE.md
│
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── experiments/
│
├── src/
│   └── asl_training/
│       ├── __init__.py
│       │
│       ├── models/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── experiments/
│       └── utils/
│
├── scripts/
│   ├── audit_dataset.py
│   ├── build_manifests.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   └── run_robustness.py
│
├── notebooks/
│   ├── colab/
│   └── kaggle/
│
├── tests/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── integration/
│
├── artifacts/
│   ├── manifests/
│   ├── label_maps/
│   ├── audits/
│   └── reports/
│
├── outputs/
│   └── .gitkeep
│
└── data/
    └── .gitkeep
```

`CLAUDE.md` and `.gitignore` live at the parent workspace root rather than inside `ASL_training/`, because the Git repository root is the parent workspace. See D-001 in `docs/DECISIONS.md`.

The repository may evolve, but changes to the major layer boundaries require an explicit architectural decision.

## Model Layer

### Responsibility

The model layer owns the neural network architectures used for isolated sign classification.

Initial architectures:

* VideoMAE-Base
* Video Swin-Tiny

The model layer should provide a shared logical contract while containing architecture-specific details internally.

### Inputs

The model layer receives a standardized batch of RGB video clips.

Canonical logical shape:

```text
[batch, frames, channels, height, width]
```

The canonical representation exists at the project boundary.

An adapter may rearrange dimensions internally if a model implementation expects another format.

### Outputs

Every classifier should return:

```text
logits: [batch, number_of_classes]
```

It may also return:

* loss, when labels are provided
* intermediate metadata needed for debugging
* hidden representations when explicitly requested

Architecture-specific output objects should not leak into the training and evaluation layers.

### Model-Layer Ownership

The model layer owns:

* pretrained checkpoint loading
* classifier-head replacement
* architecture-specific input adaptation
* model configuration
* full or partial fine-tuning controls
* model checkpoint state loading
* dummy-forward validation
* parameter-count reporting

It does not own:

* video decoding
* label-map creation
* dataset splits
* optimization loops
* metric computation
* confidence thresholds
* experiment interpretation

### Initial Model Components

Expected modules may include:

```text
src/asl_training/models/
├── base.py
├── factory.py
├── outputs.py
├── videomae.py
└── video_swin.py
```

Avoid deep inheritance hierarchies.

A small shared protocol or base class is preferable to an elaborate framework.

## Data Layer

### Responsibility

The data layer converts raw isolated-sign datasets into standardized, reproducible model inputs.

The primary dataset is ASL Citizen.

WLASL may be added later as a separate dataset adapter.

### Data-Layer Ownership

The data layer owns:

* annotation parsing
* manifest generation
* stable label mapping
* split preservation
* signer metadata
* corruption detection
* video decoding
* temporal sampling
* spatial preprocessing
* training augmentation
* deterministic evaluation transforms
* batch collation
* dataset audit reports

It does not own:

* model selection
* optimizer configuration
* loss optimization
* confidence calibration
* checkpoint selection
* experimental conclusions

### Raw Data Location

Raw datasets are external to the repository.

Possible runtime roots include:

```text
Local:
<external-path>/asl_citizen

Kaggle:
/kaggle/input/<dataset-name>

Google Colab:
/content/<dataset-name>
```

The root must be passed through configuration or environment variables.

Reusable modules must not hardcode any one environment.

### Manifest as Source of Truth

Training code should consume manifests rather than infer dataset structure repeatedly.

A manifest record should identify at least:

* sample ID
* video path
* gloss
* class ID
* signer ID
* split
* dataset source

Optional audited metadata may include:

* duration
* frame rate
* frame count
* resolution
* corruption status
* handedness, if available
* mirroring status, if known

### Split Integrity

ASL Citizen’s official signer-independent split is authoritative.

The data layer must make it possible to verify that:

* train, validation, and test records are distinct
* signers do not leak across incompatible splits
* the same video does not appear more than once
* label IDs remain stable across splits
* both model families use the same records

### Transform Separation

Training transforms and evaluation transforms must be separate.

Training may use controlled randomness.

Validation and test transforms must be deterministic.

All spatial transformations must remain temporally consistent across the frames of a clip. A random crop must use one crop window for the entire clip, not a different crop for every frame.

## Training Layer

### Responsibility

The training layer fine-tunes a configured model using batches supplied by the data layer.

### Training-Layer Ownership

The training layer owns:

* cross-entropy optimization
* optimizer construction
* learning-rate scheduling
* mixed precision
* gradient accumulation
* gradient clipping
* distributed-training integration, if needed
* checkpoint creation
* checkpoint resume
* early stopping, if configured
* validation scheduling
* training logs
* reproducibility controls
* run-state capture

It does not own:

* raw annotation parsing
* model architecture definitions
* test-set interpretation
* dataset label harmonization
* robustness transformation definitions
* production inference

### Training Contract

The default initial training process is:

```text
pretrained video model
        ↓
replace classification head
        ↓
full-model supervised fine-tuning
        ↓
multiclass logits
        ↓
cross-entropy loss
```

The baseline should use minimal augmentation.

Aggressive robustness augmentation belongs to later, separate experiments.

### Checkpointing

A checkpoint should preserve enough information to resume training and reproduce evaluation.

Expected contents include:

* model state
* optimizer state
* scheduler state
* epoch or step
* best validation metric
* random state where feasible
* experiment configuration
* label-map identity
* preprocessing identity
* source Git commit

Checkpoints must not be committed to Git.

## Evaluation Layer

### Responsibility

The evaluation layer measures model quality without modifying model weights.

It should support:

* validation during training
* final clean evaluation
* confidence calibration
* selective prediction
* robustness evaluation
* cross-dataset analysis

### Evaluation-Layer Ownership

The evaluation layer owns:

* top-1 accuracy
* top-5 accuracy
* macro F1
* weighted F1
* mean class accuracy
* per-class metrics
* per-signer metrics
* confusion analysis
* validation logit export
* negative log-likelihood
* expected calibration error
* reliability analysis
* temperature scaling
* threshold analysis
* coverage and selective accuracy
* robustness stress-test results

It does not own:

* gradient updates
* model checkpoint selection using test data
* dataset split creation
* training augmentation
* production monitoring

### Clean and Robustness Evaluation

Clean test metrics and robustness metrics must remain separate.

Example:

```text
Clean evaluation:
Official ASL Citizen test split

Robustness evaluation:
Derived mirrored, degraded, or temporally perturbed test conditions

External evaluation:
Verified WLASL overlap
```

Do not merge these into one headline metric.

### Calibration Boundary

Temperature scaling is fit using validation logits.

Confidence thresholds are selected using validation results.

The untouched test split is used only to report the final selected operating behavior.

## Experiment Layer

### Responsibility

The experiment layer defines and records controlled model-development runs.

It should answer:

* What hypothesis is being tested?
* Which dataset and model are used?
* What differs from the baseline?
* Which configuration produced the run?
* Which checkpoint and metrics resulted?
* What conclusion is supported?
* What should happen next?

### Experiment-Layer Ownership

The experiment layer owns:

* experiment configuration composition
* run identifiers
* hypotheses
* baseline references
* seeds
* hardware metadata
* dataset and checkpoint versions
* result summaries
* interpretation
* next-step decisions

It should not contain reusable model, dataset, training, or metric logic.

### Executable and Explanatory Sources

YAML or equivalent configuration files are the executable source of truth.

Markdown experiment records are the explanatory source of truth.

Example:

```text
configs/experiments/exp-001-videomae-baseline.yaml
docs/experiments/EXP-001-videomae-baseline.md
```

## Configuration Architecture

Configuration should be explicit and versionable.

Recommended configuration groups:

```text
configs/
├── datasets/
│   ├── asl_citizen.yaml
│   └── wlasl.yaml
├── models/
│   ├── videomae_base.yaml
│   └── video_swin_tiny.yaml
├── training/
│   ├── baseline.yaml
│   └── robustness.yaml
├── evaluation/
│   ├── clean.yaml
│   └── robustness.yaml
└── experiments/
    ├── exp-001-videomae-baseline.yaml
    └── exp-002-swin-baseline.yaml
```

The implementation may use a configuration library later, but the first version should prioritize readability over advanced composition.

## Artifact Architecture

`ASL_training` should produce several categories of artifacts.

### Repository Artifacts

Safe to commit when licensing permits:

* source code
* configuration files
* manifests without private or restricted content
* label maps
* audit summaries
* test fixtures
* experiment Markdown records
* lightweight metric summaries

### External Runtime Artifacts

Do not commit:

* raw videos
* dataset archives
* extracted data
* decoded-frame caches
* model checkpoints
* optimizer states
* large prediction files
* secrets
* API tokens

### Training-to-Serving Handoff

Future `ASL_serving` work should consume an exported bundle from `ASL_training`.

The logical bundle should include:

```text
selected_model/
├── model weights
├── model configuration
├── stable label map
├── preprocessing specification
├── calibration parameters
├── confidence threshold policy
├── training metadata
└── evaluation summary
```

The exact file format will be decided later.

The serving project must reproduce this contract rather than reimplement preprocessing from memory.

## Notebook Boundary

Kaggle and Colab notebooks are launch environments, not the main implementation.

Notebooks may:

* clone or install the repository
* authenticate to data or artifact services
* resolve environment-specific paths
* start a training or evaluation command
* save external artifacts
* display summary results

Notebooks should not contain the authoritative:

* model definitions
* dataset loaders
* training loop
* metric implementations
* label mapping
* experiment logic

Those belong in `src/`, configuration files, and scripts.

## Testing Architecture

Testing is divided by layer.

### Model Tests

* pretrained model construction
* classifier-head dimensions
* dummy forward pass
* canonical input adaptation
* logits shape
* checkpoint loading

### Data Tests

* manifest parsing
* label-map stability
* split integrity
* signer leakage detection
* deterministic evaluation sampling
* short-video behavior
* corrupted-video reporting
* temporally consistent transforms

### Training Tests

* one-batch optimization
* loss computation
* gradient update
* checkpoint save and resume
* configuration capture
* reduced smoke-run completion

### Evaluation Tests

* metric correctness
* per-class aggregation
* per-signer aggregation
* calibration calculations
* threshold and coverage calculations
* no-gradient behavior

### Integration Tests

* manifest to model forward pass
* small end-to-end smoke run
* checkpoint to evaluation report
* comparable model evaluation under one shared protocol

Tests establish software correctness, not model quality.

## Error Handling

The system should fail clearly when experimental validity is at risk.

Examples that should produce explicit errors or audit failures:

* unknown class IDs
* duplicate class mappings
* signer leakage
* missing manifest columns
* incompatible checkpoint vocabulary
* inconsistent frame counts after preprocessing
* unsupported video decoding
* test-set threshold selection
* mismatched model and label-map sizes

Corrupted samples may be skipped only through a documented, counted policy.

## Reproducibility

Every real experiment should preserve:

* Git commit
* configuration
* random seed
* dataset version
* manifest identity
* label-map identity
* dependency versions
* CUDA and GPU information
* pretrained checkpoint source
* augmentation policy
* metric results
* output checkpoint reference

A result without this metadata should not be treated as fully reproducible.

## Parallel Development

The intended development order is:

```text
basic model layer
        ↓
data layer
        ↓
training integration
        ↓
first full training runs
```

While training runs execute, separate work may continue on:

* evaluation metrics
* calibration
* experiment reporting
* robustness test definitions

Parallel work must not modify the active training path without coordination.

## Architectural Non-Goals

The current architecture should not attempt to solve:

* model serving
* user authentication
* frontend state
* webcam browser compatibility
* production latency optimization
* live video segmentation
* deployment scaling
* model registry infrastructure
* automated retraining
* drift response
* continuous ASL
* translation

These belong to future decisions, primarily in `ASL_serving`.
