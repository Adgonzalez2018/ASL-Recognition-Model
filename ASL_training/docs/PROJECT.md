# Isolated ASL Recognition

## Project Summary

This project trains a video-classification model to recognize one isolated American Sign Language sign from a short video clip.

The model receives a video containing a single sign and produces:

* a predicted sign or gloss
* a confidence score

The project focuses first on model training, evaluation, calibration, and robustness. Serving will be designed later around the model and preprocessing pipeline that perform best.

## Problem Definition

This is a closed-set multiclass isolated sign recognition problem.

Given a video (X), the model learns:

```text
video clip
    ↓
video classifier
    ↓
class logits
    ↓
predicted ASL gloss and confidence
```

Each input should contain one isolated sign.

The initial classifier assumes that the correct sign belongs to the configured vocabulary. Invalid clips, incomplete signs, random gestures, and signs outside the vocabulary will be addressed through later robustness and out-of-distribution evaluation.

## Scope

### Included

* isolated ASL video classification
* motion-aware video modeling
* supervised fine-tuning of pretrained video architectures
* ASL Citizen dataset integration
* VideoMAE and Video Swin comparison
* signer-independent training and evaluation
* clean baseline experiments
* confidence calibration
* selective prediction analysis
* per-class and per-signer error analysis
* robustness stress testing
* targeted augmentation experiments
* optional WLASL secondary benchmarking
* reproducible experiment configuration
* checkpoint and metric export

### Not Included Yet

* continuous sign recognition
* sentence segmentation
* ASL-to-English translation
* language modeling
* conversational interpretation
* facial grammar analysis
* frontend development
* webcam capture
* API serving
* mobile deployment
* production monitoring
* canary deployment
* drift detection
* real-time inference optimization

These may become later project stages, but they must not constrain the initial training architecture unnecessarily.

## Primary Goal

Train and select a model that can classify isolated ASL signs from short videos while generalizing to signers not seen during training.

The selected model should provide:

* strong top-1 classification performance
* balanced performance across classes
* reasonable generalization across unseen signers
* calibrated confidence estimates
* a useful accuracy-versus-coverage tradeoff
* a reproducible preprocessing contract
* an exportable checkpoint for later inference work

## Initial Models

Two pretrained video architectures will be compared.

### VideoMAE-Base

VideoMAE is the primary candidate.

It provides a pretrained spatial-temporal video representation that can be adapted to ASL classification by replacing the original classification head and fine-tuning the full model.

### Video Swin-Tiny

Video Swin-Tiny is the comparison architecture.

It provides hierarchical spatial-temporal attention and a different accuracy, memory, and throughput profile from VideoMAE.

The project should initially compare these architectures under the same dataset and evaluation protocol.

## Primary Dataset

### ASL Citizen

ASL Citizen is the primary training and evaluation dataset.

Reasons for selecting it include:

* isolated sign videos
* large vocabulary
* webcam-like recordings
* multiple signers
* varied environments
* signer-independent evaluation
* close alignment with the intended future application

The official train, validation, and test structure should be preserved.

The full vocabulary should be used for real baseline experiments unless a run is explicitly labeled as a smoke test or subset experiment.

## Secondary Dataset

### WLASL

WLASL may be used later in two ways.

#### Independent benchmark

Train and evaluate each model using WLASL’s own splits and vocabulary.

This determines whether the architecture ranking observed on ASL Citizen also holds on a second isolated-sign dataset.

#### External generalization test

Train on ASL Citizen and evaluate on verified overlapping WLASL labels.

This requires an explicit label-harmonization process and should be reported separately from the primary ASL Citizen results.

WLASL must not be treated as a drop-in ASL Citizen test set without resolving:

* vocabulary overlap
* gloss naming differences
* regional or lexical variants
* ambiguous matches
* unavailable source videos
* differing collection conditions

## Training Method

The project does not train the video architectures from scratch.

The main training process is supervised fine-tuning:

1. Load a pretrained video model.
2. Replace its original classification head.
3. Set the output dimension to the ASL vocabulary size.
4. Process ASL Citizen video clips into fixed-size tensors.
5. Train using multiclass cross-entropy.
6. Update the full model unless an experiment explicitly tests another strategy.
7. validate during training.
8. Save the best checkpoint using validation metrics.
9. Evaluate once on the untouched test set.

The full ASL training run is itself the fine-tuning stage.

## Baseline Strategy

The first experiments should establish clean baselines for both models.

Baseline training should use:

* official signer-independent splits
* standardized frame sampling
* standard model normalization
* deterministic validation and test transforms
* only minimal training augmentation
* no aggressive robustness augmentation
* consistent evaluation metrics
* reproducible configurations

The baseline exists as a control. It should not be overwritten by later robustness-trained checkpoints.

## Robustness Strategy

Robustness should be studied after the clean baseline is established.

### Step 1: Stress-test the baseline

Evaluate the baseline under controlled perturbations such as:

* mirrored videos
* temporal shifts
* speed changes
* clipped beginnings or endings
* lower resolution
* blur
* compression
* lighting changes

This measures where the baseline is fragile without changing the trained model.

### Step 2: Targeted robustness training

Introduce only augmentations justified by measured weaknesses.

Examples may include:

* random temporal cropping
* mild speed jitter
* conservative frame dropping
* mild visual degradation
* controlled spatial translation
* class-aware horizontal flipping

Each robustness strategy should be trained as a separate experiment and compared against the clean baseline.

### Horizontal flipping

Horizontal flipping must not be globally assumed to preserve every ASL label.

The project should first determine:

* whether videos are already mirrored
* whether handedness metadata exists
* whether a sign is invariant under mirroring
* whether direction or body side changes meaning

A class-aware flip policy may be introduced later after review.

## Data Standardization

The original videos do not need to be permanently converted to identical files.

The data pipeline should dynamically standardize model input by:

* decoding the source video
* preserving chronological frame order
* sampling a fixed number of frames
* resizing and cropping consistently
* converting to RGB
* applying model-compatible normalization
* returning a fixed-size tensor
* mapping the gloss to a stable class ID

Training transformations may be randomized.

Validation and test transformations must be deterministic.

The exact preprocessing contract must be versioned and exported with the selected checkpoint so future inference reproduces the same behavior.

## Confidence

The model’s raw confidence begins as the maximum softmax score.

Raw softmax values should not automatically be treated as reliable probabilities.

After training:

1. Collect validation logits.
2. Fit temperature scaling using validation data.
3. Measure calibration.
4. Select confidence thresholds using validation data.
5. Report final threshold behavior on test data.

Confidence evaluation should include:

* expected calibration error
* negative log-likelihood
* reliability analysis
* selective accuracy
* prediction coverage

An eventual system should be able to reject uncertain predictions rather than always returning a class.

## Initial Success Targets

These are practical project targets rather than universal safety standards.

### Research checkpoint

* top-1 accuracy at or above approximately 60%
* top-5 accuracy at or above approximately 85%
* macro F1 at or above approximately 0.55
* calibrated ECE at or below approximately 0.08

### Initial serving candidate

* overall top-1 accuracy at or above approximately 70%
* macro F1 at or above approximately 0.65
* calibrated ECE at or below approximately 0.05
* at least 90% accuracy among accepted predictions
* at least 50% coverage at the selected threshold

Serving readiness should be determined by the accuracy-versus-coverage tradeoff, not by a single raw confidence number.

## Evaluation

Primary clean metrics:

* top-1 accuracy
* top-5 accuracy
* macro F1
* weighted F1
* mean per-class accuracy
* per-class precision and recall
* per-signer accuracy
* worst-signer accuracy
* confusion matrix
* negative log-likelihood

Calibration metrics:

* expected calibration error
* reliability curves
* selective accuracy
* coverage
* accuracy at confidence thresholds

Robustness metrics:

* clean-to-perturbed accuracy drop
* per-condition macro F1
* calibration under perturbation
* false acceptance of invalid inputs
* external WLASL overlap performance

Primary ASL Citizen results, WLASL results, and robustness results must remain separately reported.

## Development Order

The authoritative, numbered implementation order is defined in:

```text
docs/ROADMAP.md
```

This document defines project scope. It deliberately does not restate phase numbers, so that only one numbering exists.

The broad sequence is:

```text
repository foundation
        ↓
basic model layer
        ↓
data audit and data layer
        ↓
training layer
        ↓
evaluation layer
        ↓
baseline experiments
        ↓
robustness evaluation and targeted robustness training
        ↓
secondary and external benchmarking
        ↓
training handoff package
```

The granular plan for the active phase lives in `docs/CURRENT_PHASE.md`.

## Repository Principles

The repository contains code and metadata, not datasets.

Expected repository content:

* source code
* configurations
* tests
* documentation
* label maps where licensing permits
* manifest-building scripts
* small synthetic fixtures
* experiment records
* lightweight metric summaries

Excluded from Git:

* raw ASL videos
* dataset archives
* extracted datasets
* large caches
* checkpoints
* optimizer states
* private credentials

The project should run in multiple environments by changing configuration rather than source code.

Expected execution environments include:

* local development
* Kaggle notebooks
* Google Colab

The dataset may be downloaded from Kaggle into a temporary Colab runtime. Checkpoints and logs should be saved to persistent storage.

## Project Workflow

The project uses a planner-and-executor workflow.

### Planner responsibilities

* maintain project scope
* define phases
* establish contracts
* choose experimental questions
* set acceptance criteria
* interpret model results
* decide follow-up experiments

### Claude responsibilities

* implement bounded roadmap tasks
* preserve architectural contracts
* add focused tests
* run validation
* report assumptions and deviations
* avoid unapproved experimental changes
* keep documentation synchronized when explicitly requested

### Human responsibilities

* review dataset and label assumptions
* approve experimental methodology
* interpret accuracy and robustness tradeoffs
* decide whether augmentations preserve ASL meaning
* approve transitions between project phases

Passing software tests establishes pipeline correctness. It does not establish that an experiment was successful.

## Current Starting Point

The project should begin with the basic model layer.

The immediate objective is to establish a shared classification contract and verify that both VideoMAE and Video Swin:

* load pretrained weights
* accept the intended logical video input
* replace their classification heads
* return logits for the configured number of ASL classes
* pass model-level smoke tests

The complete data, training, evaluation, and experiment systems should follow in later phases.
