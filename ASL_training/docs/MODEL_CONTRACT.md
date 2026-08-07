# Model Contract

## Purpose

This document defines the shared model interface for:

```text
ASL PROJECT/ASL_training/
```

The model layer supports isolated ASL video classification.

Initial architectures:

* VideoMAE-Base
* Video Swin-Tiny

Both models must expose the same logical behavior to the data, training, and evaluation layers.

## Scope

The model layer owns:

* pretrained checkpoint loading
* classification-head replacement
* model-specific input adaptation
* forward-pass behavior
* trainable-layer configuration
* model state loading
* parameter reporting
* model construction through configuration

The model layer does not own:

* dataset parsing
* video decoding
* frame sampling
* training loops
* optimizer construction
* metrics
* confidence calibration
* experiment interpretation
* serving or deployment

## Canonical Input

The project-level video batch contract is:

```text
[batch, frames, channels, height, width]
```

Expected properties:

* RGB input
* three channels
* fixed frame count
* fixed spatial resolution
* floating-point values
* preprocessing consistent with the selected pretrained model

Initial experiments are expected to use:

```text
frames: 16
resolution: 224 × 224
```

These values must remain configurable.

The model adapter may rearrange tensor dimensions internally when required by a specific architecture.

Architecture-specific tensor layouts must not leak into the shared training or evaluation layers.

## Canonical Output

Every model must return classification logits with shape:

```text
[batch, number_of_classes]
```

The shared output should contain:

* `logits`
* optional `loss`, when labels are supplied

Architecture-specific third-party output objects must be converted into this shared result before leaving the model layer.

Softmax probabilities and confidence calibration do not belong in the core model forward pass.

## Labels

When labels are provided, they must be integer class IDs with shape:

```text
[batch]
```

Valid labels are:

```text
0 through number_of_classes - 1
```

The model output dimension must match:

* the configured class count
* the label-map size
* the checkpoint vocabulary

A mismatch must fail clearly.

## Initial Architectures

### VideoMAE-Base

The VideoMAE adapter must:

* load the configured pretrained checkpoint
* replace the original classification head
* set the output size to the configured ASL class count
* accept the canonical video batch
* return shared classification output
* support full-model fine-tuning
* support checkpoint loading

### Video Swin-Tiny

The Video Swin adapter must:

* load the configured pretrained weights
* replace the original classification head
* set the output size to the configured ASL class count
* adapt the canonical input to the architecture’s expected tensor layout
* return shared classification output
* support full-model fine-tuning
* support checkpoint loading

## Model Factory

Models should be created through one shared construction interface.

The factory receives a resolved configuration containing at least:

* architecture name
* pretrained checkpoint or weights
* number of classes
* fine-tuning strategy
* dropout or head configuration
* model-specific options

Supported initial architecture names should be explicit and stable.

Unknown architectures must raise a clear error.

The training layer should not contain separate model-construction logic for VideoMAE and Video Swin.

## Classification Head

The original pretrained classification head must be replaced with a new head matching the configured ASL vocabulary.

The replacement head must be trainable.

The project should record:

* input feature dimension
* output class count
* dropout configuration
* head parameter count

The classification head must not infer its output size from a training batch.

Its size must come from validated configuration and the label map.

## Fine-Tuning Strategies

The default strategy is full-model fine-tuning.

Under full fine-tuning:

* the pretrained backbone is trainable
* the classification head is trainable

The model layer may later support:

* head-only training
* partial backbone freezing
* gradual unfreezing
* continued fine-tuning from an ASL checkpoint

Non-default strategies must be explicitly configured.

The model layer must provide a way to inspect which parameters are trainable.

It must never silently freeze layers.

## Pretrained Weight Loading

Pretrained weights must be loaded from a configured source.

The model construction report should identify:

* architecture
* checkpoint source
* whether pretrained loading succeeded
* missing keys
* unexpected keys
* intentionally replaced classification-head parameters

Classification-head mismatches caused by replacing the original task vocabulary are expected.

Unexpected backbone mismatches must not be silently ignored.

## Model-Specific Preprocessing

The model layer may expose metadata required by the data layer, including:

* expected normalization values
* expected frame count
* expected spatial resolution
* expected tensor layout
* patch or tubelet constraints

The model layer should not decode videos or apply random training augmentation.

The data layer remains responsible for producing the canonical processed video tensor.

Any difference in normalization between VideoMAE and Video Swin must be explicit and recorded in experiment configuration.

## Forward Behavior

The forward pass must:

1. Validate the logical input dimensions.
2. Adapt the tensor layout when required.
3. Run the selected architecture.
4. Return logits through the shared output contract.
5. Return loss when labels are supplied and supported.

The forward pass must not:

* apply softmax automatically
* select a predicted class
* apply confidence thresholds
* perform temperature scaling
* mutate labels
* perform dataset-specific preprocessing
* write checkpoints
* log experiment conclusions

## Training and Evaluation Modes

The model must respect standard training and evaluation modes.

In training mode:

* trainable layers receive gradients
* configured dropout and stochastic model behavior may be active

In evaluation mode:

* dropout and equivalent stochastic training behavior must be disabled
* no model weights may be modified

The evaluation layer is responsible for disabling gradient computation.

## Checkpoint Compatibility

Before loading a model checkpoint, the system should validate:

* architecture
* class count
* label-map identity where available
* classification-head shape
* model configuration
* fine-tuning strategy where relevant

The system must distinguish between:

### Exact Resume

Load model state as part of resuming the same training run.

### Model Evaluation

Load a completed ASL checkpoint for validation or testing.

### Continued Fine-Tuning

Load an ASL checkpoint as the starting point for a new experiment.

### Pretrained Transfer

Load a generic pretrained video checkpoint while replacing its original classification head.

These operations must not share ambiguous behavior.

## Parameter Reporting

Model construction should report:

* total parameter count
* trainable parameter count
* frozen parameter count
* classification-head parameter count
* approximate model size where practical

This information should be stored with real experiment metadata.

## Device and Precision

The model layer must support movement to the configured device.

Initial expected devices:

* CPU for smoke tests
* CUDA GPU for training

The model must support, where compatible:

* FP32
* FP16
* BF16

Precision orchestration belongs to the training layer.

The model layer should not independently enable autocasting.

## Memory Features

The model layer may expose optional support for:

* gradient checkpointing
* attention or memory-efficient implementation choices
* disabling unused outputs

These features must be configuration-driven.

They must not silently change the model architecture or experimental meaning.

## Common Model Interface

The shared model interface should support the equivalent of:

```text
build model from config
forward video batch
return logits and optional loss
report parameters
load checkpoint state
configure trainable layers
```

The interface should remain small.

Avoid deep inheritance or unnecessary framework abstractions.

Only add shared behavior that is genuinely required by both initial architectures.

## Validation Before Training

Before a real training run, the model layer must pass a preflight check.

The preflight must verify:

* architecture construction
* pretrained weight loading
* configured class count
* classification-head replacement
* canonical input acceptance
* output-logit dimensions
* optional loss computation
* trainable parameter presence
* device movement
* configured precision compatibility where practical

A failure must block the full run.

## Tests

The model layer should include focused tests for:

* supported architecture construction
* unknown architecture rejection
* classifier-head output size
* dummy forward pass
* canonical input adaptation
* logits shape
* label and class-count mismatch
* trainable parameter configuration
* full fine-tuning behavior
* checkpoint state loading
* evaluation mode behavior
* parameter reporting

Tests may use synthetic video tensors.

Real dataset videos are not required for model-layer unit tests.

## Failure Conditions

The model layer must fail clearly when:

* an unsupported architecture is requested
* pretrained weights cannot be loaded
* backbone weights have unexplained mismatches
* class count is invalid
* input tensor rank is incorrect
* channel count is not supported
* frame count violates model constraints
* output dimensions do not match the label map
* labels fall outside the valid class range
* no parameters are trainable
* checkpoint architecture is incompatible
* checkpoint classification-head shape is incompatible

## Initial Implementation Boundary

Phase 1 should implement only:

* the shared model interface
* VideoMAE-Base adapter
* Video Swin-Tiny adapter
* model factory
* classification-head replacement
* full fine-tuning configuration
* parameter reporting
* dummy-forward tests

Phase 1 should not implement:

* ASL Citizen parsing
* video decoding
* data augmentation
* training orchestration
* evaluation metrics
* confidence calibration
* serving

## Completion Criteria

The model layer is complete for the initial phase when both VideoMAE-Base and Video Swin-Tiny:

* load pretrained weights
* accept the canonical dummy video batch
* expose the same logical interface
* produce logits with the configured ASL class count
* support full-model fine-tuning
* report trainable parameters
* load compatible state dictionaries
* pass focused model tests

## Related Documents

Project scope:

```text
PROJECT.md
```

Architecture and implementation order:

```text
docs/ARCHITECTURE.md
docs/ROADMAP.md
```

Data behavior:

```text
docs/DATA_CONTRACT.md
```

Training behavior:

```text
docs/TRAINING_CONTRACT.md
```

Evaluation behavior will be defined in:

```text
docs/EVALUATION_CONTRACT.md
```

Serving remains outside this project:

```text
ASL PROJECT/ASL_serving/
```
