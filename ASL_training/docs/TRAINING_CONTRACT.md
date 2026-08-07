# Training Contract

## Purpose

This document defines the required behavior of the supervised training system in:

```text
ASL PROJECT/ASL_training/
```

The training system fine-tunes pretrained video-classification models for isolated American Sign Language recognition.

The initial supported architectures are:

* VideoMAE-Base
* Video Swin-Tiny

The training layer must remain independent from future serving concerns. It should produce reproducible checkpoints and metadata that can later be handed to `ASL_serving`.

## Training Objective

The model receives one short video containing one isolated ASL sign and predicts one class from the configured vocabulary.

For a batch of video clips, the model produces:

```text
logits: [batch_size, number_of_classes]
```

The default optimization objective is multiclass cross-entropy.

The training system is responsible for adapting pretrained video models to the ASL Citizen classification task. The full ASL Citizen training run is the supervised fine-tuning stage.

## Scope

The training layer owns:

* training orchestration
* loss computation
* gradient updates
* optimizer construction
* learning-rate scheduling
* mixed-precision execution
* gradient accumulation
* gradient clipping
* checkpoint creation
* checkpoint resume
* validation scheduling
* best-checkpoint selection
* training logs
* run metadata capture
* reproducibility controls
* training failure reporting

The training layer does not own:

* raw dataset download
* annotation interpretation
* manifest generation
* label-map creation
* video decoding semantics
* model architecture definitions
* evaluation metric definitions
* confidence calibration
* robustness-test construction
* experiment interpretation
* model serving
* frontend behavior
* deployment

## Layer Dependencies

The training layer may depend on:

* model interfaces
* dataset and data-loader interfaces
* shared configuration utilities
* checkpoint utilities
* evaluation interfaces used during validation
* logging and reproducibility utilities

The training layer must not require knowledge of:

* Kaggle dataset identifiers
* Google Drive paths
* browser video formats
* serving API contracts
* frontend state
* production monitoring

Environment-specific paths must be resolved through configuration before entering the core training logic.

## Required Inputs

Every real training run must receive a resolved configuration containing at least:

### Experiment Identity

* experiment name
* run name or run ID
* random seed
* output directory
* Git commit, when available

### Dataset Identity

* training manifest
* validation manifest
* label map
* dataset name
* dataset version or source identity
* manifest identity or checksum
* expected number of classes
* expected training sample count
* expected validation sample count

### Model Configuration

* architecture
* pretrained checkpoint source
* number of output classes
* fine-tuning strategy
* model-specific options
* expected input frame count
* expected spatial resolution

### Optimization Configuration

* loss function
* optimizer
* learning rate
* optional classifier-head learning rate
* weight decay
* scheduler
* warmup policy
* epoch or step limit
* batch size
* gradient accumulation
* gradient clipping
* precision mode

### Runtime Configuration

* device
* number of data-loader workers
* checkpoint frequency
* validation frequency
* logging frequency
* resume checkpoint, when applicable
* deterministic or performance-oriented runtime settings

A real run must fail before training if required configuration is missing or inconsistent.

## Model Contract

The training layer must interact with models through a shared logical interface.

The model must accept:

```text
video batch
optional class labels
```

The model must return:

```text
logits
optional loss
```

The training layer must not depend directly on architecture-specific output objects.

If a third-party model returns architecture-specific outputs, the model adapter must convert them into the shared training contract.

The training layer must verify before optimization that:

* model output size matches the label-map size
* model output size matches the configured number of classes
* the configured frame count is supported
* the batch tensor can complete a forward pass
* pretrained weights loaded as expected
* the new classification head is trainable

## Fine-Tuning Strategy

The default strategy is full-model supervised fine-tuning.

Under the default strategy:

* the pretrained backbone is trainable
* the replacement classification head is trainable
* gradients propagate through the complete network

Alternative strategies may be supported through explicit configuration:

* head-only training
* partial layer freezing
* gradual unfreezing
* continued fine-tuning from an ASL checkpoint

Alternative strategies must never activate silently.

Every checkpoint and experiment record must state the fine-tuning strategy used.

## Baseline Training Policy

The initial baseline experiments should use restrained training augmentation.

The baseline exists to measure the clean performance of each pretrained architecture before targeted robustness intervention.

Baseline training may include:

* random temporal selection required for ordinary training
* mild random resized crop
* standard spatial resizing
* model-compatible normalization
* mild appearance variation when explicitly configured

The baseline should not include by default:

* global horizontal flipping
* class-aware horizontal flipping
* speed jitter
* frame dropping
* synthetic compression
* artificial blur
* aggressive lighting distortion
* strong rotation
* large spatial translations
* heavy random erasing
* robustness-specific perturbations

Any augmentation beyond the baseline policy must be explicit in the run configuration.

## Loss Contract

The default loss is multiclass cross-entropy.

The loss must compare:

```text
model logits
against
integer class labels
```

Labels must be valid integers in:

```text
0 through number_of_classes - 1
```

The training system must fail clearly if:

* labels fall outside the configured class range
* labels use an incompatible data type
* model and label-map dimensions differ
* a batch contains no valid samples
* loss becomes non-finite

Optional loss variants may include:

* label-smoothed cross-entropy
* class-weighted cross-entropy
* focal loss

Non-default losses require explicit experiment configuration and documentation.

The training layer must not silently infer class weights or imbalance policies.

## Optimizer Contract

The default optimizer should be AdamW unless an experiment specifies otherwise.

Optimizer construction must be configuration-driven.

The system should support, where needed:

* one learning rate for the full model
* a separate learning rate for the classification head
* parameter groups
* layer-wise learning-rate decay
* exclusion of selected parameters from weight decay

Model-specific parameter grouping may be implemented through model adapters or dedicated optimizer helpers.

Every trainable parameter must either:

* belong to an optimizer parameter group, or
* be explicitly frozen

The preflight process should verify this.

## Scheduler Contract

The scheduler must be selected explicitly through configuration.

The initial expected scheduler is cosine decay with optional warmup.

The scheduler configuration must define whether it advances:

* per optimizer step
* per epoch

This behavior must not be ambiguous.

When gradient accumulation is enabled, step-based scheduling must advance only when the optimizer updates, not after every micro-batch.

Checkpoint resume must restore scheduler state.

## Batch and Gradient Accumulation

The physical batch size is the number of samples processed in one forward and backward pass.

The effective batch size is:

```text
physical batch size
× gradient accumulation steps
× distributed worker count
```

The training system must report both physical and effective batch size.

When accumulation is enabled:

* gradients accumulate across the configured number of micro-batches
* optimizer updates occur only at accumulation boundaries
* scheduler updates follow optimizer updates
* gradient clipping occurs immediately before the optimizer update
* logging distinguishes micro-steps from optimizer steps
* partial accumulation at the end of an epoch is handled explicitly

The implementation must not accidentally divide or scale the loss twice.

## Mixed Precision

The training system should support:

* full precision
* FP16 mixed precision
* BF16 mixed precision

The selected precision must be explicit in configuration and captured in run metadata.

The implementation should prefer BF16 when the hardware supports it reliably.

For FP16:

* gradient scaling should be used
* overflow or skipped-step behavior should be logged
* non-finite loss must trigger clear handling

Mixed precision must not change label tensors or integer metadata types.

Validation should use the configured inference precision where appropriate but must not update weights.

## Gradient Clipping

Gradient clipping should be configurable.

When enabled, clipping must occur:

1. after backward propagation
2. after gradient unscaling, when using FP16
3. after the final accumulation micro-batch
4. before the optimizer step

The configured clipping norm must be recorded in the run metadata.

## Epoch and Step Semantics

The training system must use explicit counters for:

* epoch
* batch or micro-step
* optimizer step
* validation event
* checkpoint event

Logs and checkpoints must not use an ambiguous generic `step` without documenting which counter it represents.

A run may be configured by:

* maximum epochs
* maximum optimizer steps

If both are supplied, the stopping behavior must be explicit.

## Validation During Training

Validation must:

* use the configured validation manifest
* use deterministic evaluation preprocessing
* disable gradient computation
* place the model in evaluation mode
* restore training mode afterward
* avoid updating optimizer or scheduler state
* preserve sample identifiers where required
* compute only approved validation metrics

Validation must never use the test split.

Validation frequency may be:

* once per epoch
* every configured number of optimizer steps

The selected frequency must be recorded.

## Checkpoint Selection

The best checkpoint must be selected using a configured validation metric.

Examples include:

* macro F1
* top-1 accuracy
* validation loss

The primary selection metric and optimization direction must be explicit:

```text
maximize macro F1
```

or:

```text
minimize validation loss
```

Test metrics must never influence checkpoint selection.

When multiple checkpoints tie, the tie-breaking rule should be deterministic, such as:

* earliest checkpoint
* lower validation loss
* higher secondary metric

## Checkpoint Contents

A resumable training checkpoint should include:

* model state
* optimizer state
* scheduler state
* precision scaler state, when applicable
* epoch
* micro-step
* optimizer step
* best validation metric
* best-checkpoint reference
* experiment configuration
* label-map identity
* manifest identities
* preprocessing identity
* random seed
* random-number-generator states where feasible
* Git commit
* dependency and hardware summary where feasible

A model-only export may be produced separately.

The model-only export does not replace the resumable training checkpoint.

## Checkpoint Types

The training system should distinguish:

### Latest Checkpoint

The most recent resumable state.

Used for recovery after interruption.

### Best Checkpoint

The checkpoint with the best configured validation metric.

Used for final evaluation and model comparison.

### Periodic Checkpoint

Optional historical snapshots retained at configured intervals.

### Model Export

A lightweight model artifact intended for evaluation or future serving handoff.

File names and metadata should clearly identify the checkpoint type.

## Resume Contract

Resume must restore:

* model weights
* optimizer state
* scheduler state
* precision scaler state
* epoch and optimizer-step counters
* best metric state
* configured random state where feasible

The training system must validate checkpoint compatibility before resuming.

Compatibility checks should include:

* architecture
* number of classes
* label-map identity
* preprocessing contract
* optimizer type where optimizer state is resumed
* scheduler type
* fine-tuning strategy

A mismatch must fail clearly unless an explicit override is supported.

The system must distinguish:

* exact training resume
* loading model weights for a new experiment
* continued fine-tuning
* transfer to a new vocabulary

These operations must not share one ambiguous `resume` behavior.

## Reproducibility Contract

Every real run must capture:

* complete resolved configuration
* Git commit
* random seed
* Python version
* PyTorch version
* Transformers or Torchvision version
* CUDA version
* GPU type
* precision mode
* dataset identity
* manifest identity
* label-map identity
* pretrained checkpoint source
* trainable parameter count
* total parameter count
* augmentation policy
* physical and effective batch size

The training system should seed:

* Python
* NumPy
* PyTorch CPU
* PyTorch CUDA

Perfect numerical determinism is not required unless explicitly configured, but the level of determinism must be documented.

A run without sufficient metadata must not be treated as fully reproducible.

## Logging Contract

Training logs should include:

* experiment and run identity
* epoch
* micro-step
* optimizer step
* learning rate
* training loss
* moving-average loss where useful
* gradient norm when available
* throughput
* GPU memory use when available
* validation metrics
* checkpoint events
* skipped batches
* corrupt-sample counts
* non-finite values
* runtime warnings

Logs should distinguish:

* smoke tests
* preflight benchmarks
* subset experiments
* full experiments

The system must not label a reduced-data run as a full baseline.

## Preflight Contract

Before a full training run, the project should support a preflight mode.

The preflight should verify:

* model construction
* data-loader construction
* one batch forward pass
* one backward pass
* optimizer update
* scheduler update
* mixed-precision behavior
* memory use
* checkpoint save
* checkpoint load
* short validation pass

The preflight should report:

* GPU type
* peak allocated memory
* physical batch size
* effective batch size
* videos processed per second
* estimated epoch duration
* checkpoint size
* data-loading bottlenecks
* whether gradient accumulation is active

A preflight must use a clearly limited sample or step count.

It must not be interpreted as a real experiment result.

## Smoke-Test Contract

A smoke run exists to validate the software path.

A smoke run may use:

* synthetic videos
* a small controlled subset
* very few optimizer steps
* one short validation pass

A smoke run must not be used to:

* compare architectures
* report model accuracy
* select hyperparameters
* claim convergence
* choose a final checkpoint

Smoke-run outputs should be stored separately from real experiment outputs.

## Real Experiment Contract

A run qualifies as a real experiment only when:

* the intended dataset scope is used
* the run configuration is versioned
* the run identity is unique
* the dataset and label map are validated
* the complete training policy is recorded
* checkpointing and resume are enabled
* validation metrics are captured
* hardware and dependency metadata are saved
* the experiment record distinguishes it from smoke and preflight runs

The experiment record should identify any interruption, failed epoch, missing sample, or configuration change.

## Failure Handling

The training system must fail clearly for conditions that threaten validity.

Examples include:

* signer leakage detected before training
* model output dimension mismatch
* label-map mismatch
* missing manifest
* empty dataset
* invalid class labels
* incompatible checkpoint
* non-finite loss
* no trainable parameters
* missing pretrained weights
* unsupported tensor dimensions
* deterministic validation failure
* accidental test-manifest use

Recoverable runtime issues may include:

* individual corrupted videos
* temporary data-loader worker failures
* unavailable optional logging services

Recoverable issues must follow an explicit configured policy.

The system must report:

* number of skipped samples
* sample identifiers
* reason for skipping
* whether the run remains valid

Silent skipping is prohibited.

## Data Reduction

The training system must never silently:

* reduce the vocabulary
* exclude rare classes
* shorten the dataset
* cap samples per class
* change the split
* drop signers
* use only a subset of videos

Any reduction must be explicit in the configuration and run name.

Examples:

```text
smoke-500-samples
subset-100-classes
preflight-50-steps
```

A reduced run must not be compared directly against a full-dataset baseline without clear qualification.

## Test-Set Isolation

The test split must not be used for:

* optimizer decisions
* early stopping
* learning-rate choices
* augmentation choices
* architecture changes
* temperature fitting
* confidence-threshold selection
* checkpoint selection

During development, the default training command should not require or load the test manifest.

Final test evaluation belongs to the evaluation workflow after model and threshold choices are fixed.

## Augmentation Experiment Contract

Robustness augmentations belong to explicit experiment configurations.

A targeted augmentation run must state:

* measured weakness motivating the augmentation
* transformation type
* probability
* parameter range
* whether the transformation is believed label-preserving
* baseline experiment used for comparison

Robustness experiments must start from:

* the same original pretrained checkpoint as the baseline, by default

They should not continue from the baseline ASL checkpoint unless the experiment explicitly studies continued fine-tuning.

Each augmentation run must preserve the clean baseline checkpoint.

## Model Comparison Contract

VideoMAE and Video Swin experiments must share, where practical:

* dataset manifests
* label map
* split definitions
* input frame count
* spatial resolution
* evaluation transforms
* primary metrics
* seed policy
* experiment reporting structure

Model-specific differences are allowed for:

* tensor layout
* supported pretrained normalization
* optimizer parameter grouping
* learning-rate scale
* memory-driven physical batch size

When physical batch sizes differ, effective batch size and optimization differences must be reported.

A model comparison must not conceal architecture-specific deviations.

## Output Structure

A real training run should write to a unique run directory.

The logical structure should contain:

```text
run/
├── resolved_config
├── run_metadata
├── logs
├── checkpoints
│   ├── latest
│   ├── best
│   └── periodic
├── validation
├── environment
└── status
```

The exact file formats may evolve, but the content categories must remain identifiable.

Run directories must not overwrite prior experiments unless explicitly configured.

## Completion Criteria

The training layer is considered complete when it can:

* train both supported model architectures
* consume the shared ASL Citizen data contract
* perform full-model fine-tuning
* use cross-entropy correctly
* run mixed precision
* support gradient accumulation
* validate deterministically
* select a best checkpoint from validation metrics
* save and resume full training state
* record reproducibility metadata
* distinguish smoke, preflight, subset, and full runs
* fail clearly on invalid experimental conditions
* execute from a normal command-line entry point
* run in local, Kaggle, and Colab environments through configuration

## Initial Implementation Priority

The initial training implementation should prioritize:

1. Correct end-to-end behavior.
2. Clear model and data contracts.
3. Reliable checkpoint resume.
4. Reproducible run metadata.
5. One shared path for both architectures.
6. Simple configuration.
7. Clear failure reporting.

Do not prioritize initially:

* distributed multi-node training
* advanced hyperparameter search
* automated experiment scheduling
* model registries
* production orchestration
* custom CUDA kernels
* serving optimization

## Relationship to Other Contracts

This document depends on:

* `PROJECT.md`
* `docs/ARCHITECTURE.md`
* `docs/MODEL_CONTRACT.md`
* `docs/DATA_CONTRACT.md`

Evaluation behavior is defined separately in:

```text
docs/EVALUATION_CONTRACT.md
```

Experiment naming, interpretation, and recordkeeping are defined separately in:

```text
docs/experiments/
```

Serving behavior belongs to the future sibling project:

```text
ASL PROJECT/ASL_serving/
```
