# Data Contract

## Purpose

This document defines the required behavior of the data system in:

```text
ASL PROJECT/ASL_training/
```

The data layer converts raw isolated-sign video datasets into standardized, reproducible inputs for model training and evaluation.

The primary dataset is ASL Citizen.

WLASL may be integrated later as a separate dataset source and external benchmark.

The data layer must preserve dataset integrity, signer-independent splits, stable class mappings, and reproducible preprocessing.

## Scope

The data layer owns:

* dataset metadata inspection
* annotation parsing
* manifest generation
* label-map generation
* dataset split preservation
* signer metadata
* duplicate detection
* signer-leakage validation
* video-path resolution
* video decoding
* frame-order preservation
* temporal sampling
* spatial preprocessing
* training augmentation
* deterministic evaluation transforms
* batch collation
* corrupted-sample reporting
* dataset audit reports
* dataset identity and manifest identity

The data layer does not own:

* model architecture definitions
* classifier-head construction
* loss computation
* optimizer behavior
* checkpoint selection
* confidence calibration
* experiment interpretation
* model serving
* webcam recording
* browser video handling
* deployment-time preprocessing

## Primary Dataset

ASL Citizen is the authoritative primary dataset for initial training and evaluation.

The data layer must use the official dataset annotations and signer-independent split definitions unless an explicit experiment defines another protocol.

The initial full-dataset baseline should use all valid samples from the configured official vocabulary.

The data layer must not silently:

* remove classes
* merge classes
* rename glosses
* cap samples per class
* rebalance the dataset
* move signers between splits
* regenerate random splits
* discard difficult videos
* substitute WLASL examples

Any dataset reduction or alternate split must be explicit in configuration and experiment metadata.

## Secondary Dataset

WLASL may be added later through its own dataset adapter.

WLASL must retain:

* its own source identity
* its own manifests
* its own label map
* its own split definitions
* its own corruption and availability audit
* its own preprocessing audit

ASL Citizen and WLASL must not share one implicit label map.

Cross-dataset evaluation requires a separate reviewed label-harmonization artifact.

## External Dataset Location

Raw datasets must remain outside version control.

Expected runtime environments may include:

```text
Local:
<external-root>/asl_citizen

Kaggle:
/kaggle/input/<dataset-name>

Google Colab:
/content/<dataset-name>
```

Dataset roots must be supplied through:

* configuration
* environment variables
* notebook setup
* command-line overrides

Reusable source modules must not hardcode environment-specific absolute paths.

The core data loader should receive resolved paths rather than contain Kaggle- or Colab-specific logic.

## Dataset Substrate

A run may read either the original dataset or a re-encoded copy of it. Which one it read is part of the experiment.

A substrate is a substitute for the source only when it preserves:

* every relative video path and file name
* every clip's frame count
* the split files, copied rather than regenerated
* the set of samples, exactly

Path and record preservation keeps the manifest identity unchanged, which is what proves the experiment's structure was not altered. Frame-count preservation matters because manifests record a count per clip and the temporal sampler indexes against it; drift there corrupts sampling silently rather than failing.

Geometry is not preserved and is not required to be. A substrate may re-encode at a lower resolution, provided the short side is at least the spatial preprocessing's resize target, so that the random crop retains its freedom to move.

Re-encoding is lossy. Therefore:

* every split must use the same substrate, without exception
* a baseline and the robustness or cross-dataset work compared against it must use the same substrate
* run metadata must record which substrate was used
* results from different substrates must not be compared

The current substrate is recorded in `docs/DECISIONS.md`. `scripts/build_video_mirror.py` produces one and verifies each clip; `--verify-only` re-checks an existing one without modifying it.

## Repository Data Boundary

The repository may contain:

* manifest schemas
* manifest-building code
* label-map-building code
* dataset audit summaries
* small synthetic fixtures
* small metadata fixtures
* configuration files
* reviewed label-overlap tables
* checksums or dataset identities

The repository must not contain:

* raw dataset videos
* downloaded archives
* extracted video collections
* large frame caches
* copied Kaggle datasets
* private dataset credentials
* generated tensors from the full dataset
* unrestricted samples that violate dataset licensing

Whether generated manifests and label maps may be committed depends on the source dataset license and the contents of those artifacts.

## Dataset Identity

Every real experiment must record the dataset identity.

Dataset identity should include, where available:

* dataset name
* dataset version
* official source
* download date
* archive checksum
* annotation-file checksum
* split-file checksum
* hosted-copy identity
* dataset root used at runtime
* total discovered videos
* total usable videos
* total excluded videos
* exclusion reasons

When using a third-party Kaggle copy, the data audit must compare its metadata and structure against the official dataset source.

A hosted copy must not be assumed identical solely because it has the same name.

## Data Audit Contract

Before full training, the project must produce a dataset audit.

The audit should report:

* discovered annotation files
* discovered video directories
* expected sample count
* discovered video count
* usable sample count
* missing video count
* corrupted video count
* duplicate sample count
* class count
* signer count
* samples by split
* classes by split
* signers by split
* samples per class
* samples per signer
* class imbalance statistics
* video duration distribution
* frame-count distribution
* frame-rate distribution
* resolution distribution
* codec distribution where practical
* unknown or malformed metadata
* handedness metadata availability
* mirroring metadata availability
* split-integrity results

The audit should distinguish:

* annotation-level missing records
* missing files
* unreadable files
* partially decodable files
* zero-frame videos
* invalid labels
* duplicate videos
* duplicate annotation rows

The audit must not alter raw dataset files.

## Manifest Contract

The manifest is the source of truth consumed by the training and evaluation systems.

Training code must not repeatedly rediscover labels, signers, or splits from raw directory names.

Each manifest row represents one isolated-sign video sample.

## Required Manifest Fields

Every record must include:

* `sample_id`
* `video_path`
* `gloss`
* `class_id`
* `signer_id`
* `split`
* `dataset_name`

Recommended additional fields include:

* `source_annotation_id`
* `duration_seconds`
* `frame_count`
* `fps`
* `width`
* `height`
* `codec`
* `corruption_status`
* `handedness`
* `mirroring_status`
* `dataset_version`

Fields without reliable source information should remain absent or explicitly unknown.

Do not invent handedness or mirroring metadata.

## Sample Identity

`sample_id` must be stable and unique within the dataset version.

It should be derived from an authoritative source identifier where possible.

If no stable source identifier exists, the project may derive one from immutable metadata such as:

```text
dataset name
+ source-relative video path
+ annotation identifier
```

The generated identity method must be documented.

Changing runtime root paths must not change `sample_id`.

## Video Paths

Manifest video paths should be either:

* relative to the configured dataset root, or
* represented using a source-relative path field

Absolute environment-specific paths should not be committed into reusable manifests.

At runtime:

```text
configured dataset root
+ source-relative video path
→ resolved video path
```

The loader must verify that the resolved path remains inside the expected dataset root.

## Split Contract

Allowed initial split values are:

* `train`
* `validation`
* `test`

Aliases such as `val`, `dev`, or numeric split IDs must be normalized during manifest creation, not handled inconsistently throughout the project.

The original split value may be retained as separate metadata.

## Signer-Independent Split Integrity

The ASL Citizen official signer-independent split is authoritative.

The manifest system must verify:

* no sample appears in multiple splits
* no duplicate video appears in multiple splits
* no prohibited signer overlap exists
* every record has a valid split
* all configured splits use the same label map
* class IDs have the same meaning in every split

If the official protocol allows any particular overlap, that exception must be documented explicitly.

A signer-leakage failure must block full training.

## Label-Map Contract

The label map defines the classifier output vocabulary.

It must provide:

```text
gloss → class ID
class ID → gloss
```

Class IDs must be:

* unique
* integer-valued
* zero-indexed
* contiguous
* stable across training, validation, and test
* stable across both initial model architectures

For `N` classes, valid IDs are:

```text
0 through N - 1
```

The model output dimension must equal the label-map size.

## Label Stability

Once baseline experiments begin, the label map must not change without:

* a new label-map version
* new manifest identities
* new experiment configurations
* checkpoint compatibility invalidation
* a documented decision

Alphabetical order is acceptable if applied deterministically, but the chosen construction rule must be documented.

The map must not depend on directory iteration order, hash-map ordering, or which split is loaded first.

## Gloss Preservation

The original dataset gloss should be retained.

A normalized gloss may also be created for search, analysis, or cross-dataset comparison.

Normalization may include:

* whitespace normalization
* consistent case
* controlled punctuation handling

Normalization must not silently merge distinct source labels.

Examples of potentially unsafe automatic merging include:

* synonyms
* regional variants
* plural and singular concepts
* compound glosses
* labels differing by directional meaning
* labels with numbered variants

Any semantic merge requires an explicit reviewed mapping.

## Manifest Versioning

Every manifest set should have an identity.

This may be based on:

* version string
* generated timestamp
* source checksums
* configuration checksum
* manifest file checksum
* code Git commit

A manifest identity should change when:

* records are added or removed
* paths change semantically
* splits change
* labels change
* corruption policy changes
* metadata affecting preprocessing changes

Formatting-only changes should not necessarily create a new semantic version, but file checksums may still differ.

## Duplicate Detection

The data audit should detect duplicates using multiple signals where practical:

* repeated annotation IDs
* repeated source-relative paths
* repeated sample IDs
* repeated file checksums
* repeated video fingerprints
* identical signer, gloss, and source metadata

Duplicate handling must be explicit.

Possible statuses include:

* confirmed duplicate
* probable duplicate
* repeated annotation
* shared source video
* unresolved

Confirmed duplicates must not silently enter multiple splits.

## Corrupted and Missing Sample Policy

Missing or corrupted samples must be visible.

Possible statuses include:

* usable
* missing
* unreadable
* zero frames
* partial decode
* invalid duration
* malformed annotation
* invalid label
* unsupported codec

The default full-training policy should fail during manifest validation when the usable sample set differs unexpectedly from the audited set.

A configured skip policy may permit known exclusions when:

* excluded sample IDs are recorded
* reasons are recorded
* counts are reported
* manifests remain stable
* the run metadata references the exclusion policy

Training-time silent skipping is prohibited.

## Video Decoding Contract

The decoding layer converts a source video into an ordered RGB frame sequence.

The logical decoded representation should include:

* frames
* frame order
* frame count
* frame rate, when available
* duration, when available
* original resolution
* decoding status

The canonical decoded frame shape should be documented.

A recommended internal representation is:

```text
[total_frames, channels, height, width]
```

with:

```text
channels = 3
color space = RGB
```

The exact tensor data type may differ before and after normalization, but the stage must be explicit.

## Color Contract

Model inputs must use RGB.

Decoding libraries that return BGR, YUV, or another color representation must convert explicitly to RGB.

Color conversion must be tested.

The project must not assume a decoding backend already returns RGB without verification.

## Frame-Order Contract

Chronological order must be preserved.

The data layer must not:

* sort frames lexicographically by generated filename
* reverse clips
* shuffle frame order
* sample unordered indices
* apply independent temporal transforms per frame

Temporal sampling indices must remain ordered.

## Orientation and Rotation Metadata

Video orientation metadata must be handled consistently.

The decoder should account for stored rotation metadata where supported.

The audit should identify videos whose decoded orientation may differ across backends.

Evaluation should not depend on one backend accidentally applying rotation while another does not.

## Temporal Sampling Contract

Every model input must contain exactly the configured number of frames.

The initial expected frame counts may include:

* 16 frames
* 32 frames in later experiments

Frame count is an experiment-level configuration and part of the preprocessing identity.

## Baseline Temporal Sampling

The clean baseline should use one clearly defined sampling policy.

Candidate policies include:

* uniform sampling across the full clip
* random ordered sampling during training
* random contiguous temporal window
* deterministic uniform sampling during evaluation

The initial policy must be selected explicitly in configuration.

The implementation must not ambiguously switch policies based on clip length.

## Evaluation Temporal Sampling

Validation and test sampling must be deterministic.

Given:

* the same video
* the same preprocessing configuration
* the same software version

the selected frame indices should be reproducible.

Evaluation must not use unseeded random temporal sampling.

## Short-Video Policy

Videos shorter than the configured frame count require an explicit policy.

Supported policies may include:

* repeated frame indices through uniform sampling
* repeat-last-frame padding
* looped sampling
* temporal interpolation

The initial policy should favor preserving the original chronology without inventing reversed motion.

The chosen short-video policy must be:

* documented
* deterministic for evaluation
* tested
* included in preprocessing metadata

Short videos must not be silently dropped unless the manifest exclusion policy says so.

## Long-Video Policy

Long videos must be reduced to the configured frame count according to the selected temporal sampler.

The implementation should avoid decoding unnecessary frames where efficient random access is reliable, but optimization must not change selected frames.

Performance optimizations must preserve the sampling contract.

## Clip Boundary Assumption

The initial dataset contract assumes each source video contains one isolated sign.

The data layer does not initially perform:

* continuous-sign segmentation
* automatic sign-boundary detection
* sentence segmentation
* multi-sign extraction

If ASL Citizen includes substantial leading or trailing idle content, that should be documented in the audit and addressed through an explicit temporal-sampling experiment.

## Spatial Preprocessing Contract

Every sampled clip must be transformed to the configured model input resolution.

The spatial pipeline should define:

* resize policy
* crop policy
* final height
* final width
* interpolation method
* aspect-ratio behavior
* normalization
* tensor data type

The initial expected spatial resolution is commonly:

```text
224 × 224
```

but the actual value belongs to configuration.

## Temporal Consistency of Spatial Transforms

Spatial transforms applied to a video clip must use the same random parameters across all frames.

Examples include:

* crop coordinates
* resize scale
* rotation angle
* horizontal flip decision
* color transformation parameters where appropriate

A different random crop per frame would create artificial camera motion and violate the intended baseline.

## Training and Evaluation Transform Separation

Training and evaluation transforms must be constructed separately.

### Training transforms

May contain configured randomness.

### Validation and test transforms

Must be deterministic.

The loader must not use a generic transform object whose behavior depends only on global model mode unless that distinction is clear and tested.

## Baseline Training Transform Policy

The first baseline should use only restrained augmentation.

Potentially allowed:

* required resize
* mild random resized crop
* deterministic or lightly randomized temporal sampling
* model-compatible normalization
* mild explicitly configured color variation

Initially disabled:

* horizontal flipping
* speed jitter
* frame dropping
* blur
* compression simulation
* strong brightness changes
* strong contrast changes
* strong saturation changes
* heavy rotation
* aggressive random erasing
* background replacement

The exact enabled baseline transforms must appear in the resolved experiment configuration.

## Robustness Transform Boundary

Robustness perturbations have two separate uses.

### Evaluation perturbations

Applied to validation or test copies without changing model weights.

Purpose:

* measure sensitivity
* produce robustness profiles

### Training augmentations

Applied stochastically during separate retraining experiments.

Purpose:

* improve measured weaknesses

These two categories must not share ambiguous configuration names.

For example:

```text
training_augmentation.speed_jitter
```

and:

```text
robustness_evaluation.speed_jitter
```

should be distinct.

## Horizontal Flip Contract

Horizontal flipping is disabled for the initial baseline.

It must not be enabled globally without review.

Before any horizontal-flip training experiment, the project should determine:

* whether stored videos are mirrored
* whether the dataset interface showed mirrored previews
* whether handedness metadata exists
* whether each affected class is believed mirror-invariant
* whether direction or body side changes meaning

A future class-aware policy may define:

* safe to flip
* unsafe to flip
* unresolved

Unresolved classes should not be flipped by default.

## Speed and Temporal Perturbation Contract

Speed changes, frame drops, and temporal truncation are robustness-specific until evidence supports training use.

Any such transformation must define:

* probability
* parameter range
* frame-selection behavior
* interpolation behavior
* label-preservation assumption

Time reversal is prohibited for ordinary augmentation unless a specific reviewed experiment justifies it.

## Appearance Augmentation Contract

Appearance transforms should be realistic and conservative.

Possible later transforms include:

* mild brightness variation
* mild contrast variation
* mild saturation variation
* low-probability blur
* lower-resolution simulation
* compression simulation

Transforms must not routinely erase defining finger configurations or hand boundaries.

Parameter ranges must be explicit.

## Model-Specific Normalization

Each pretrained model may require particular normalization values or processor behavior.

The data layer should expose a canonical decoded and sampled clip.

Model-specific adaptation may then occur through:

* a configured transform profile
* a model processor
* a model adapter

The project must avoid accidentally giving the two models materially different input information during architecture comparison.

Any required model-specific normalization difference must be documented.

## Canonical Sample Contract

A dataset sample returned before batching should contain:

* processed video tensor
* integer class label
* sample ID
* signer ID
* gloss
* split
* dataset name

Optional metadata may include:

* source-relative path
* duration
* frame rate
* selected frame indices
* transform metadata for debugging

The logical sample should resemble:

```text
pixel_values
label
sample_id
signer_id
gloss
dataset_name
```

The exact programming structure may evolve.

## Canonical Video Tensor

The project-level canonical logical layout should be:

```text
[frames, channels, height, width]
```

A collated batch should be:

```text
[batch, frames, channels, height, width]
```

Architecture adapters may convert internally to another format.

The data layer should not expose a different canonical shape for each model.

## Label Tensor Contract

Labels must be integer class IDs.

Recommended type:

```text
signed 64-bit integer
```

The label tensor must not be normalized, one-hot encoded, or cast to a floating type for standard cross-entropy training.

## Batch Collation Contract

The batch collation layer must:

* stack fixed-size video tensors
* stack integer labels
* preserve ordered metadata lists
* surface failed samples according to policy
* produce a predictable batch structure
* avoid architecture-specific dimension permutations

The batch should retain enough metadata for:

* per-signer evaluation
* per-class evaluation
* per-example prediction export
* corruption tracing

## Batch Failure Policy

If one sample fails during batch construction, behavior must follow an explicit policy.

Possible policies:

* fail the batch
* skip known pre-audited exclusions
* return a structured failure for controlled handling

The default should not silently shrink batches.

Any skipped runtime sample must be logged with:

* sample ID
* video path
* failure reason
* split
* cumulative skipped count

A high runtime failure rate should invalidate the run.

## DataLoader Contract

The DataLoader or equivalent batching system should support:

* configurable batch size
* configurable worker count
* configurable shuffling
* pinned memory where useful
* reproducible worker seeding
* drop-last behavior
* persistent workers where supported

Expected split behavior:

### Train

* shuffle enabled
* stochastic training transforms
* optional drop-last configured explicitly

### Validation

* shuffle disabled
* deterministic transforms
* no drop-last by default

### Test

* shuffle disabled
* deterministic transforms
* no drop-last

## Worker Seeding

Randomized data transformations must be reproducible to the configured extent.

Worker initialization should derive seeds from:

* experiment seed
* worker identity
* epoch where needed

Multiple workers must not unintentionally produce identical random sequences.

The exact reproducibility guarantees should be documented.

## Distributed Sampling

If distributed training is later introduced:

* each training sample should be assigned appropriately across workers
* epoch-level sampler reseeding must be handled
* validation duplication must not corrupt metric aggregation
* effective dataset coverage must remain known

Distributed support is not required for the first implementation.

## Class Imbalance Data Support

The data layer should report class-frequency statistics.

It may expose information needed for:

* class-weight calculation
* weighted sampling
* class-aware batching

The data layer must not activate imbalance correction automatically.

Imbalance treatment belongs to explicit training experiments.

## Caching Contract

Caching is optional.

Possible caches include:

* audited metadata
* frame indices
* decoded clips
* resized clips
* extracted frames
* dataset fingerprints

A cache must be identified by:

* dataset identity
* preprocessing identity
* decoder version
* frame-sampling configuration
* spatial-transform configuration where applicable

Stale caches must not be reused silently.

Large caches must remain outside Git.

The initial implementation should prefer correctness and simplicity over aggressive caching.

## Kaggle and Colab Boundary

Kaggle and Colab notebooks may:

* resolve dataset roots
* download or attach datasets
* clone the repository
* install dependencies
* invoke audit, training, or evaluation commands

Notebook code must not become the authoritative implementation of:

* annotation parsing
* video decoding
* transforms
* label mapping
* split validation
* batch collation

Those belong in `ASL_training/src/`.

## Dataset Download Boundary

Dataset download logic should remain separate from dataset interpretation.

The training data modules should not call Kaggle APIs, mount Google Drive, or download archives automatically.

A separate setup script or notebook may prepare the external dataset location.

This separation prevents training jobs from unexpectedly downloading tens of gigabytes.

## Preprocessing Identity

Every preprocessing configuration used in a real experiment must have an identity.

It should include:

* frame count
* temporal sampling strategy
* short-video policy
* resize policy
* crop policy
* final resolution
* color space
* normalization values
* horizontal-flip policy
* enabled training augmentations
* evaluation transform policy
* decoder backend
* relevant library versions

The selected model checkpoint must reference this identity.

## Train-to-Serving Preprocessing Handoff

The current project does not implement serving.

However, the data layer must eventually export a deterministic evaluation preprocessing specification that future `ASL_serving` can reproduce.

The handoff should state:

* expected clip semantics
* supported input color space
* orientation handling
* frame count
* temporal sampling method
* short-video policy
* resize behavior
* crop behavior
* normalization
* tensor layout
* mirroring policy

Random training augmentations are not part of the serving contract.

The serving project should reproduce the deterministic validation and test pipeline.

## Data Validation Before Training

A full training run must not start until the following validations pass:

* manifests exist
* required fields exist
* all class IDs are valid
* class IDs are contiguous
* label-map size matches configuration
* every record has a valid split
* every runtime path resolves
* no prohibited signer leakage exists
* no sample appears in multiple splits
* sample IDs are unique
* known exclusions match the audited exclusion list
* the expected sample counts match
* at least one batch can be decoded and transformed
* processed tensor dimensions match the model contract

A configurable override may exist for development, but a run using it must not be labeled a valid full baseline.

## Test Requirements

The data layer should include focused tests for:

* annotation parsing
* manifest schema validation
* label-map determinism
* contiguous class IDs
* duplicate sample IDs
* duplicate video paths
* signer leakage
* split normalization
* dataset-root path resolution
* RGB conversion
* frame-order preservation
* fixed-frame output
* short-video handling
* deterministic evaluation sampling
* temporally consistent crop parameters
* correct tensor shape
* correct label type
* batch collation
* corrupted-video reporting
* worker seed behavior where practical
* preprocessing identity generation

Tests should use small synthetic fixtures or legally redistributable sample media.

The full dataset is not required for the ordinary unit-test suite.

## Failure Conditions

The data layer must fail clearly for conditions that threaten validity.

Examples include:

* missing required annotations
* missing manifest columns
* unknown split value
* duplicate class IDs
* non-contiguous label IDs
* label map inconsistent with manifests
* signer leakage
* unresolved runtime paths
* zero usable samples
* unsupported video decoding
* incorrect output frame count
* non-RGB output
* inconsistent tensor shape
* evaluation randomness
* test transforms using training augmentation
* dataset identity mismatch
* unexpected exclusion-count change

Warnings may be appropriate for:

* uncommon codecs
* unusually long videos
* unusually short videos
* uncertain orientation metadata
* unknown handedness
* unknown mirroring status

Warnings must not conceal hard integrity failures.

## Completion Criteria

The ASL Citizen data layer is complete when it can:

* audit the configured dataset source
* verify the official splits
* detect signer leakage
* generate stable manifests
* generate a stable label map
* identify missing and corrupted videos
* resolve runtime paths across local, Kaggle, and Colab environments
* decode videos consistently
* preserve chronological frame order
* sample the configured number of frames
* handle short videos deterministically
* apply temporally consistent training transforms
* apply deterministic validation and test transforms
* produce canonical video tensors
* produce batches accepted by both initial model adapters
* report dataset and preprocessing identities
* pass focused data tests
* expose no dependency on future serving code

## Initial Implementation Priority

The first data-layer implementation should prioritize:

1. Accurate ASL Citizen audit.
2. Stable manifests and labels.
3. Official split preservation.
4. Signer-leakage detection.
5. Reliable decoding.
6. Deterministic evaluation sampling.
7. Minimal baseline transforms.
8. Compatibility with both model adapters.
9. Clear failure reporting.
10. Reproducible preprocessing metadata.

Do not prioritize initially:

* WLASL merging
* automatic sign segmentation
* pose extraction
* background replacement
* aggressive caching
* distributed data loading
* class-aware flipping
* advanced motion detection
* deployment video formats
* live webcam preprocessing

## Relationship to Other Documents

This document depends on:

* `PROJECT.md`
* `docs/ARCHITECTURE.md`
* `docs/ROADMAP.md`

The model boundary is defined in:

```text
docs/MODEL_CONTRACT.md
```

Training behavior is defined in:

```text
docs/TRAINING_CONTRACT.md
```

Evaluation behavior is defined in:

```text
docs/EVALUATION_CONTRACT.md
```

Future inference-time preprocessing belongs to:

```text
ASL PROJECT/ASL_serving/
```

The serving project should consume the deterministic preprocessing contract exported by `ASL_training`.
