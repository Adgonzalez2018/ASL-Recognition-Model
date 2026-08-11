"""Data layer: dataset parsing, manifests, decoding, sampling, and transforms.

    label_map    the classifier's output vocabulary and its identity
    manifest     record schema, split integrity, signer-leakage detection
    asl_citizen  annotation parsing for the primary dataset
    audit        the pre-training dataset audit
    decode       video to ordered RGB frames
    sampling     temporal frame selection
    transforms   spatial preprocessing, temporally consistent
    dataset      manifest records to canonical model batches

The canonical batch leaving this layer is:

    [batch, frames, channels, height, width]

Architecture-specific layouts are the model layer's business.

See docs/DATA_CONTRACT.md.
"""

from .asl_citizen import DatasetStructureError, parse_annotations, resolve_layout
from .audit import DatasetAudit, audit_dataset, probe_video
from .dataset import (
    FAILURE_POLICIES,
    LoaderConfig,
    PreprocessingSpec,
    SampleFailure,
    VideoClipDataset,
    collate_clips,
    worker_init_fn,
)
from .decode import DecodedClip, VideoDecodeError, count_frames, decode_clip
from .label_map import CONSTRUCTION_RULE, LabelMap, normalize_gloss
from .manifest import (
    REQUIRED_FIELDS,
    SPLITS,
    Manifest,
    ManifestRecord,
    ManifestValidationError,
    ValidationReport,
    normalize_split,
)
from .mirror import (
    SCALE_FILTER,
    SHORT_SIDE,
    MirrorError,
    detect_fps_flag,
    encode_clip,
    encode_command,
    ffmpeg_available,
    probe_dimensions,
    verify_clip,
)
from .sampling import SHORT_VIDEO_POLICY, STRATEGIES, TemporalSampler
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EvalTransform,
    TrainTransform,
)

__all__ = [
    "CONSTRUCTION_RULE",
    "FAILURE_POLICIES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "REQUIRED_FIELDS",
    "SCALE_FILTER",
    "SHORT_SIDE",
    "SHORT_VIDEO_POLICY",
    "SPLITS",
    "STRATEGIES",
    "DatasetAudit",
    "DatasetStructureError",
    "DecodedClip",
    "EvalTransform",
    "LabelMap",
    "LoaderConfig",
    "Manifest",
    "ManifestRecord",
    "ManifestValidationError",
    "MirrorError",
    "PreprocessingSpec",
    "SampleFailure",
    "TemporalSampler",
    "TrainTransform",
    "ValidationReport",
    "VideoClipDataset",
    "VideoDecodeError",
    "audit_dataset",
    "collate_clips",
    "count_frames",
    "decode_clip",
    "detect_fps_flag",
    "encode_clip",
    "encode_command",
    "ffmpeg_available",
    "normalize_gloss",
    "normalize_split",
    "parse_annotations",
    "probe_dimensions",
    "probe_video",
    "resolve_layout",
    "verify_clip",
    "worker_init_fn",
]
