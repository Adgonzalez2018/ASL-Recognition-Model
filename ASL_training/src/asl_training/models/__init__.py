"""Model layer: architecture adapters and the shared classification contract.

Canonical input:

    [batch, frames, channels, height, width]

Canonical output:

    logits: [batch, num_classes]

Architecture-specific tensor layouts and output objects stay inside adapters.
See docs/MODEL_CONTRACT.md.
"""

from .base import BaseVideoClassifier, ParameterReport, PreprocessingRequirements
from .config import ModelConfig
from .factory import (
    ARCHITECTURES,
    available_architectures,
    build_model,
    build_model_from_yaml,
    load_checkpoint_state,
)
from .outputs import VideoClassifierOutput
from .video_swin import VideoSwinClassifier
from .videomae import VideoMAEClassifier

__all__ = [
    "ARCHITECTURES",
    "BaseVideoClassifier",
    "ModelConfig",
    "ParameterReport",
    "PreprocessingRequirements",
    "VideoClassifierOutput",
    "VideoMAEClassifier",
    "VideoSwinClassifier",
    "available_architectures",
    "build_model",
    "build_model_from_yaml",
    "load_checkpoint_state",
]
