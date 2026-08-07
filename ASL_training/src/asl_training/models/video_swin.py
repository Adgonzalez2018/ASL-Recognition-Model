"""Video Swin-Tiny adapter, backed by torchvision.

torchvision's Swin3D expects ``[batch, channels, frames, height, width]``. The
adapter permutes from the project canonical ``[batch, frames, channels, height,
width]`` internally; the transposed layout never leaves this module.

Weight source and the 16-frame decision are recorded as D-002 and D-003 in
docs/DECISIONS.md.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .base import BaseVideoClassifier, PreprocessingRequirements

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "KINETICS400_V1"

# Matches VideoMAE's normalization, so the architecture comparison is not
# confounded by preprocessing differences.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# torchvision swin3d_t pretrains at 32 frames. Running the baseline at 16 holds
# the data protocol constant across architectures; see D-003.
PRETRAINED_NUM_FRAMES = 32

# Patch embedding is 2x4x4, and the backbone downsamples spatially by 32.
TEMPORAL_PATCH = 2
SPATIAL_DIVISOR = 32


class VideoSwinClassifier(BaseVideoClassifier):
    """Video Swin-Tiny with a replaced ASL classification head."""

    def _build(self) -> None:
        try:
            from torchvision.models.video import Swin3D_T_Weights, swin3d_t
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "Video Swin requires 'torchvision>=0.19'. Install the project "
                "dependencies from pyproject.toml."
            ) from exc

        checkpoint = self.config.checkpoint or DEFAULT_CHECKPOINT
        self.checkpoint_source = checkpoint

        self._validate_input_divisibility()

        if self.config.pretrained:
            try:
                weights = Swin3D_T_Weights[checkpoint]
            except KeyError as exc:
                available = ", ".join(w.name for w in Swin3D_T_Weights)
                raise ValueError(
                    f"unknown Swin3D_T weights {checkpoint!r}; available: {available}"
                ) from exc

            # Build with the original Kinetics head, then replace it, so the
            # pretrained state dict loads cleanly and the head replacement stays
            # explicit rather than relying on mismatch tolerance.
            self.backbone = swin3d_t(weights=weights, **self.config.options)
            self.pretrained_load_report = {
                "checkpoint": checkpoint,
                "loaded": True,
                "replaced_head": True,
                "source": "torchvision.models.video.swin3d_t",
                "pretrained_num_frames": PRETRAINED_NUM_FRAMES,
            }
            logger.info("Loaded pretrained Swin3D-T weights: %s", checkpoint)

            if self.config.num_frames != PRETRAINED_NUM_FRAMES:
                logger.warning(
                    "Swin3D-T was pretrained at %d frames but is configured for %d. "
                    "This is the documented baseline choice (D-003) and must be "
                    "recorded in the experiment record.",
                    PRETRAINED_NUM_FRAMES,
                    self.config.num_frames,
                )
        else:
            self.backbone = swin3d_t(weights=None, **self.config.options)
            self.pretrained_load_report = {
                "checkpoint": checkpoint,
                "loaded": False,
                "reason": "pretrained=False; weights are randomly initialized",
            }

        self._replace_head()

    def _validate_input_divisibility(self) -> None:
        """Reject input dimensions the architecture cannot tile evenly."""
        frames = self.config.num_frames
        size = self.config.image_size

        if frames % TEMPORAL_PATCH != 0:
            raise ValueError(
                f"Video Swin uses a temporal patch size of {TEMPORAL_PATCH}; "
                f"num_frames must be divisible by it, got {frames}."
            )
        if size % SPATIAL_DIVISOR != 0:
            raise ValueError(
                f"Video Swin downsamples spatially by {SPATIAL_DIVISOR}; image_size "
                f"must be divisible by it, got {size}."
            )

    def _replace_head(self) -> None:
        """Replace the Kinetics-400 head with an ASL head of the configured size."""
        in_features = self.backbone.head.in_features
        head = nn.Linear(in_features, self.config.num_classes)

        if self.config.dropout > 0.0:
            self.backbone.head = nn.Sequential(nn.Dropout(self.config.dropout), head)
        else:
            self.backbone.head = head

        self.head_in_features = in_features

    def classification_head(self) -> nn.Module:
        return self.backbone.head

    def _forward_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Canonical [B, T, C, H, W] -> torchvision's [B, C, T, H, W].
        adapted = pixel_values.permute(0, 2, 1, 3, 4)
        return self.backbone(adapted)

    def preprocessing(self) -> PreprocessingRequirements:
        return PreprocessingRequirements(
            num_frames=self.config.num_frames,
            image_size=self.config.image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
