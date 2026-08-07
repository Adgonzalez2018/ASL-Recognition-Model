"""Shared fixtures for model-layer tests.

Tests use tiny randomly initialized backbones so the default suite runs offline
on CPU in seconds. Pretrained construction is covered separately by tests marked
``pretrained``. See D-005 in docs/DECISIONS.md.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.models import ModelConfig

NUM_CLASSES = 7
NUM_FRAMES = 16
IMAGE_SIZE = 224

# A minimal VideoMAE that preserves the real architecture's structure and
# tubelet/position-embedding behavior while training in milliseconds.
TINY_VIDEOMAE_OPTIONS = {
    "hidden_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "intermediate_size": 192,
}


@pytest.fixture
def videomae_config() -> ModelConfig:
    return ModelConfig(
        architecture="videomae_base",
        num_classes=NUM_CLASSES,
        pretrained=False,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        options=dict(TINY_VIDEOMAE_OPTIONS),
    )


@pytest.fixture
def swin_config() -> ModelConfig:
    return ModelConfig(
        architecture="video_swin_tiny",
        num_classes=NUM_CLASSES,
        pretrained=False,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )


@pytest.fixture(params=["videomae", "swin"])
def any_config(request, videomae_config, swin_config) -> ModelConfig:
    """Both architectures, so contract tests run against each."""
    return videomae_config if request.param == "videomae" else swin_config


def make_batch(config: ModelConfig, batch_size: int = 2) -> torch.Tensor:
    """A canonical [batch, frames, channels, height, width] float batch."""
    return torch.randn(batch_size, config.num_frames, 3, config.image_size, config.image_size)


def make_labels(config: ModelConfig, batch_size: int = 2) -> torch.Tensor:
    return torch.randint(0, config.num_classes, (batch_size,), dtype=torch.long)
