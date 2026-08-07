"""Fixtures for training tests.

Uses a tiny model and synthetic tensors so the full loop runs in seconds. No
dataset and no network are required.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from asl_training.models import ModelConfig, build_model

NUM_CLASSES = 4
FRAMES = 4
SIZE = 32

TINY_VIDEOMAE = {
    "hidden_size": 48,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 96,
}


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        architecture="videomae_base",
        num_classes=NUM_CLASSES,
        pretrained=False,
        num_frames=FRAMES,
        image_size=SIZE,
        options=dict(TINY_VIDEOMAE),
    )


@pytest.fixture
def model(model_config):
    return build_model(model_config)


class SyntheticClips(Dataset):
    """Learnable synthetic clips.

    Each class gets a distinct constant offset, so a working training loop
    reduces the loss. That makes "loss decreases" a real signal rather than an
    accident of random data.
    """

    def __init__(self, size: int = 16, num_classes: int = NUM_CLASSES, seed: int = 0):
        self.size = size
        self.num_classes = num_classes
        generator = torch.Generator().manual_seed(seed)

        self.labels = torch.randint(0, num_classes, (size,), generator=generator)
        self.clips = []
        for label in self.labels:
            noise = torch.randn(FRAMES, 3, SIZE, SIZE, generator=generator) * 0.1
            self.clips.append(noise + float(label))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "pixel_values": self.clips[index],
            "label": int(self.labels[index]),
            "sample_id": f"synthetic:{index:04d}",
            "signer_id": f"signer{index % 3:02d}",
            "gloss": f"GLOSS_{int(self.labels[index])}",
            "split": "train",
            "dataset_name": "synthetic",
        }


def make_loader(size: int = 16, batch_size: int = 4, **kwargs) -> DataLoader:
    from asl_training.data import collate_clips

    return DataLoader(
        SyntheticClips(size=size, **kwargs),
        batch_size=batch_size,
        collate_fn=collate_clips,
        num_workers=0,
    )


@pytest.fixture
def train_loader() -> DataLoader:
    return make_loader(size=16, batch_size=4)


@pytest.fixture
def val_loader() -> DataLoader:
    return make_loader(size=8, batch_size=4, seed=1)
