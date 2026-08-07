"""Architecture-specific adapter behavior.

The shared contract is covered in test_model_contract.py. These tests cover the
guards and layout adaptation unique to each architecture.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.models import ModelConfig, build_model

from .conftest import TINY_VIDEOMAE_OPTIONS, make_batch

# VideoMAE --------------------------------------------------------------------


def test_videomae_rejects_frames_not_divisible_by_tubelet(videomae_config):
    config = ModelConfig.from_dict({**videomae_config.to_dict(), "num_frames": 15})
    with pytest.raises(ValueError, match="divisible by tubelet_size"):
        build_model(config)


def test_videomae_rejects_frame_count_mismatched_to_checkpoint(videomae_config):
    """Position embeddings are not interpolated, so this must fail loudly.

    The guard matters on the pretrained path, where the checkpoint's own config
    fixes the frame count. Exercised here by re-running the check against a
    backbone config that disagrees, which is what loading such a checkpoint
    produces.
    """
    model = build_model(videomae_config)
    model.backbone.config.num_frames = 32

    with pytest.raises(ValueError, match="Position embeddings are not interpolated"):
        model._validate_temporal_compatibility()


def test_videomae_rejects_reserved_options(videomae_config):
    """Options must not shadow fields that define the preprocessing identity."""
    config = ModelConfig.from_dict(
        {
            **videomae_config.to_dict(),
            "options": {**TINY_VIDEOMAE_OPTIONS, "num_frames": 32},
        }
    )
    with pytest.raises(ValueError, match="options may not override num_frames"):
        build_model(config)


def test_videomae_accepts_canonical_layout_without_permutation(videomae_config):
    """VideoMAE's native layout is the canonical one."""
    model = build_model(videomae_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(videomae_config))
    assert out.logits.shape[1] == videomae_config.num_classes


def test_videomae_reports_unloaded_pretrained_state(videomae_config):
    model = build_model(videomae_config)
    assert model.pretrained_load_report["loaded"] is False
    assert "randomly initialized" in model.pretrained_load_report["reason"]


# Video Swin ------------------------------------------------------------------


def test_swin_rejects_odd_frame_count(swin_config):
    config = ModelConfig.from_dict({**swin_config.to_dict(), "num_frames": 15})
    with pytest.raises(ValueError, match="temporal patch size"):
        build_model(config)


def test_swin_rejects_resolution_not_divisible_by_32(swin_config):
    config = ModelConfig.from_dict({**swin_config.to_dict(), "image_size": 100})
    with pytest.raises(ValueError, match="downsamples spatially"):
        build_model(config)


def test_swin_rejects_unknown_weight_name(swin_config):
    config = ModelConfig.from_dict(
        {**swin_config.to_dict(), "pretrained": True, "checkpoint": "IMAGENET_V9"}
    )
    with pytest.raises(ValueError, match="unknown Swin3D_T weights"):
        build_model(config)


def test_swin_permutes_to_channels_first_internally(swin_config):
    """The adapter must transpose; the canonical layout never leaves the module."""
    model = build_model(swin_config)
    captured = {}

    original = model.backbone.forward

    def spy(x, *args, **kwargs):
        captured["shape"] = tuple(x.shape)
        return original(x, *args, **kwargs)

    model.backbone.forward = spy
    model.eval()
    with torch.no_grad():
        model(make_batch(swin_config, batch_size=2))

    batch, channels, frames, height, width = captured["shape"]
    assert (batch, channels, frames) == (2, 3, swin_config.num_frames)
    assert (height, width) == (swin_config.image_size, swin_config.image_size)


def test_swin_head_replaced_with_configured_size(swin_config):
    model = build_model(swin_config)
    assert model.backbone.head.out_features == swin_config.num_classes
    assert model.head_in_features == 768


def test_swin_accepts_pretrained_frame_count(swin_config):
    """32 frames is the backbone's native resolution and must remain usable."""
    config = ModelConfig.from_dict({**swin_config.to_dict(), "num_frames": 32})
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(config, batch_size=1))
    assert out.logits.shape == (1, config.num_classes)


# Cross-architecture ----------------------------------------------------------


def test_both_architectures_share_normalization(videomae_config, swin_config):
    """Differing normalization would confound the Phase 5 comparison."""
    mae = build_model(videomae_config).preprocessing()
    swin = build_model(swin_config).preprocessing()
    assert mae.mean == swin.mean
    assert mae.std == swin.std
    assert mae.canonical_layout == swin.canonical_layout


def test_both_architectures_produce_identical_output_shapes(videomae_config, swin_config):
    mae = build_model(videomae_config)
    swin = build_model(swin_config)
    mae.eval()
    swin.eval()
    with torch.no_grad():
        mae_out = mae(make_batch(videomae_config, batch_size=2))
        swin_out = swin(make_batch(swin_config, batch_size=2))
    assert mae_out.logits.shape == swin_out.logits.shape
    assert type(mae_out) is type(swin_out)
