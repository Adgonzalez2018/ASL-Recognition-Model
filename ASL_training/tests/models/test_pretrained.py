"""Pretrained weight loading.

Excluded from the default run because these download real checkpoints. See D-005
in docs/DECISIONS.md.

    pytest tests/models -m pretrained

These are the tests that catch a renamed checkpoint or a changed state-dict
layout, which the offline structural suite cannot see. Model preflight is the
gate that must pass before any real training run.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.models import ModelConfig, build_model, build_model_from_yaml

from .conftest import make_batch

pytestmark = pytest.mark.pretrained

NUM_CLASSES = 2731  # ASL Citizen full vocabulary


@pytest.mark.slow
def test_videomae_loads_pretrained_and_replaces_head():
    config = ModelConfig(
        architecture="videomae_base",
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model = build_model(config)

    report = model.pretrained_load_report
    assert report["loaded"] is True
    assert report["replaced_head"] is True
    assert model.classification_head().out_features == NUM_CLASSES

    # Every missing key must be explained. An unexplained one means part of the
    # backbone is randomly initialized, which would invalidate a "pretrained"
    # baseline. See docs/MODEL_CONTRACT.md.
    assert report["unexplained_missing_keys"] == []
    assert report["unexplained_unused_keys"] == []
    assert "classifier.weight" in report["expected_missing_keys"]


@pytest.mark.slow
def test_videomae_attention_biases_are_actually_loaded():
    """Guards against the silent legacy-name bias drop.

    Published VideoMAE checkpoints store attention biases as ``q_bias``/``v_bias``.
    Some transformers versions expect ``query.bias``/``value.bias`` and do not
    translate, leaving them zero while reporting a successful load. See D-006.
    """
    model = build_model(ModelConfig(architecture="videomae_base", num_classes=10, pretrained=True))
    state = model.state_dict()
    prefix = "backbone.videomae.encoder.layer.0.attention.attention"

    # Query and value biases are learned and must be non-zero after loading.
    assert state[f"{prefix}.query.bias"].norm() > 0
    assert state[f"{prefix}.value.bias"].norm() > 0

    # VideoMAE defines no key bias; zero is correct here.
    assert state[f"{prefix}.key.bias"].norm() == 0

    report = model.pretrained_load_report
    if report["repaired_keys"]:
        # Two repaired tensors per encoder layer when the rename applies.
        assert len(report["repaired_keys"]) == 2 * model.backbone.config.num_hidden_layers


@pytest.mark.slow
def test_swin_loads_pretrained_and_replaces_head():
    config = ModelConfig(
        architecture="video_swin_tiny",
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model = build_model(config)

    assert model.pretrained_load_report["loaded"] is True
    assert model.pretrained_load_report["pretrained_num_frames"] == 32
    assert model.backbone.head.out_features == NUM_CLASSES

    model.eval()
    with torch.no_grad():
        out = model(make_batch(config, batch_size=1))
    assert out.logits.shape == (1, NUM_CLASSES)
    assert torch.isfinite(out.logits).all()


@pytest.mark.slow
def test_pretrained_backbone_differs_from_random_init():
    """Confirms weights were actually loaded, not silently skipped."""
    shared = {"architecture": "video_swin_tiny", "num_classes": 10}
    pre = build_model(ModelConfig(**shared, pretrained=True))
    rand = build_model(ModelConfig(**shared, pretrained=False))

    name = "backbone.patch_embed.proj.weight"
    pre_w = dict(pre.named_parameters())[name]
    rand_w = dict(rand.named_parameters())[name]
    assert not torch.allclose(pre_w, rand_w)


@pytest.mark.slow
@pytest.mark.parametrize("config_name", ["videomae_base.yaml", "video_swin_tiny.yaml"])
def test_shipped_configs_construct_with_real_weights(config_name):
    """The committed configs must work end to end, not just parse."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "models" / config_name
    model = build_model_from_yaml(path, num_classes=NUM_CLASSES)

    model.eval()
    with torch.no_grad():
        out = model(make_batch(model.config, batch_size=1))
    assert out.logits.shape == (1, NUM_CLASSES)


@pytest.mark.slow
def test_both_pretrained_architectures_agree_on_preprocessing():
    """A normalization difference would confound the Phase 5 comparison."""
    mae = build_model(
        ModelConfig(architecture="videomae_base", num_classes=10, pretrained=True)
    ).preprocessing()
    swin = build_model(
        ModelConfig(architecture="video_swin_tiny", num_classes=10, pretrained=True)
    ).preprocessing()

    assert mae.mean == swin.mean
    assert mae.std == swin.std
    assert mae.num_frames == swin.num_frames
    assert mae.image_size == swin.image_size
