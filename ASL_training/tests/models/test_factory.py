"""Model factory and checkpoint state loading."""

from __future__ import annotations

import pytest
import torch

from asl_training.models import (
    ModelConfig,
    VideoMAEClassifier,
    VideoSwinClassifier,
    available_architectures,
    build_model,
    build_model_from_yaml,
    load_checkpoint_state,
)

from .conftest import NUM_CLASSES, TINY_VIDEOMAE_OPTIONS, make_batch


def test_registered_architectures_are_stable():
    assert available_architectures() == ["video_swin_tiny", "videomae_base"]


def test_factory_dispatches_to_the_right_adapter(videomae_config, swin_config):
    assert isinstance(build_model(videomae_config), VideoMAEClassifier)
    assert isinstance(build_model(swin_config), VideoSwinClassifier)


def test_unknown_architecture_is_rejected_clearly():
    config = ModelConfig(architecture="i3d", num_classes=10)
    with pytest.raises(ValueError, match="unknown architecture 'i3d'"):
        build_model(config)


def test_unknown_architecture_error_lists_supported_names():
    config = ModelConfig(architecture="slowfast", num_classes=10)
    with pytest.raises(ValueError, match="videomae_base"):
        build_model(config)


# YAML construction -----------------------------------------------------------


def test_build_from_yaml_with_runtime_num_classes(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "model:\n  architecture: videomae_base\n  pretrained: false\n  num_frames: 16\n"
    )
    model = build_model_from_yaml(path, num_classes=13, options=dict(TINY_VIDEOMAE_OPTIONS))
    assert model.num_classes == 13


def test_build_from_yaml_rejects_unknown_override(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("model:\n  architecture: videomae_base\n  pretrained: false\n")
    with pytest.raises(ValueError, match="unknown override keys"):
        build_model_from_yaml(path, num_classes=10, num_frame=16)


# Checkpoint state loading ----------------------------------------------------


def test_round_trip_state_dict_restores_weights(videomae_config, tmp_path):
    source = build_model(videomae_config)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": videomae_config.architecture,
            "num_classes": videomae_config.num_classes,
        },
        path,
    )

    target = build_model(videomae_config)
    batch = make_batch(videomae_config)

    source.eval()
    target.eval()
    with torch.no_grad():
        before = target(batch).logits
        expected = source(batch).logits
    assert not torch.allclose(before, expected), "fixture models started identical"

    report = load_checkpoint_state(target, path)
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []

    target.eval()
    with torch.no_grad():
        after = target(batch).logits
    assert torch.allclose(after, expected, atol=1e-6)


def test_accepts_bare_state_dict(videomae_config, tmp_path):
    source = build_model(videomae_config)
    path = tmp_path / "bare.pt"
    torch.save(source.state_dict(), path)

    target = build_model(videomae_config)
    report = load_checkpoint_state(target, path)
    assert report["missing_keys"] == []


def test_missing_checkpoint_raises(videomae_config, tmp_path):
    model = build_model(videomae_config)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint_state(model, tmp_path / "absent.pt")


def test_rejects_mismatched_class_count(videomae_config, tmp_path):
    """Loading a differently sized vocabulary would silently change class meaning."""
    other = ModelConfig.from_dict({**videomae_config.to_dict(), "num_classes": 3})
    source = build_model(other)
    path = tmp_path / "wrong_classes.pt"
    torch.save({"model_state": source.state_dict(), "num_classes": 3}, path)

    target = build_model(videomae_config)
    with pytest.raises(ValueError, match="3 classes but the model is configured for 7"):
        load_checkpoint_state(target, path)


def test_detects_head_mismatch_without_metadata(videomae_config, tmp_path):
    """Head width is inferred from the state dict when metadata is absent."""
    other = ModelConfig.from_dict({**videomae_config.to_dict(), "num_classes": 3})
    source = build_model(other)
    path = tmp_path / "no_meta.pt"
    torch.save(source.state_dict(), path)

    target = build_model(videomae_config)
    with pytest.raises(ValueError, match="3 outputs but the model is configured for 7"):
        load_checkpoint_state(target, path)


def test_rejects_mismatched_architecture(videomae_config, swin_config, tmp_path):
    source = build_model(swin_config)
    path = tmp_path / "swin.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": "video_swin_tiny",
            "num_classes": NUM_CLASSES,
        },
        path,
    )

    target = build_model(videomae_config)
    with pytest.raises(ValueError, match="does not match the constructed model"):
        load_checkpoint_state(target, path)


def test_reads_architecture_from_nested_config_metadata(videomae_config, tmp_path):
    source = build_model(videomae_config)
    path = tmp_path / "nested.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"architecture": "video_swin_tiny", "num_classes": NUM_CLASSES},
        },
        path,
    )
    target = build_model(videomae_config)
    with pytest.raises(ValueError, match="does not match the constructed model"):
        load_checkpoint_state(target, path)


def test_preserves_checkpoint_metadata_in_report(videomae_config, tmp_path):
    source = build_model(videomae_config)
    path = tmp_path / "meta.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": videomae_config.architecture,
            "num_classes": NUM_CLASSES,
            "label_map_identity": "asl-citizen-v1-sha256:abc",
            "epoch": 4,
        },
        path,
    )
    target = build_model(videomae_config)
    report = load_checkpoint_state(target, path)
    assert report["checkpoint_metadata"]["label_map_identity"] == "asl-citizen-v1-sha256:abc"
    assert report["checkpoint_metadata"]["epoch"] == 4
    assert "model_state" not in report["checkpoint_metadata"]
