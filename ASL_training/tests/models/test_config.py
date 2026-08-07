"""Model configuration validation."""

from __future__ import annotations

import pytest

from asl_training.models import ModelConfig


def test_minimal_config_applies_documented_defaults():
    config = ModelConfig(architecture="videomae_base", num_classes=100)
    assert config.num_frames == 16
    assert config.image_size == 224
    assert config.fine_tuning == "full"
    assert config.pretrained is True
    assert config.dropout == 0.0


@pytest.mark.parametrize("num_classes", [0, 1, -5])
def test_rejects_class_count_below_two(num_classes):
    with pytest.raises(ValueError, match="at least 2"):
        ModelConfig(architecture="videomae_base", num_classes=num_classes)


def test_rejects_bool_class_count():
    # bool is an int subclass, so this would otherwise pass as num_classes=1.
    with pytest.raises(TypeError, match="must be an int"):
        ModelConfig(architecture="videomae_base", num_classes=True)


def test_rejects_float_class_count():
    with pytest.raises(TypeError, match="must be an int"):
        ModelConfig(architecture="videomae_base", num_classes=10.0)


def test_rejects_empty_architecture():
    with pytest.raises(ValueError, match="non-empty string"):
        ModelConfig(architecture="", num_classes=10)


def test_rejects_unknown_fine_tuning_strategy():
    with pytest.raises(ValueError, match="unknown fine_tuning"):
        ModelConfig(architecture="videomae_base", num_classes=10, fine_tuning="partial")


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_rejects_out_of_range_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(architecture="videomae_base", num_classes=10, dropout=dropout)


@pytest.mark.parametrize("frames", [0, -1])
def test_rejects_invalid_frame_count(frames):
    with pytest.raises(ValueError, match="num_frames"):
        ModelConfig(architecture="videomae_base", num_classes=10, num_frames=frames)


def test_from_dict_rejects_unknown_keys():
    # A typo must fail loudly rather than leaving a setting at its default,
    # which would make the recorded config not describe the actual run.
    with pytest.raises(ValueError, match="unknown model config keys"):
        ModelConfig.from_dict({"architecture": "videomae_base", "num_classes": 10, "num_frame": 16})


def test_round_trips_through_dict():
    original = ModelConfig(
        architecture="video_swin_tiny",
        num_classes=2731,
        num_frames=32,
        dropout=0.3,
        fine_tuning="head_only",
    )
    assert ModelConfig.from_dict(original.to_dict()) == original


def test_from_yaml_reads_model_section(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("model:\n  architecture: videomae_base\n  num_classes: 50\n  num_frames: 16\n")
    config = ModelConfig.from_yaml(path)
    assert config.architecture == "videomae_base"
    assert config.num_classes == 50


def test_from_yaml_requires_model_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("architecture: videomae_base\nnum_classes: 50\n")
    with pytest.raises(ValueError, match="missing required top-level 'model' key"):
        ModelConfig.from_yaml(path)


def test_shipped_configs_are_valid_and_consistent():
    """The committed model configs must parse and agree with D-003."""
    from pathlib import Path

    configs_dir = Path(__file__).resolve().parents[2] / "configs" / "models"
    files = sorted(configs_dir.glob("*.yaml"))
    assert files, "no model configs found"

    for path in files:
        section = _yaml_model_section(path)
        # num_classes comes from the label map at runtime, never the config file.
        assert "num_classes" not in section, f"{path.name} must not pin num_classes"

        config = ModelConfig.from_dict({**section, "num_classes": 10})
        # D-003: both architectures share the baseline input protocol.
        assert config.num_frames == 16
        assert config.image_size == 224
        assert config.fine_tuning == "full"


def _yaml_model_section(path):
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)["model"]
