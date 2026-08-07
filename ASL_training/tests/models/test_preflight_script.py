"""Model preflight script.

docs/MODEL_CONTRACT.md requires preflight to pass before a real training run and
requires a failure to block the run, so the exit code matters as much as the
checks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from asl_training.models import ModelConfig

from .conftest import TINY_VIDEOMAE_OPTIONS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "model_preflight.py"


@pytest.fixture(scope="module")
def preflight():
    spec = importlib.util.spec_from_file_location("model_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_preflight"] = module
    spec.loader.exec_module(module)
    return module


def test_passes_on_a_valid_configuration(preflight):
    config = ModelConfig(
        architecture="videomae_base",
        num_classes=7,
        pretrained=False,
        options=dict(TINY_VIDEOMAE_OPTIONS),
    )
    report = preflight.run_preflight(config, batch_size=2, device="cpu")

    assert report["logits_shape"] == [2, 7]
    assert report["parameters"]["trainable"] == report["parameters"]["total"]
    assert report["parameters_with_gradients"] > 0
    assert set(report["checks"].values()) == {"ok"}
    assert "backbone_trainable" in report["checks"]
    assert "head_trainable" in report["checks"]


def test_reports_initial_loss_near_uniform_prior(preflight):
    import math

    config = ModelConfig(
        architecture="videomae_base",
        num_classes=7,
        pretrained=False,
        options=dict(TINY_VIDEOMAE_OPTIONS),
    )
    report = preflight.run_preflight(config, batch_size=4, device="cpu")
    assert report["initial_loss"] == pytest.approx(math.log(7), abs=1.5)


def test_head_only_strategy_reports_frozen_backbone(preflight):
    config = ModelConfig(
        architecture="videomae_base",
        num_classes=7,
        pretrained=False,
        fine_tuning="head_only",
        options=dict(TINY_VIDEOMAE_OPTIONS),
    )
    report = preflight.run_preflight(config, batch_size=2, device="cpu")
    assert report["parameters"]["frozen"] > 0
    assert report["parameters"]["trainable"] == report["parameters"]["head"]


def test_cli_exits_nonzero_on_unknown_architecture(preflight, capsys):
    """A preflight failure must block the run."""
    code = preflight.main(
        ["--architecture", "videomae_base", "--num-classes", "1", "--no-pretrained"]
    )
    assert code == 1


def test_cli_requires_a_config_or_architecture(preflight):
    with pytest.raises(SystemExit):
        preflight.parse_args(["--num-classes", "10"])


def test_cli_rejects_both_config_and_architecture(preflight):
    with pytest.raises(SystemExit):
        preflight.parse_args(
            ["--config", "x.yaml", "--architecture", "videomae_base", "--num-classes", "10"]
        )
