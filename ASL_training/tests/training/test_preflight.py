"""Training preflight.

Measures cost rather than quality. The properties under test are that the
numbers are sound (warmup discarded, data separated from compute, epoch time
extrapolated from the real step count) and that a preflight cannot be mistaken
for an experiment.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from asl_training.models import build_model
from asl_training.training import TrainingConfig, run_preflight
from asl_training.training.config import OptimizerConfig, SchedulerConfig
from asl_training.training.preflight import WARMUP_STEPS

from .conftest import make_loader
from .test_checkpoint import make_metadata


def make_config(**overrides) -> TrainingConfig:
    defaults = {
        "experiment": "preflight",
        "run_name": "preflight",
        "run_kind": "preflight",
        "epochs": 10,
        "batch_size": 4,
        "precision": "fp32",
        "device": "cpu",
        "optimizer": OptimizerConfig(lr=1e-3),
        "scheduler": SchedulerConfig(name="cosine", warmup_steps=0),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def preflight(model_config, steps=4, loaders=None, **overrides):
    train_loader, val_loader = loaders or (make_loader(32, 4), None)
    return run_preflight(
        build_model(model_config),
        make_config(**overrides),
        train_loader,
        val_loader,
        steps=steps,
        metadata=make_metadata(),
    )


# Measurement ------------------------------------------------------------------


def test_reports_throughput_and_step_time(model_config):
    report = preflight(model_config)
    m = report.measurements

    assert m["videos_per_second"] > 0
    assert m["seconds_per_optimizer_step"] > 0
    assert m["measured_steps"] == 4


def test_warmup_steps_are_discarded(model_config):
    """The first steps allocate and compile, so timing them misleads."""
    report = preflight(model_config, steps=5)

    assert report.measurements["warmup_steps_discarded"] == WARMUP_STEPS
    assert report.measurements["measured_steps"] == 5


def test_data_and_compute_are_timed_separately(model_config):
    """Which one dominates is the actionable output."""
    m = preflight(model_config).measurements

    assert m["data_seconds_per_step"] >= 0
    assert m["compute_seconds_per_step"] > 0
    assert m["seconds_per_optimizer_step"] == pytest.approx(
        m["data_seconds_per_step"] + m["compute_seconds_per_step"], abs=1e-6
    )
    assert 0 <= m["data_share_percent"] <= 100


def test_bottleneck_is_named(model_config):
    m = preflight(model_config).measurements
    assert m["bottleneck"] in ("data loading", "compute")

    expected = "data loading" if m["data_share_percent"] > 50 else "compute"
    assert m["bottleneck"] == expected


def test_epoch_estimate_uses_the_real_step_count(model_config):
    """Extrapolated from measured step time and the dataset's actual size."""
    report = preflight(model_config, loaders=(make_loader(32, 4), None))
    m = report.measurements

    assert m["train_samples"] == 32
    assert m["steps_per_epoch"] == 8  # 32 samples / batch 4
    assert m["estimated_epoch_seconds"] == pytest.approx(
        m["steps_per_epoch"] * m["seconds_per_optimizer_step"], rel=0.05
    )


def test_run_estimate_scales_with_epochs(model_config):
    m = preflight(model_config, epochs=10).measurements
    assert m["epochs"] == 10
    assert m["estimated_run_seconds"] == pytest.approx(m["estimated_epoch_seconds"] * 10, rel=0.05)


def test_accumulation_is_accounted_for_in_step_time(model_config):
    """An optimizer step under accumulation costs several micro-batches."""
    without = preflight(model_config, batch_size=4).measurements
    accumulated = preflight(model_config, batch_size=4, gradient_accumulation_steps=2).measurements

    assert accumulated["effective_batch_size"] == 8
    assert accumulated["seconds_per_optimizer_step"] > without["seconds_per_optimizer_step"]


def test_measurements_record_the_configuration_they_describe(model_config):
    m = preflight(model_config, batch_size=4, gradient_accumulation_steps=2).measurements

    assert m["physical_batch_size"] == 4
    assert m["effective_batch_size"] == 8
    assert m["gradient_accumulation_steps"] == 2
    assert m["precision_requested"] == "fp32"
    assert m["precision_active"] == "fp32"
    assert "num_workers" in m


# Full path --------------------------------------------------------------------


def test_exercises_the_whole_training_path(model_config):
    report = preflight(model_config)

    for check in ("construction", "forward_backward", "optimizer_step", "scheduler_step"):
        assert report.checks[check] == "ok", f"{check} was not exercised"


def test_checkpoint_round_trip_is_measured(model_config, tmp_path):
    """Checkpoint size drives the Drive budget, so it is measured not guessed."""
    report = run_preflight(
        build_model(model_config),
        make_config(),
        make_loader(32, 4),
        None,
        steps=4,
        metadata=make_metadata(),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert report.checks["checkpoint_save"] == "ok"
    assert report.checks["checkpoint_resume"] == "ok"
    assert report.measurements["checkpoint_mb"] > 0
    assert report.measurements["checkpoint_save_seconds"] >= 0


def test_checkpoint_write_volume_is_projected(model_config, tmp_path):
    report = run_preflight(
        build_model(model_config),
        make_config(checkpoint_every_minutes=20),
        make_loader(32, 4),
        None,
        steps=4,
        metadata=make_metadata(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    m = report.measurements
    assert m["checkpoint_write_mb_per_hour"] == pytest.approx(m["checkpoint_mb"] * 3, rel=0.05)


def test_validation_pass_is_exercised_and_priced(model_config):
    report = preflight(model_config, loaders=(make_loader(32, 4), make_loader(16, 4, seed=1)))

    assert report.checks["validation_pass"] == "ok"
    assert report.measurements["validation_samples"] == 16
    assert report.measurements["estimated_validation_minutes"] >= 0


def test_validation_is_optional(model_config):
    report = preflight(model_config, loaders=(make_loader(32, 4), None))
    assert "validation_pass" not in report.checks
    assert "estimated_validation_minutes" not in report.measurements


# It is not an experiment ------------------------------------------------------


def test_report_says_it_is_not_an_experiment(model_config):
    payload = preflight(model_config).to_dict()

    assert payload["kind"] == "preflight"
    assert "never model quality" in payload["not_an_experiment"]


def test_summary_repeats_the_caveat(model_config):
    assert "not an experiment" in preflight(model_config).summary()


def test_no_loss_or_accuracy_is_reported(model_config):
    """A preflight measures cost. Reporting quality would invite misreading."""
    m = preflight(model_config).measurements

    for forbidden in ("loss", "accuracy", "macro_f1", "top1_accuracy"):
        assert forbidden not in m, f"preflight reported {forbidden}"


def test_run_kind_is_carried_into_the_report(model_config):
    report = preflight(model_config)
    assert report.config["run_kind"] == "preflight"
    assert report.config["is_reduced"] is True


# Warnings ---------------------------------------------------------------------


def test_warns_when_data_loading_dominates(model_config, monkeypatch):
    """The actionable case: raise workers rather than change the model."""
    report = preflight(model_config)

    if report.measurements["data_share_percent"] > 50:
        assert any("num-workers" in w for w in report.warnings)
    else:
        assert not any("num-workers" in w for w in report.warnings)


class SlowLoader:
    """A loader whose decoding dominates the step, as on a CPU-starved host.

    Wraps a real loader rather than raising its ``num_workers``, so the delay
    stays in this process and no worker is spawned to serve it.
    """

    def __init__(self, num_workers: int, delay: float = 0.03):
        self._loader = make_loader(32, 4)
        self.dataset = self._loader.dataset
        self.drop_last = self._loader.drop_last
        self.num_workers = num_workers
        self._delay = delay

    def __iter__(self):
        for batch in self._loader:
            time.sleep(self._delay)
            yield batch


def data_bound_warning(model_config, monkeypatch, *, workers, cpus):
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    report = preflight(model_config, steps=4, loaders=(SlowLoader(workers), None))

    assert report.measurements["data_share_percent"] > 50
    warning = next(w for w in report.warnings if "data loading is" in w)
    return report, warning


def test_data_loading_warning_recommends_more_workers_when_cores_are_free(
    model_config, monkeypatch
):
    """Two workers on eight cores: there is real headroom to spend."""
    _, warning = data_bound_warning(model_config, monkeypatch, workers=2, cpus=8)

    assert "Raising --num-workers above 2" in warning
    assert "CPU-bound" not in warning


def test_data_loading_warning_does_not_oversubscribe_the_cores(model_config, monkeypatch):
    """Four workers on four cores, as on Kaggle: more workers measured slower."""
    report, warning = data_bound_warning(model_config, monkeypatch, workers=4, cpus=4)

    assert "CPU-bound" in warning
    assert "Raising --num-workers" not in warning
    assert report.measurements["cpu_count"] == 4


def test_warns_when_validation_is_expensive_relative_to_training(model_config):
    """A large validation split can quietly dominate epoch time."""
    report = preflight(
        model_config,
        loaders=(make_loader(8, 4), make_loader(64, 4, seed=1)),
        steps=4,
    )
    m = report.measurements

    if m["estimated_validation_minutes"] > m["estimated_epoch_minutes"] * 0.25:
        assert any("validate_every_epochs" in w for w in report.warnings)


def test_summary_is_readable(model_config):
    summary = preflight(model_config).summary()

    for fragment in ("Preflight", "throughput", "bottleneck", "epoch"):
        assert fragment in summary


def test_report_is_serializable(model_config):
    import json

    payload = json.loads(json.dumps(preflight(model_config).to_dict()))
    assert payload["measurements"]["videos_per_second"] > 0


# Failure ----------------------------------------------------------------------


def test_small_dataset_warns_that_throughput_is_optimistic(model_config):
    """Repeated batches decode from a warm cache, so the figure is not real."""
    report = preflight(model_config, steps=6, loaders=(make_loader(8, 4), None))

    assert report.measurements["loader_restarts"] > 0
    assert any("optimistic" in w for w in report.warnings)


def test_a_large_enough_dataset_does_not_restart(model_config):
    report = preflight(model_config, steps=4, loaders=(make_loader(64, 4), None))

    assert report.measurements["loader_restarts"] == 0
    assert not any("optimistic" in w for w in report.warnings)


def test_model_is_left_in_training_mode(model_config):
    """Preflight ends mid-training-path; it must not leave eval mode behind."""
    model = build_model(model_config)
    run_preflight(
        model,
        make_config(),
        make_loader(32, 4),
        make_loader(16, 4, seed=1),
        steps=4,
        metadata=make_metadata(),
    )
    assert model.training


def test_no_gradients_leak_from_the_validation_pass(model_config):
    report = preflight(model_config, loaders=(make_loader(32, 4), make_loader(16, 4, seed=1)))
    assert report.checks["validation_pass"] == "ok"
    assert torch.is_grad_enabled()
