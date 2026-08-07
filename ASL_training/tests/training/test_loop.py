"""Training loop behavior.

Covers the properties that would corrupt an experiment silently if wrong:
gradient accumulation arithmetic, scheduler cadence, validation leaving weights
untouched, and run metadata being complete enough to reproduce the run.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import json

import pytest
import torch

from asl_training.models import build_model
from asl_training.training import (
    Trainer,
    TrainingConfig,
    default_metrics,
    resolve_precision,
)
from asl_training.training.config import OptimizerConfig, SchedulerConfig

from .conftest import NUM_CLASSES, make_loader
from .test_checkpoint import make_metadata


def make_config(**overrides) -> TrainingConfig:
    defaults = {
        "experiment": "test",
        "run_name": "run-001",
        "run_kind": "smoke",
        "epochs": 1,
        "batch_size": 4,
        "precision": "fp32",
        "device": "cpu",
        "checkpoint_every_minutes": None,
        "log_every_steps": 1000,
        "optimizer": OptimizerConfig(lr=1e-3),
        "scheduler": SchedulerConfig(name="cosine", warmup_steps=0),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def make_trainer(model_config, tmp_path, loaders=None, **overrides) -> Trainer:
    train_loader, val_loader = loaders or (make_loader(16, 4), make_loader(8, 4, seed=1))
    return Trainer(
        build_model(model_config),
        make_config(**overrides),
        train_loader,
        val_loader,
        tmp_path / "run",
        metadata=make_metadata(),
    )


# Optimization -----------------------------------------------------------------


def test_loss_decreases_on_a_learnable_task(model_config, tmp_path):
    """The synthetic clips are separable, so a working loop must reduce loss."""
    trainer = make_trainer(model_config, tmp_path, epochs=6)
    history = trainer.train()

    assert len(history) == 6
    assert history[-1].train_loss < history[0].train_loss, (
        f"loss did not decrease: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}"
    )


def test_weights_actually_change(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.train()

    changed = sum(
        1
        for k, v in trainer.model.state_dict().items()
        if v.dtype.is_floating_point and not torch.allclose(v, before[k])
    )
    assert changed > 0


def test_counters_advance_consistently(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=2, batch_size=4)
    trainer.train()

    assert trainer.state.epoch == 2
    assert trainer.state.micro_step == 8  # 16 samples / batch 4, twice
    assert trainer.state.optimizer_step == 8
    assert trainer.state.samples_seen == 32


# Gradient accumulation --------------------------------------------------------


def test_accumulation_reduces_optimizer_steps(model_config, tmp_path):
    """Micro-steps count batches; optimizer steps count updates."""
    trainer = make_trainer(
        model_config, tmp_path, epochs=1, batch_size=4, gradient_accumulation_steps=2
    )
    trainer.train()

    assert trainer.state.micro_step == 4
    assert trainer.state.optimizer_step == 2


def test_partial_accumulation_window_still_steps(model_config, tmp_path):
    """4 batches with accumulation 3 gives one full window and one partial."""
    trainer = make_trainer(
        model_config,
        tmp_path,
        loaders=(make_loader(16, 4), None),
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=3,
    )
    trainer.train()

    assert trainer.state.micro_step == 4
    assert trainer.state.optimizer_step == 2


def test_accumulation_matches_a_larger_batch(model_config, tmp_path):
    """Averaging must be right: scaling twice is a classic silent error."""
    torch.manual_seed(0)
    big = Trainer(
        build_model(model_config),
        make_config(epochs=1, batch_size=8, seed=0),
        make_loader(8, 8),
        None,
        tmp_path / "big",
        metadata=make_metadata(),
    )
    big.train()

    torch.manual_seed(0)
    accumulated = Trainer(
        build_model(model_config),
        make_config(epochs=1, batch_size=4, gradient_accumulation_steps=2, seed=0),
        make_loader(8, 4),
        None,
        tmp_path / "accumulated",
        metadata=make_metadata(),
    )
    accumulated.train()

    assert big.state.optimizer_step == accumulated.state.optimizer_step == 1

    # Same effective batch and same data, so the updated weights should agree
    # closely. A doubled or missing 1/N would move them far apart.
    for key, value in big.model.state_dict().items():
        if not value.dtype.is_floating_point:
            continue
        other = accumulated.model.state_dict()[key]
        assert torch.allclose(value, other, atol=1e-4), f"{key} diverged"


# Scheduler cadence ------------------------------------------------------------


def test_scheduler_advances_per_optimizer_step_not_per_batch(model_config, tmp_path):
    """Under accumulation, stepping per batch would compress the schedule."""
    trainer = make_trainer(
        model_config,
        tmp_path,
        loaders=(make_loader(16, 4), None),
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=2,
    )
    trainer.train()

    assert trainer.scheduler.last_epoch == trainer.state.optimizer_step
    assert trainer.scheduler.last_epoch < trainer.state.micro_step


def test_max_steps_stops_the_run(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=10, max_steps=3)
    trainer.train()
    assert trainer.state.optimizer_step == 3


# Validation -------------------------------------------------------------------


def test_validation_does_not_change_weights(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    trainer.validate()

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed during validation"


def test_validation_does_not_advance_the_optimizer(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    step = trainer.state.optimizer_step
    lr = trainer.scheduler.get_last_lr()

    trainer.validate()

    assert trainer.state.optimizer_step == step
    assert trainer.scheduler.get_last_lr() == lr


def test_validation_restores_training_mode(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path)
    trainer.model.train()
    trainer.validate()
    assert trainer.model.training


def test_validation_produces_metrics(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    metrics = trainer.validate()

    assert "top1_accuracy" in metrics
    assert "loss" in metrics
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0


def test_run_without_validation_is_supported(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, loaders=(make_loader(16, 4), None), epochs=1)
    history = trainer.train()
    assert history[0].validation == {}


def test_best_checkpoint_tracks_the_selection_metric(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=3)
    trainer.train()

    assert trainer.state.best_metric is not None
    assert trainer.state.best_epoch is not None
    assert trainer.checkpoints.best_path.exists()

    recorded = [h.validation["top1_accuracy"] for h in trainer.history if h.validation]
    assert trainer.state.best_metric == pytest.approx(max(recorded))


# Precision --------------------------------------------------------------------


def test_fp32_disables_autocast():
    enabled, dtype = resolve_precision("fp32", torch.device("cpu"))
    assert not enabled
    assert dtype is None


def test_mixed_precision_downgrades_loudly_off_cuda(caplog):
    """A silent downgrade would make the recorded precision wrong."""
    with caplog.at_level("WARNING"):
        enabled, dtype = resolve_precision("bf16", torch.device("cpu"))

    assert not enabled
    assert dtype is None
    assert "running in fp32" in caplog.text


def test_scaler_is_only_created_for_fp16(model_config, tmp_path):
    """bf16 needs no loss scaling; a scaler there would be a silent no-op."""
    trainer = make_trainer(model_config, tmp_path, precision="fp32")
    assert trainer.scaler is None


# Run records ------------------------------------------------------------------


def test_run_metadata_captures_reproducibility_fields(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    trainer.train()

    payload = json.loads((trainer.output_dir / "run_metadata.json").read_text())

    for key in (
        "experiment",
        "run_name",
        "run_kind",
        "is_reduced",
        "config",
        "metadata",
        "physical_batch_size",
        "effective_batch_size",
        "precision_requested",
        "precision_active",
        "train_samples",
        "environment",
    ):
        assert key in payload, f"run metadata is missing {key}"

    assert payload["environment"]["torch"]
    assert payload["config"]["seed"] == 42


def test_run_metadata_records_both_batch_sizes(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, batch_size=4, gradient_accumulation_steps=4)
    trainer.train()

    payload = json.loads((trainer.output_dir / "run_metadata.json").read_text())
    assert payload["physical_batch_size"] == 4
    assert payload["effective_batch_size"] == 16


def test_reduced_run_is_labeled_in_metadata(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, run_kind="smoke")
    trainer.train()

    payload = json.loads((trainer.output_dir / "run_metadata.json").read_text())
    assert payload["run_kind"] == "smoke"
    assert payload["is_reduced"] is True


def test_reduced_run_warns(model_config, tmp_path, caplog):
    """A smoke run must not be mistaken for a baseline."""
    with caplog.at_level("WARNING"):
        make_trainer(model_config, tmp_path, run_kind="subset")
    assert "not a full baseline" in caplog.text


def test_history_is_written_per_epoch(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=2)
    trainer.train()

    history = json.loads((trainer.output_dir / "history.json").read_text())
    assert len(history) == 2
    for entry in history:
        assert "train_loss" in entry
        assert "learning_rates" in entry
        assert "optimizer_steps" in entry


def test_epoch_result_reports_skips_and_non_finite(model_config, tmp_path):
    trainer = make_trainer(model_config, tmp_path, epochs=1)
    result = trainer.train()[0]
    assert result.skipped_batches == 0
    assert result.non_finite_losses == 0


# Metrics ----------------------------------------------------------------------


def test_default_metrics_on_perfect_predictions():
    logits = torch.eye(NUM_CLASSES) * 10
    labels = torch.arange(NUM_CLASSES)
    assert default_metrics(logits, labels)["top1_accuracy"] == 1.0


def test_default_metrics_on_wrong_predictions():
    logits = torch.zeros(4, NUM_CLASSES)
    logits[:, 0] = 10.0
    labels = torch.tensor([1, 1, 2, 3])
    assert default_metrics(logits, labels)["top1_accuracy"] == 0.0


def test_top5_is_omitted_when_there_are_too_few_classes():
    """Reporting 1.0 for an undefined metric would be misleading."""
    metrics = default_metrics(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
    assert "top5_accuracy" not in metrics


def test_top5_is_reported_when_defined():
    metrics = default_metrics(torch.randn(4, 10), torch.tensor([0, 1, 2, 3]))
    assert "top5_accuracy" in metrics
    assert metrics["top5_accuracy"] >= metrics["top1_accuracy"]
