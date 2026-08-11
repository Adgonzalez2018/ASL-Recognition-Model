"""Checkpointing and resume.

On Colab, interruption is the normal path rather than a failure mode, so resume
correctness is a blocking criterion. See D-004.

Verified by interrupting a real run and continuing it, not by inspecting the
saved file alone.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.training import (
    CheckpointError,
    CheckpointManager,
    CheckpointMetadata,
    Trainer,
    TrainingConfig,
    TrainingState,
    build_optimizer,
    build_scheduler,
    is_better,
    restore,
    validate_resume_compatibility,
)
from asl_training.training.checkpoint import PREVIOUS
from asl_training.training.config import OptimizerConfig, SchedulerConfig

from .conftest import NUM_CLASSES


def make_metadata(**overrides) -> CheckpointMetadata:
    defaults = {
        "architecture": "videomae_base",
        "num_classes": NUM_CLASSES,
        "label_map_identity": "asl_citizen:4:sha256:abc",
        "preprocessing_identity": "preprocessing:train:sha256:def",
        "dataset_substrate": "source",
        "fine_tuning": "full",
        "optimizer_name": "adamw",
        "scheduler_name": "cosine",
    }
    defaults.update(overrides)
    return CheckpointMetadata(**defaults)


def make_config(tmp_path, **overrides) -> TrainingConfig:
    defaults = {
        "experiment": "test",
        "run_name": "run-001",
        "run_kind": "smoke",
        "epochs": 2,
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


# Round trip -------------------------------------------------------------------


def test_saves_and_reloads_every_component(model, tmp_path):
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)

    state = TrainingState(epoch=3, micro_step=30, optimizer_step=10, best_metric=0.42)
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=state,
        metadata=make_metadata(),
        config={"seed": 42},
    )

    payload = manager.load()
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "training_state",
        "metadata",
        "config",
        "rng_state",
    ):
        assert key in payload, f"missing {key}"

    assert payload["training_state"]["epoch"] == 3
    assert payload["training_state"]["best_metric"] == 0.42


def test_restore_returns_the_saved_state(model, tmp_path):
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=TrainingState(epoch=5, optimizer_step=25, best_metric=0.9, best_epoch=4),
        metadata=make_metadata(),
        config={},
    )

    state = restore(manager.load(), model=model, optimizer=optimizer, scheduler=scheduler)
    assert state.epoch == 5
    assert state.optimizer_step == 25
    assert state.best_metric == 0.9
    assert state.best_epoch == 4


def test_best_checkpoint_is_written_separately(model, tmp_path):
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=TrainingState(epoch=1),
        metadata=make_metadata(),
        config={},
        is_best=True,
    )
    assert manager.best_path.exists()
    assert manager.latest_path.exists()


# Write safety -----------------------------------------------------------------


def test_previous_checkpoint_is_retained(model, tmp_path):
    """One bad write must not end a run."""
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)

    for epoch in (1, 2):
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            state=TrainingState(epoch=epoch),
            metadata=make_metadata(),
            config={},
        )

    assert (tmp_path / PREVIOUS).exists()
    previous = torch.load(tmp_path / PREVIOUS, map_location="cpu", weights_only=False)
    assert previous["training_state"]["epoch"] == 1


def test_falls_back_to_previous_when_latest_is_corrupt(model, tmp_path):
    """A session killed mid-write must not strand the run."""
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)

    for epoch in (1, 2):
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            state=TrainingState(epoch=epoch),
            metadata=make_metadata(),
            config={},
        )

    manager.latest_path.write_bytes(b"truncated garbage")

    payload = manager.load()
    assert payload["training_state"]["epoch"] == 1


def test_no_temporary_files_remain(model, tmp_path):
    optimizer = build_optimizer(model, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)
    manager = CheckpointManager(tmp_path)
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=TrainingState(),
        metadata=make_metadata(),
        config={},
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError, match="no readable checkpoint"):
        CheckpointManager(tmp_path).load()


# Compatibility ----------------------------------------------------------------


def test_compatible_metadata_passes():
    payload = {"metadata": make_metadata().to_dict()}
    assert validate_resume_compatibility(payload, make_metadata()) == []


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("architecture", "video_swin_tiny", "architecture"),
        ("num_classes", 99, "num_classes"),
        ("label_map_identity", "asl_citizen:9:sha256:zzz", "label_map_identity"),
        ("preprocessing_identity", "preprocessing:train:sha256:zzz", "preprocessing"),
        ("fine_tuning", "head_only", "fine_tuning"),
        ("optimizer_name", "sgd", "optimizer"),
        ("scheduler_name", "linear", "scheduler"),
        # A mirror preserves paths and frame counts, so manifest identity is
        # identical to the source's and cannot catch this. Resuming across the
        # two would mix lossy-different copies of the dataset inside one run.
        ("dataset_substrate", "mirror:short_side=256:crf=20", "dataset_substrate"),
    ],
)
def test_incompatible_checkpoint_is_rejected(field_name, value, expected):
    """Resuming across any of these would produce uninterpretable results."""
    payload = {"metadata": make_metadata(**{field_name: value}).to_dict()}
    with pytest.raises(CheckpointError, match=expected):
        validate_resume_compatibility(payload, make_metadata())


def test_incompatibility_can_be_inspected_without_raising():
    payload = {"metadata": make_metadata(num_classes=99).to_dict()}
    differences = validate_resume_compatibility(payload, make_metadata(), strict=False)
    assert len(differences) == 1
    assert "num_classes" in differences[0]


def test_error_distinguishes_resume_from_transfer():
    """Resume and 'start a new experiment from these weights' are different."""
    payload = {"metadata": make_metadata(num_classes=99).to_dict()}
    with pytest.raises(CheckpointError, match="new experiment"):
        validate_resume_compatibility(payload, make_metadata())


def test_restore_rejects_mismatched_model_state(model, tmp_path, model_config):
    from asl_training.models import ModelConfig, build_model

    other = build_model(ModelConfig.from_dict({**model_config.to_dict(), "num_classes": 9}))
    optimizer = build_optimizer(other, OptimizerConfig())
    scheduler = build_scheduler(optimizer, SchedulerConfig(), total_steps=10)

    manager = CheckpointManager(tmp_path)
    manager.save(
        model=other,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=TrainingState(),
        metadata=make_metadata(num_classes=9),
        config={},
    )

    with pytest.raises((CheckpointError, RuntimeError)):
        restore(manager.load(), model=model)


def test_restore_requires_core_keys(model):
    with pytest.raises(CheckpointError, match="missing required key"):
        restore({"model_state": {}}, model=model)


# Best-checkpoint selection ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "best", "mode", "expected"),
    [
        (0.5, None, "max", True),
        (0.6, 0.5, "max", True),
        (0.4, 0.5, "max", False),
        (0.5, 0.5, "max", False),  # ties keep the earlier checkpoint
        (0.4, 0.5, "min", True),
        (0.6, 0.5, "min", False),
        (0.5, 0.5, "min", False),
    ],
)
def test_improvement_comparison(value, best, mode, expected):
    assert is_better(value, best, mode) is expected


# End-to-end resume ------------------------------------------------------------


def test_interrupted_run_resumes_and_continues(model_config, train_loader, val_loader, tmp_path):
    """The blocking criterion: interrupt a real run, then continue it."""
    from asl_training.models import build_model

    output = tmp_path / "run"

    first = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=1),
        train_loader,
        val_loader,
        output,
        metadata=make_metadata(),
    )
    first.train()

    interrupted_step = first.state.optimizer_step
    interrupted_epoch = first.state.epoch
    assert interrupted_epoch == 1
    assert interrupted_step > 0

    # A fresh process would construct everything again and find the checkpoint.
    second = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=3),
        train_loader,
        val_loader,
        output,
        metadata=make_metadata(),
    )
    assert second.maybe_resume()

    assert second.state.epoch == interrupted_epoch
    assert second.state.optimizer_step == interrupted_step

    second.train()
    assert second.state.epoch == 3
    assert second.state.optimizer_step > interrupted_step


def test_resume_restores_weights_exactly(model_config, train_loader, tmp_path):
    from asl_training.models import build_model

    output = tmp_path / "run"
    first = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=1),
        train_loader,
        None,
        output,
        metadata=make_metadata(),
    )
    first.train()
    expected = {k: v.clone() for k, v in first.model.state_dict().items()}

    second = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=1),
        train_loader,
        None,
        output,
        metadata=make_metadata(),
    )
    second.maybe_resume()

    for key, value in second.model.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6), f"{key} differs after resume"


def test_resume_restores_optimizer_and_scheduler_state(model_config, train_loader, tmp_path):
    """Losing optimizer momentum on resume would perturb the run invisibly."""
    from asl_training.models import build_model

    output = tmp_path / "run"
    first = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=1),
        train_loader,
        None,
        output,
        metadata=make_metadata(),
    )
    first.train()
    expected_lr = first.scheduler.get_last_lr()

    second = Trainer(
        build_model(model_config),
        make_config(tmp_path, epochs=2),
        train_loader,
        None,
        output,
        metadata=make_metadata(),
    )
    second.maybe_resume()

    assert second.scheduler.get_last_lr() == pytest.approx(expected_lr)
    assert second.optimizer.state_dict()["state"], "optimizer moment state was not restored"


def test_fresh_run_does_not_resume(model_config, train_loader, tmp_path):
    from asl_training.models import build_model

    trainer = Trainer(
        build_model(model_config),
        make_config(tmp_path),
        train_loader,
        None,
        tmp_path / "fresh",
        metadata=make_metadata(),
    )
    assert not trainer.maybe_resume()
    assert trainer.state.epoch == 0
