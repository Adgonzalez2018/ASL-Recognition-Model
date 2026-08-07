"""Optimizer groups, scheduling, and training configuration.

Two properties are contractual: every trainable parameter belongs to a group,
and the schedule advances per optimizer step rather than per micro-batch.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import pytest

from asl_training.models import ModelConfig, build_model
from asl_training.training import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    build_optimizer,
    build_parameter_groups,
    build_scheduler,
    compute_total_steps,
)

# Parameter groups -------------------------------------------------------------


def test_every_trainable_parameter_is_assigned(model):
    groups = build_parameter_groups(model, OptimizerConfig())

    assigned = sum(len(g["params"]) for g in groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert assigned == trainable


def test_biases_and_norms_are_excluded_from_weight_decay(model):
    groups = build_parameter_groups(model, OptimizerConfig(weight_decay=0.05))
    by_name = {g["name"]: g for g in groups}

    assert by_name["backbone_no_decay"]["weight_decay"] == 0.0
    assert by_name["backbone_decay"]["weight_decay"] == 0.05
    assert by_name["backbone_no_decay"]["params"], "no parameters exempted from decay"


def test_head_can_take_a_separate_learning_rate(model):
    groups = build_parameter_groups(model, OptimizerConfig(lr=1e-4, head_lr=1e-3))
    by_name = {g["name"]: g for g in groups}

    assert by_name["head_decay"]["lr"] == 1e-3
    assert by_name["backbone_decay"]["lr"] == 1e-4


def test_head_defaults_to_the_base_learning_rate(model):
    groups = build_parameter_groups(model, OptimizerConfig(lr=5e-5))
    assert all(g["lr"] == 5e-5 for g in groups)


def test_head_only_training_assigns_only_head_parameters(model_config):
    config = ModelConfig.from_dict({**model_config.to_dict(), "fine_tuning": "head_only"})
    model = build_model(config)

    groups = build_parameter_groups(model, OptimizerConfig())
    assigned = sum(len(g["params"]) for g in groups)
    head = sum(1 for _ in model.classification_head().parameters())
    assert assigned == head


def test_frozen_model_is_rejected(model):
    for param in model.parameters():
        param.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_parameter_groups(model, OptimizerConfig())


@pytest.mark.parametrize("name", ["adamw", "sgd"])
def test_supported_optimizers_construct(model, name):
    optimizer = build_optimizer(model, OptimizerConfig(name=name))
    assert optimizer.param_groups


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        OptimizerConfig(name="lamb")


# Scheduling -------------------------------------------------------------------


def test_warmup_rises_then_cosine_decays(model):
    optimizer = build_optimizer(model, OptimizerConfig(lr=1.0))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(name="cosine", warmup_steps=10), total_steps=100
    )

    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[0] < rates[5] < rates[9]  # warming up
    assert rates[10] == pytest.approx(1.0, abs=1e-6)  # peak at the base rate
    assert rates[50] < rates[10]  # decaying
    assert rates[99] < rates[50]


def test_cosine_decays_to_the_configured_floor(model):
    optimizer = build_optimizer(model, OptimizerConfig(lr=1.0))
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", warmup_steps=0, min_lr_ratio=0.1),
        total_steps=50,
    )
    for _ in range(50):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)


def test_constant_schedule_holds_after_warmup(model):
    optimizer = build_optimizer(model, OptimizerConfig(lr=1.0))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(name="constant", warmup_steps=5), total_steps=50
    )
    for _ in range(20):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_warmup_ratio_resolves_against_total_steps():
    assert SchedulerConfig(warmup_ratio=0.1).resolve_warmup(1000) == 100


def test_explicit_warmup_steps_win_over_ratio():
    assert SchedulerConfig(warmup_steps=7, warmup_ratio=0.5).resolve_warmup(1000) == 7


def test_warmup_cannot_exceed_the_run():
    assert SchedulerConfig(warmup_steps=500).resolve_warmup(100) == 100


def test_per_epoch_scheduling_is_rejected():
    """Step and epoch intervals must not be ambiguous."""
    with pytest.raises(ValueError, match="must be 'step'"):
        SchedulerConfig(interval="epoch")


def test_zero_total_steps_is_rejected(model):
    optimizer = build_optimizer(model, OptimizerConfig())
    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        build_scheduler(optimizer, SchedulerConfig(), total_steps=0)


# Step accounting --------------------------------------------------------------


def test_total_steps_accounts_for_accumulation():
    """Accumulation reduces optimizer steps; the schedule must match."""
    without = compute_total_steps(100, batch_size=10, gradient_accumulation_steps=1, epochs=1)
    with_accumulation = compute_total_steps(
        100, batch_size=10, gradient_accumulation_steps=2, epochs=1
    )
    assert without == 10
    assert with_accumulation == 5


def test_total_steps_counts_a_partial_final_batch():
    assert compute_total_steps(105, batch_size=10, gradient_accumulation_steps=1, epochs=1) == 11


def test_drop_last_discards_the_partial_batch():
    assert (
        compute_total_steps(
            105, batch_size=10, gradient_accumulation_steps=1, epochs=1, drop_last=True
        )
        == 10
    )


def test_total_steps_counts_a_partial_accumulation_window():
    """A partial window at epoch end still produces an optimizer step."""
    assert compute_total_steps(100, batch_size=10, gradient_accumulation_steps=3, epochs=1) == 4


def test_max_steps_caps_the_total():
    assert (
        compute_total_steps(
            1000, batch_size=10, gradient_accumulation_steps=1, epochs=10, max_steps=50
        )
        == 50
    )


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="dataset_size must be positive"):
        compute_total_steps(0, batch_size=10, gradient_accumulation_steps=1, epochs=1)


def test_dataset_smaller_than_batch_still_yields_a_step():
    assert compute_total_steps(3, batch_size=10, gradient_accumulation_steps=1, epochs=1) == 1


# Training configuration -------------------------------------------------------


def base_config(**overrides) -> TrainingConfig:
    defaults = {"experiment": "exp", "run_name": "run"}
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_effective_batch_size_is_reported():
    config = base_config(batch_size=4, gradient_accumulation_steps=8)
    assert config.effective_batch_size == 32


def test_reduced_runs_are_flagged():
    assert base_config(run_kind="full").is_reduced is False
    for kind in ("smoke", "subset", "preflight"):
        assert base_config(run_kind=kind).is_reduced is True


def test_unknown_run_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown run_kind"):
        base_config(run_kind="production")


@pytest.mark.parametrize(
    ("field_name", "value", "pattern"),
    [
        ("precision", "int8", "unknown precision"),
        ("loss", "hinge", "unknown loss"),
        ("selection_mode", "highest", "selection_mode"),
        ("epochs", 0, "epochs must be at least 1"),
        ("batch_size", 0, "batch_size must be at least 1"),
        ("gradient_accumulation_steps", 0, "gradient_accumulation_steps"),
        ("grad_clip_norm", -1.0, "grad_clip_norm must be positive"),
        ("max_steps", 0, "max_steps must be at least 1"),
        ("validate_every_epochs", 0, "validate_every_epochs"),
    ],
)
def test_invalid_configuration_is_rejected(field_name, value, pattern):
    with pytest.raises(ValueError, match=pattern):
        base_config(**{field_name: value})


def test_experiment_and_run_name_are_required():
    with pytest.raises(ValueError, match="experiment and run_name are required"):
        TrainingConfig(experiment="", run_name="run")


def test_label_smoothing_without_the_matching_loss_is_rejected():
    """A setting with no effect must not pass silently."""
    with pytest.raises(ValueError, match="has no effect"):
        base_config(label_smoothing=0.1, loss="cross_entropy")


def test_label_smoothing_with_the_matching_loss_is_accepted():
    config = base_config(label_smoothing=0.1, loss="label_smoothing_cross_entropy")
    assert config.label_smoothing == 0.1


def test_config_round_trips():
    original = base_config(batch_size=16, gradient_accumulation_steps=2, precision="fp16")
    restored = TrainingConfig.from_dict(original.to_dict())
    assert restored.batch_size == 16
    assert restored.effective_batch_size == 32
    assert restored.precision == "fp16"


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown training config key"):
        TrainingConfig.from_dict({"experiment": "e", "run_name": "r", "learning_rate": 1e-4})


def test_nested_configs_load_from_mappings():
    config = TrainingConfig.from_dict(
        {
            "experiment": "e",
            "run_name": "r",
            "optimizer": {"name": "adamw", "lr": 3e-4, "head_lr": 1e-3},
            "scheduler": {"name": "linear", "warmup_ratio": 0.1},
        }
    )
    assert config.optimizer.lr == 3e-4
    assert config.optimizer.head_lr == 1e-3
    assert config.scheduler.name == "linear"


def test_config_loads_from_yaml(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text(
        "training:\n"
        "  experiment: exp-001\n"
        "  run_name: videomae-baseline\n"
        "  epochs: 20\n"
        "  optimizer:\n"
        "    lr: 0.0001\n"
    )
    config = TrainingConfig.from_yaml(path)
    assert config.experiment == "exp-001"
    assert config.epochs == 20
    assert config.optimizer.lr == 1e-4


def test_yaml_requires_the_training_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: e\n")
    with pytest.raises(ValueError, match="missing required top-level 'training' key"):
        TrainingConfig.from_yaml(path)


# Shipped configuration --------------------------------------------------------


def test_shipped_baseline_config_is_valid():
    """The committed baseline must parse and carry the intended policy."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "training" / "baseline.yaml"
    config = TrainingConfig.from_yaml(path, experiment="exp", run_name="run")

    assert config.run_kind == "full"
    assert config.optimizer.name == "adamw"
    assert config.scheduler.interval == "step"
    assert config.checkpoint_every_minutes, "Colab runs need wall-clock checkpointing"


def test_shipped_selection_metric_is_actually_produced():
    """A selection metric the trainer cannot compute would select nothing."""
    from pathlib import Path

    import torch

    from asl_training.training import default_metrics

    path = Path(__file__).resolve().parents[2] / "configs" / "training" / "baseline.yaml"
    config = TrainingConfig.from_yaml(path, experiment="exp", run_name="run")

    available = default_metrics(torch.randn(20, 10), torch.randint(0, 10, (20,)))
    assert config.selection_metric in available, (
        f"baseline.yaml selects on {config.selection_metric!r}, which the trainer "
        f"does not produce. Available: {sorted(available)}"
    )
