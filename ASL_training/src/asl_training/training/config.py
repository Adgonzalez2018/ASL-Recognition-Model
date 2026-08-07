"""Training configuration.

Every value that affects what a run means is explicit here and recorded in run
metadata. Nothing is inferred from the environment, and nothing defaults
silently to a different experiment than the one requested.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "linear", "constant")
PRECISIONS = ("fp32", "fp16", "bf16")
LOSSES = ("cross_entropy", "label_smoothing_cross_entropy")

# How a run is labeled. A reduced-data run must never be reported as a full
# baseline, so the kind is part of the run's identity.
RUN_KINDS = ("full", "subset", "smoke", "preflight")

# Direction for the checkpoint-selection metric.
SELECTION_MODES = ("max", "min")


@dataclass
class OptimizerConfig:
    """Optimizer construction.

    Attributes:
        name: One of ``OPTIMIZERS``.
        lr: Base learning rate for the backbone.
        head_lr: Learning rate for the classification head. ``None`` uses ``lr``.
            A freshly initialized head often benefits from a higher rate than a
            pretrained backbone.
        weight_decay: Applied to weights only; norms and biases are excluded.
        betas: AdamW betas.
        momentum: SGD momentum.
    """

    name: str = "adamw"
    lr: float = 1e-4
    head_lr: float | None = None
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.name!r}; supported: {', '.join(OPTIMIZERS)}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.head_lr is not None and self.head_lr <= 0:
            raise ValueError(f"head_lr must be positive, got {self.head_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


@dataclass
class SchedulerConfig:
    """Learning-rate schedule.

    Attributes:
        name: One of ``SCHEDULERS``.
        warmup_steps: Optimizer steps spent warming up from zero.
        warmup_ratio: Warmup as a fraction of total steps, used when
            ``warmup_steps`` is unset.
        min_lr_ratio: Floor as a fraction of the base rate.

    The schedule advances per *optimizer* step, never per micro-batch. With
    gradient accumulation those differ, and conflating them silently compresses
    the schedule.
    """

    name: str = "cosine"
    warmup_steps: int | None = None
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    interval: str = "step"

    def __post_init__(self) -> None:
        if self.name not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.name!r}; supported: {', '.join(SCHEDULERS)}")
        if self.interval != "step":
            raise ValueError(
                f"scheduler interval must be 'step', got {self.interval!r}. Per-epoch "
                f"scheduling is not supported; see docs/TRAINING_CONTRACT.md."
            )
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")

    def resolve_warmup(self, total_steps: int) -> int:
        """Warmup steps for a run of ``total_steps`` optimizer steps."""
        if self.warmup_steps is not None:
            return min(self.warmup_steps, total_steps)
        return min(int(total_steps * self.warmup_ratio), total_steps)


@dataclass
class TrainingConfig:
    """A complete, resolved training run.

    Attributes:
        experiment: Experiment name, shared across related runs.
        run_name: Unique name for this run.
        run_kind: One of ``RUN_KINDS``. A reduced run must not be labeled full.
        seed: Random seed for Python, NumPy, and torch.
        epochs: Maximum epochs.
        max_steps: Optional optimizer-step limit. When both are set, whichever
            is reached first stops the run, and that is recorded.
        batch_size: Physical batch size, per forward and backward pass.
        gradient_accumulation_steps: Micro-batches per optimizer update.
        precision: One of ``PRECISIONS``.
        grad_clip_norm: Gradient-norm clip. ``None`` disables clipping.
        loss: One of ``LOSSES``.
        label_smoothing: Used only by the label-smoothing loss.
        selection_metric: Validation metric driving best-checkpoint selection.
        selection_mode: ``"max"`` or ``"min"`` for that metric.
        validate_every_epochs: Validation cadence.
        checkpoint_every_minutes: Wall-clock checkpoint cadence, in addition to
            per-epoch. Colab interruption is expected, so a bounded amount of
            work should be at risk; see D-004.
        keep_periodic: Retain periodic checkpoints. Off by default because they
            multiply Drive usage quickly; see D-007.
        log_every_steps: Logging cadence, in micro-steps.
        device: ``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``.
        resume_from: Checkpoint to resume. Exact resume only, not transfer.
    """

    experiment: str
    run_name: str
    run_kind: str = "full"
    seed: int = 42

    epochs: int = 10
    max_steps: int | None = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    precision: str = "bf16"
    grad_clip_norm: float | None = 1.0

    loss: str = "cross_entropy"
    label_smoothing: float = 0.0

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    selection_metric: str = "top1_accuracy"
    selection_mode: str = "max"
    validate_every_epochs: int = 1

    checkpoint_every_minutes: float | None = 30.0
    keep_periodic: bool = False

    log_every_steps: int = 50
    device: str = "auto"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment or not self.run_name:
            raise ValueError("experiment and run_name are required for a real run")

        if self.run_kind not in RUN_KINDS:
            raise ValueError(
                f"unknown run_kind {self.run_kind!r}; supported: {', '.join(RUN_KINDS)}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"unknown precision {self.precision!r}; supported: {', '.join(PRECISIONS)}"
            )
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; supported: {', '.join(LOSSES)}")
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {SELECTION_MODES}, got {self.selection_mode!r}"
            )

        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be at least 1, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.validate_every_epochs < 1:
            raise ValueError(
                f"validate_every_epochs must be at least 1, got {self.validate_every_epochs}"
            )

        if self.label_smoothing > 0 and self.loss != "label_smoothing_cross_entropy":
            raise ValueError(
                f"label_smoothing={self.label_smoothing} has no effect with loss "
                f"{self.loss!r}. Set loss='label_smoothing_cross_entropy' or remove it."
            )

    @property
    def effective_batch_size(self) -> int:
        """Samples contributing to one optimizer update.

        Reported alongside the physical batch size, because two runs with the
        same physical size but different accumulation are not comparable.
        """
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def is_reduced(self) -> bool:
        """Whether this run must not be compared against a full baseline."""
        return self.run_kind != "full"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_batch_size"] = self.effective_batch_size
        payload["is_reduced"] = self.is_reduced
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        payload = dict(raw)
        # Derived fields are outputs, not inputs.
        payload.pop("effective_batch_size", None)
        payload.pop("is_reduced", None)

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"unknown training config key(s): {', '.join(sorted(unknown))}; "
                f"supported: {', '.join(sorted(known))}"
            )

        if isinstance(payload.get("optimizer"), dict):
            payload["optimizer"] = OptimizerConfig(**payload["optimizer"])
        if isinstance(payload.get("scheduler"), dict):
            payload["scheduler"] = SchedulerConfig(**payload["scheduler"])
        if isinstance(payload.get("optimizer"), OptimizerConfig) and isinstance(
            payload["optimizer"].betas, list
        ):
            payload["optimizer"].betas = tuple(payload["optimizer"].betas)

        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainingConfig:
        path = Path(path)
        with path.open() as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict) or "training" not in raw:
            raise ValueError(f"{path}: missing required top-level 'training' key")

        merged = {**raw["training"], **overrides}
        try:
            return cls.from_dict(merged)
        except (ValueError, TypeError) as exc:
            raise type(exc)(f"{path}: {exc}") from exc
