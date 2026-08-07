"""Training layer: supervised fine-tuning orchestration.

    config      resolved run configuration
    optim       optimizer groups and per-optimizer-step scheduling
    checkpoint  atomic checkpointing, resume, compatibility validation
    loop        the training orchestration itself

One path serves both architectures. See docs/TRAINING_CONTRACT.md.
"""

from .checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointMetadata,
    TrainingState,
    is_better,
    restore,
    validate_resume_compatibility,
)
from .config import OptimizerConfig, SchedulerConfig, TrainingConfig
from .loop import (
    EpochResult,
    Trainer,
    default_metrics,
    environment_summary,
    resolve_device,
    resolve_precision,
    seed_everything,
)
from .optim import (
    build_optimizer,
    build_parameter_groups,
    build_scheduler,
    compute_total_steps,
    current_lrs,
)

__all__ = [
    "CheckpointError",
    "CheckpointManager",
    "CheckpointMetadata",
    "EpochResult",
    "OptimizerConfig",
    "SchedulerConfig",
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "build_optimizer",
    "build_parameter_groups",
    "build_scheduler",
    "compute_total_steps",
    "current_lrs",
    "default_metrics",
    "environment_summary",
    "is_better",
    "resolve_device",
    "resolve_precision",
    "restore",
    "seed_everything",
    "validate_resume_compatibility",
]
