"""Optimizer and scheduler construction.

Two properties are enforced rather than assumed:

* every trainable parameter belongs to exactly one optimizer group
* the schedule advances per optimizer step, never per micro-batch

The second matters under gradient accumulation, where conflating the two
compresses the schedule by the accumulation factor without any error.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..models.base import BaseVideoClassifier
from .config import OptimizerConfig, SchedulerConfig

logger = logging.getLogger(__name__)


def _excluded_from_weight_decay(name: str, param: torch.Tensor) -> bool:
    """Whether a parameter should be exempt from weight decay.

    Biases and normalization scales are conventionally exempt: decaying them
    fights the normalization rather than regularizing the representation.
    """
    if param.ndim <= 1:
        return True
    lowered = name.lower()
    return any(token in lowered for token in ("bias", "norm", "bn", "ln", "embedding"))


def build_parameter_groups(
    model: BaseVideoClassifier,
    config: OptimizerConfig,
) -> list[dict[str, Any]]:
    """Split trainable parameters into optimizer groups.

    Four groups: head and backbone, each split by whether weight decay applies.
    A separate head learning rate is supported because a freshly initialized head
    often wants a higher rate than a pretrained backbone.

    Raises:
        ValueError: If no parameter is trainable, or if any trainable parameter
            would be left out of every group.
    """
    head_ids = {id(p) for p in model.classification_head().parameters()}
    head_lr = config.head_lr if config.head_lr is not None else config.lr

    buckets: dict[str, dict[str, Any]] = {
        "backbone_decay": {"params": [], "lr": config.lr, "weight_decay": config.weight_decay},
        "backbone_no_decay": {"params": [], "lr": config.lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": head_lr, "weight_decay": config.weight_decay},
        "head_no_decay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }

    assigned = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        section = "head" if id(param) in head_ids else "backbone"
        decay = "no_decay" if _excluded_from_weight_decay(name, param) else "decay"
        buckets[f"{section}_{decay}"]["params"].append(param)
        assigned += 1

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError(
            "no trainable parameters; the model was constructed with every parameter frozen"
        )
    if assigned != trainable:
        raise ValueError(
            f"{trainable - assigned} trainable parameter(s) were not assigned to an "
            f"optimizer group. Every trainable parameter must belong to a group or be "
            f"explicitly frozen."
        )

    groups = [{"name": name, **bucket} for name, bucket in buckets.items() if bucket["params"]]

    logger.info(
        "Optimizer groups: %s",
        ", ".join(
            f"{g['name']}={sum(p.numel() for p in g['params']):,} "
            f"(lr={g['lr']:g}, wd={g['weight_decay']:g})"
            for g in groups
        ),
    )
    return groups


def build_optimizer(model: BaseVideoClassifier, config: OptimizerConfig) -> Optimizer:
    """Construct the configured optimizer over grouped parameters."""
    groups = build_parameter_groups(model, config)

    if config.name == "adamw":
        return torch.optim.AdamW(groups, lr=config.lr, betas=tuple(config.betas))
    if config.name == "sgd":
        return torch.optim.SGD(groups, lr=config.lr, momentum=config.momentum)

    raise ValueError(f"unknown optimizer {config.name!r}")  # pragma: no cover


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> LambdaLR:
    """Construct a per-optimizer-step schedule.

    Args:
        optimizer: The optimizer to schedule.
        config: Schedule settings.
        total_steps: Total *optimizer* steps for the run, not micro-batches.

    The returned scheduler must be stepped only when the optimizer steps.
    """
    if total_steps < 1:
        raise ValueError(
            f"total_steps must be at least 1, got {total_steps}. A run with no "
            f"optimizer steps usually means the dataset is smaller than the "
            f"effective batch size."
        )

    warmup = config.resolve_warmup(total_steps)
    floor = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        # Warm up linearly from zero, so a freshly initialized head does not
        # take a large first step against pretrained weights.
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        if config.name == "constant":
            return 1.0

        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)

        if config.name == "cosine":
            decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # linear
            decayed = 1.0 - progress

        return floor + (1.0 - floor) * decayed

    logger.info(
        "Scheduler: %s over %d optimizer steps, %d warmup, floor %.3g",
        config.name,
        total_steps,
        warmup,
        floor,
    )
    return LambdaLR(optimizer, lr_lambda)


def compute_total_steps(
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_steps: int | None = None,
    drop_last: bool = False,
) -> int:
    """Total optimizer steps for a run.

    The schedule must span the run's real length, so this accounts for
    accumulation and for whether a partial final batch is kept.
    """
    if dataset_size < 1:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")

    if drop_last:
        batches_per_epoch = dataset_size // batch_size
    else:
        batches_per_epoch = math.ceil(dataset_size / batch_size)

    if batches_per_epoch < 1:
        raise ValueError(
            f"dataset of {dataset_size} sample(s) yields no batches at batch_size "
            f"{batch_size} with drop_last={drop_last}"
        )

    # A partial accumulation window at the end of an epoch still produces an
    # optimizer step, so it is counted rather than dropped.
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    total = steps_per_epoch * epochs

    if max_steps is not None:
        total = min(total, max_steps)
    return max(total, 1)


def current_lrs(optimizer: Optimizer) -> dict[str, float]:
    """Current learning rate per named group, for logging."""
    return {
        group.get("name", f"group_{index}"): group["lr"]
        for index, group in enumerate(optimizer.param_groups)
    }
