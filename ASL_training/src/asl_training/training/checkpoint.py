"""Checkpointing and resume.

On Colab, interruption is the normal execution path rather than a failure mode,
so resume correctness is a blocking requirement. See D-004.

Two properties matter most:

* writes are atomic, so a session killed mid-write cannot leave a truncated file
  as the only resume point
* a previous checkpoint is retained, so one bad write does not end a run

Resume is distinguished from loading weights for a new experiment. Conflating
them silently produces a run whose optimizer state belongs to a different
configuration.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

LATEST = "latest.pt"
BEST = "best.pt"
PREVIOUS = "latest.previous.pt"

CHECKPOINT_VERSION = 1


class CheckpointError(Exception):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


@dataclass
class TrainingState:
    """Counters and best-metric state carried across an interruption."""

    epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    samples_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingState:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class CheckpointMetadata:
    """Identities that determine whether a checkpoint may be resumed.

    A mismatch in any of these means the checkpoint belongs to a different
    experiment, and resuming would produce results that cannot be interpreted.
    """

    architecture: str
    num_classes: int
    label_map_identity: str | None = None
    manifest_identity: str | None = None
    preprocessing_identity: str | None = None
    fine_tuning: str = "full"
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    git_commit: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def capture_rng_state() -> dict[str, Any]:
    """Snapshot random state, so a resumed run continues the same sequence."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot from ``capture_rng_state``.

    Missing entries are skipped rather than raising: a checkpoint written on a
    GPU host must still resume on a CPU host.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
        except (RuntimeError, ValueError) as exc:
            logger.warning("could not restore CUDA RNG state: %s", exc)
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except (ImportError, ValueError) as exc:  # pragma: no cover
            logger.warning("could not restore NumPy RNG state: %s", exc)


def _as_byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.cpu().to(torch.uint8)


class CheckpointManager:
    """Writes and reads training checkpoints for one run.

    Args:
        directory: Where checkpoints live. Created if absent.
        keep_periodic: Retain per-epoch snapshots. Off by default, because they
            multiply storage quickly; see D-007.
    """

    def __init__(self, directory: str | Path, *, keep_periodic: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_periodic = keep_periodic

    @property
    def latest_path(self) -> Path:
        return self.directory / LATEST

    @property
    def best_path(self) -> Path:
        return self.directory / BEST

    def has_checkpoint(self) -> bool:
        return self.latest_path.exists() or (self.directory / PREVIOUS).exists()

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        state: TrainingState,
        metadata: CheckpointMetadata,
        config: dict[str, Any],
        is_best: bool = False,
        include_rng: bool = True,
    ) -> Path:
        """Write a resumable checkpoint.

        Writes to a temporary file and renames, so an interrupted write cannot
        corrupt the resume point. The prior checkpoint is retained first.
        """
        payload = {
            "version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "training_state": state.to_dict(),
            "metadata": metadata.to_dict(),
            "config": config,
        }
        if include_rng:
            payload["rng_state"] = capture_rng_state()

        # Keep the previous checkpoint before replacing it. A single slot is one
        # bad write away from losing the run.
        if self.latest_path.exists():
            shutil.copy2(self.latest_path, self.directory / PREVIOUS)

        self._atomic_write(payload, self.latest_path)

        if is_best:
            self._atomic_write(payload, self.best_path)
            logger.info(
                "New best checkpoint at epoch %d (%s)",
                state.epoch,
                f"{state.best_metric:.4f}" if state.best_metric is not None else "n/a",
            )

        if self.keep_periodic:
            self._atomic_write(payload, self.directory / f"epoch_{state.epoch:04d}.pt")

        return self.latest_path

    def _atomic_write(self, payload: dict[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)

        # Force the bytes to disk before the rename, so a crash cannot leave a
        # renamed file whose contents never landed.
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())

        temporary.replace(path)

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Read a checkpoint, falling back to the retained previous one.

        Raises:
            CheckpointError: If no readable checkpoint exists.
        """
        candidates = [Path(path)] if path else [self.latest_path, self.directory / PREVIOUS]

        errors = []
        for candidate in candidates:
            if not candidate.exists():
                errors.append(f"{candidate.name}: not found")
                continue
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
            except Exception as exc:
                errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
                continue

            if candidate.name == PREVIOUS:
                logger.warning(
                    "latest checkpoint was unreadable; resuming from the retained "
                    "previous checkpoint. Some progress since it was written is lost."
                )
            return payload

        raise CheckpointError(f"no readable checkpoint in {self.directory}: {'; '.join(errors)}")


def validate_resume_compatibility(
    payload: dict[str, Any],
    metadata: CheckpointMetadata,
    *,
    strict: bool = True,
) -> list[str]:
    """Check that a checkpoint may be resumed under the current configuration.

    Args:
        payload: Loaded checkpoint.
        metadata: What the current run expects.
        strict: Raise on mismatch. When false, differences are returned.

    Returns:
        Human-readable differences, empty when compatible.

    Raises:
        CheckpointError: On mismatch when ``strict``.
    """
    recorded = CheckpointMetadata.from_dict(payload.get("metadata", {}))
    differences: list[str] = []

    # Architecture and class count make the weights meaningless if they differ.
    if recorded.architecture != metadata.architecture:
        differences.append(
            f"architecture: checkpoint {recorded.architecture!r}, run {metadata.architecture!r}"
        )
    if recorded.num_classes != metadata.num_classes:
        differences.append(
            f"num_classes: checkpoint {recorded.num_classes}, run {metadata.num_classes}"
        )

    # Identities make the numbers uninterpretable if they differ, even when the
    # weights load cleanly.
    for label, was, now in (
        ("label_map_identity", recorded.label_map_identity, metadata.label_map_identity),
        (
            "preprocessing_identity",
            recorded.preprocessing_identity,
            metadata.preprocessing_identity,
        ),
        ("fine_tuning", recorded.fine_tuning, metadata.fine_tuning),
    ):
        if was and now and was != now:
            differences.append(f"{label}: checkpoint {was!r}, run {now!r}")

    # Optimizer and scheduler state cannot be restored across a type change.
    for label, was, now in (
        ("optimizer", recorded.optimizer_name, metadata.optimizer_name),
        ("scheduler", recorded.scheduler_name, metadata.scheduler_name),
    ):
        if was and now and was != now:
            differences.append(f"{label}: checkpoint {was!r}, run {now!r}")

    if differences and strict:
        listed = "\n  - ".join(differences)
        raise CheckpointError(
            f"checkpoint is incompatible with this run:\n  - {listed}\n"
            f"Resume restores optimizer and scheduler state and assumes the same "
            f"experiment. To start a new experiment from these weights, load model "
            f"state only; see docs/MODEL_CONTRACT.md."
        )

    return differences


def restore(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> TrainingState:
    """Restore a run from a checkpoint.

    Returns the training state to continue from.
    """
    missing = [k for k in ("model_state", "training_state") if k not in payload]
    if missing:
        raise CheckpointError(f"checkpoint is missing required key(s): {', '.join(missing)}")

    result = model.load_state_dict(payload["model_state"], strict=True)
    if getattr(result, "missing_keys", None) or getattr(result, "unexpected_keys", None):
        raise CheckpointError(
            f"model state does not match the constructed model: "
            f"missing={list(result.missing_keys)[:5]}, "
            f"unexpected={list(result.unexpected_keys)[:5]}"
        )

    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])

    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])

    state = TrainingState.from_dict(payload["training_state"])
    logger.info(
        "Resumed at epoch %d, optimizer step %d (best %s)",
        state.epoch,
        state.optimizer_step,
        f"{state.best_metric:.4f}" if state.best_metric is not None else "n/a",
    )
    return state


def is_better(value: float, best: float | None, mode: str) -> bool:
    """Whether ``value`` improves on ``best``.

    Ties do not count as improvements, so the earliest checkpoint achieving a
    given value is retained. That makes best-checkpoint selection deterministic.
    """
    if best is None:
        return True
    return value > best if mode == "max" else value < best
