"""Training orchestration.

One supervised fine-tuning path shared by both architectures. The loop owns
optimization, validation scheduling, checkpoint selection, and run metadata; it
does not define metrics, transforms, or dataset semantics.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.base import BaseVideoClassifier
from .checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    TrainingState,
    is_better,
    restore,
    validate_resume_compatibility,
)
from .config import TrainingConfig
from .optim import build_optimizer, build_scheduler, compute_total_steps, current_lrs

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the configured device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> tuple[bool, torch.dtype | None]:
    """Resolve autocast settings, downgrading unsupported requests loudly.

    Silently training in a different precision than requested would make the run
    metadata wrong, so every downgrade is logged and recorded.
    """
    if precision == "fp32":
        return False, None

    if device.type != "cuda":
        logger.warning(
            "precision %s requested but device is %s; running in fp32",
            precision,
            device.type,
        )
        return False, None

    if precision == "bf16":
        if _has_native_bf16(device):
            return True, torch.bfloat16
        logger.warning(
            "bf16 is not supported natively on this GPU (compute capability %d.%d); "
            "falling back to fp16, which uses tensor cores. torch.cuda."
            "is_bf16_supported() reports True here via emulation, but emulated bf16 "
            "runs without tensor-core acceleration and is far slower.",
            *torch.cuda.get_device_capability(device),
        )
        return True, torch.float16

    return True, torch.float16


def _has_native_bf16(device: torch.device) -> bool:
    """Whether the GPU supports bf16 in hardware rather than by emulation.

    torch.cuda.is_bf16_supported() defaults to including_emulation=True, so it
    returns True on pre-Ampere cards such as the T4. Emulated bf16 bypasses the
    tensor cores entirely and runs roughly at fp32 speed, which is a large and
    silent throughput loss. Native bf16 requires compute capability 8.0.
    """
    try:
        return torch.cuda.get_device_capability(device)[0] >= 8
    except (RuntimeError, AssertionError):  # pragma: no cover
        return False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


def git_commit() -> str | None:
    """Current commit, when the working tree is a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def environment_summary(device: torch.device) -> dict[str, Any]:
    """Hardware and dependency versions, recorded per run.

    Colab may assign a different GPU on resume, so this is captured per session
    rather than once per run.
    """
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torchvision

        summary["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import transformers

        summary["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass

    if device.type == "cuda":
        summary["cuda"] = torch.version.cuda
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 2
        )
    return summary


@dataclass
class EpochResult:
    """Outcome of one epoch."""

    epoch: int
    train_loss: float
    validation: dict[str, float] = field(default_factory=dict)
    learning_rates: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    optimizer_steps: int = 0
    skipped_batches: int = 0
    non_finite_losses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "validation": {k: round(v, 6) for k, v in self.validation.items()},
            "learning_rates": self.learning_rates,
            "duration_seconds": round(self.duration_seconds, 2),
            "optimizer_steps": self.optimizer_steps,
            "skipped_batches": self.skipped_batches,
            "non_finite_losses": self.non_finite_losses,
        }


class Trainer:
    """Supervised fine-tuning for one run.

    Args:
        model: A constructed classifier.
        config: Resolved training configuration.
        train_loader: Training batches.
        val_loader: Validation batches. Must never be the test split.
        output_dir: Run directory. Checkpoints, logs, and metadata land here.
        metadata: Identities recorded with every checkpoint.
        metric_fn: Maps logits and labels to validation metrics. Defaults to
            top-1 accuracy; the evaluation layer supplies the real set in
            Phase 4.
    """

    def __init__(
        self,
        model: BaseVideoClassifier,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        output_dir: str | Path,
        *,
        metadata: CheckpointMetadata | None = None,
        metric_fn: Any = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(config.device)
        self.use_autocast, self.autocast_dtype = resolve_precision(config.precision, self.device)

        seed_everything(config.seed)

        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, config.optimizer)

        dataset_size = len(train_loader.dataset)
        self.total_steps = compute_total_steps(
            dataset_size=dataset_size,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            epochs=config.epochs,
            max_steps=config.max_steps,
            drop_last=bool(getattr(train_loader, "drop_last", False)),
        )
        self.scheduler = build_scheduler(self.optimizer, config.scheduler, self.total_steps)

        # GradScaler is fp16-only. bf16 has enough exponent range that loss
        # scaling is unnecessary, and enabling it would be a silent no-op.
        self.scaler = (
            torch.amp.GradScaler(self.device.type) if self.autocast_dtype is torch.float16 else None
        )

        self.criterion = self._build_criterion()
        self.metric_fn = metric_fn or default_metrics

        self.checkpoints = CheckpointManager(
            self.output_dir / "checkpoints", keep_periodic=config.keep_periodic
        )
        self.metadata = metadata or CheckpointMetadata(
            architecture=model.config.architecture,
            num_classes=model.num_classes,
            fine_tuning=model.config.fine_tuning,
            optimizer_name=config.optimizer.name,
            scheduler_name=config.scheduler.name,
            git_commit=git_commit(),
            environment=environment_summary(self.device),
        )

        self.state = TrainingState()
        self.history: list[EpochResult] = []
        self._last_checkpoint_time = time.monotonic()

        if config.is_reduced:
            logger.warning(
                "run_kind=%r: this is not a full baseline and must not be compared "
                "against one without qualification",
                config.run_kind,
            )

    def _build_criterion(self) -> nn.Module:
        if self.config.loss == "label_smoothing_cross_entropy":
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        return nn.CrossEntropyLoss()

    # Resume -------------------------------------------------------------------

    def maybe_resume(self) -> bool:
        """Resume when a checkpoint exists, or when one was explicitly named.

        Returns whether a checkpoint was loaded.
        """
        explicit = self.config.resume_from
        if not explicit and not self.checkpoints.has_checkpoint():
            return False

        payload = self.checkpoints.load(explicit)
        validate_resume_compatibility(payload, self.metadata, strict=True)

        self.state = restore(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

        # Hardware can change between Colab sessions, so the environment is
        # recorded per session rather than once per run.
        self.metadata.environment = environment_summary(self.device)
        return True

    # Training -----------------------------------------------------------------

    def train(self) -> list[EpochResult]:
        """Run training to completion, or until the step limit is reached."""
        resumed = self.maybe_resume()

        logger.info(
            "%s run %r: %d epoch(s), %d optimizer steps, batch %d x %d accumulation "
            "= %d effective, %s on %s%s",
            self.config.run_kind,
            self.config.run_name,
            self.config.epochs,
            self.total_steps,
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.config.effective_batch_size,
            self.config.precision,
            self.device,
            " (resumed)" if resumed else "",
        )
        self._write_run_metadata()

        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            if self.config.max_steps and self.state.optimizer_step >= self.config.max_steps:
                logger.info("reached max_steps=%d; stopping", self.config.max_steps)
                break

            result = self._train_epoch(epoch)

            if self.val_loader is not None and (epoch + 1) % self.config.validate_every_epochs == 0:
                result.validation = self.validate()

            self.state.epoch = epoch + 1
            self._finish_epoch(result)

        self._write_run_metadata()
        return self.history

    def _train_epoch(self, epoch: int) -> EpochResult:
        self.model.train()
        started = time.monotonic()

        total_loss = 0.0
        counted = 0
        skipped = 0
        non_finite = 0
        steps_this_epoch = 0

        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.gradient_accumulation_steps
        batches = len(self.train_loader)

        for batch_index, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.use_autocast,
            ):
                output = self.model(pixel_values)
                loss = self.criterion(output.logits, labels)

            if not torch.isfinite(loss):
                # Skipping the batch keeps a single bad step from poisoning the
                # weights, but it is counted and reported rather than hidden.
                non_finite += 1
                skipped += 1
                logger.warning(
                    "non-finite loss at epoch %d batch %d; skipping this batch (%d so far)",
                    epoch,
                    batch_index,
                    non_finite,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            total_loss += float(loss.detach())
            counted += 1

            # Scale so accumulated gradients average rather than sum. Scaling
            # twice is a classic silent error, so it happens exactly here.
            scaled = loss / accumulation

            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            self.state.micro_step += 1
            self.state.samples_seen += pixel_values.shape[0]

            is_boundary = (batch_index + 1) % accumulation == 0
            is_last = batch_index + 1 == batches

            # A partial accumulation window at the end of an epoch still
            # produces an optimizer step rather than being discarded.
            if is_boundary or is_last:
                self._optimizer_step()
                steps_this_epoch += 1

                if self.config.max_steps and self.state.optimizer_step >= self.config.max_steps:
                    logger.info("reached max_steps during epoch %d", epoch)
                    break

            if self.state.micro_step % self.config.log_every_steps == 0:
                logger.info(
                    "epoch %d | micro-step %d | optimizer step %d | loss %.4f | lr %.3g",
                    epoch,
                    self.state.micro_step,
                    self.state.optimizer_step,
                    total_loss / max(counted, 1),
                    next(iter(current_lrs(self.optimizer).values())),
                )

            self._maybe_checkpoint_on_time()

        return EpochResult(
            epoch=epoch,
            train_loss=total_loss / max(counted, 1),
            learning_rates=current_lrs(self.optimizer),
            duration_seconds=time.monotonic() - started,
            optimizer_steps=steps_this_epoch,
            skipped_batches=skipped,
            non_finite_losses=non_finite,
        )

    def _optimizer_step(self) -> None:
        """One optimizer update: unscale, clip, step, schedule, zero."""
        if self.config.grad_clip_norm is not None:
            # Unscale before clipping, or the clip threshold would apply to
            # scaled gradients and mean nothing.
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # The schedule advances per optimizer step, never per micro-batch.
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimizer_step += 1

    # Validation ---------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on the validation split without touching weights."""
        if self.val_loader is None:
            return {}

        was_training = self.model.training
        self.model.eval()

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        batches = 0

        for batch in self.val_loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.use_autocast,
            ):
                output = self.model(pixel_values)
                loss = self.criterion(output.logits, labels)

            total_loss += float(loss)
            batches += 1
            all_logits.append(output.logits.float().cpu())
            all_labels.append(labels.cpu())

        if was_training:
            self.model.train()

        if not batches:
            return {}

        metrics = self.metric_fn(torch.cat(all_logits), torch.cat(all_labels))
        metrics["loss"] = total_loss / batches
        return metrics

    # Checkpointing ------------------------------------------------------------

    def _finish_epoch(self, result: EpochResult) -> None:
        self.history.append(result)

        # A selection metric the metric function never produces would mean no
        # checkpoint is ever marked best, and the run would finish looking
        # successful with nothing selected.
        if result.validation and self.config.selection_metric not in result.validation:
            raise ValueError(
                f"selection_metric {self.config.selection_metric!r} is not produced "
                f"by the validation metrics. Available: "
                f"{', '.join(sorted(result.validation))}. No checkpoint could ever "
                f"be selected as best."
            )

        selected = result.validation.get(self.config.selection_metric)
        improved = False
        if selected is not None and is_better(
            selected, self.state.best_metric, self.config.selection_mode
        ):
            self.state.best_metric = selected
            self.state.best_epoch = result.epoch
            improved = True

        self._save(is_best=improved)
        self._last_checkpoint_time = time.monotonic()

        logger.info(
            "epoch %d done in %.1fs | train loss %.4f%s",
            result.epoch,
            result.duration_seconds,
            result.train_loss,
            (
                " | " + ", ".join(f"{k} {v:.4f}" for k, v in result.validation.items())
                if result.validation
                else ""
            ),
        )
        self._write_history()

    def _maybe_checkpoint_on_time(self) -> None:
        """Checkpoint on wall-clock cadence.

        Per-epoch alone is not enough on Colab: an epoch on the full dataset can
        exceed the time a session survives, so an interruption could lose all of
        it. See D-004.
        """
        interval = self.config.checkpoint_every_minutes
        if not interval:
            return
        if time.monotonic() - self._last_checkpoint_time < interval * 60:
            return

        self._save(is_best=False)
        self._last_checkpoint_time = time.monotonic()
        logger.info("periodic checkpoint at optimizer step %d", self.state.optimizer_step)

    def _save(self, *, is_best: bool) -> None:
        self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.state,
            metadata=self.metadata,
            config=self.config.to_dict(),
            is_best=is_best,
        )

    # Run records --------------------------------------------------------------

    def _write_run_metadata(self) -> None:
        payload = {
            "experiment": self.config.experiment,
            "run_name": self.config.run_name,
            "run_kind": self.config.run_kind,
            "is_reduced": self.config.is_reduced,
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "total_optimizer_steps_planned": self.total_steps,
            "physical_batch_size": self.config.batch_size,
            "effective_batch_size": self.config.effective_batch_size,
            "precision_requested": self.config.precision,
            "precision_active": (
                str(self.autocast_dtype).replace("torch.", "") if self.use_autocast else "fp32"
            ),
            "train_samples": len(self.train_loader.dataset),
            "val_samples": len(self.val_loader.dataset) if self.val_loader else 0,
            # Promoted out of metadata because it is the field most likely to be
            # read on its own: results from different substrates are not
            # comparable, and manifest identity cannot tell them apart. See D-011.
            "dataset_substrate": self.metadata.dataset_substrate,
            "environment": environment_summary(self.device),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps([r.to_dict() for r in self.history], indent=2) + "\n"
        )


def default_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Validation metrics available for checkpoint selection.

    A restricted set, which docs/EVALUATION_CONTRACT.md permits for in-training
    validation: aggregate values only, without the per-class, per-signer, or
    confusion breakdowns. Full evaluation runs separately from a checkpoint.

    Macro F1 and mean per-class accuracy are included because ASL Citizen's
    class support is uneven, and top-1 can rise while the tail is neglected.
    """
    from ..evaluation.metrics import (
        macro_f1,
        mean_per_class_accuracy,
        negative_log_likelihood,
        per_class_metrics,
        top1_accuracy,
        top_k_accuracy,
    )

    if logits.numel() == 0:
        return {}

    per_class = per_class_metrics(logits, labels, logits.shape[1])
    macro, _ = macro_f1(per_class)
    balanced, _ = mean_per_class_accuracy(per_class)

    metrics = {
        "top1_accuracy": top1_accuracy(logits, labels),
        "macro_f1": macro,
        "mean_per_class_accuracy": balanced,
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
    }

    top5 = top_k_accuracy(logits, labels, 5)
    if top5 is not None:
        metrics["top5_accuracy"] = top5
    return metrics
