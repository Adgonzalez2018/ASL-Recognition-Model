"""Training preflight.

Measures what a real run will cost before committing to it: throughput, memory,
epoch duration, and whether the data loader or the GPU is the limiting factor.

A preflight is not an experiment. It uses a deliberately limited step count and
its output must never be reported as a result.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..models.base import BaseVideoClassifier
from .checkpoint import CheckpointManager, CheckpointMetadata, TrainingState, restore
from .config import TrainingConfig
from .loop import environment_summary, resolve_device, resolve_precision
from .optim import build_optimizer, build_scheduler, compute_total_steps

logger = logging.getLogger(__name__)

# Steps discarded before timing. The first pass allocates caching-allocator
# blocks, compiles kernels, and fills the loader's prefetch queue, so timing it
# would report a number no later step reproduces.
WARMUP_STEPS = 3


@dataclass
class Timing:
    """Wall-clock accounting for one measured step."""

    data_seconds: float
    compute_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.data_seconds + self.compute_seconds


@dataclass
class PreflightReport:
    """What a real run will cost, measured rather than estimated."""

    config: dict[str, Any]
    environment: dict[str, Any]
    checks: dict[str, str] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "preflight",
            "not_an_experiment": (
                "A preflight uses a limited step count to measure cost. Its numbers "
                "describe throughput and memory, never model quality."
            ),
            "config": self.config,
            "environment": self.environment,
            "checks": self.checks,
            "measurements": self.measurements,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        m = self.measurements
        lines = [
            "Preflight",
            f"  device            {self.environment.get('device')}",
        ]
        if self.environment.get("gpu"):
            lines.append(
                f"  gpu               {self.environment['gpu']} "
                f"({self.environment.get('gpu_memory_gb')} GB)"
            )
        lines += [
            f"  precision         {m.get('precision_active')} "
            f"(requested {m.get('precision_requested')})",
            f"  batch size        {m.get('physical_batch_size')} physical, "
            f"{m.get('effective_batch_size')} effective",
            f"  workers           {m.get('num_workers')}",
            "",
            f"  throughput        {m.get('videos_per_second', 0):.1f} videos/s",
            f"  per optimizer step{m.get('seconds_per_optimizer_step', 0):>8.3f} s",
            f"    data loading    {m.get('data_seconds_per_step', 0):>8.3f} s "
            f"({m.get('data_share_percent', 0):.0f}%)",
            f"    compute         {m.get('compute_seconds_per_step', 0):>8.3f} s "
            f"({100 - m.get('data_share_percent', 0):.0f}%)",
            f"  bottleneck        {m.get('bottleneck')}",
        ]

        if m.get("peak_memory_gb") is not None:
            lines.append(
                f"  peak memory       {m['peak_memory_gb']:.2f} GB of "
                f"{self.environment.get('gpu_memory_gb')} GB"
            )
        if m.get("checkpoint_mb") is not None:
            lines.append(f"  checkpoint        {_size(m['checkpoint_mb'])}")

        if m.get("estimated_epoch_minutes") is not None:
            lines += [
                "",
                f"  train samples     {m.get('train_samples'):,}",
                f"  optimizer steps   {m.get('steps_per_epoch'):,} per epoch, "
                f"{m.get('total_optimizer_steps'):,} total",
                f"  epoch             ~{_duration(m['estimated_epoch_seconds'])}",
                f"  full run          ~{_duration(m.get('estimated_run_seconds', 0))} "
                f"over {m.get('epochs')} epochs",
            ]
            if m.get("estimated_sessions") is not None:
                lines.append(f"  colab sessions    ~{m['estimated_sessions']:.1f} at 12 h each")

        if self.warnings:
            lines.append("")
            lines.append(f"  {len(self.warnings)} warning(s):")
            lines.extend(f"    - {w}" for w in self.warnings)

        lines.append("")
        lines.append("  This is a cost measurement, not an experiment.")
        return "\n".join(lines)


def _duration(seconds: float) -> str:
    """Format a duration at a scale a reader can act on."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _size(megabytes: float) -> str:
    if megabytes < 1024:
        return f"{megabytes:.0f} MB"
    return f"{megabytes / 1024:.2f} GB"


def run_preflight(
    model: BaseVideoClassifier,
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    steps: int = 20,
    metadata: CheckpointMetadata | None = None,
    checkpoint_dir: Any = None,
) -> PreflightReport:
    """Measure the cost of a training run without performing one.

    Exercises the full path — forward, backward, optimizer step, scheduler step,
    checkpoint save, checkpoint load, and a short validation pass — while timing
    data loading separately from compute.

    Args:
        model: A constructed classifier.
        config: The configuration the real run would use.
        train_loader: Training batches. Timing is only meaningful if this is
            configured exactly as the real run would configure it.
        val_loader: Optional validation batches for a short pass.
        steps: Measured optimizer steps, after ``WARMUP_STEPS`` discarded.
        metadata: Identities, for the checkpoint round trip.
        checkpoint_dir: Where to write the throwaway checkpoint.

    Returns:
        The report. It measures cost, never quality.
    """
    device = resolve_device(config.device)
    use_autocast, autocast_dtype = resolve_precision(config.precision, device)

    environment = environment_summary(device)
    report = PreflightReport(config=config.to_dict(), environment=environment)
    warnings = report.warnings

    model = model.to(device)
    optimizer = build_optimizer(model, config.optimizer)

    train_samples = len(train_loader.dataset)  # type: ignore[arg-type]
    total_steps = compute_total_steps(
        dataset_size=train_samples,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        epochs=config.epochs,
        max_steps=config.max_steps,
        drop_last=bool(getattr(train_loader, "drop_last", False)),
    )
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps)
    scaler = torch.amp.GradScaler(device.type) if autocast_dtype is torch.float16 else None
    criterion = torch.nn.CrossEntropyLoss()

    report.checks["construction"] = "ok"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[Timing] = []
    accumulation = config.gradient_accumulation_steps
    measured_target = WARMUP_STEPS + steps

    model.train()
    optimizer.zero_grad(set_to_none=True)

    iterator = iter(train_loader)
    micro_batches = 0
    optimizer_steps = 0
    non_finite = 0
    wraps = 0

    # Timing loop. The clock starts before the batch is pulled, so time spent
    # waiting on the loader is attributed to data rather than compute.
    while optimizer_steps < measured_target:
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            # The dataset is smaller than the measured span. Restarting keeps
            # the measurement going, but the repeated batches decode from a warm
            # page cache and will look faster than reality, so it is recorded.
            wraps += 1
            iterator = iter(train_loader)
            batch = next(iterator)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        _synchronize(device)
        data_seconds = time.perf_counter() - data_started

        compute_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = criterion(model(pixel_values).logits, labels)

        if not torch.isfinite(loss):
            non_finite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled = loss / accumulation
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        micro_batches += 1

        if micro_batches % accumulation == 0:
            if config.grad_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        _synchronize(device)
        compute_seconds = time.perf_counter() - compute_started

        if optimizer_steps > WARMUP_STEPS:
            timings.append(Timing(data_seconds, compute_seconds))

    report.checks["forward_backward"] = "ok"
    report.checks["optimizer_step"] = "ok"
    report.checks["scheduler_step"] = "ok"

    if not timings:
        raise RuntimeError(
            f"no steps were measured. The dataset may be too small for {steps} "
            f"steps at batch size {config.batch_size}."
        )

    # Per optimizer step, which is the unit the schedule and the run length use.
    micro_per_step = accumulation
    data_per_micro = sum(t.data_seconds for t in timings) / len(timings)
    compute_per_micro = sum(t.compute_seconds for t in timings) / len(timings)

    data_per_step = data_per_micro * micro_per_step
    compute_per_step = compute_per_micro * micro_per_step
    seconds_per_step = data_per_step + compute_per_step

    videos_per_second = config.effective_batch_size / seconds_per_step
    data_share = 100 * data_per_step / seconds_per_step

    steps_per_epoch = compute_total_steps(
        dataset_size=train_samples,
        batch_size=config.batch_size,
        gradient_accumulation_steps=accumulation,
        epochs=1,
        drop_last=bool(getattr(train_loader, "drop_last", False)),
    )
    epoch_seconds = steps_per_epoch * seconds_per_step

    measurements: dict[str, Any] = {
        "measured_steps": len(timings),
        "warmup_steps_discarded": WARMUP_STEPS,
        "physical_batch_size": config.batch_size,
        "effective_batch_size": config.effective_batch_size,
        "gradient_accumulation_steps": accumulation,
        "num_workers": getattr(train_loader, "num_workers", 0),
        "precision_requested": config.precision,
        "precision_active": (str(autocast_dtype).replace("torch.", "") if use_autocast else "fp32"),
        "videos_per_second": round(videos_per_second, 3),
        "seconds_per_optimizer_step": round(seconds_per_step, 6),
        "data_seconds_per_step": round(data_per_step, 6),
        "compute_seconds_per_step": round(compute_per_step, 6),
        "data_share_percent": round(data_share, 1),
        "bottleneck": "data loading" if data_share > 50 else "compute",
        "train_samples": train_samples,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "epochs": config.epochs,
        "estimated_epoch_seconds": round(epoch_seconds, 4),
        "estimated_epoch_minutes": round(epoch_seconds / 60, 2),
        "estimated_run_seconds": round(epoch_seconds * config.epochs, 4),
        "estimated_run_hours": round(epoch_seconds * config.epochs / 3600, 3),
        "non_finite_losses": non_finite,
        "loader_restarts": wraps,
    }

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        measurements["peak_memory_gb"] = round(peak, 3)
        total_gb = environment.get("gpu_memory_gb")
        if total_gb:
            headroom = 100 * (1 - peak / total_gb)
            measurements["memory_headroom_percent"] = round(headroom, 1)
            if headroom < 15:
                warnings.append(
                    f"only {headroom:.0f}% GPU memory headroom at batch size "
                    f"{config.batch_size}. A longer clip or a larger batch will "
                    f"likely run out. Reduce batch_size and raise "
                    f"gradient_accumulation_steps by the same factor to keep the "
                    f"effective batch comparable."
                )
        measurements["estimated_sessions"] = round(epoch_seconds * config.epochs / 3600 / 12, 2)

    # Bottleneck guidance. Video decoding is CPU-bound, so more workers help
    # only while cores remain. Kaggle gives 4; oversubscribing them measured
    # slower on both the step and the data share (see D-009, Phase 5 appendix).
    if data_share > 50:
        workers = measurements["num_workers"]
        cpus = os.cpu_count()
        measurements["cpu_count"] = cpus
        head = (
            f"data loading is {data_share:.0f}% of each step, so the GPU is "
            f"idle most of the time. "
        )
        if cpus is not None and workers >= cpus:
            warnings.append(
                head + f"--num-workers is already {workers} on {cpus} CPU core(s), "
                f"so this floor is CPU-bound: more workers will oversubscribe the "
                f"cores and are likely to make the step slower, not faster. Reduce "
                f"decoding cost instead, through fewer frames, a smaller decode "
                f"resolution, or a faster clip source."
            )
        else:
            warnings.append(
                head + f"Raising --num-workers above {workers} is likely to help "
                f"more than any model or batch-size change."
            )

    if wraps:
        warnings.append(
            f"the data loader restarted {wraps} time(s) during measurement, so some "
            f"clips were decoded more than once from a warm cache. Throughput is "
            f"optimistic. Use fewer --steps, or a larger dataset, for a real figure."
        )

    if non_finite:
        warnings.append(
            f"{non_finite} non-finite loss(es) in {len(timings)} measured step(s). "
            f"Investigate before a real run."
        )

    # Checkpoint round trip on real artifacts, not synthetic ones.
    if checkpoint_dir is not None and metadata is not None:
        manager = CheckpointManager(checkpoint_dir)
        state = TrainingState(epoch=0, optimizer_step=optimizer_steps)

        save_started = time.perf_counter()
        path = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            metadata=metadata,
            config=config.to_dict(),
        )
        measurements["checkpoint_save_seconds"] = round(time.perf_counter() - save_started, 2)
        measurements["checkpoint_mb"] = round(path.stat().st_size / 1024**2, 2)
        report.checks["checkpoint_save"] = "ok"

        restore(manager.load(), model=model, optimizer=optimizer, scheduler=scheduler)
        report.checks["checkpoint_resume"] = "ok"

        if config.checkpoint_every_minutes:
            per_checkpoint = measurements["checkpoint_mb"]
            writes_per_hour = 60 / config.checkpoint_every_minutes
            measurements["checkpoint_write_mb_per_hour"] = round(
                per_checkpoint * writes_per_hour, 1
            )

    # Short validation pass, to confirm it runs and to price it.
    if val_loader is not None:
        val_started = time.perf_counter()
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ):
                    model(batch["pixel_values"].to(device))
                seen += batch["pixel_values"].shape[0]
                if seen >= config.batch_size * 5:
                    break
        model.train()

        elapsed = time.perf_counter() - val_started
        val_samples = len(val_loader.dataset)  # type: ignore[arg-type]
        measurements["validation_samples"] = val_samples
        measurements["estimated_validation_minutes"] = round(
            (elapsed / max(seen, 1)) * val_samples / 60, 1
        )
        report.checks["validation_pass"] = "ok"

        per_epoch = measurements["estimated_validation_minutes"]
        if per_epoch > measurements["estimated_epoch_minutes"] * 0.25:
            warnings.append(
                f"validation adds ~{per_epoch:.0f} min per epoch against a "
                f"~{measurements['estimated_epoch_minutes']:.0f} min training epoch. "
                f"Consider validate_every_epochs > 1."
            )

    report.measurements = measurements
    return report


def _synchronize(device: torch.device) -> None:
    """Wait for queued device work, so timing measures work rather than queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
