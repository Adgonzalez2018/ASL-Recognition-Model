"""Shared model interface.

Every adapter accepts the canonical video batch and returns
:class:`~asl_training.models.outputs.VideoClassifierOutput`. Architecture-specific
tensor layouts are handled inside adapters and must not leak outward.

Canonical input:

    [batch, frames, channels, height, width]

Canonical output:

    logits: [batch, num_classes]

See docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .config import ModelConfig
from .outputs import VideoClassifierOutput

CANONICAL_CHANNELS = 3
CANONICAL_RANK = 5


@dataclass
class ParameterReport:
    """Parameter counts recorded with real experiment metadata."""

    total: int
    trainable: int
    frozen: int
    head: int

    @property
    def approx_fp32_mb(self) -> float:
        """Approximate parameter size in MiB at 4 bytes per parameter.

        Excludes optimizer state, gradients, and activations, so it is not a
        memory-planning figure.
        """
        return self.total * 4 / (1024**2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "head": self.head,
            "approx_fp32_mb": round(self.approx_fp32_mb, 2),
        }


@dataclass
class PreprocessingRequirements:
    """Input requirements the data layer must satisfy for this architecture.

    The model layer publishes these; it does not decode video or apply
    augmentation. Differences between architectures must be recorded in
    experiment configuration so an architecture comparison cannot be confounded
    by preprocessing.
    """

    num_frames: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    canonical_layout: str = "BTCHW"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "canonical_layout": self.canonical_layout,
        }


class BaseVideoClassifier(nn.Module, abc.ABC):
    """Base class for ASL video classifiers.

    Subclasses implement :meth:`_build`, :meth:`_forward_backbone`,
    :meth:`classification_head`, and :meth:`preprocessing`.

    The base class owns input validation, label validation, loss computation,
    parameter reporting, and trainable-layer configuration, so that both
    architectures fail identically on identical mistakes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._build()
        self._apply_fine_tuning_strategy()

    # Subclass responsibilities ------------------------------------------------

    @abc.abstractmethod
    def _build(self) -> None:
        """Construct the backbone and replace the classification head."""

    @abc.abstractmethod
    def _forward_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the architecture on a validated canonical batch and return logits.

        Args:
            pixel_values: ``[batch, frames, channels, height, width]``.

        Returns:
            ``[batch, num_classes]``.
        """

    @abc.abstractmethod
    def classification_head(self) -> nn.Module:
        """Return the replaced classification head."""

    @abc.abstractmethod
    def preprocessing(self) -> PreprocessingRequirements:
        """Return this architecture's input requirements."""

    # Shared behavior ----------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def validate_input(self, pixel_values: torch.Tensor) -> None:
        """Validate the canonical batch, raising on any contract violation."""
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                f"pixel_values must be a torch.Tensor, got {type(pixel_values).__name__}"
            )

        if pixel_values.ndim != CANONICAL_RANK:
            raise ValueError(
                f"expected a rank-{CANONICAL_RANK} tensor "
                f"[batch, frames, channels, height, width], got shape "
                f"{tuple(pixel_values.shape)}. If this is [batch, channels, frames, "
                f"height, width], the data layer is emitting a non-canonical layout."
            )

        batch, frames, channels, height, width = pixel_values.shape

        if batch == 0:
            raise ValueError("batch is empty; a batch must contain at least one clip")

        if channels != CANONICAL_CHANNELS:
            raise ValueError(
                f"expected {CANONICAL_CHANNELS} RGB channels at dim 2, got {channels}. "
                f"Full shape {tuple(pixel_values.shape)}. A value of {frames} here "
                f"would indicate a transposed [batch, channels, frames, ...] layout."
            )

        expected_frames = self.config.num_frames
        if frames != expected_frames:
            raise ValueError(
                f"expected {expected_frames} frames, got {frames}. Frame count is part "
                f"of the preprocessing identity and must match the configuration."
            )

        expected_size = self.config.image_size
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"expected {expected_size}x{expected_size} spatial resolution, "
                f"got {height}x{width}."
            )

        if not pixel_values.is_floating_point():
            raise TypeError(
                f"pixel_values must be a floating-point tensor, got {pixel_values.dtype}. "
                f"The data layer is responsible for normalization."
            )

    def validate_labels(self, labels: torch.Tensor, batch: int) -> None:
        """Validate integer class labels against the configured class count."""
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels must be a torch.Tensor, got {type(labels).__name__}")

        if labels.ndim != 1:
            raise ValueError(
                f"labels must have shape [batch], got {tuple(labels.shape)}. "
                f"One-hot or float labels are not used for standard cross-entropy."
            )

        if labels.shape[0] != batch:
            raise ValueError(f"labels length {labels.shape[0]} does not match batch size {batch}")

        if labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(
                f"labels must be an integer dtype, got {labels.dtype}. See docs/DATA_CONTRACT.md."
            )

        if labels.numel():
            low = int(labels.min())
            high = int(labels.max())
            if low < 0 or high >= self.num_classes:
                raise ValueError(
                    f"labels out of range: found [{low}, {high}], valid range is "
                    f"[0, {self.num_classes - 1}]. This usually means the label map "
                    f"and the configured class count disagree."
                )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> VideoClassifierOutput:
        """Classify a batch of clips.

        Args:
            pixel_values: ``[batch, frames, channels, height, width]``, float.
            labels: Optional integer class IDs, shape ``[batch]``.

        Returns:
            Logits, and cross-entropy loss when labels were supplied.
        """
        self.validate_input(pixel_values)
        if labels is not None:
            self.validate_labels(labels, pixel_values.shape[0])

        logits = self._forward_backbone(pixel_values)

        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"{type(self).__name__} produced logits of shape {tuple(logits.shape)}, "
                f"expected [{pixel_values.shape[0]}, {self.num_classes}]. This is an "
                f"adapter bug, not a configuration error."
            )

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.long())

        return VideoClassifierOutput(logits=logits, loss=loss)

    # Trainable-layer configuration -------------------------------------------

    def _apply_fine_tuning_strategy(self) -> None:
        """Apply the configured strategy. Never freezes layers silently."""
        strategy = self.config.fine_tuning

        if strategy == "full":
            for param in self.parameters():
                param.requires_grad_(True)
        elif strategy == "head_only":
            for param in self.parameters():
                param.requires_grad_(False)
            for param in self.classification_head().parameters():
                param.requires_grad_(True)
        else:  # pragma: no cover - ModelConfig validates this
            raise ValueError(f"unknown fine_tuning strategy {strategy!r}")

        if not any(p.requires_grad for p in self.parameters()):
            raise RuntimeError(f"no trainable parameters after applying strategy {strategy!r}")

    def trainable_parameter_names(self) -> list[str]:
        """Names of parameters that will receive gradients."""
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def frozen_parameter_names(self) -> list[str]:
        """Names of parameters that will not receive gradients."""
        return [name for name, param in self.named_parameters() if not param.requires_grad]

    def parameter_report(self) -> ParameterReport:
        """Count total, trainable, frozen, and head parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.classification_head().parameters())
        return ParameterReport(
            total=total,
            trainable=trainable,
            frozen=total - trainable,
            head=head,
        )

    def describe(self) -> dict[str, Any]:
        """Summary for run metadata and checkpoint records."""
        return {
            "config": self.config.to_dict(),
            "parameters": self.parameter_report().to_dict(),
            "preprocessing": self.preprocessing().to_dict(),
        }
