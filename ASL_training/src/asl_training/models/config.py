"""Model configuration.

The classification head takes its size from validated configuration and the label
map, never from an observed training batch. See docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FINE_TUNING_STRATEGIES = ("full", "head_only")


@dataclass
class ModelConfig:
    """Resolved configuration for constructing one model.

    Attributes:
        architecture: Registered architecture name, e.g. ``"videomae_base"``.
        num_classes: Output dimension. Must equal the label-map size.
        pretrained: Whether to load pretrained backbone weights.
        checkpoint: Architecture-specific pretrained source identifier. ``None``
            uses the adapter's documented default.
        num_frames: Frames per clip in the canonical input.
        image_size: Spatial resolution of the canonical input.
        fine_tuning: One of ``FINE_TUNING_STRATEGIES``. Defaults to full
            fine-tuning; anything else must be configured explicitly and is
            recorded in experiment metadata.
        dropout: Dropout applied before the classification head.
        options: Architecture-specific options, passed through to the adapter.
    """

    architecture: str
    num_classes: int
    pretrained: bool = True
    checkpoint: str | None = None
    num_frames: int = 16
    image_size: int = 224
    fine_tuning: str = "full"
    dropout: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be a non-empty string")

        # Guard against bool, which is an int subclass and would silently pass.
        if isinstance(self.num_classes, bool) or not isinstance(self.num_classes, int):
            raise TypeError(f"num_classes must be an int, got {type(self.num_classes).__name__}")
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2 for multiclass classification, "
                f"got {self.num_classes}"
            )

        if not isinstance(self.num_frames, int) or self.num_frames < 1:
            raise ValueError(f"num_frames must be a positive int, got {self.num_frames!r}")
        if not isinstance(self.image_size, int) or self.image_size < 1:
            raise ValueError(f"image_size must be a positive int, got {self.image_size!r}")

        if self.fine_tuning not in FINE_TUNING_STRATEGIES:
            raise ValueError(
                f"unknown fine_tuning strategy {self.fine_tuning!r}; "
                f"supported: {', '.join(FINE_TUNING_STRATEGIES)}"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Build from a mapping, rejecting unknown keys.

        Unknown keys are an error rather than a warning: a typo in a config file
        would otherwise silently leave a setting at its default and produce a run
        whose recorded configuration does not describe what actually executed.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown model config keys: {', '.join(sorted(unknown))}; "
                f"supported: {', '.join(sorted(known))}"
            )
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> ModelConfig:
        """Load from a YAML file containing a top-level ``model`` mapping.

        Overrides are merged before validation, because values resolved at
        runtime are deliberately absent from the committed configs. Chief among
        them is ``num_classes``, which comes from the label map so that the model
        and the label map cannot disagree.
        """
        path = Path(path)
        with path.open() as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        if "model" not in raw:
            raise ValueError(f"{path}: missing required top-level 'model' key")

        section = raw["model"]
        if not isinstance(section, dict):
            raise ValueError(f"{path}: 'model' must be a mapping")

        merged = {**section, **overrides}
        if "num_classes" not in merged:
            raise ValueError(
                f"{path}: num_classes was not supplied. It is resolved at runtime "
                f"from the label map and must be passed as an override."
            )

        try:
            return cls.from_dict(merged)
        except (ValueError, TypeError) as exc:
            raise type(exc)(f"{path}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize for run metadata and checkpoint records."""
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "checkpoint": self.checkpoint,
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "fine_tuning": self.fine_tuning,
            "dropout": self.dropout,
            "options": dict(self.options),
        }
