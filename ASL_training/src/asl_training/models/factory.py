"""Model factory.

One shared construction interface for every architecture. The training layer must
not contain per-architecture construction logic. See docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base import BaseVideoClassifier
from .config import ModelConfig
from .video_swin import VideoSwinClassifier
from .videomae import VideoMAEClassifier

logger = logging.getLogger(__name__)

# Registered architecture names are explicit and stable. Adding one is a
# deliberate change, not a side effect of importing a module.
ARCHITECTURES: dict[str, type[BaseVideoClassifier]] = {
    "videomae_base": VideoMAEClassifier,
    "video_swin_tiny": VideoSwinClassifier,
}


def available_architectures() -> list[str]:
    """Registered architecture names, sorted."""
    return sorted(ARCHITECTURES)


def build_model(config: ModelConfig) -> BaseVideoClassifier:
    """Construct a model from a resolved configuration.

    Args:
        config: Validated model configuration.

    Returns:
        A constructed classifier with its ASL head in place and the configured
        fine-tuning strategy applied.

    Raises:
        ValueError: If the architecture is not registered.
    """
    if config.architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {config.architecture!r}; "
            f"supported: {', '.join(available_architectures())}"
        )

    model_cls = ARCHITECTURES[config.architecture]
    model = model_cls(config)

    report = model.parameter_report()
    logger.info(
        "Built %s: %d classes, %s params (%s trainable, %s frozen), head %s",
        config.architecture,
        config.num_classes,
        f"{report.total:,}",
        f"{report.trainable:,}",
        f"{report.frozen:,}",
        f"{report.head:,}",
    )
    return model


def build_model_from_yaml(path: str | Path, **overrides: Any) -> BaseVideoClassifier:
    """Construct a model from a YAML config, with optional overrides.

    Overrides exist for values resolved at runtime rather than stored in the
    config file, chiefly ``num_classes``, which comes from the label map.
    """
    known = set(ModelConfig.__dataclass_fields__)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(
            f"unknown override keys: {', '.join(sorted(unknown))}; "
            f"supported: {', '.join(sorted(known))}"
        )
    return build_model(ModelConfig.from_yaml(path, **overrides))


def load_checkpoint_state(
    model: BaseVideoClassifier,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load ASL model weights into a constructed model.

    This is model-state loading only. It is not training resume, which also
    restores optimizer, scheduler, and counter state and belongs to the training
    layer. Keeping them separate avoids one ambiguous ``resume`` behavior, per
    docs/MODEL_CONTRACT.md.

    Args:
        model: A model already constructed with the intended configuration.
        checkpoint_path: Path to a checkpoint file.
        strict: Require exact state-dict key agreement.
        map_location: Device to map storage to.

    Returns:
        A report containing any missing and unexpected keys.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint is incompatible with the model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        metadata = {k: v for k, v in payload.items() if k != "model_state"}
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state = payload
        metadata = {}

    _validate_checkpoint_compatibility(model, state, metadata, checkpoint_path)

    result = model.load_state_dict(state, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    if missing or unexpected:
        logger.warning(
            "Loaded %s with %d missing and %d unexpected keys",
            checkpoint_path,
            len(missing),
            len(unexpected),
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_metadata": metadata,
    }


def _validate_checkpoint_compatibility(
    model: BaseVideoClassifier,
    state: dict[str, Any],
    metadata: dict[str, Any],
    path: Path,
) -> None:
    """Fail clearly on incompatible checkpoints before loading.

    A silent vocabulary or architecture mismatch would produce a model whose
    predictions do not mean what the label map says they mean.
    """
    recorded_arch = metadata.get("architecture") or (metadata.get("config") or {}).get(
        "architecture"
    )
    if recorded_arch and recorded_arch != model.config.architecture:
        raise ValueError(
            f"{path}: checkpoint architecture {recorded_arch!r} does not match the "
            f"constructed model {model.config.architecture!r}."
        )

    recorded_classes = metadata.get("num_classes") or (metadata.get("config") or {}).get(
        "num_classes"
    )
    if recorded_classes and recorded_classes != model.num_classes:
        raise ValueError(
            f"{path}: checkpoint has {recorded_classes} classes but the model is "
            f"configured for {model.num_classes}. Loading would silently change the "
            f"meaning of every class ID."
        )

    # Infer the head width from the state dict, which catches mismatches even when
    # the checkpoint carries no metadata.
    head_weight_keys = [
        key for key in state if key.endswith("head.weight") or key.endswith("classifier.weight")
    ]
    for key in head_weight_keys:
        out_features = state[key].shape[0]
        if out_features != model.num_classes:
            raise ValueError(
                f"{path}: classification head {key!r} has {out_features} outputs but "
                f"the model is configured for {model.num_classes} classes."
            )
