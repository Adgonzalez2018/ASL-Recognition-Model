"""VideoMAE-Base adapter.

VideoMAE accepts ``[batch, frames, channels, height, width]`` natively, which is
the project canonical layout, so no permutation is required.

See docs/MODEL_CONTRACT.md and D-003 in docs/DECISIONS.md.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .base import BaseVideoClassifier, PreprocessingRequirements

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "MCG-NJU/videomae-base"

# VideoMAE was pretrained with ImageNet normalization. torchvision's Swin3D uses
# the same values, so the two architectures share preprocessing and the Phase 5
# comparison is not confounded by it.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# VideoMAE's attention stores a query bias and a value bias but no key bias, and
# published checkpoints use the original names 'q_bias' and 'v_bias'. Some
# transformers versions expect the standard 'query.bias' and 'value.bias' and do
# not translate, which silently leaves those biases at zero. _repair_attention_bias
# below detects and fixes that.
_BIAS_RENAMES = {
    "attention.attention.query.bias": "attention.attention.q_bias",
    "attention.attention.value.bias": "attention.attention.v_bias",
}

# Keys genuinely absent from every published checkpoint, rather than signs of a
# failed load: the deliberately replaced head, and the key bias VideoMAE's
# attention does not define.
_EXPECTED_MISSING_SUFFIXES = ("attention.attention.key.bias",)
_HEAD_KEYS = ("classifier.weight", "classifier.bias")


# Keys present in a VideoMAE checkpoint that a classifier legitimately does not
# use: the self-supervised reconstruction decoder, and the legacy-named attention
# biases consumed by _repair_attention_bias.
_EXPECTED_UNUSED_PREFIXES = ("decoder.", "encoder_to_decoder.")
_EXPECTED_UNUSED_KEYS = ("mask_token",)
_EXPECTED_UNUSED_SUFFIXES = ("q_bias", "v_bias")


def _is_expected_missing(key: str) -> bool:
    return key in _HEAD_KEYS or key.endswith(_EXPECTED_MISSING_SUFFIXES)


def _is_expected_unused(key: str) -> bool:
    return (
        key in _EXPECTED_UNUSED_KEYS
        or key.startswith(_EXPECTED_UNUSED_PREFIXES)
        or key.endswith(_EXPECTED_UNUSED_SUFFIXES)
    )


def _renamed_source(key: str) -> str | None:
    """Map an expected model key to its checkpoint name, if it was renamed."""
    for suffix, source_suffix in _BIAS_RENAMES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)] + source_suffix
    return None


def _load_raw_state_dict(checkpoint: str) -> dict[str, torch.Tensor] | None:
    """Read a checkpoint's raw tensors, for repairing renamed keys.

    Returns ``None`` when the weights cannot be located, in which case the caller
    reports the unrepaired mismatch rather than failing.
    """
    from pathlib import Path

    candidates = ("model.safetensors", "pytorch_model.bin")
    local = Path(checkpoint)

    for filename in candidates:
        try:
            if local.is_dir():
                path = local / filename
                if not path.exists():
                    continue
                path = str(path)
            else:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(checkpoint, filename)

            if path.endswith(".safetensors"):
                from safetensors.torch import load_file

                return load_file(path)
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            continue

    return None


class VideoMAEClassifier(BaseVideoClassifier):
    """VideoMAE-Base with a replaced ASL classification head."""

    def _build(self) -> None:
        try:
            from transformers import (
                VideoMAEConfig,
                VideoMAEForVideoClassification,
            )
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "VideoMAE requires the 'transformers' package. Install the project "
                "dependencies from pyproject.toml."
            ) from exc

        checkpoint = self.config.checkpoint or DEFAULT_CHECKPOINT
        self.checkpoint_source = checkpoint

        # These are owned by ModelConfig and part of the preprocessing identity.
        # Accepting them through `options` would create two sources of truth.
        reserved = {"image_size", "num_frames", "num_labels"} & set(self.config.options)
        if reserved:
            raise ValueError(
                f"model options may not override {', '.join(sorted(reserved))}; "
                f"set the corresponding top-level config fields instead."
            )

        if self.config.pretrained:
            self.backbone = self._load_pretrained(VideoMAEForVideoClassification, checkpoint)
        else:
            # Random initialization, for offline structural tests. Not a valid
            # starting point for a real run; see D-005 in docs/DECISIONS.md.
            hf_config = VideoMAEConfig(
                image_size=self.config.image_size,
                num_frames=self.config.num_frames,
                num_labels=self.config.num_classes,
                **self.config.options,
            )
            self.backbone = VideoMAEForVideoClassification(hf_config)
            self.pretrained_load_report = {
                "checkpoint": checkpoint,
                "loaded": False,
                "reason": "pretrained=False; weights are randomly initialized",
            }

        self._validate_temporal_compatibility()
        self._install_dropout()

    def _load_pretrained(self, model_cls, checkpoint: str):
        """Load pretrained weights, replacing the original head.

        A classification-head mismatch is expected, since the original Kinetics
        head is deliberately discarded. Backbone mismatches are not expected and
        are surfaced rather than swallowed.
        """
        model, loading_info = model_cls.from_pretrained(
            checkpoint,
            num_labels=self.config.num_classes,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
            **self.config.options,
        )

        missing = list(loading_info.get("missing_keys", []))
        unexpected = list(loading_info.get("unexpected_keys", []))

        repaired = self._repair_attention_bias(model, checkpoint, missing)

        remaining = [k for k in missing if k not in repaired]
        expected_missing = [k for k in remaining if _is_expected_missing(k)]
        unexplained = [k for k in remaining if not _is_expected_missing(k)]

        self.pretrained_load_report = {
            "checkpoint": checkpoint,
            "loaded": True,
            "replaced_head": True,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "expected_missing_keys": expected_missing,
            "unexplained_missing_keys": unexplained,
            "repaired_keys": repaired,
            "note": (
                "Classification-head parameters were replaced for the ASL vocabulary. "
                "VideoMAE defines no key attention bias, so that key is absent from "
                "every published checkpoint by design."
            ),
        }

        if unexplained:
            # Not fatal: transformers may rename keys across versions, and failing
            # here would block a run for a cosmetic difference. But it must be
            # visible and recorded, per docs/MODEL_CONTRACT.md.
            logger.warning(
                "VideoMAE checkpoint %s left %d backbone parameter(s) randomly "
                "initialized that are not a known VideoMAE quirk: %s. Verify this "
                "before treating the run as a valid pretrained baseline.",
                checkpoint,
                len(unexplained),
                ", ".join(unexplained[:10]),
            )
        unexplained_unused = [k for k in unexpected if not _is_expected_unused(k)]
        self.pretrained_load_report["unexplained_unused_keys"] = unexplained_unused

        if unexplained_unused:
            logger.warning(
                "VideoMAE checkpoint %s contained %d parameter(s) the classifier did "
                "not consume and that are not part of the known self-supervised "
                "decoder: %s",
                checkpoint,
                len(unexplained_unused),
                ", ".join(unexplained_unused[:10]),
            )

        logger.info("Loaded pretrained VideoMAE weights from %s", checkpoint)
        return model

    def _repair_attention_bias(self, model, checkpoint: str, missing: list[str]) -> list[str]:
        """Restore query and value attention biases stored under legacy names.

        Some transformers versions expect ``query.bias`` and ``value.bias`` but
        published VideoMAE checkpoints store ``q_bias`` and ``v_bias``, and no
        translation happens. The biases are then left at zero while the loader
        reports success, so a run would begin from a partially unloaded backbone
        with no visible error.

        Returns the keys that were repaired.
        """
        renamable = {k: _renamed_source(k) for k in missing if _renamed_source(k)}
        if not renamable:
            return []

        raw = _load_raw_state_dict(checkpoint)
        if raw is None:
            logger.warning(
                "VideoMAE checkpoint %s uses legacy attention-bias names, but its raw "
                "weights could not be read to repair them. %d bias tensor(s) remain "
                "zero-initialized.",
                checkpoint,
                len(renamable),
            )
            return []

        state = model.state_dict()
        repaired: list[str] = []

        with torch.no_grad():
            for target, source in renamable.items():
                if source not in raw or target not in state:
                    continue
                tensor = raw[source]
                if tensor.shape != state[target].shape:
                    logger.warning(
                        "Shape mismatch repairing %s from %s: %s vs %s",
                        target,
                        source,
                        tuple(tensor.shape),
                        tuple(state[target].shape),
                    )
                    continue
                state[target].copy_(tensor)
                repaired.append(target)

        if repaired:
            logger.info(
                "Repaired %d VideoMAE attention-bias tensor(s) stored under legacy "
                "names in %s. Without this they would have remained zero.",
                len(repaired),
                checkpoint,
            )

        return repaired

    def _validate_temporal_compatibility(self) -> None:
        """Check the frame count against tubelet size and position embeddings.

        VideoMAE embeds frames in tubelets and builds fixed position embeddings
        from the configured frame count. A mismatch would otherwise surface as an
        opaque shape error deep inside the backbone.
        """
        hf_config = self.backbone.config
        tubelet = getattr(hf_config, "tubelet_size", 2)
        frames = self.config.num_frames

        if frames % tubelet != 0:
            raise ValueError(
                f"VideoMAE requires num_frames divisible by tubelet_size; "
                f"got num_frames={frames}, tubelet_size={tubelet}."
            )

        pretrained_frames = getattr(hf_config, "num_frames", frames)
        if pretrained_frames != frames:
            raise ValueError(
                f"VideoMAE checkpoint '{self.checkpoint_source}' was configured for "
                f"{pretrained_frames} frames but num_frames={frames} was requested. "
                f"Position embeddings are not interpolated, so this would either fail "
                f"or silently misalign temporal positions. Either set num_frames="
                f"{pretrained_frames} or select a checkpoint trained at {frames} frames."
            )

    def _install_dropout(self) -> None:
        """Insert dropout before the head when configured."""
        if self.config.dropout <= 0.0:
            self._head_dropout = None
            return

        head = self.backbone.classifier
        self._head_dropout = nn.Dropout(self.config.dropout)
        self.backbone.classifier = nn.Sequential(self._head_dropout, head)

    def classification_head(self) -> nn.Module:
        return self.backbone.classifier

    def _forward_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # VideoMAE's native layout is the canonical one; no permutation needed.
        # Loss is computed by the base class so both architectures share one path.
        output = self.backbone(pixel_values=pixel_values)
        return output.logits

    def preprocessing(self) -> PreprocessingRequirements:
        return PreprocessingRequirements(
            num_frames=self.config.num_frames,
            image_size=self.config.image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
