"""Spatial transforms.

Every spatial transform is applied to a whole clip with one set of parameters.
A per-frame random crop would invent camera motion that was never recorded, so
crop windows and scales are drawn once per clip and applied to all frames.

Training and evaluation transforms are separate objects. Evaluation transforms
contain no randomness at all, rather than randomness disabled by a mode flag.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Both architectures use ImageNet normalization, so preprocessing does not
# confound the Phase 5 comparison. See docs/DECISIONS.md D-002.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INTERPOLATION = "bilinear"


def _resize(clip: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize every frame of a ``[T, C, H, W]`` clip identically."""
    return F.interpolate(
        clip,
        size=(height, width),
        mode=INTERPOLATION,
        align_corners=False,
        antialias=True,
    )


def resize_short_side(clip: torch.Tensor, size: int) -> torch.Tensor:
    """Scale so the shorter side equals ``size``, preserving aspect ratio."""
    _, _, height, width = clip.shape
    if height <= width:
        new_height = size
        new_width = max(1, round(width * size / height))
    else:
        new_width = size
        new_height = max(1, round(height * size / width))
    return _resize(clip, new_height, new_width)


def center_crop(clip: torch.Tensor, size: int) -> torch.Tensor:
    """Crop the centre ``size x size`` region from every frame."""
    _, _, height, width = clip.shape
    if height < size or width < size:
        clip = resize_short_side(clip, size)
        _, _, height, width = clip.shape

    top = (height - size) // 2
    left = (width - size) // 2
    return clip[:, :, top : top + size, left : left + size]


def crop(clip: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """Crop one window from every frame of the clip."""
    return clip[:, :, top : top + height, left : left + width]


def to_unit_float(clip: torch.Tensor) -> torch.Tensor:
    """Convert uint8 pixels to float in [0, 1].

    Deliberately explicit rather than inferred from dtype inside ``normalize``.
    A conditional conversion silently skips the 1/255 scaling whenever a caller
    has already cast to float, producing tensors ~255x too large — which looks
    like a diverging model rather than a preprocessing bug.
    """
    if clip.dtype == torch.uint8:
        return clip.float().div_(255.0)

    if clip.is_floating_point():
        return clip.float()

    raise TypeError(f"expected uint8 or floating-point pixels, got {clip.dtype}")


def normalize(
    clip: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Standardize per channel.

    Expects float pixels already scaled to [0, 1]; call ``to_unit_float`` first.
    """
    clip = clip.float()
    mean_t = torch.tensor(mean, dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    return (clip - mean_t) / std_t


@dataclass(frozen=True)
class EvalTransform:
    """Deterministic preprocessing for validation and test.

    Resize the shorter side, centre crop, normalize. No randomness of any kind:
    the same clip always produces the same tensor.
    """

    crop_size: int = 224
    resize_size: int = 256
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.resize_size < self.crop_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) must be at least crop_size "
                f"({self.crop_size}); a smaller resize would upscale during crop"
            )

    @property
    def is_deterministic(self) -> bool:
        return True

    def __call__(self, clip: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
        """Transform a ``[T, C, H, W]`` uint8 clip. ``rng`` is ignored."""
        _validate_clip(clip)
        # Scale to [0, 1] first, so every downstream step operates on one
        # known range rather than inferring it.
        clip = to_unit_float(clip)
        clip = resize_short_side(clip, self.resize_size)
        clip = center_crop(clip, self.crop_size)
        return normalize(clip, self.mean, self.std)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "eval",
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": INTERPOLATION,
            "deterministic": True,
        }


@dataclass(frozen=True)
class TrainTransform:
    """Restrained training augmentation for the clean baseline.

    Only a mild random resized crop, per the baseline transform policy. Flipping,
    speed jitter, blur, compression, and strong colour changes are deliberately
    absent: they belong to targeted robustness experiments justified by measured
    weaknesses, not to the control.

    Crop parameters are drawn once per clip.
    """

    crop_size: int = 224
    resize_size: int = 256
    scale: tuple[float, float] = (0.8, 1.0)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    # Explicitly disabled for the baseline. Present so the resolved configuration
    # records what was off, rather than leaving it unstated.
    disabled: tuple[str, ...] = field(
        default_factory=lambda: (
            "horizontal_flip",
            "speed_jitter",
            "frame_dropping",
            "blur",
            "compression",
            "brightness",
            "contrast",
            "saturation",
            "rotation",
            "random_erasing",
        )
    )

    def __post_init__(self) -> None:
        low, high = self.scale
        if not 0 < low <= high <= 1.0:
            raise ValueError(f"scale must satisfy 0 < low <= high <= 1.0, got {self.scale}")
        if self.resize_size < self.crop_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) must be at least crop_size ({self.crop_size})"
            )

    @property
    def is_deterministic(self) -> bool:
        return False

    def __call__(self, clip: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
        """Transform a ``[T, C, H, W]`` uint8 clip.

        Args:
            clip: Source frames.
            rng: Seeded generator. Required, so training randomness is
                reproducible from the experiment seed.
        """
        _validate_clip(clip)
        if rng is None:
            raise ValueError(
                "TrainTransform requires an rng so that augmentation is reproducible "
                "from the experiment seed"
            )

        clip = to_unit_float(clip)
        clip = resize_short_side(clip, self.resize_size)
        _, _, height, width = clip.shape

        # One crop window for the whole clip. Drawing per frame would synthesize
        # camera motion the recording never contained.
        area_scale = rng.uniform(*self.scale)
        side = max(self.crop_size, round(min(height, width) * area_scale))
        side = min(side, height, width)

        top = rng.randint(0, height - side)
        left = rng.randint(0, width - side)

        clip = crop(clip, top, left, side, side)
        if side != self.crop_size:
            clip = _resize(clip, self.crop_size, self.crop_size)

        return normalize(clip, self.mean, self.std)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "train",
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "scale": list(self.scale),
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": INTERPOLATION,
            "deterministic": False,
            "disabled_augmentations": list(self.disabled),
        }


def _validate_clip(clip: torch.Tensor) -> None:
    if not isinstance(clip, torch.Tensor):
        raise TypeError(f"clip must be a torch.Tensor, got {type(clip).__name__}")
    if clip.ndim != 4:
        raise ValueError(f"clip must be [frames, channels, height, width], got {tuple(clip.shape)}")
    if clip.shape[1] != 3:
        raise ValueError(f"clip must have 3 RGB channels, got {clip.shape[1]}")
    if clip.shape[0] == 0:
        raise ValueError("clip has no frames")
