"""Spatial transforms.

The property that matters most: one set of spatial parameters per clip. A
per-frame crop would synthesize camera motion the recording never contained.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import random

import pytest
import torch

from asl_training.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EvalTransform,
    TrainTransform,
    center_crop,
    normalize,
    resize_short_side,
    to_unit_float,
)


def make_clip(frames=16, height=120, width=160, fill=None) -> torch.Tensor:
    """A uint8 [T, C, H, W] clip."""
    if fill is not None:
        return torch.full((frames, 3, height, width), fill, dtype=torch.uint8)
    return torch.randint(0, 256, (frames, 3, height, width), dtype=torch.uint8)


def make_static_clip(frames=16, height=120, width=160) -> torch.Tensor:
    """A clip whose frames are identical.

    Any spatial variation in the output must then come from the transform, which
    is exactly what a per-frame crop bug would produce.
    """
    single = torch.randint(0, 256, (1, 3, height, width), dtype=torch.uint8)
    return single.repeat(frames, 1, 1, 1)


# Output shape -----------------------------------------------------------------


@pytest.mark.parametrize("transform", [EvalTransform(), TrainTransform()])
def test_produces_the_configured_crop_size(transform):
    clip = make_clip()
    out = transform(clip, random.Random(0))
    assert out.shape == (16, 3, 224, 224)


@pytest.mark.parametrize("size", [(80, 60), (400, 700), (224, 224), (300, 300)])
def test_handles_varied_source_resolutions(size):
    height, width = size
    out = EvalTransform()(make_clip(height=height, width=width))
    assert out.shape == (16, 3, 224, 224)


def test_output_is_float(sample_transforms=None):
    out = EvalTransform()(make_clip())
    assert out.dtype == torch.float32


def test_preserves_frame_count():
    for frames in (1, 8, 16, 32):
        out = EvalTransform()(make_clip(frames=frames))
        assert out.shape[0] == frames


# Temporal consistency ---------------------------------------------------------


def test_train_crop_is_identical_across_frames():
    """The central guarantee: one crop window per clip, not per frame."""
    clip = make_static_clip()
    out = TrainTransform()(clip, random.Random(0))

    first = out[0]
    for index in range(1, out.shape[0]):
        assert torch.allclose(out[index], first, atol=1e-5), (
            f"frame {index} differs from frame 0 on an identical-frame clip; "
            f"spatial parameters are being drawn per frame"
        )


def test_eval_crop_is_identical_across_frames():
    clip = make_static_clip()
    out = EvalTransform()(clip)
    for index in range(1, out.shape[0]):
        assert torch.allclose(out[index], out[0], atol=1e-5)


def test_train_transform_varies_across_clips_but_not_within():
    """Augmentation must vary between samples while staying fixed within one."""
    clip = make_static_clip()
    transform = TrainTransform()

    a = transform(clip, random.Random(1))
    b = transform(clip, random.Random(2))

    assert not torch.allclose(a, b), "different seeds produced identical crops"
    assert torch.allclose(a[0], a[-1], atol=1e-5)
    assert torch.allclose(b[0], b[-1], atol=1e-5)


# Determinism ------------------------------------------------------------------


def test_eval_transform_is_deterministic():
    clip = make_clip()
    transform = EvalTransform()
    assert torch.allclose(transform(clip), transform(clip))


def test_eval_transform_ignores_rng():
    """Evaluation output must not depend on random state."""
    clip = make_clip()
    transform = EvalTransform()
    assert torch.allclose(transform(clip, random.Random(1)), transform(clip, random.Random(99)))


def test_eval_transform_declares_itself_deterministic():
    assert EvalTransform().is_deterministic
    assert not TrainTransform().is_deterministic


def test_train_transform_reproduces_from_the_same_seed():
    clip = make_clip()
    transform = TrainTransform()
    assert torch.allclose(transform(clip, random.Random(7)), transform(clip, random.Random(7)))


def test_train_transform_requires_an_rng():
    """Unseeded augmentation would make a run unreproducible."""
    with pytest.raises(ValueError, match="requires an rng"):
        TrainTransform()(make_clip())


# Normalization ----------------------------------------------------------------


def test_normalization_produces_expected_statistics():
    """A mid-grey clip should land near the ImageNet-normalized value."""
    clip = make_clip(fill=128)
    out = EvalTransform()(clip)

    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert float(out[:, channel].mean()) == pytest.approx(expected, abs=0.05)


def test_to_unit_float_scales_uint8_into_zero_one():
    clip = torch.full((2, 3, 8, 8), 255, dtype=torch.uint8)
    out = to_unit_float(clip)
    assert out.dtype == torch.float32
    assert float(out.max()) == pytest.approx(1.0)

    assert float(to_unit_float(torch.zeros(2, 3, 8, 8, dtype=torch.uint8)).max()) == 0.0


def test_normalize_standardizes_unit_range_input():
    clip = torch.ones(2, 3, 8, 8)
    out = normalize(clip)
    for channel in range(3):
        expected = (1.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert float(out[:, channel].mean()) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("transform", [EvalTransform(), TrainTransform()])
def test_output_lands_in_the_expected_normalized_range(transform):
    """Regression: a skipped 1/255 scaling put tensors ~255x too large.

    That failure surfaces as a diverging model rather than a preprocessing bug,
    so the output range is asserted directly.
    """
    out = transform(make_clip(), random.Random(0))

    # ImageNet-normalized pixels span roughly [-2.2, 2.7].
    assert float(out.min()) > -3.0, f"minimum {float(out.min())} is out of range"
    assert float(out.max()) < 3.0, f"maximum {float(out.max())} is out of range"
    assert abs(float(out.mean())) < 1.5


def test_to_unit_float_rejects_non_pixel_dtypes():
    with pytest.raises(TypeError, match="expected uint8 or floating-point"):
        to_unit_float(torch.zeros(2, 3, 8, 8, dtype=torch.int32))


def test_both_transforms_use_the_same_normalization():
    """A normalization difference would confound the architecture comparison."""
    assert EvalTransform().mean == TrainTransform().mean
    assert EvalTransform().std == TrainTransform().std


# Geometry helpers -------------------------------------------------------------


def test_resize_short_side_preserves_aspect_ratio():
    clip = make_clip(height=100, width=200).float()
    out = resize_short_side(clip, 50)
    assert out.shape[2] == 50
    assert out.shape[3] == 100


def test_resize_short_side_handles_portrait():
    clip = make_clip(height=200, width=100).float()
    out = resize_short_side(clip, 50)
    assert out.shape[3] == 50
    assert out.shape[2] == 100


def test_center_crop_takes_the_middle():
    clip = torch.zeros(2, 3, 10, 10)
    clip[:, :, 4:6, 4:6] = 1.0
    out = center_crop(clip, 2)
    assert out.shape == (2, 3, 2, 2)
    assert torch.all(out == 1.0)


def test_center_crop_upscales_when_the_source_is_too_small():
    out = center_crop(make_clip(height=10, width=10).float(), 20)
    assert out.shape[2:] == (20, 20)


# Baseline policy --------------------------------------------------------------


def test_baseline_disables_flipping_and_aggressive_augmentation():
    """The baseline is the control; robustness augmentation belongs elsewhere."""
    disabled = TrainTransform().disabled
    for name in (
        "horizontal_flip",
        "speed_jitter",
        "frame_dropping",
        "blur",
        "compression",
        "rotation",
        "random_erasing",
    ):
        assert name in disabled


def test_disabled_augmentations_are_recorded_in_the_spec():
    """The resolved configuration must state what was off, not leave it implied."""
    spec = TrainTransform().to_dict()
    assert "horizontal_flip" in spec["disabled_augmentations"]
    assert spec["deterministic"] is False


def test_train_crop_scale_is_mild():
    """A restrained baseline: crops keep most of the frame."""
    low, _ = TrainTransform().scale
    assert low >= 0.7


# Configuration ----------------------------------------------------------------


@pytest.mark.parametrize("transform_cls", [EvalTransform, TrainTransform])
def test_resize_smaller_than_crop_is_rejected(transform_cls):
    with pytest.raises(ValueError, match="at least crop_size"):
        transform_cls(crop_size=224, resize_size=200)


def test_invalid_scale_is_rejected():
    with pytest.raises(ValueError, match="scale must satisfy"):
        TrainTransform(scale=(1.2, 1.5))


def test_spec_records_interpolation_and_geometry():
    spec = EvalTransform().to_dict()
    assert spec["crop_size"] == 224
    assert spec["resize_size"] == 256
    assert spec["interpolation"] == "bilinear"
    assert spec["deterministic"] is True


# Input validation -------------------------------------------------------------


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[frames, channels, height, width\]"):
        EvalTransform()(torch.zeros(3, 224, 224))


def test_rejects_non_rgb_channel_count():
    with pytest.raises(ValueError, match="3 RGB channels"):
        EvalTransform()(torch.zeros(4, 1, 100, 100, dtype=torch.uint8))


def test_rejects_empty_clip():
    with pytest.raises(ValueError, match="no frames"):
        EvalTransform()(torch.zeros(0, 3, 100, 100, dtype=torch.uint8))


def test_rejects_non_tensor():
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        EvalTransform()([[1, 2, 3]])
