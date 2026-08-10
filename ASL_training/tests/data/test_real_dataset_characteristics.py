"""Pipeline behaviour against the real ASL Citizen distribution.

The values here come from the Phase 2A audit of the actual dataset, not from
guesses. They exist so that a pipeline change cannot quietly break a case the
real data contains.

Audit, 2026-08-07, mirror `abd0kamel/asl-citizen`:

    83,399 videos, 2,731 classes, 52 signers, all decodable
    frame count   min 3, median 75, max 680
    duration      min 0.064s, median 2.57s, max 22.6s
    fps           min 11.3, median 30.0, max 120.0
    resolutions   640x480 (80,184), 960x540 (3,211), 480x640 (4, portrait)
    codecs        h264 (79,873), mpeg4 (3,526)
    rotation      none

See docs/DATA_CONTRACT.md and docs/phases/ for the audit record.
"""

from __future__ import annotations

import random

import av
import numpy as np
import pytest
import torch

from asl_training.data import EvalTransform, TemporalSampler, TrainTransform, decode_clip

FRAMES = 16
CROP = 224


def write_video(path, frames: int, width: int, height: int, fps: int = 30, codec: str = "libx264"):
    """Encode a clip with the given real-world properties."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.zeros((height, width, 3), dtype=np.uint8)
            array[:, :, 0] = int(index * (250 / max(frames - 1, 1)))
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return path


def process(path, total_frames: int, transform):
    """Decode and transform exactly as the dataset would."""
    sampler = TemporalSampler(num_frames=FRAMES, strategy="uniform")
    indices = sampler.indices(total_frames)
    clip = decode_clip(path, indices, expected_frames=FRAMES)
    return transform(clip.frames, random.Random(0))


# Frame count extremes ---------------------------------------------------------


@pytest.mark.parametrize("frames", [3, 8, 15, 16, 75, 680])
def test_real_frame_counts_produce_valid_tensors(frames, tmp_path):
    """3 is the shortest clip in the dataset; 680 the longest."""
    path = write_video(tmp_path / f"f{frames}.mp4", frames, 640, 480)
    out = process(path, frames, EvalTransform())

    assert out.shape == (FRAMES, 3, CROP, CROP)
    assert torch.isfinite(out).all()


def test_shortest_clip_repeats_without_reversing(tmp_path):
    """18 clips are under 16 frames. The policy repeats, never reverses."""
    sampler = TemporalSampler(num_frames=FRAMES, strategy="uniform")
    indices = sampler.indices(3)

    assert len(indices) == FRAMES
    assert indices == sorted(indices), "chronology must be preserved"
    assert set(indices) == {0, 1, 2}, "every source frame should appear"

    path = write_video(tmp_path / "short.mp4", 3, 640, 480)
    clip = decode_clip(path, indices, expected_frames=FRAMES)
    assert clip.frames.shape[0] == FRAMES


def test_longest_clip_is_sampled_across_its_whole_span(tmp_path):
    """A 22.6s clip must not be represented by its first second."""
    sampler = TemporalSampler(num_frames=FRAMES, strategy="uniform")
    indices = sampler.indices(680)

    assert indices[0] < 68, "sampling should start near the beginning"
    assert indices[-1] > 612, "sampling should reach the end"


# Resolutions ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "count"),
    [(640, 480, 80184), (960, 540, 3211), (480, 640, 4)],
)
def test_real_resolutions_produce_valid_tensors(width, height, count, tmp_path):
    """Every resolution present in the dataset, including the 4 portrait clips."""
    path = write_video(tmp_path / f"{width}x{height}.mp4", 40, width, height)
    out = process(path, 40, EvalTransform())

    assert out.shape == (FRAMES, 3, CROP, CROP), f"{width}x{height} ({count} videos)"


def test_portrait_clips_are_cropped_vertically(tmp_path):
    """The 4 portrait videos crop top and bottom, which could clip raised hands.

    Documented rather than corrected: 4 of 83,399 videos is not worth a
    resolution-dependent code path, and a per-aspect-ratio policy would be an
    unreviewed preprocessing difference between samples.
    """
    from asl_training.data.transforms import center_crop, resize_short_side

    clip = torch.zeros(2, 3, 640, 480)  # portrait
    resized = resize_short_side(clip, 256)

    assert resized.shape[3] == 256, "short side is the width for portrait"
    assert resized.shape[2] > 256, "height stays longer"

    cropped = center_crop(resized, CROP)
    assert cropped.shape[2:] == (CROP, CROP)


# Codecs and frame rates -------------------------------------------------------


@pytest.mark.parametrize("codec", ["libx264", "mpeg4"])
def test_both_dataset_codecs_decode(codec, tmp_path):
    """3,526 clips are mpeg4 rather than h264."""
    path = write_video(tmp_path / f"{codec}.mp4", 40, 640, 480, codec=codec)
    out = process(path, 40, EvalTransform())
    assert out.shape == (FRAMES, 3, CROP, CROP)


@pytest.mark.parametrize("fps", [11, 25, 30, 120])
def test_real_frame_rates_decode(fps, tmp_path):
    """Frame rate spans 11.3 to 120 in the dataset.

    Sampling is index-based across the whole clip, so a fixed frame count
    always covers the entire sign regardless of the source rate.
    """
    path = write_video(tmp_path / f"fps{fps}.mp4", 40, 640, 480, fps=fps)
    out = process(path, 40, EvalTransform())
    assert out.shape == (FRAMES, 3, CROP, CROP)


# Training path ----------------------------------------------------------------


@pytest.mark.parametrize("frames", [3, 40, 680])
def test_training_transform_handles_the_real_range(frames, tmp_path):
    path = write_video(tmp_path / f"t{frames}.mp4", frames, 640, 480)
    out = process(path, frames, TrainTransform())

    assert out.shape == (FRAMES, 3, CROP, CROP)
    assert torch.isfinite(out).all()


# Vocabulary -------------------------------------------------------------------


def test_label_map_handles_the_real_vocabulary_size():
    """2,731 classes, verified against the official count by the audit."""
    from asl_training.data import LabelMap

    label_map = LabelMap.from_glosses([f"GLOSS_{i:04d}" for i in range(2731)])

    assert label_map.num_classes == 2731
    assert list(label_map.class_ids) == list(range(2731))
    assert label_map.identity.startswith("asl_citizen:2731:")


def test_model_accepts_the_real_class_count():
    """The head must size to 2,731 without special handling."""
    from asl_training.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="videomae_base",
            num_classes=2731,
            pretrained=False,
            num_frames=FRAMES,
            image_size=CROP,
            options={
                "hidden_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 96,
            },
        )
    )
    assert model.classification_head().out_features == 2731
