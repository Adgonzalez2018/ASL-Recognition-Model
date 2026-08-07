"""Video decoding.

RGB output and chronological frame order are contractual, so both are verified
against encoded files whose pixel content identifies each frame.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import av
import numpy as np
import pytest
import torch

from asl_training.data.decode import VideoDecodeError, count_frames, decode_clip


def write_marked_video(path, frames=20, width=64, height=48, fps=25):
    """Encode a video whose red channel increases with frame index.

    Frame order is then recoverable from pixel content, which is what makes an
    order guarantee testable rather than assumed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}  # near-lossless, so markers survive

        for index in range(frames):
            array = np.zeros((height, width, 3), dtype=np.uint8)
            array[:, :, 0] = int(index * (250 / max(frames - 1, 1)))  # red ramp
            array[:, :, 2] = 40  # constant blue, to catch channel swaps
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))

        container.mux(stream.encode())
    return path


@pytest.fixture
def marked_video(tmp_path):
    return write_marked_video(tmp_path / "marked.mp4", frames=20)


# Basic decoding ---------------------------------------------------------------


def test_decodes_all_frames(marked_video):
    clip = decode_clip(marked_video)
    assert clip.frames.shape == (20, 3, 48, 64)
    assert clip.source_frame_count == 20
    assert clip.fps == pytest.approx(25.0)


def test_output_is_uint8(marked_video):
    assert decode_clip(marked_video).frames.dtype == torch.uint8


def test_decodes_only_requested_indices(marked_video):
    clip = decode_clip(marked_video, [0, 5, 10, 15])
    assert clip.frames.shape[0] == 4
    assert clip.selected_indices == [0, 5, 10, 15]


# Frame order ------------------------------------------------------------------


def test_frames_come_out_in_chronological_order(marked_video):
    """The red ramp increases with time, so order is verifiable from pixels."""
    clip = decode_clip(marked_video)
    reds = [float(clip.frames[i, 0].float().mean()) for i in range(clip.frames.shape[0])]
    assert reds == sorted(reds)
    assert reds[-1] > reds[0] + 100


def test_requested_order_is_honoured(marked_video):
    clip = decode_clip(marked_video, [2, 8, 14])
    reds = [float(clip.frames[i, 0].float().mean()) for i in range(3)]
    assert reds[0] < reds[1] < reds[2]


def test_repeated_indices_are_honoured(marked_video):
    """The short-video policy depends on repeats being returned, not collapsed."""
    clip = decode_clip(marked_video, [0, 0, 1, 1, 2, 2])
    assert clip.frames.shape[0] == 6
    assert torch.equal(clip.frames[0], clip.frames[1])
    assert torch.equal(clip.frames[2], clip.frames[3])
    assert not torch.equal(clip.frames[0], clip.frames[2])


# Colour space -----------------------------------------------------------------


def test_output_is_rgb_not_bgr(marked_video):
    """The decoder's native format is YUV; conversion must be explicit."""
    clip = decode_clip(marked_video, [19])
    frame = clip.frames[0].float()

    red = float(frame[0].mean())
    green = float(frame[1].mean())
    blue = float(frame[2].mean())

    # Encoded as a high red ramp, low green, constant low blue.
    assert red > 200, f"channel 0 should be red-dominant, got {red}"
    assert green < 100
    assert blue < 100
    assert red > blue, "channels appear swapped (BGR rather than RGB)"


def test_channel_count_is_three(marked_video):
    assert decode_clip(marked_video).frames.shape[1] == 3


# Failures ---------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoDecodeError, match="video not found"):
        decode_clip(tmp_path / "absent.mp4")


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a video at all")
    with pytest.raises(VideoDecodeError, match="could not decode"):
        decode_clip(path)


def test_index_beyond_the_file_raises_with_context(marked_video):
    """A manifest frame count disagreeing with the file must be visible."""
    with pytest.raises(VideoDecodeError, match="disagrees with the file"):
        decode_clip(marked_video, [0, 5, 999])


def test_expected_frame_count_is_enforced(marked_video):
    with pytest.raises(VideoDecodeError, match="expected 16"):
        decode_clip(marked_video, [0, 1, 2], expected_frames=16)


def test_expected_frame_count_passes_when_matched(marked_video):
    clip = decode_clip(marked_video, list(range(16)), expected_frames=16)
    assert clip.frames.shape[0] == 16


# Frame counting ---------------------------------------------------------------


def test_counts_frames(marked_video):
    assert count_frames(marked_video) == 20


def test_count_frames_missing_file_raises(tmp_path):
    with pytest.raises(VideoDecodeError, match="video not found"):
        count_frames(tmp_path / "absent.mp4")


def test_count_frames_corrupt_file_raises(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"nope")
    with pytest.raises(VideoDecodeError, match="could not read"):
        count_frames(path)


# Short videos -----------------------------------------------------------------


def test_decodes_a_very_short_video(tmp_path):
    path = write_marked_video(tmp_path / "short.mp4", frames=3)
    assert decode_clip(path).frames.shape[0] == 3


def test_short_video_supports_repeat_sampling(tmp_path):
    """A 3-frame clip must still fill a 16-frame model input."""
    from asl_training.data.sampling import uniform_indices

    path = write_marked_video(tmp_path / "short.mp4", frames=3)
    indices = uniform_indices(3, 16)
    clip = decode_clip(path, indices, expected_frames=16)
    assert clip.frames.shape[0] == 16
