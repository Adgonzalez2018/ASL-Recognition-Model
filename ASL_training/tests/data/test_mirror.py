"""Mirror encoding settings and per-clip verification.

The mirror replaces the dataset a training run reads. Two properties make that
substitution safe — unchanged frame counts and a 256 short side — and this module
is where both are enforced. A clip that decodes but violates either would train
without error against the wrong data.

No ffmpeg is needed: the subprocess boundary is faked so the settings themselves
are what gets tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asl_training.data import mirror


class FakeCompleted:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


# Encoding settings ---------------------------------------------------------


def test_short_side_matches_the_training_transform():
    """The transform resizes to 256 then crops 224.

    Encoding below 256 would leave the random crop no room to move, removing the
    spatial augmentation rather than just shrinking the file.
    """
    assert mirror.SHORT_SIDE == 256


def test_scale_filter_is_orientation_aware():
    """A fixed -2:256 would scale portrait clips by their long side instead."""
    assert "gt(iw,ih)" in mirror.SCALE_FILTER
    assert mirror.SCALE_FILTER.count(str(mirror.SHORT_SIDE)) == 2


def test_encode_command_passes_frame_rate_through():
    command = mirror.encode_command(Path("in.mp4"), Path("out.mp4"), 20, "-vsync 0")

    assert "-vsync" in command
    # An -r override would resample and change the frame count.
    assert "-r" not in command


def test_encode_command_carries_crf_and_drops_audio():
    command = mirror.encode_command(Path("in.mp4"), Path("out.mp4"), 18, "-vsync 0")

    assert command[command.index("-crf") + 1] == "18"
    assert "-an" in command
    assert command[command.index("-threads") + 1] == "1"


def test_encode_command_applies_the_scale_filter():
    command = mirror.encode_command(Path("in.mp4"), Path("out.mp4"), 20, "-vsync 0")

    assert command[command.index("-vf") + 1] == mirror.SCALE_FILTER


# Frame-rate flag detection -------------------------------------------------


def test_detect_fps_flag_prefers_the_modern_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror.subprocess, "run", lambda *a, **k: FakeCompleted(0))

    assert mirror.detect_fps_flag(Path("in.mp4"), tmp_path) == "-fps_mode passthrough"


def test_detect_fps_flag_falls_back_on_older_ffmpeg(tmp_path, monkeypatch):
    def only_vsync(command, **kwargs):
        return FakeCompleted(0 if "-vsync" in command else 1, "Unrecognized option")

    monkeypatch.setattr(mirror.subprocess, "run", only_vsync)

    assert mirror.detect_fps_flag(Path("in.mp4"), tmp_path) == "-vsync 0"


def test_detect_fps_flag_raises_when_neither_works(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror.subprocess, "run", lambda *a, **k: FakeCompleted(1, "nope"))

    with pytest.raises(mirror.MirrorError, match="ffmpeg build"):
        mirror.detect_fps_flag(Path("in.mp4"), tmp_path)


def test_detect_fps_flag_leaves_no_probe_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror.subprocess, "run", lambda *a, **k: FakeCompleted(0))
    mirror.detect_fps_flag(Path("in.mp4"), tmp_path)

    assert not (tmp_path / ".fps-probe.mp4").exists()


# Encoding ------------------------------------------------------------------


def test_encode_clip_removes_the_output_on_failure(tmp_path, monkeypatch):
    """A partial file left behind would later be skipped as already done."""
    target = tmp_path / "out.mp4"

    def failing(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return FakeCompleted(1, "Invalid data found\n")

    monkeypatch.setattr(mirror.subprocess, "run", failing)

    with pytest.raises(mirror.MirrorError, match="Invalid data"):
        mirror.encode_clip(Path("in.mp4"), target, 20, "-vsync 0")
    assert not target.exists()


def test_encode_clip_creates_parent_directories(tmp_path, monkeypatch):
    target = tmp_path / "videos" / "nested" / "out.mp4"

    def succeed(command, **kwargs):
        Path(command[-1]).write_bytes(b"ok")
        return FakeCompleted(0)

    monkeypatch.setattr(mirror.subprocess, "run", succeed)
    mirror.encode_clip(Path("in.mp4"), target, 20, "-vsync 0")

    assert target.exists()


# Verification --------------------------------------------------------------


@pytest.fixture
def written(tmp_path):
    target = tmp_path / "out.mp4"
    target.write_bytes(b"video")
    return target


def test_verify_accepts_a_correct_clip(written, monkeypatch):
    monkeypatch.setattr(mirror, "count_frames", lambda path: 75)
    monkeypatch.setattr(mirror, "probe_dimensions", lambda path: (342, 256))

    mirror.verify_clip(written, 75)


def test_verify_rejects_frame_drift(written, monkeypatch):
    """The manifests index against the frame count, so drift corrupts sampling."""
    monkeypatch.setattr(mirror, "count_frames", lambda path: 74)
    monkeypatch.setattr(mirror, "probe_dimensions", lambda path: (342, 256))

    with pytest.raises(mirror.MirrorError, match="74 frame"):
        mirror.verify_clip(written, 75)


def test_verify_rejects_wrong_short_side(written, monkeypatch):
    """Geometry errors are silent at training time; this is the only gate."""
    monkeypatch.setattr(mirror, "count_frames", lambda path: 75)
    monkeypatch.setattr(mirror, "probe_dimensions", lambda path: (455, 342))

    with pytest.raises(mirror.MirrorError, match="short side"):
        mirror.verify_clip(written, 75)


def test_verify_accepts_portrait_geometry(written, monkeypatch):
    """The dataset holds a few 480x640 clips; 256 must land on the width."""
    monkeypatch.setattr(mirror, "count_frames", lambda path: 75)
    monkeypatch.setattr(mirror, "probe_dimensions", lambda path: (256, 342))

    mirror.verify_clip(written, 75)


def test_verify_rejects_a_missing_clip(tmp_path):
    with pytest.raises(mirror.MirrorError, match="was not written"):
        mirror.verify_clip(tmp_path / "absent.mp4", 75)


def test_verify_reports_an_undecodable_clip(written, monkeypatch):
    def broken(path):
        raise mirror.VideoDecodeError("moov atom not found")

    monkeypatch.setattr(mirror, "count_frames", broken)

    with pytest.raises(mirror.MirrorError, match="does not decode"):
        mirror.verify_clip(written, 75)
