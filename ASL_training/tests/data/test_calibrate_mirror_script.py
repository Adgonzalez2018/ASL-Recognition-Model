"""Mirror calibration script.

The script's job is to decide whether re-encoding the dataset is worth doing, so
the parts under test are the ones that could make a bad idea look good: the
scaling filter, frame-count preservation, and the exit code that gates the build.

These tests need no ffmpeg and no dataset. They cover the reporting and command
construction; the measurement itself is only meaningful against real video.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_video_mirror.py"


@pytest.fixture(scope="module")
def calibrate():
    spec = importlib.util.spec_from_file_location("calibrate_video_mirror", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_video_mirror"] = module
    spec.loader.exec_module(module)
    return module


def clip(calibrate, **overrides):
    defaults = {
        "sample_id": "sample-1",
        "source_bytes": 1000,
        "output_bytes": 300,
        "encode_seconds": 0.4,
        "source_frames": 75,
        "output_frames": 75,
        "resolution": "640x480",
    }
    return calibrate.ClipResult(**{**defaults, **overrides})


# Encoding ------------------------------------------------------------------


def test_scale_filter_is_orientation_aware(calibrate):
    """640x480 and 480x640 must both end up with a 256 short side.

    A fixed -2:256 would stretch the four portrait clips in the dataset to a
    256-pixel long side instead, silently changing their scale relative to
    everything else.
    """
    assert "gt(iw,ih)" in calibrate.SCALE_FILTER
    assert str(calibrate.SHORT_SIDE) in calibrate.SCALE_FILTER


def test_short_side_matches_the_transform(calibrate):
    """Encoding below 256 would remove the random crop's freedom to move."""
    assert calibrate.SHORT_SIDE == 256


def test_encode_command_preserves_frame_rate(calibrate):
    command = calibrate.encode_command(Path("in.mp4"), Path("out.mp4"), 20, "-fps_mode passthrough")
    assert "-fps_mode" in command
    assert "passthrough" in command
    # A frame-rate override would change the frame count the manifests record.
    assert "-r" not in command


def test_encode_command_drops_audio_and_pins_threads(calibrate):
    command = calibrate.encode_command(Path("in.mp4"), Path("out.mp4"), 20, "-vsync 0")
    assert "-an" in command
    assert command[command.index("-threads") + 1] == "1"


def test_encode_command_carries_crf(calibrate):
    command = calibrate.encode_command(Path("in.mp4"), Path("out.mp4"), 18, "-vsync 0")
    assert command[command.index("-crf") + 1] == "18"


# Frame preservation --------------------------------------------------------


def test_frames_match_detects_drift(calibrate):
    assert clip(calibrate).frames_match
    assert not clip(calibrate, output_frames=74).frames_match


def test_mismatched_frames_are_reported(calibrate):
    report = calibrate.Report()
    report.clips = [clip(calibrate), clip(calibrate, sample_id="s2", output_frames=70)]

    assert [c.sample_id for c in report.mismatched] == ["s2"]
    assert report.to_dict()["frame_mismatches"] == ["s2"]


# Projections ---------------------------------------------------------------


def test_size_ratio_and_projection(calibrate):
    report = calibrate.Report()
    report.clips = [clip(calibrate, source_bytes=1000, output_bytes=250)]

    assert report.size_ratio == pytest.approx(0.25)
    expected_gb = 250 * calibrate.TOTAL_VIDEOS / 1024**3
    assert report.projected_gb == pytest.approx(expected_gb)


def test_encode_projection_divides_by_jobs(calibrate):
    report = calibrate.Report(jobs=4)
    report.clips = [clip(calibrate, encode_seconds=1.0)]

    assert report.projected_encode_hours == pytest.approx(calibrate.TOTAL_VIDEOS / 4 / 3600)


def test_failed_clips_excluded_from_projections(calibrate):
    """A failure contributes no size, so counting it would understate the total."""
    report = calibrate.Report()
    report.clips = [
        clip(calibrate, source_bytes=1000, output_bytes=250),
        clip(calibrate, sample_id="s2", output_bytes=0, ok=False, error="boom"),
    ]

    assert len(report.encoded) == 1
    assert report.size_ratio == pytest.approx(0.25)
    assert report.to_dict()["encode_failures"][0]["sample_id"] == "s2"


def test_decode_speedup(calibrate):
    report = calibrate.Report()
    report.decodes = [
        calibrate.DecodeResult(sample_id="s1", source_seconds=0.10, mirror_seconds=0.04),
        calibrate.DecodeResult(sample_id="s2", source_seconds=0.10, mirror_seconds=0.06),
    ]

    assert report.decode_speedup == pytest.approx(0.20 / 0.10)


def test_empty_report_does_not_divide_by_zero(calibrate):
    report = calibrate.Report()

    assert report.size_ratio == 0.0
    assert report.projected_gb == 0.0
    assert report.projected_encode_hours == 0.0
    assert report.decode_speedup == 0.0
    assert "Mirror calibration" in report.summary()


def test_summary_reports_the_gating_number(calibrate):
    report = calibrate.Report()
    report.clips = [clip(calibrate)]
    report.decodes = [
        calibrate.DecodeResult(sample_id="s1", source_seconds=0.10, mirror_seconds=0.05)
    ]

    summary = report.summary()
    assert "DECODE SPEEDUP" in summary
    assert "2.00x" in summary


# Entry point ---------------------------------------------------------------


def test_missing_dataset_root_exits_two(calibrate, tmp_path, monkeypatch):
    monkeypatch.delenv("ASL_DATASET_ROOT", raising=False)
    assert calibrate.main(["--work-dir", str(tmp_path)]) == 2


def test_missing_manifest_exits_one(calibrate, tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "ffmpeg_available", lambda: True)
    code = calibrate.main(
        [
            "--dataset-root",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert code == 1
