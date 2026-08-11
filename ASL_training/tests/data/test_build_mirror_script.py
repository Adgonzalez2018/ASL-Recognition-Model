"""Mirror build script.

The build spans several hours and several sessions, so the behaviour that matters
is what happens on the second run: work already done must be skipped, and work
that failed must not be mistaken for done.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_video_mirror.py"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_video_mirror", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_video_mirror"] = module
    spec.loader.exec_module(module)
    return module


class FakeRecord:
    def __init__(self, root, frame_count=75):
        self.sample_id = "sample-1"
        self.video_path = "videos/sample-1.mp4"
        self.frame_count = frame_count
        self._root = root

    def resolve_path(self, dataset_root):
        return self._root / self.video_path


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source" / "videos" / "sample-1.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    return tmp_path / "source"


def run(build, record, dataset, mirror_root, verify_only=False):
    return build.process(record, dataset, mirror_root, 20, "-vsync 0", verify_only)


# Resume --------------------------------------------------------------------


def test_existing_verified_clip_is_skipped(build, dataset, tmp_path, monkeypatch):
    """The whole point of resume: never re-encode what is already correct."""
    mirror_root = tmp_path / "mirror"
    target = mirror_root / "videos" / "sample-1.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"done")

    monkeypatch.setattr(build, "verify_clip", lambda *a: None)
    monkeypatch.setattr(build, "encode_clip", lambda *a: pytest.fail("re-encoded an existing clip"))

    assert run(build, FakeRecord(dataset), dataset, mirror_root).status == "skipped"


def test_existing_bad_clip_is_replaced(build, dataset, tmp_path, monkeypatch):
    """A truncated clip from an interrupted session must not survive as 'done'."""
    mirror_root = tmp_path / "mirror"
    target = mirror_root / "videos" / "sample-1.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")

    calls = {"verify": 0}

    def verify(path, frames):
        calls["verify"] += 1
        if calls["verify"] == 1:
            raise build.MirrorError("frame drift")

    monkeypatch.setattr(build, "verify_clip", verify)
    monkeypatch.setattr(build, "encode_clip", lambda *a: target.write_bytes(b"good"))

    assert run(build, FakeRecord(dataset), dataset, mirror_root).status == "encoded"


# Failures ------------------------------------------------------------------


def test_failed_encode_leaves_nothing_behind(build, dataset, tmp_path, monkeypatch):
    mirror_root = tmp_path / "mirror"
    target = mirror_root / "videos" / "sample-1.mp4"

    def failing(source, dest, crf, flag):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial")
        raise build.MirrorError("ffmpeg failed")

    monkeypatch.setattr(build, "encode_clip", failing)

    outcome = run(build, FakeRecord(dataset), dataset, mirror_root)

    assert outcome.status == "failed"
    assert not target.exists()


def test_verification_failure_after_encode_removes_the_clip(build, dataset, tmp_path, monkeypatch):
    """An encode that succeeds but verifies wrong is still a failure."""
    mirror_root = tmp_path / "mirror"
    target = mirror_root / "videos" / "sample-1.mp4"

    def encode(source, dest, crf, flag):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wrong size")

    monkeypatch.setattr(build, "encode_clip", encode)
    monkeypatch.setattr(
        build, "verify_clip", lambda *a: (_ for _ in ()).throw(build.MirrorError("short side"))
    )

    outcome = run(build, FakeRecord(dataset), dataset, mirror_root)

    assert outcome.status == "failed"
    assert not target.exists()


def test_unreadable_source_is_reported_not_raised(build, tmp_path, monkeypatch):
    record = FakeRecord(tmp_path / "source", frame_count=None)
    monkeypatch.setattr(
        build,
        "count_frames",
        lambda path: (_ for _ in ()).throw(build.VideoDecodeError("no such file")),
    )

    outcome = run(build, record, tmp_path / "source", tmp_path / "mirror")

    assert outcome.status == "failed"
    assert "source unreadable" in outcome.error


# Verify-only ---------------------------------------------------------------


def test_verify_only_flags_a_missing_clip(build, dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(
        build, "encode_clip", lambda *a: pytest.fail("encoded during --verify-only")
    )

    outcome = run(build, FakeRecord(dataset), dataset, tmp_path / "mirror", verify_only=True)

    assert outcome.status == "failed"
    assert "missing" in outcome.error


def test_verify_only_flags_a_bad_clip(build, dataset, tmp_path, monkeypatch):
    mirror_root = tmp_path / "mirror"
    target = mirror_root / "videos" / "sample-1.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")

    monkeypatch.setattr(
        build, "verify_clip", lambda *a: (_ for _ in ()).throw(build.MirrorError("drift"))
    )

    outcome = run(build, FakeRecord(dataset), dataset, mirror_root, verify_only=True)

    assert outcome.status == "failed"
    # The clip is reported, not deleted: --verify-only must not mutate.
    assert target.exists()


# Reporting -----------------------------------------------------------------


def test_report_counts_by_status(build):
    report = build.BuildReport()
    report.outcomes = [
        build.Outcome("a", "encoded"),
        build.Outcome("b", "skipped"),
        build.Outcome("c", "failed", "boom"),
    ]

    assert report.count("encoded") == 1
    assert report.to_dict()["failures"] == [{"sample_id": "c", "error": "boom"}]


def test_missing_dataset_root_exits_two(build, tmp_path, monkeypatch):
    monkeypatch.delenv("ASL_DATASET_ROOT", raising=False)
    assert build.main(["--mirror-root", str(tmp_path)]) == 2
