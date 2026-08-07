"""Dataset audit.

The audit's job is to report what the dataset actually contains, including the
parts nobody wants to hear about. These tests assert that problems are surfaced
rather than smoothed over.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import json

import pytest

from asl_training.data.asl_citizen import parse_annotations
from asl_training.data.audit import audit_dataset, probe_video
from asl_training.data.label_map import LabelMap
from asl_training.data.manifest import Manifest, ManifestRecord

from .conftest import write_video

# Video probing ----------------------------------------------------------------


def test_probes_a_real_video(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=20, width=64, height=48, fps=25)
    probe = probe_video(path)

    assert probe.exists
    assert probe.status == "usable"
    assert probe.frame_count == 20
    assert probe.fps == pytest.approx(25.0)
    assert (probe.width, probe.height) == (64, 48)
    assert probe.codec == "h264"
    assert probe.duration_seconds == pytest.approx(0.8, abs=0.1)


def test_missing_file_is_reported_not_raised(tmp_path):
    probe = probe_video(tmp_path / "absent.mp4")
    assert not probe.exists
    assert probe.status == "missing"


def test_corrupt_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"this is not a video")

    probe = probe_video(path)
    assert probe.exists
    assert probe.status == "unreadable"
    assert probe.error


def test_truncated_video_is_reported(tmp_path):
    """A partially written file, as an interrupted download produces."""
    path = write_video(tmp_path / "clip.mp4", frames=20)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])

    probe = probe_video(path)
    assert probe.status != "usable"


def test_probes_short_video(tmp_path):
    path = write_video(tmp_path / "short.mp4", frames=4)
    assert probe_video(path).frame_count == 4


# Audit over a clean dataset ---------------------------------------------------


@pytest.fixture
def clean_audit(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    return audit_dataset(
        parsed.manifest,
        parsed.label_map,
        synthetic_root,
        annotation_rows=parsed.rows_read,
    )


def test_clean_dataset_audits_without_integrity_errors(clean_audit):
    assert clean_audit.report["integrity"]["errors"] == []
    assert clean_audit.report["integrity"]["signer_independent"]


def test_reports_counts(clean_audit):
    counts = clean_audit.report["counts"]
    assert counts["manifest_records"] == 12
    assert counts["classes"] == 3
    assert counts["signers"] == 4
    assert counts["by_split"] == {"train": 6, "validation": 3, "test": 3}
    assert counts["signers_by_split"] == {"train": 2, "validation": 1, "test": 1}


def test_reports_media_distributions(clean_audit):
    media = clean_audit.report["media"]
    assert media["probed"] == 12
    assert media["complete"]
    assert media["status_counts"] == {"usable": 12}
    assert media["frame_count"]["median"] == 20
    assert media["fps"]["median"] == pytest.approx(25.0)
    assert "64x48" in media["resolutions"]
    assert "h264" in media["codecs"]


def test_records_identities_for_reproducibility(clean_audit):
    assert clean_audit.report["manifest_identity"].startswith("asl_citizen:12:")
    assert clean_audit.report["label_map_identity"].startswith("asl_citizen:3:")
    assert clean_audit.report["generated_at"]


def test_does_not_invent_handedness_or_mirroring(clean_audit):
    """Absent metadata must stay absent, per docs/DATA_CONTRACT.md."""
    availability = clean_audit.report["metadata_availability"]
    assert availability["handedness"] is None
    assert availability["mirroring_status"] is None


def test_audit_never_modifies_the_dataset(synthetic_root):
    before = sorted(p.stat().st_mtime_ns for p in synthetic_root.rglob("*") if p.is_file())
    parsed = parse_annotations(synthetic_root)
    audit_dataset(parsed.manifest, parsed.label_map, synthetic_root)
    after = sorted(p.stat().st_mtime_ns for p in synthetic_root.rglob("*") if p.is_file())
    assert before == after


# Problems are surfaced --------------------------------------------------------


def test_reports_missing_video_files(synthetic_root, tmp_path):
    """A manifest row whose file is absent must be reported, not skipped."""
    parsed = parse_annotations(synthetic_root)
    manifest = Manifest(
        records=[
            *parsed.manifest.records,
            ManifestRecord(
                sample_id="asl_citizen:ghost",
                video_path="videos/ghost.mp4",
                gloss="APPLE",
                class_id=parsed.label_map.to_id("APPLE"),
                signer_id="signer01",
                split="train",
                dataset_name="asl_citizen",
            ),
        ]
    )
    audit = audit_dataset(manifest, parsed.label_map, synthetic_root)

    assert audit.report["media"]["status_counts"]["missing"] == 1
    assert any("not usable" in p for p in audit.problems)


def test_reports_class_count_mismatch_without_correcting_it(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(parsed.manifest, parsed.label_map, synthetic_root, expected_classes=2731)
    assert any("2731 was expected" in p for p in audit.problems)
    # The dataset governs; the label map is unchanged.
    assert audit.report["counts"]["classes"] == 3


def test_reports_partial_audit_as_insufficient(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(parsed.manifest, parsed.label_map, synthetic_root, probe_limit=3)

    assert audit.report["media"]["probed"] == 3
    assert not audit.report["media"]["complete"]
    assert any("partial audit" in p for p in audit.problems)


def test_reports_signer_leakage(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    leaked = [
        *parsed.manifest.records,
        ManifestRecord(
            sample_id="asl_citizen:leak",
            video_path="videos/clip001.mp4",
            gloss="APPLE",
            class_id=parsed.label_map.to_id("APPLE"),
            signer_id="signer01",
            split="test",
            dataset_name="asl_citizen",
        ),
    ]
    audit = audit_dataset(Manifest(records=leaked), parsed.label_map, synthetic_root)

    assert not audit.report["integrity"]["signer_independent"]
    assert any("signer01" in e for e in audit.report["integrity"]["errors"])


def test_reports_short_videos_against_the_frame_policy(tmp_path):
    """Videos below the configured frame count drive the short-video policy."""
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    for index, frames in enumerate([4, 20, 40], start=1):
        write_video(videos / f"c{index}.mp4", frames=frames)

    label_map = LabelMap.from_glosses(["APPLE", "BOOK", "CAT"])
    records = [
        ManifestRecord(
            sample_id=f"asl_citizen:c{i}",
            video_path=f"videos/c{i}.mp4",
            gloss=gloss,
            class_id=label_map.to_id(gloss),
            signer_id="signer01",
            split="train",
            dataset_name="asl_citizen",
        )
        for i, gloss in enumerate(["APPLE", "BOOK", "CAT"], start=1)
    ]
    audit = audit_dataset(Manifest(records=records), label_map, tmp_path)

    assert audit.report["media"]["videos_under_16_frames"] == 1
    assert audit.report["media"]["videos_under_32_frames"] == 2
    assert any("fewer than the configured 16 frames" in p for p in audit.problems)


def test_short_video_threshold_follows_the_configured_frame_count(tmp_path):
    """Clips short of a frame count nobody trains at are information, not a problem."""
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    for index in range(2):
        write_video(videos / f"c{index}.mp4", frames=20)

    label_map = LabelMap.from_glosses(["APPLE", "BOOK"])
    records = [
        ManifestRecord(
            sample_id=f"asl_citizen:c{i}",
            video_path=f"videos/c{i}.mp4",
            gloss=gloss,
            class_id=label_map.to_id(gloss),
            signer_id="signer01",
            split="train",
            dataset_name="asl_citizen",
        )
        for i, gloss in enumerate(["APPLE", "BOOK"])
    ]
    manifest = Manifest(records=records)

    # 20-frame clips are fine at 16 frames, but short at 32.
    at_16 = audit_dataset(manifest, label_map, tmp_path, configured_frames=16)
    assert not any("short-video policy" in p for p in at_16.problems)
    assert at_16.report["media"]["videos_under_32_frames"] == 2

    at_32 = audit_dataset(manifest, label_map, tmp_path, configured_frames=32)
    assert any("fewer than the configured 32 frames" in p for p in at_32.problems)


def test_reports_classes_with_no_training_samples(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    train_only = [r for r in parsed.manifest.records if r.split != "train"]
    audit = audit_dataset(Manifest(records=train_only), parsed.label_map, synthetic_root)
    assert any("no training samples" in p for p in audit.problems)


def test_reports_class_imbalance(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(parsed.manifest, parsed.label_map, synthetic_root)

    balance = audit.report["class_balance"]
    assert balance["min_samples_per_class"] == 4
    assert balance["max_samples_per_class"] == 4
    assert balance["imbalance_ratio"] == 1.0


def test_reports_singleton_training_classes(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    for index in range(3):
        write_video(videos / f"c{index}.mp4", frames=20)

    label_map = LabelMap.from_glosses(["APPLE", "BOOK"])
    records = [
        ManifestRecord(
            sample_id=f"asl_citizen:c{i}",
            video_path=f"videos/c{i}.mp4",
            gloss=gloss,
            class_id=label_map.to_id(gloss),
            signer_id="signer01",
            split="train",
            dataset_name="asl_citizen",
        )
        for i, gloss in enumerate(["APPLE", "APPLE", "BOOK"])
    ]
    audit = audit_dataset(Manifest(records=records), label_map, tmp_path)
    assert any("exactly one training sample" in p for p in audit.problems)


# Report output ----------------------------------------------------------------


def test_report_is_serializable_and_reloadable(clean_audit, tmp_path):
    path = clean_audit.save(tmp_path / "audits" / "report.json")
    reloaded = json.loads(path.read_text())
    assert reloaded["counts"]["manifest_records"] == 12
    assert reloaded["audit_version"] == 1


def test_summary_is_human_readable(clean_audit):
    summary = clean_audit.summary()
    assert "Dataset audit: asl_citizen" in summary
    assert "classes" in summary
    assert "no problems found" in summary


def test_summary_surfaces_problems(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(parsed.manifest, parsed.label_map, synthetic_root, expected_classes=99)
    assert "PROBLEMS" in audit.summary()


def test_reconciles_annotation_rows_against_records(clean_audit):
    counts = clean_audit.report["counts"]
    assert counts["annotation_rows_total"] == 12
    assert counts["rows_dropped"] == 0


def test_reports_rows_that_produced_no_record(synthetic_root):
    """A row silently failing to become a record must be caught."""
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(
        parsed.manifest,
        parsed.label_map,
        synthetic_root,
        # Claim more rows were read than records exist.
        annotation_rows={"train": 9, "validation": 3, "test": 3},
    )
    assert audit.report["counts"]["rows_dropped"] == 3
    assert any("produced no manifest record" in p for p in audit.problems)
    assert any("split 'train': 9 annotation row(s) but 6" in p for p in audit.problems)


def test_records_source_identity_for_a_mirror(synthetic_root):
    """A hosted mirror's identity must be recorded, not assumed."""
    parsed = parse_annotations(synthetic_root)
    audit = audit_dataset(
        parsed.manifest,
        parsed.label_map,
        synthetic_root,
        extra={"source_id": "someuser/asl-citizen"},
    )
    assert audit.report["source_identity"]["source_id"] == "someuser/asl-citizen"
