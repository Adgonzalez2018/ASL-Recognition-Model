"""Manifest schema, split integrity, and signer-leakage detection.

The validation under test is the project's main defense against experimentally
invalid states. These tests assert that invalid states are rejected, not merely
noted.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import pytest

from asl_training.data.label_map import LabelMap
from asl_training.data.manifest import (
    Manifest,
    ManifestRecord,
    ManifestValidationError,
    normalize_split,
)

GLOSSES = ["APPLE", "BOOK", "CAT"]


def make_record(
    sample_id: str = "s1",
    *,
    gloss: str = "APPLE",
    class_id: int = 0,
    signer: str = "signer01",
    split: str = "train",
    path: str | None = None,
    **extra,
) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        video_path=path or f"videos/{sample_id}.mp4",
        gloss=gloss,
        class_id=class_id,
        signer_id=signer,
        split=split,
        dataset_name="asl_citizen",
        **extra,
    )


def make_valid_manifest() -> Manifest:
    """A small, signer-independent, three-split manifest."""
    records = []
    plan = [
        ("train", ["signer01", "signer02"]),
        ("validation", ["signer03"]),
        ("test", ["signer04"]),
    ]
    counter = 0
    for split, signers in plan:
        for signer in signers:
            for class_id, gloss in enumerate(GLOSSES):
                counter += 1
                records.append(
                    make_record(
                        f"s{counter:03d}",
                        gloss=gloss,
                        class_id=class_id,
                        signer=signer,
                        split=split,
                    )
                )
    return Manifest(records=records)


# Split normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("train", "train"),
        ("TRAIN", "train"),
        ("training", "train"),
        ("val", "validation"),
        ("dev", "validation"),
        ("valid", "validation"),
        (" Validation ", "validation"),
        ("test", "test"),
    ],
)
def test_split_aliases_normalize(alias, expected):
    assert normalize_split(alias) == expected


def test_unknown_split_is_rejected():
    """An unrecognized split must never be silently dropped or defaulted."""
    with pytest.raises(ValueError, match="unknown split"):
        normalize_split("holdout")


# Record validation ------------------------------------------------------------


def test_valid_record_constructs():
    record = make_record()
    assert record.is_usable
    assert record.split == "train"


def test_rejects_absolute_video_path():
    """Absolute paths are environment-specific and must not enter a manifest."""
    with pytest.raises(ValueError, match="must be relative to the dataset root"):
        make_record(path="/kaggle/input/asl/videos/x.mp4")


def test_rejects_path_traversal():
    with pytest.raises(ValueError, match="must not escape"):
        make_record(path="../../etc/passwd")


def test_rejects_negative_class_id():
    with pytest.raises(ValueError, match="non-negative"):
        make_record(class_id=-1)


def test_rejects_bool_class_id():
    with pytest.raises(TypeError, match="must be an int"):
        make_record(class_id=True)


@pytest.mark.parametrize(
    "field_name", ["sample_id", "gloss", "signer_id", "video_path", "dataset_name"]
)
def test_rejects_empty_required_string(field_name):
    payload = {
        "sample_id": "s1",
        "video_path": "videos/s1.mp4",
        "gloss": "APPLE",
        "class_id": 0,
        "signer_id": "signer01",
        "split": "train",
        "dataset_name": "asl_citizen",
    }
    payload[field_name] = "  "
    with pytest.raises(ValueError, match=f"{field_name} must be a non-empty string"):
        ManifestRecord(**payload)


def test_rejects_unnormalized_split():
    with pytest.raises(ValueError, match="split must be one of"):
        make_record(split="val")


def test_from_dict_requires_all_required_fields():
    with pytest.raises(ValueError, match="missing required field"):
        ManifestRecord.from_dict({"sample_id": "s1", "gloss": "CAT"})


def test_from_dict_rejects_unknown_fields():
    payload = make_record().to_dict()
    payload["mystery_column"] = "x"
    with pytest.raises(ValueError, match="unknown manifest field"):
        ManifestRecord.from_dict(payload)


def test_from_dict_normalizes_split_alias():
    payload = make_record().to_dict()
    payload["split"] = "dev"
    assert ManifestRecord.from_dict(payload).split == "validation"


# Path resolution --------------------------------------------------------------


def test_resolves_against_runtime_root(tmp_path):
    record = make_record(path="videos/clip.mp4")
    assert record.resolve_path(tmp_path) == (tmp_path / "videos/clip.mp4").resolve()


def test_resolution_is_root_independent():
    """Changing the runtime root must not change sample identity."""
    record = make_record(path="videos/clip.mp4")
    assert record.resolve_path("/a").name == record.resolve_path("/b").name
    assert record.sample_id == "s1"


# Signer leakage ---------------------------------------------------------------


def test_clean_manifest_validates():
    report = make_valid_manifest().validate(LabelMap.from_glosses(GLOSSES))
    assert report.ok, report.errors
    report.raise_if_invalid()


def test_detects_signer_leakage_between_train_and_test():
    """The failure this project most needs to catch."""
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record("leak1", gloss="CAT", class_id=2, signer="signer01", split="test")
    )

    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert not report.ok
    assert any("signer01" in e and "train and test" in e for e in report.errors)


def test_detects_signer_leakage_between_train_and_validation():
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record("leak2", gloss="CAT", class_id=2, signer="signer02", split="validation")
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("train and validation" in e for e in report.errors)


def test_signer_leakage_raises():
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record("leak3", gloss="CAT", class_id=2, signer="signer01", split="test")
    )
    with pytest.raises(ManifestValidationError, match="signer"):
        manifest.validate(LabelMap.from_glosses(GLOSSES)).raise_if_invalid()


def test_signer_overlap_can_be_permitted_explicitly():
    """An override must exist, must be explicit, and must still be visible."""
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record("leak4", gloss="CAT", class_id=2, signer="signer01", split="test")
    )

    report = manifest.validate(LabelMap.from_glosses(GLOSSES), allow_signer_overlap=True)
    assert report.ok
    assert any("permitted by configuration" in w for w in report.warnings)


# Duplicates -------------------------------------------------------------------


def test_detects_duplicate_sample_ids():
    manifest = make_valid_manifest()
    manifest.records.append(make_record("s001", gloss="APPLE", class_id=0, signer="signer01"))
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("duplicate sample_id" in e for e in report.errors)


def test_detects_duplicate_video_paths():
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record("unique1", gloss="APPLE", class_id=0, signer="signer01", path="videos/s001.mp4")
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("duplicate video_path" in e for e in report.errors)


def test_detects_same_video_in_multiple_splits():
    """Distinct IDs pointing at one video still contaminates evaluation."""
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record(
            "dup1",
            gloss="APPLE",
            class_id=0,
            signer="signer04",
            split="test",
            path="videos/s001.mp4",
        )
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("multiple splits" in e for e in report.errors)


# Label agreement --------------------------------------------------------------


def test_detects_class_id_outside_label_map():
    manifest = make_valid_manifest()
    manifest.records.append(make_record("oob1", gloss="APPLE", class_id=99, signer="signer01"))
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("outside the label map" in e for e in report.errors)


def test_detects_gloss_to_id_disagreement_with_label_map():
    """A record claiming a different ID than the label map assigns."""
    manifest = Manifest(
        records=[
            make_record("a", gloss="APPLE", class_id=0, signer="s1"),
            make_record("b", gloss="BOOK", class_id=2, signer="s1"),
        ]
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("maps to class 1 in the label map" in e for e in report.errors)


def test_detects_one_gloss_with_two_ids():
    manifest = Manifest(
        records=[
            make_record("a", gloss="APPLE", class_id=0, signer="s1"),
            make_record("b", gloss="APPLE", class_id=1, signer="s1"),
        ]
    )
    report = manifest.validate()
    assert any("map to multiple IDs" in e for e in report.errors)


def test_detects_one_id_with_two_glosses():
    manifest = Manifest(
        records=[
            make_record("a", gloss="APPLE", class_id=0, signer="s1"),
            make_record("b", gloss="BOOK", class_id=0, signer="s1"),
        ]
    )
    report = manifest.validate()
    assert any("carry multiple glosses" in e for e in report.errors)


def test_warns_on_unused_label_map_classes():
    manifest = Manifest(records=[make_record("a", gloss="APPLE", class_id=0, signer="s1")])
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("no records in this manifest" in w for w in report.warnings)


def test_warns_on_non_contiguous_ids_without_label_map():
    manifest = Manifest(
        records=[
            make_record("a", gloss="APPLE", class_id=0, signer="s1"),
            make_record("b", gloss="CAT", class_id=5, signer="s1"),
        ]
    )
    report = manifest.validate()
    assert any("not contiguous" in w for w in report.warnings)


# Corruption -------------------------------------------------------------------


def test_warns_on_non_usable_records():
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record(
            "bad1", gloss="CAT", class_id=2, signer="signer01", corruption_status="unreadable"
        )
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert any("non-usable records" in w for w in report.warnings)


def test_corruption_warning_does_not_mask_a_hard_error():
    """Warnings must never conceal an integrity failure."""
    manifest = make_valid_manifest()
    manifest.records.append(
        make_record(
            "bad2",
            gloss="CAT",
            class_id=2,
            signer="signer01",
            split="test",
            corruption_status="unreadable",
        )
    )
    report = manifest.validate(LabelMap.from_glosses(GLOSSES))
    assert report.warnings
    assert not report.ok  # signer leakage still fails


# Empty and structural ---------------------------------------------------------


def test_empty_manifest_is_invalid():
    report = Manifest(records=[]).validate()
    assert not report.ok
    assert any("empty" in e for e in report.errors)


def test_warns_on_missing_split():
    manifest = Manifest(records=[make_record("a", signer="s1")])
    report = manifest.validate()
    assert any("'validation' has no records" in w for w in report.warnings)


# Queries ----------------------------------------------------------------------


def test_split_and_signer_queries():
    manifest = make_valid_manifest()
    assert manifest.split_counts() == {"train": 6, "validation": 3, "test": 3}
    assert manifest.signers("train") == {"signer01", "signer02"}
    assert manifest.signers("test") == {"signer04"}
    assert len(manifest.for_split("val")) == 3


def test_class_distribution():
    manifest = make_valid_manifest()
    assert manifest.class_distribution("train") == {0: 2, 1: 2, 2: 2}


def test_validation_report_counts():
    report = make_valid_manifest().validate(LabelMap.from_glosses(GLOSSES))
    assert report.counts["total_records"] == 12
    assert report.counts["classes"] == 3
    assert report.counts["signers"] == 4
    assert report.counts["signers_by_split"]["train"] == 2


# Identity ---------------------------------------------------------------------


def test_identity_is_order_independent():
    manifest = make_valid_manifest()
    shuffled = Manifest(records=list(reversed(manifest.records)))
    assert manifest.identity == shuffled.identity


def test_identity_changes_when_a_record_moves_split():
    manifest = make_valid_manifest()
    before = manifest.identity

    original = manifest.records[0]
    manifest.records[0] = make_record(
        original.sample_id,
        gloss=original.gloss,
        class_id=original.class_id,
        signer=original.signer_id,
        split="test",
        path=original.video_path,
    )
    assert manifest.identity != before


def test_identity_ignores_audited_metadata():
    """Resolution and codec describe videos, not experiment structure."""
    base = make_record("a", signer="s1")
    annotated = make_record("a", signer="s1", width=640, height=480, codec="h264")
    assert Manifest(records=[base]).identity == Manifest(records=[annotated]).identity


# CSV round trip ---------------------------------------------------------------


def test_csv_round_trip_preserves_identity(tmp_path):
    manifest = make_valid_manifest()
    path = manifest.to_csv(tmp_path / "train.csv")
    assert Manifest.from_csv(path).identity == manifest.identity


def test_csv_round_trip_preserves_optional_metadata(tmp_path):
    manifest = Manifest(
        records=[make_record("a", signer="s1", fps=29.97, frame_count=80, width=640)]
    )
    path = manifest.to_csv(tmp_path / "m.csv")
    restored = Manifest.from_csv(path).records[0]

    assert restored.fps == pytest.approx(29.97)
    assert restored.frame_count == 80
    assert restored.width == 640
    assert restored.height is None  # absent stays absent, never invented


def test_csv_rejects_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("sample_id,gloss\ns1,CAT\n")
    with pytest.raises(ValueError, match="missing required column"):
        Manifest.from_csv(path)


def test_csv_reports_the_offending_line(tmp_path):
    manifest = make_valid_manifest()
    path = manifest.to_csv(tmp_path / "m.csv")

    lines = path.read_text().splitlines()
    lines[2] = lines[2].replace("train", "holdout")
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="line 3"):
        Manifest.from_csv(path)


def test_csv_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("sample_id,video_path,gloss,class_id,signer_id,split,dataset_name\n")
    with pytest.raises(ValueError, match="contains no records"):
        Manifest.from_csv(path)


def test_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        Manifest.from_csv(tmp_path / "absent.csv")
