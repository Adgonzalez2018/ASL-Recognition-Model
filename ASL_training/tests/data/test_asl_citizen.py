"""ASL Citizen annotation parsing.

Exercised against a synthetic dataset with real encoded video files, so path
resolution and structure discovery are tested against actual files.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from asl_training.data.asl_citizen import (
    DatasetStructureError,
    parse_annotations,
    resolve_columns,
    resolve_layout,
)

from .conftest import GLOSSES, write_video

# Layout discovery -------------------------------------------------------------


def test_resolves_split_files_and_video_dir(synthetic_root):
    layout = resolve_layout(synthetic_root)
    assert set(layout.split_files) == {"train", "validation", "test"}
    assert layout.split_files["validation"].name == "val.csv"
    assert layout.video_dir.name == "videos"


def test_finds_dataset_nested_inside_the_root(synthetic_root, tmp_path):
    """Mirrors commonly nest the dataset a directory or two down."""
    import shutil

    nested = tmp_path / "outer" / "asl_citizen_v1"
    nested.parent.mkdir(parents=True)
    shutil.copytree(synthetic_root, nested)

    layout = resolve_layout(tmp_path / "outer")
    assert set(layout.split_files) == {"train", "validation", "test"}


def test_missing_root_raises_actionable_error(tmp_path):
    with pytest.raises(DatasetStructureError, match="ASL_DATASET_ROOT"):
        resolve_layout(tmp_path / "nope")


def test_root_without_csv_files_raises(tmp_path):
    (tmp_path / "videos").mkdir()
    with pytest.raises(DatasetStructureError, match="no CSV annotation files"):
        resolve_layout(tmp_path)


def test_unrecognized_split_names_raise_with_the_names_found(tmp_path):
    (tmp_path / "everything.csv").write_text("a,b\n1,2\n")
    with pytest.raises(DatasetStructureError, match="everything"):
        resolve_layout(tmp_path)


def test_ambiguous_split_files_raise_rather_than_guessing(synthetic_root, tmp_path):
    """Two train.csv files must not be silently disambiguated."""
    import shutil

    outer = tmp_path / "outer"
    shutil.copytree(synthetic_root, outer / "copy_a")
    shutil.copytree(synthetic_root, outer / "copy_b")

    with pytest.raises(DatasetStructureError, match="ambiguous split files"):
        resolve_layout(outer)


# Column resolution ------------------------------------------------------------


def test_resolves_official_column_names():
    mapping = resolve_columns(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
    assert mapping.video_path == "Video file"
    assert mapping.gloss == "Gloss"
    assert mapping.signer_id == "Participant ID"


@pytest.mark.parametrize(
    "header",
    [
        ["signer_id", "video_path", "gloss"],
        ["SignerID", "Filename", "Label"],
        ["user", "video", "sign"],
    ],
)
def test_resolves_common_mirror_renamings(header):
    mapping = resolve_columns(header)
    assert mapping.video_path is not None
    assert mapping.gloss is not None


def test_unresolvable_columns_raise_rather_than_guessing():
    """Guessing the label column risks training against the wrong target."""
    with pytest.raises(DatasetStructureError, match="could not identify required column"):
        resolve_columns(["col_a", "col_b", "col_c"])


def test_missing_signer_column_is_tolerated_but_recorded():
    mapping = resolve_columns(["Video file", "Gloss"])
    assert mapping.signer_id is None


# Parsing ----------------------------------------------------------------------


def test_parses_all_splits(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    assert len(parsed.manifest) == 12
    assert parsed.manifest.split_counts() == {"train": 6, "validation": 3, "test": 3}
    assert not parsed.problems


def test_builds_one_label_map_across_every_split(synthetic_root):
    """Class IDs must mean the same thing in train, validation, and test."""
    parsed = parse_annotations(synthetic_root)
    assert parsed.label_map.num_classes == len(GLOSSES)
    assert set(parsed.label_map.glosses) == set(GLOSSES)

    for split in ("train", "validation", "test"):
        for record in parsed.manifest.for_split(split):
            assert parsed.label_map.to_id(record.gloss) == record.class_id


def test_parsed_manifest_passes_validation(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    report = parsed.manifest.validate(parsed.label_map)
    assert report.ok, report.errors


def test_preserves_signer_ids(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    assert parsed.manifest.signers("train") == {"signer01", "signer02"}
    assert parsed.manifest.signers("test") == {"signer04"}


def test_splits_are_signer_independent(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    train = parsed.manifest.signers("train")
    assert not train & parsed.manifest.signers("test")
    assert not train & parsed.manifest.signers("validation")


def test_video_paths_are_relative_and_resolve(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    for record in parsed.manifest:
        assert not Path(record.video_path).is_absolute()
        assert record.resolve_path(synthetic_root).exists()


def test_video_prefix_is_not_doubled(synthetic_root):
    """A bare filename gets the video dir; an already-relative path does not."""
    parsed = parse_annotations(synthetic_root)
    for record in parsed.manifest:
        assert record.video_path.count("videos/") == 1


def test_sample_ids_are_unique_and_root_independent(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    ids = [r.sample_id for r in parsed.manifest]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("asl_citizen:") for i in ids)
    # No runtime path component leaks into identity.
    assert all(str(synthetic_root) not in i for i in ids)


def test_records_source_annotation_line(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    record = parsed.manifest.records[0]
    assert record.source_annotation_id.split(":")[0] in {"train", "validation", "test"}


def test_records_rows_read_per_split(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    assert parsed.rows_read == {"train": 6, "validation": 3, "test": 3}


# Malformed input --------------------------------------------------------------


def _write_dataset(root: Path, rows: list[dict], header: list[str]) -> Path:
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row.get("Video file")
        if name:
            write_video(videos / name, frames=8)

    path = root / "train.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


HEADER = ["Participant ID", "Video file", "Gloss"]


def test_reports_rows_with_empty_gloss(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {"Participant ID": "s1", "Video file": "a.mp4", "Gloss": "APPLE"},
            {"Participant ID": "s1", "Video file": "b.mp4", "Gloss": ""},
            {"Participant ID": "s1", "Video file": "c.mp4", "Gloss": "BOOK"},
        ],
        HEADER,
    )
    parsed = parse_annotations(tmp_path)

    assert len(parsed.manifest) == 2
    assert any("empty gloss" in p for p in parsed.problems)
    assert any("missing video path or gloss" in p for p in parsed.problems)


def test_reports_rows_with_missing_video_path(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {"Participant ID": "s1", "Video file": "a.mp4", "Gloss": "APPLE"},
            {"Participant ID": "s1", "Video file": "", "Gloss": "BOOK"},
        ],
        HEADER,
    )
    parsed = parse_annotations(tmp_path)
    assert len(parsed.manifest) == 1
    assert any("line 3" in p for p in parsed.problems)


def test_missing_signer_becomes_unknown_not_invented(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {"Video file": "a.mp4", "Gloss": "APPLE"},
            {"Video file": "b.mp4", "Gloss": "BOOK"},
        ],
        ["Video file", "Gloss"],
    )
    parsed = parse_annotations(tmp_path)
    assert parsed.manifest.signers() == {"unknown"}
    assert parsed.columns.signer_id is None


def test_all_rows_unusable_raises(tmp_path):
    _write_dataset(
        tmp_path,
        [{"Participant ID": "s1", "Video file": "a.mp4", "Gloss": ""}],
        HEADER,
    )
    with pytest.raises(DatasetStructureError, match="no usable glosses"):
        parse_annotations(tmp_path)


def test_handles_utf8_bom(tmp_path):
    """Exported CSVs frequently carry a BOM, which corrupts the first header."""
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    write_video(videos / "a.mp4", frames=8)
    write_video(videos / "b.mp4", frames=8)

    (tmp_path / "train.csv").write_text(
        "﻿Participant ID,Video file,Gloss\ns1,a.mp4,APPLE\ns1,b.mp4,BOOK\n",
        encoding="utf-8",
    )
    parsed = parse_annotations(tmp_path)
    assert len(parsed.manifest) == 2
    assert parsed.columns.signer_id == "Participant ID"


def test_handles_windows_path_separators(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    write_video(videos / "a.mp4", frames=8)

    (tmp_path / "train.csv").write_text(
        "Participant ID,Video file,Gloss\ns1,videos\\a.mp4,APPLE\ns1,b.mp4,BOOK\n"
    )
    parsed = parse_annotations(tmp_path)
    assert parsed.manifest.records[0].video_path == "videos/a.mp4"
