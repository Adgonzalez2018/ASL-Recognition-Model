"""Audit script entry point.

The exit code is the contract: an integrity failure must block a training run,
so it must be non-zero and must not be reported as success.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_dataset.py"


@pytest.fixture(scope="module")
def audit_script():
    spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_dataset"] = module
    spec.loader.exec_module(module)
    return module


def test_clean_dataset_exits_zero(audit_script, synthetic_root, tmp_path):
    code = audit_script.main(["--dataset-root", str(synthetic_root), "--output-dir", str(tmp_path)])
    assert code == 0


def test_writes_report(audit_script, synthetic_root, tmp_path):
    audit_script.main(["--dataset-root", str(synthetic_root), "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / "audits" / "asl_citizen_audit.json").read_text())
    assert report["counts"]["manifest_records"] == 12
    assert report["counts"]["rows_dropped"] == 0


def test_writes_manifests_and_label_map(audit_script, synthetic_root, tmp_path):
    audit_script.main(
        [
            "--dataset-root",
            str(synthetic_root),
            "--output-dir",
            str(tmp_path),
            "--write-manifests",
        ]
    )

    label_map = tmp_path / "label_maps" / "asl_citizen.json"
    assert label_map.exists()

    for split, expected in (("train", 6), ("validation", 3), ("test", 3)):
        path = tmp_path / "manifests" / f"asl_citizen_{split}.csv"
        assert path.exists()
        assert len(path.read_text().splitlines()) == expected + 1  # header


def test_written_manifests_reload_and_validate(audit_script, synthetic_root, tmp_path):
    """The artifacts must be usable by the training layer, not just written."""
    from asl_training.data import LabelMap, Manifest

    audit_script.main(
        [
            "--dataset-root",
            str(synthetic_root),
            "--output-dir",
            str(tmp_path),
            "--write-manifests",
        ]
    )

    label_map = LabelMap.load(tmp_path / "label_maps" / "asl_citizen.json")
    records = []
    for split in ("train", "validation", "test"):
        records.extend(Manifest.from_csv(tmp_path / "manifests" / f"asl_citizen_{split}.csv"))

    report = Manifest(records=records).validate(label_map)
    assert report.ok, report.errors


def test_does_not_overwrite_an_existing_label_map(audit_script, synthetic_root, tmp_path):
    """Overwriting would invalidate checkpoints trained against the old map."""
    args = [
        "--dataset-root",
        str(synthetic_root),
        "--output-dir",
        str(tmp_path),
        "--write-manifests",
    ]
    audit_script.main(args)
    path = tmp_path / "label_maps" / "asl_citizen.json"
    original = path.read_text()

    audit_script.main(args)  # second run must not clobber it
    assert path.read_text() == original


def test_signer_leakage_exits_nonzero(audit_script, synthetic_root, tmp_path):
    """The failure that must block training."""
    import csv
    import shutil

    root = tmp_path / "leaky"
    shutil.copytree(synthetic_root, root)

    path = root / "test.csv"
    rows = list(csv.DictReader(path.open()))
    header = list(rows[0])
    rows.append(
        {
            "Participant ID": "signer01",  # already in train
            "Video file": "clip001.mp4",
            "Gloss": "APPLE",
            "ASL-LEX Code": "lex_apple",
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    code = audit_script.main(["--dataset-root", str(root), "--output-dir", str(tmp_path / "out")])
    assert code == 1


def test_missing_dataset_root_exits_nonzero(audit_script, tmp_path):
    assert audit_script.main(["--dataset-root", str(tmp_path / "absent")]) == 1


def test_no_dataset_root_at_all_exits_nonzero(audit_script, monkeypatch):
    monkeypatch.delenv("ASL_DATASET_ROOT", raising=False)
    assert audit_script.parse_args([]).dataset_root is None


def test_layout_only_stops_before_parsing(audit_script, synthetic_root, tmp_path):
    code = audit_script.main(
        ["--dataset-root", str(synthetic_root), "--output-dir", str(tmp_path), "--layout-only"]
    )
    assert code == 0
    assert not (tmp_path / "audits").exists()


def test_probe_limit_produces_a_partial_audit(audit_script, synthetic_root, tmp_path):
    audit_script.main(
        [
            "--dataset-root",
            str(synthetic_root),
            "--output-dir",
            str(tmp_path),
            "--probe-limit",
            "2",
        ]
    )
    report = json.loads((tmp_path / "audits" / "asl_citizen_audit.json").read_text())
    assert report["media"]["probed"] == 2
    assert not report["media"]["complete"]
    assert any("partial audit" in p for p in report["problems"])
