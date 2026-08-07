#!/usr/bin/env python3
"""Audit an ASL Citizen dataset root and write a versioned report.

Reads the dataset's own split files, builds the label map and manifests, probes
the video files, and reports what the dataset actually contains. Nothing is
modified.

`docs/DATA_CONTRACT.md` requires this audit before full training. It is intended
to run on Kaggle, where the mirror is already attached read-only, per D-007.

Examples:

    # Full audit, writing manifests and label map alongside the report
    python scripts/audit_dataset.py \\
        --dataset-root /kaggle/input/asl-citizen \\
        --output-dir artifacts --write-manifests

    # Fast structural check: parse and validate, probe only 200 videos
    python scripts/audit_dataset.py \\
        --dataset-root /kaggle/input/asl-citizen --probe-limit 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_training.data.asl_citizen import (
    DatasetStructureError,
    parse_annotations,
    resolve_layout,
)
from asl_training.data.audit import audit_dataset
from asl_training.data.manifest import SPLITS, Manifest

logger = logging.getLogger("audit_dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("ASL_DATASET_ROOT"),
        help="dataset root; defaults to $ASL_DATASET_ROOT",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="where to write the audit report and, optionally, manifests",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=None,
        help=(
            "probe at most N videos. Omit for a complete audit; a limit produces a "
            "partial audit, which is not sufficient for a full training run."
        ),
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=None,
        help="verify the class count against a documented expectation",
    )
    parser.add_argument(
        "--configured-frames",
        type=int,
        default=16,
        help=(
            "frame count the baseline will train at; clips shorter than this are "
            "reported as needing the short-video policy (default: 16, per D-003)"
        ),
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="mirror identity, e.g. a Kaggle dataset slug, recorded in the report",
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="write per-split manifests and the label map",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="resolve and print the dataset layout, then stop",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset_root is None:
        logger.error(
            "no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT. "
            "Reusable modules never guess environment-specific paths."
        )
        return 2

    try:
        layout = resolve_layout(args.dataset_root)
    except DatasetStructureError as exc:
        logger.error("could not resolve dataset layout: %s", exc)
        return 1

    print("\nResolved layout:")
    print(f"  root       {layout.root}")
    for split, path in sorted(layout.split_files.items()):
        print(f"  {split:<10} {path.relative_to(layout.root)}")
    print(f"  videos     {layout.video_dir.relative_to(layout.root) if layout.video_dir else '?'}")

    missing_splits = [s for s in SPLITS if s not in layout.split_files]
    if missing_splits:
        print(f"\n  WARNING: no split file found for: {', '.join(missing_splits)}")

    if args.layout_only:
        return 0

    try:
        parsed = parse_annotations(args.dataset_root, layout=layout)
    except DatasetStructureError as exc:
        logger.error("could not parse annotations: %s", exc)
        return 1

    print("\nResolved columns:")
    for name, value in parsed.columns.to_dict().items():
        if name != "available_columns":
            print(f"  {name:<12} {value}")
    print(f"  available    {parsed.columns.available}")

    if parsed.problems:
        print(f"\nParse problems ({len(parsed.problems)}):")
        for problem in parsed.problems[:10]:
            print(f"  - {problem}")

    audit = audit_dataset(
        parsed.manifest,
        parsed.label_map,
        args.dataset_root,
        probe_limit=args.probe_limit,
        expected_classes=args.expected_classes,
        configured_frames=args.configured_frames,
        annotation_rows=parsed.rows_read,
        extra=_source_identity(args, parsed),
    )

    print()
    print(audit.summary())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit.save(args.output_dir / "audits" / "asl_citizen_audit.json")
    print(f"\nReport written to {report_path}")

    if args.write_manifests:
        _write_manifests(parsed, args.output_dir)

    # A structural failure blocks training. A partial audit does not fail the
    # command, but the report records that it was partial.
    if audit.report["integrity"]["errors"]:
        logger.error(
            "audit found %d integrity error(s). These block a full training run.",
            len(audit.report["integrity"]["errors"]),
        )
        return 1

    return 0


def _source_identity(args: argparse.Namespace, parsed) -> dict:
    return {
        "source_id": args.source_id,
        "annotation_rows": parsed.rows_read,
        "parse_problems": len(parsed.problems),
        "note": (
            "A hosted mirror must not be assumed identical to the official release. "
            "Compare these counts against official metadata before treating a run as "
            "a valid baseline."
        ),
    }


def _write_manifests(parsed, output_dir: Path) -> None:
    label_map_path = output_dir / "label_maps" / "asl_citizen.json"
    if label_map_path.exists():
        existing_note = " (already exists, not overwritten)"
        logger.warning(
            "label map already exists at %s; not overwriting, because that would "
            "invalidate checkpoints trained against it",
            label_map_path,
        )
    else:
        parsed.label_map.save(label_map_path)
        existing_note = ""
    print(f"Label map: {label_map_path}{existing_note}")
    print(f"  identity {parsed.label_map.identity}")

    for split in SPLITS:
        records = parsed.manifest.for_split(split)
        if not records:
            continue
        path = Manifest(records=records, dataset_name=parsed.manifest.dataset_name).to_csv(
            output_dir / "manifests" / f"asl_citizen_{split}.csv"
        )
        print(f"Manifest: {path} ({len(records)} records)")

    print(f"  manifest identity {parsed.manifest.identity}")


if __name__ == "__main__":
    raise SystemExit(main())
