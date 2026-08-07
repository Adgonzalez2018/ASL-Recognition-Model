"""ASL Citizen annotation parsing.

This module reads the dataset's own split files and turns them into manifest
records. It deliberately *discovers* structure rather than assuming it: column
names, file locations, and split file names all vary between the official release
and third-party mirrors, and `docs/DATA_CONTRACT.md` requires that a mirror be
audited rather than trusted to match.

Every assumption this module makes is either verified against the files or
reported. Nothing is silently defaulted.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .label_map import LabelMap
from .manifest import Manifest, ManifestRecord, normalize_split

logger = logging.getLogger(__name__)

DATASET_NAME = "asl_citizen"

# Candidate column names, lowercased and stripped of non-alphanumerics for
# matching. Mirrors rename these freely; the resolved mapping is reported so a
# reviewer can confirm the parser read what they think it read.
COLUMN_CANDIDATES = {
    "video_path": ("videofile", "video", "videoname", "filename", "file", "path", "videopath"),
    "gloss": ("gloss", "label", "sign", "word", "aslLex", "asllexcode", "signgloss"),
    "signer_id": (
        "participantid",
        "participant",
        "signerid",
        "signer",
        "userid",
        "user",
        "subject",
    ),
}

# Split files, by canonical split name. Matched case-insensitively on stem.
SPLIT_FILE_CANDIDATES = {
    "train": ("train", "training"),
    "validation": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing"),
}

VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi", ".mkv")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _canonical(name: str) -> str:
    return _NON_ALNUM.sub("", name.strip().casefold())


class DatasetStructureError(Exception):
    """Raised when the dataset layout cannot be resolved unambiguously."""


@dataclass
class ColumnMapping:
    """Resolved annotation columns, recorded so a reviewer can verify them."""

    video_path: str
    gloss: str
    signer_id: str | None
    available: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_path": self.video_path,
            "gloss": self.gloss,
            "signer_id": self.signer_id,
            "available_columns": self.available,
        }


@dataclass
class DatasetLayout:
    """Resolved locations within a dataset root."""

    root: Path
    split_files: dict[str, Path]
    video_dir: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "split_files": {k: str(v.relative_to(self.root)) for k, v in self.split_files.items()},
            "video_dir": str(self.video_dir.relative_to(self.root)) if self.video_dir else None,
        }


def resolve_layout(dataset_root: str | Path) -> DatasetLayout:
    """Locate split files and the video directory beneath a dataset root.

    Searches recursively, because mirrors commonly nest the real dataset one or
    two directories down inside the archive they were built from.

    Raises:
        DatasetStructureError: If the root is missing, no split files are found,
            or a split matches ambiguously.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetStructureError(
            f"dataset root does not exist or is not a directory: {root}. "
            f"Set ASL_DATASET_ROOT or pass --dataset-root."
        )

    csv_files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not csv_files:
        raise DatasetStructureError(f"no CSV annotation files found under {root}")

    split_files: dict[str, Path] = {}
    ambiguous: dict[str, list[Path]] = {}

    for split, candidates in SPLIT_FILE_CANDIDATES.items():
        matches = [p for p in csv_files if _canonical(p.stem) in candidates]
        if len(matches) == 1:
            split_files[split] = matches[0]
        elif len(matches) > 1:
            ambiguous[split] = matches

    if ambiguous:
        detail = "; ".join(
            f"{split}: {[str(p.relative_to(root)) for p in paths]}"
            for split, paths in ambiguous.items()
        )
        raise DatasetStructureError(
            f"ambiguous split files under {root} ({detail}). Point --dataset-root at "
            f"the specific directory containing one set of split files."
        )

    if not split_files:
        found = sorted({p.stem for p in csv_files})[:10]
        raise DatasetStructureError(
            f"no recognizable split files under {root}. Found CSV files named: "
            f"{', '.join(found)}. Expected names like train.csv, val.csv, test.csv."
        )

    video_dir = _find_video_dir(root)
    return DatasetLayout(root=root, split_files=split_files, video_dir=video_dir)


def _find_video_dir(root: Path) -> Path | None:
    """Locate the directory holding the video files, if there is an obvious one."""
    for name in ("videos", "video", "clips", "data"):
        candidate = root / name
        if candidate.is_dir():
            return candidate

    # Fall back to whichever directory holds the most video files.
    counts: dict[Path, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES:
            counts[path.parent] = counts.get(path.parent, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda p: counts[p])


def resolve_columns(header: list[str]) -> ColumnMapping:
    """Match annotation columns to the fields the manifest requires.

    Raises:
        DatasetStructureError: If a required column cannot be identified. Guessing
            would risk training on the wrong label.
    """
    lookup = {_canonical(name): name for name in header}
    resolved: dict[str, str | None] = {}

    for field_name, candidates in COLUMN_CANDIDATES.items():
        match = next((lookup[_canonical(c)] for c in candidates if _canonical(c) in lookup), None)
        resolved[field_name] = match

    missing = [f for f in ("video_path", "gloss") if resolved[f] is None]
    if missing:
        raise DatasetStructureError(
            f"could not identify required column(s) {missing} in annotation header "
            f"{header}. Guessing risks training against the wrong label; add the "
            f"column name to COLUMN_CANDIDATES in asl_citizen.py once verified."
        )

    if resolved["signer_id"] is None:
        # Not fatal here, but signer-independent evaluation is impossible without
        # it, and the audit reports this as a hard problem.
        logger.warning(
            "no signer column identified in header %s; signer-independent "
            "validation will not be possible",
            header,
        )

    return ColumnMapping(
        video_path=resolved["video_path"],
        gloss=resolved["gloss"],
        signer_id=resolved["signer_id"],
        available=list(header),
    )


@dataclass
class ParseResult:
    """Outcome of parsing the annotation files."""

    manifest: Manifest
    label_map: LabelMap
    layout: DatasetLayout
    columns: ColumnMapping
    rows_read: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _make_sample_id(split: str, video_path: str) -> str:
    """Derive a stable sample ID.

    Built from the dataset name and the source-relative video path, so it does
    not change when the runtime root changes. The construction is documented in
    docs/DATA_CONTRACT.md.
    """
    stem = Path(video_path).stem
    return f"{DATASET_NAME}:{stem}" if stem else f"{DATASET_NAME}:{split}:{video_path}"


def parse_annotations(
    dataset_root: str | Path,
    *,
    layout: DatasetLayout | None = None,
) -> ParseResult:
    """Read ASL Citizen split files into a manifest and a label map.

    The label map is built from the union of glosses across all splits, so class
    IDs mean the same thing everywhere.

    Problems are collected rather than raised, so one audit run reports every
    issue at once. Callers validate the resulting manifest to enforce integrity.
    """
    layout = layout or resolve_layout(dataset_root)

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    problems: list[str] = []
    columns: ColumnMapping | None = None

    for split, path in sorted(layout.split_files.items()):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                problems.append(f"{path.name}: no header row")
                continue

            mapping = resolve_columns(list(reader.fieldnames))
            if columns is None:
                columns = mapping
            elif (mapping.video_path, mapping.gloss) != (columns.video_path, columns.gloss):
                problems.append(
                    f"{path.name}: column layout differs from the first split file "
                    f"({mapping.to_dict()} vs {columns.to_dict()})"
                )

            rows_by_split[split] = [dict(row) for row in reader]

    if columns is None:
        raise DatasetStructureError("no readable annotation files")

    # One label map over every split, so class IDs are split-independent.
    all_glosses = [
        (row.get(columns.gloss) or "").strip() for rows in rows_by_split.values() for row in rows
    ]
    usable_glosses = [g for g in all_glosses if g]
    if len(usable_glosses) != len(all_glosses):
        problems.append(f"{len(all_glosses) - len(usable_glosses)} row(s) have an empty gloss")

    if not usable_glosses:
        raise DatasetStructureError("no usable glosses found in any split file")

    label_map = LabelMap.from_glosses(usable_glosses, dataset_name=DATASET_NAME)

    records: list[ManifestRecord] = []
    video_prefix = layout.video_dir.relative_to(layout.root).as_posix() if layout.video_dir else ""

    for split, rows in rows_by_split.items():
        for line, row in enumerate(rows, start=2):
            raw_path = (row.get(columns.video_path) or "").strip()
            gloss = (row.get(columns.gloss) or "").strip()

            if not raw_path or not gloss:
                problems.append(f"{split} line {line}: missing video path or gloss")
                continue

            # Annotation files may store a bare filename or a path already
            # relative to the root. Only prepend the video directory when the
            # value is a bare name, so we never double up the prefix.
            path_value = raw_path.replace("\\", "/").lstrip("/")
            if video_prefix and "/" not in path_value:
                path_value = f"{video_prefix}/{path_value}"

            signer = (row.get(columns.signer_id) or "").strip() if columns.signer_id else ""
            if not signer:
                signer = "unknown"

            try:
                records.append(
                    ManifestRecord(
                        sample_id=_make_sample_id(split, path_value),
                        video_path=path_value,
                        gloss=gloss,
                        class_id=label_map.to_id(gloss),
                        signer_id=signer,
                        split=normalize_split(split),
                        dataset_name=DATASET_NAME,
                        source_annotation_id=f"{split}:{line}",
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                problems.append(f"{split} line {line}: {exc}")

    if not records:
        raise DatasetStructureError("no usable records parsed from any split file")

    return ParseResult(
        manifest=Manifest(records=records, dataset_name=DATASET_NAME),
        label_map=label_map,
        layout=layout,
        columns=columns,
        rows_read={split: len(rows) for split, rows in rows_by_split.items()},
        problems=problems,
    )
