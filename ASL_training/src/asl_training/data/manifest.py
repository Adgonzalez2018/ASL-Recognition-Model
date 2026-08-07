"""Manifests: the source of truth consumed by training and evaluation.

Training code reads manifests. It must never re-derive labels, signers, or splits
from directory names, because doing so makes the dataset definition implicit and
unversionable.

The validation here exists to make experimentally invalid states impossible to
reach quietly: signer leakage, duplicated samples across splits, class IDs that
disagree with the label map.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .label_map import LabelMap

SPLITS = ("train", "validation", "test")

# Aliases normalized once, during manifest creation, rather than handled
# inconsistently across the project.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}

REQUIRED_FIELDS = (
    "sample_id",
    "video_path",
    "gloss",
    "class_id",
    "signer_id",
    "split",
    "dataset_name",
)


def normalize_split(value: str) -> str:
    """Map a split alias to its canonical name.

    Raises:
        ValueError: If the value is not a recognized split. An unknown split is
            never silently dropped or defaulted.
    """
    if not isinstance(value, str):
        raise ValueError(f"split must be a string, got {type(value).__name__}")

    canonical = SPLIT_ALIASES.get(value.strip().casefold())
    if canonical is None:
        raise ValueError(
            f"unknown split {value!r}; recognized values are {', '.join(sorted(SPLIT_ALIASES))}"
        )
    return canonical


@dataclass(frozen=True)
class ManifestRecord:
    """One isolated-sign video sample.

    Required fields identify the sample and its experimental role. Optional
    fields carry audited metadata and are left absent rather than invented when
    the source provides no reliable value.
    """

    sample_id: str
    video_path: str
    gloss: str
    class_id: int
    signer_id: str
    split: str
    dataset_name: str

    # Audited metadata. Absent means unknown, never assumed.
    source_annotation_id: str | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    corruption_status: str = "usable"
    handedness: str | None = None
    mirroring_status: str | None = None
    dataset_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "video_path", "gloss", "signer_id", "dataset_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise TypeError(f"class_id must be an int, got {type(self.class_id).__name__}")
        if self.class_id < 0:
            raise ValueError(f"class_id must be non-negative, got {self.class_id}")

        if self.split not in SPLITS:
            raise ValueError(
                f"split must be one of {SPLITS}, got {self.split!r}. Normalize "
                f"aliases with normalize_split() during manifest creation."
            )

        if Path(self.video_path).is_absolute():
            raise ValueError(
                f"video_path must be relative to the dataset root, got the absolute "
                f"path {self.video_path!r}. Absolute paths are environment-specific "
                f"and must not enter a committed manifest."
            )

        if ".." in Path(self.video_path).parts:
            raise ValueError(f"video_path must not escape the dataset root: {self.video_path!r}")

    @property
    def is_usable(self) -> bool:
        return self.corruption_status == "usable"

    def resolve_path(self, dataset_root: str | Path) -> Path:
        """Resolve against a runtime dataset root, refusing to escape it."""
        root = Path(dataset_root).resolve()
        resolved = (root / self.video_path).resolve()

        if not resolved.is_relative_to(root):
            raise ValueError(
                f"resolved path escapes the dataset root: {resolved} is outside {root}"
            )
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestRecord:
        missing = [f for f in REQUIRED_FIELDS if f not in raw or raw[f] in (None, "")]
        if missing:
            raise ValueError(f"record is missing required field(s): {', '.join(missing)}")

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown manifest field(s): {', '.join(sorted(unknown))}")

        payload = dict(raw)
        payload["class_id"] = int(payload["class_id"])
        payload["split"] = normalize_split(payload["split"])

        for name in ("duration_seconds", "fps"):
            if payload.get(name) not in (None, ""):
                payload[name] = float(payload[name])
        for name in ("frame_count", "width", "height"):
            if payload.get(name) not in (None, ""):
                payload[name] = int(payload[name])

        return cls(**payload)


@dataclass
class ValidationReport:
    """Outcome of validating a manifest set."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Raise on any integrity failure.

        Warnings never raise. They must not be able to conceal a hard failure,
        so they are reported separately.
        """
        if self.errors:
            listed = "\n  - ".join(self.errors)
            raise ManifestValidationError(
                f"manifest validation failed with {len(self.errors)} error(s):\n  - {listed}"
            )


class ManifestValidationError(Exception):
    """Raised when a manifest set would compromise experimental validity."""


@dataclass
class Manifest:
    """An ordered collection of manifest records."""

    records: list[ManifestRecord]
    dataset_name: str = "asl_citizen"
    dataset_version: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def for_split(self, split: str) -> list[ManifestRecord]:
        canonical = normalize_split(split)
        return [r for r in self.records if r.split == canonical]

    def signers(self, split: str | None = None) -> set[str]:
        records = self.for_split(split) if split else self.records
        return {r.signer_id for r in records}

    def class_ids(self, split: str | None = None) -> set[int]:
        records = self.for_split(split) if split else self.records
        return {r.class_id for r in records}

    def split_counts(self) -> dict[str, int]:
        return dict(Counter(r.split for r in self.records))

    def class_distribution(self, split: str | None = None) -> dict[int, int]:
        records = self.for_split(split) if split else self.records
        return dict(Counter(r.class_id for r in records))

    @property
    def identity(self) -> str:
        """A fingerprint over the records' experimentally meaningful content.

        Deliberately excludes audited metadata such as resolution and codec: those
        describe the videos, not the experiment's structure. It covers sample
        identity, path, label, signer, and split, which are the fields whose
        change alters what an experiment means.
        """
        payload = json.dumps(
            sorted(
                [r.sample_id, r.video_path, r.class_id, r.signer_id, r.split] for r in self.records
            ),
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.dataset_name}:{len(self.records)}:sha256:{digest}"

    # Validation ---------------------------------------------------------------

    def validate(
        self,
        label_map: LabelMap | None = None,
        *,
        allow_signer_overlap: bool = False,
    ) -> ValidationReport:
        """Check every condition that could compromise experimental validity.

        Args:
            label_map: When supplied, class IDs are checked against it.
            allow_signer_overlap: Permit signers shared across splits. Off by
                default. ASL Citizen's official protocol is signer-independent,
                so overlap normally indicates a construction bug.

        Returns:
            A report. Call ``raise_if_invalid`` to enforce it.
        """
        report = ValidationReport()

        if not self.records:
            report.errors.append("manifest is empty")
            return report

        self._check_uniqueness(report)
        self._check_splits(report)
        self._check_signers(report, allow_signer_overlap)
        self._check_labels(report, label_map)
        self._check_corruption(report)

        report.counts = {
            "total_records": len(self.records),
            "by_split": self.split_counts(),
            "classes": len(self.class_ids()),
            "signers": len(self.signers()),
            "signers_by_split": {s: len(self.signers(s)) for s in SPLITS if self.for_split(s)},
            "manifest_identity": self.identity,
        }
        return report

    def _check_uniqueness(self, report: ValidationReport) -> None:
        for label, values in (
            ("sample_id", [r.sample_id for r in self.records]),
            ("video_path", [r.video_path for r in self.records]),
        ):
            duplicates = [v for v, n in Counter(values).items() if n > 1]
            if duplicates:
                shown = ", ".join(sorted(duplicates)[:5])
                report.errors.append(
                    f"{len(duplicates)} duplicate {label}(s): {shown}"
                    + (" ..." if len(duplicates) > 5 else "")
                )

        # The same sample appearing in more than one split contaminates
        # evaluation even when the IDs happen to differ.
        by_path: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            by_path[record.video_path].add(record.split)
        straddling = {p: s for p, s in by_path.items() if len(s) > 1}
        if straddling:
            shown = ", ".join(f"{p} in {sorted(s)}" for p, s in list(straddling.items())[:3])
            report.errors.append(f"{len(straddling)} video(s) appear in multiple splits: {shown}")

    def _check_splits(self, report: ValidationReport) -> None:
        present = set(self.split_counts())
        unknown = present - set(SPLITS)
        if unknown:
            report.errors.append(f"unknown split value(s): {', '.join(sorted(unknown))}")

        for split in SPLITS:
            if split not in present:
                report.warnings.append(f"split {split!r} has no records")

    def _check_signers(self, report: ValidationReport, allow_overlap: bool) -> None:
        by_split = {s: self.signers(s) for s in SPLITS if self.for_split(s)}

        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if a not in by_split or b not in by_split:
                continue
            shared = by_split[a] & by_split[b]
            if not shared:
                continue

            message = (
                f"{len(shared)} signer(s) appear in both {a} and {b}: "
                f"{', '.join(sorted(shared)[:5])}" + (" ..." if len(shared) > 5 else "")
            )
            if allow_overlap:
                report.warnings.append(message + " (permitted by configuration)")
            else:
                report.errors.append(
                    message + ". Signer leakage invalidates signer-independent evaluation. "
                    "If the official protocol permits this overlap, set "
                    "allow_signer_overlap and document the exception."
                )

    def _check_labels(self, report: ValidationReport, label_map: LabelMap | None) -> None:
        observed = self.class_ids()

        if label_map is not None:
            valid = set(label_map.class_ids)
            out_of_range = sorted(observed - valid)
            if out_of_range:
                report.errors.append(
                    f"{len(out_of_range)} class ID(s) outside the label map "
                    f"(0 to {label_map.num_classes - 1}): {out_of_range[:5]}"
                )

            unused = valid - observed
            if unused:
                report.warnings.append(
                    f"{len(unused)} label-map class(es) have no records in this manifest"
                )

            # A gloss must map to the same ID everywhere, or class IDs mean
            # different things in different rows.
            for record in self.records:
                if record.gloss not in label_map:
                    report.errors.append(
                        f"sample {record.sample_id}: gloss {record.gloss!r} is not in the label map"
                    )
                    break
                if label_map.to_id(record.gloss) != record.class_id:
                    report.errors.append(
                        f"sample {record.sample_id}: gloss {record.gloss!r} maps to "
                        f"class {label_map.to_id(record.gloss)} in the label map but "
                        f"the record says {record.class_id}"
                    )
                    break
        else:
            # Without a label map, contiguity is still checkable and a gap
            # usually means classes were dropped somewhere upstream.
            expected = set(range(max(observed) + 1))
            gaps = sorted(expected - observed)
            if gaps:
                report.warnings.append(
                    f"class IDs are not contiguous; {len(gaps)} unused ID(s) below the "
                    f"maximum, starting at {gaps[:5]}"
                )

        # One gloss must not map to two IDs, and one ID must not carry two glosses.
        gloss_to_ids: dict[str, set[int]] = defaultdict(set)
        id_to_glosses: dict[int, set[str]] = defaultdict(set)
        for record in self.records:
            gloss_to_ids[record.gloss].add(record.class_id)
            id_to_glosses[record.class_id].add(record.gloss)

        inconsistent = {g: i for g, i in gloss_to_ids.items() if len(i) > 1}
        if inconsistent:
            shown = ", ".join(f"{g!r} -> {sorted(i)}" for g, i in list(inconsistent.items())[:3])
            report.errors.append(f"{len(inconsistent)} gloss(es) map to multiple IDs: {shown}")

        collided = {i: g for i, g in id_to_glosses.items() if len(g) > 1}
        if collided:
            shown = ", ".join(f"{i} <- {sorted(g)}" for i, g in list(collided.items())[:3])
            report.errors.append(f"{len(collided)} class ID(s) carry multiple glosses: {shown}")

    def _check_corruption(self, report: ValidationReport) -> None:
        statuses = Counter(r.corruption_status for r in self.records)
        unusable = {s: n for s, n in statuses.items() if s != "usable"}
        if unusable:
            summary = ", ".join(f"{n} {s}" for s, n in sorted(unusable.items()))
            report.warnings.append(
                f"manifest contains non-usable records: {summary}. These must be "
                f"excluded through a counted policy, not skipped at training time."
            )

    # Serialization ------------------------------------------------------------

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [f.name for f in fields(ManifestRecord)]

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in self.records:
                writer.writerow({k: ("" if v is None else v) for k, v in record.to_dict().items()})
        return path

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        dataset_name: str = "asl_citizen",
    ) -> Manifest:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")

        records: list[ManifestRecord] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ValueError(f"{path}: manifest has no header row")
            missing = [c for c in REQUIRED_FIELDS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"{path}: manifest is missing required column(s): {', '.join(missing)}"
                )

            for line, row in enumerate(reader, start=2):
                cleaned = {k: v for k, v in row.items() if v not in ("", None)}
                try:
                    records.append(ManifestRecord.from_dict(cleaned))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path} line {line}: {exc}") from exc

        if not records:
            raise ValueError(f"{path}: manifest contains no records")

        return cls(records=records, dataset_name=dataset_name)
