"""Dataset audit.

Produces the versioned report that `docs/DATA_CONTRACT.md` requires before full
training. The audit answers what the dataset actually contains, rather than what
the paper or the mirror's description claims.

The audit never modifies raw dataset files.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .label_map import LabelMap
from .manifest import SPLITS, Manifest

logger = logging.getLogger(__name__)


@dataclass
class VideoProbe:
    """Decoded properties of one video file."""

    exists: bool
    status: str
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    rotation: int | None = None
    error: str | None = None


def probe_video(path: Path) -> VideoProbe:
    """Read a video's properties without decoding every frame.

    Uses PyAV, reading container and stream metadata. Falls back to a decode
    attempt when the container reports no frame count, which some webcam
    recordings do.
    """
    if not path.exists():
        return VideoProbe(exists=False, status="missing")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("video probing requires 'av'. Install project dependencies.") from exc

    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if not streams:
                return VideoProbe(exists=True, status="no_video_stream")

            stream = streams[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            frames = stream.frames or None

            # Some recordings report zero frames in the container header.
            # Counting packets is cheaper than full decode and usually accurate.
            if not frames:
                try:
                    frames = sum(1 for _ in container.demux(stream) if True)
                except Exception:
                    frames = None

            duration = None
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif frames and fps:
                duration = frames / fps

            rotation = None
            if stream.metadata:
                raw_rotation = stream.metadata.get("rotate")
                if raw_rotation is not None:
                    try:
                        rotation = int(raw_rotation)
                    except ValueError:
                        rotation = None

            if not frames:
                status = "zero_frames"
            elif duration is not None and duration <= 0:
                status = "invalid_duration"
            else:
                status = "usable"

            return VideoProbe(
                exists=True,
                status=status,
                frame_count=frames,
                fps=fps,
                duration_seconds=duration,
                width=stream.codec_context.width or None,
                height=stream.codec_context.height or None,
                codec=stream.codec_context.name,
                rotation=rotation,
            )
    except Exception as exc:
        return VideoProbe(exists=True, status="unreadable", error=f"{type(exc).__name__}: {exc}")


def _distribution(values: list[float]) -> dict[str, float] | None:
    """Summarize a numeric distribution."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None

    def percentile(p: float) -> float:
        index = min(int(p * (len(clean) - 1)), len(clean) - 1)
        return round(clean[index], 4)

    return {
        "count": len(clean),
        "min": round(clean[0], 4),
        "p05": percentile(0.05),
        "median": round(statistics.median(clean), 4),
        "p95": percentile(0.95),
        "max": round(clean[-1], 4),
        "mean": round(statistics.fmean(clean), 4),
    }


@dataclass
class DatasetAudit:
    """A versioned record of what the dataset actually contains."""

    report: dict[str, Any] = field(default_factory=dict)

    @property
    def problems(self) -> list[str]:
        return self.report.get("problems", [])

    @property
    def ok(self) -> bool:
        return not self.problems

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")
        return path

    def summary(self) -> str:
        """A human-readable digest for the terminal."""
        r = self.report
        counts = r.get("counts", {})
        lines = [
            f"Dataset audit: {r.get('dataset_name')}",
            f"  root            {r.get('dataset_root')}",
        ]

        if "annotation_rows_total" in counts:
            lines.append(f"  annotation rows {counts['annotation_rows_total']}")
            if counts.get("rows_dropped"):
                lines.append(f"  rows dropped    {counts['rows_dropped']}")

        lines += [
            f"  records         {counts.get('manifest_records')}",
            f"  classes         {counts.get('classes')}",
            f"  signers         {counts.get('signers')}",
        ]

        by_split = counts.get("by_split", {})
        for split in SPLITS:
            if split in by_split:
                signers = counts.get("signers_by_split", {}).get(split, "?")
                classes = counts.get("classes_by_split", {}).get(split, "?")
                lines.append(
                    f"  {split:<11} {by_split[split]:>7} samples, "
                    f"{classes} classes, {signers} signers"
                )

        media = r.get("media", {})
        if media.get("probed"):
            lines.append(f"  probed          {media['probed']} videos")
            for status, count in sorted(media.get("status_counts", {}).items()):
                lines.append(f"    {status:<13} {count}")

        integrity = r.get("integrity", {})
        if integrity.get("errors"):
            lines.append(f"  INTEGRITY ERRORS: {len(integrity['errors'])}")
            lines.extend(f"    - {e}" for e in integrity["errors"][:5])

        if self.problems:
            lines.append(f"  PROBLEMS: {len(self.problems)}")
            lines.extend(f"    - {p}" for p in self.problems[:10])
        else:
            lines.append("  no problems found")

        return "\n".join(lines)


def audit_dataset(
    manifest: Manifest,
    label_map: LabelMap,
    dataset_root: str | Path,
    *,
    probe_limit: int | None = None,
    expected_classes: int | None = None,
    configured_frames: int = 16,
    annotation_rows: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> DatasetAudit:
    """Audit a parsed dataset against its files.

    Args:
        manifest: Parsed records.
        label_map: The vocabulary built from those records.
        dataset_root: Runtime root the paths resolve against.
        probe_limit: Probe at most this many videos. ``None`` probes every video,
            which is what a real audit should do; a limit produces a partial
            audit, labeled as such in the report.
        expected_classes: Verify the class count against a documented expectation.
            A mismatch is reported, not corrected.
        configured_frames: The frame count the baseline will train at. Clips
            shorter than this are reported as needing the short-video policy.
        annotation_rows: Rows read per split. Compared against records produced,
            so rows dropped during parsing cannot pass unnoticed.
        extra: Additional identity fields, such as the mirror's Kaggle slug.

    Returns:
        The audit. It reports; it never alters the dataset or the manifest.
    """
    root = Path(dataset_root).resolve()
    problems: list[str] = []

    validation = manifest.validate(label_map)
    problems.extend(validation.errors)

    counts: dict[str, Any] = {
        "manifest_records": len(manifest),
        "classes": len(manifest.class_ids()),
        "signers": len(manifest.signers()),
        "by_split": manifest.split_counts(),
        "classes_by_split": {
            s: len(manifest.class_ids(s)) for s in SPLITS if manifest.for_split(s)
        },
        "signers_by_split": {s: len(manifest.signers(s)) for s in SPLITS if manifest.for_split(s)},
    }

    # Rows read against records produced. A gap means rows were dropped during
    # parsing, which must never pass unnoticed.
    if annotation_rows:
        counts["annotation_rows"] = dict(annotation_rows)
        counts["annotation_rows_total"] = sum(annotation_rows.values())

        dropped = counts["annotation_rows_total"] - len(manifest)
        counts["rows_dropped"] = dropped
        if dropped:
            problems.append(
                f"{dropped} annotation row(s) produced no manifest record "
                f"({counts['annotation_rows_total']} rows read, {len(manifest)} records). "
                f"Every dropped row must be accounted for before training."
            )

        for split, rows in annotation_rows.items():
            produced = len(manifest.for_split(split))
            if produced != rows:
                problems.append(
                    f"split {split!r}: {rows} annotation row(s) but {produced} record(s)"
                )

    if expected_classes is not None and label_map.num_classes != expected_classes:
        problems.append(
            f"class count is {label_map.num_classes} but {expected_classes} was expected. "
            f"The dataset governs; update the expectation only after confirming why."
        )

    # Class balance. ASL Citizen is not uniform, and the tail drives macro F1.
    distribution = manifest.class_distribution()
    per_class = sorted(distribution.values())
    class_balance = {
        "min_samples_per_class": per_class[0] if per_class else 0,
        "max_samples_per_class": per_class[-1] if per_class else 0,
        "median_samples_per_class": statistics.median(per_class) if per_class else 0,
        "classes_with_one_sample": sum(1 for c in per_class if c == 1),
        "imbalance_ratio": (
            round(per_class[-1] / per_class[0], 2) if per_class and per_class[0] else None
        ),
    }

    singletons_in_train = [
        class_id for class_id, count in manifest.class_distribution("train").items() if count == 1
    ]
    if singletons_in_train:
        problems.append(
            f"{len(singletons_in_train)} class(es) have exactly one training sample; "
            f"per-class metrics for these will be extremely noisy"
        )

    missing_in_train = set(label_map.class_ids) - manifest.class_ids("train")
    if missing_in_train:
        problems.append(
            f"{len(missing_in_train)} class(es) have no training samples but exist in "
            f"the label map; the model cannot learn them"
        )

    # Media probing.
    records = list(manifest.records)
    to_probe = records if probe_limit is None else records[:probe_limit]
    probes = [(r, probe_video(r.resolve_path(root))) for r in to_probe]

    status_counts = Counter(p.status for _, p in probes)
    unusable = [(r, p) for r, p in probes if p.status != "usable"]
    if unusable:
        shown = ", ".join(f"{r.sample_id} ({p.status})" for r, p in unusable[:5])
        problems.append(f"{len(unusable)} video(s) are not usable: {shown}")

    rotated = [r.sample_id for r, p in probes if p.rotation]
    if rotated:
        problems.append(
            f"{len(rotated)} video(s) carry rotation metadata. Decoders differ in "
            f"whether they apply it, so orientation must be verified before training."
        )

    media: dict[str, Any] = {
        "probed": len(probes),
        "complete": probe_limit is None or probe_limit >= len(records),
        "status_counts": dict(status_counts),
        "duration_seconds": _distribution([p.duration_seconds for _, p in probes]),
        "frame_count": _distribution([p.frame_count for _, p in probes]),
        "fps": _distribution([p.fps for _, p in probes]),
        "resolutions": dict(
            Counter(f"{p.width}x{p.height}" for _, p in probes if p.width).most_common(10)
        ),
        "codecs": dict(Counter(p.codec for _, p in probes if p.codec)),
        "rotation_metadata_count": len(rotated),
    }

    if not media["complete"]:
        problems.append(
            f"partial audit: only {len(probes)} of {len(records)} videos were probed. "
            f"A full training run requires a complete audit."
        )

    # Short videos matter because the short-video policy depends on them.
    # Counts are reported at both the configured frame count and the likely
    # later one, but only the configured count raises a problem: a clip shorter
    # than a frame count nobody is training at yet is information, not an issue.
    frame_counts = [p.frame_count for _, p in probes if p.frame_count]
    if frame_counts:
        for threshold in sorted({configured_frames, 32}):
            short = sum(1 for f in frame_counts if f < threshold)
            media[f"videos_under_{threshold}_frames"] = short

        below_configured = sum(1 for f in frame_counts if f < configured_frames)
        if below_configured:
            problems.append(
                f"{below_configured} probed video(s) have fewer than the configured "
                f"{configured_frames} frames; the short-video policy must handle "
                f"these explicitly rather than dropping them"
            )

    handedness = {r.handedness for r in records if r.handedness}
    mirroring = {r.mirroring_status for r in records if r.mirroring_status}

    report: dict[str, Any] = {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": manifest.dataset_name,
        "dataset_root": str(root),
        "manifest_identity": manifest.identity,
        "label_map_identity": label_map.identity,
        "counts": counts,
        "class_balance": class_balance,
        "media": media,
        "integrity": {
            "errors": validation.errors,
            "warnings": validation.warnings,
            "signer_independent": not any("signer" in e for e in validation.errors),
        },
        "metadata_availability": {
            "handedness": sorted(handedness) or None,
            "mirroring_status": sorted(mirroring) or None,
            "note": (
                "Absent means the dataset provided no reliable value. Handedness and "
                "mirroring must not be inferred; a class-aware flip policy requires "
                "review, per docs/DATA_CONTRACT.md."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "problems": problems,
    }

    if extra:
        report["source_identity"] = extra

    return DatasetAudit(report=report)
