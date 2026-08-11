#!/usr/bin/env python3
"""Build a training-resolution mirror of the dataset.

Re-encodes every video at short side 256 into a parallel directory tree, then
verifies each result. Calibration measured 2.61x cheaper decoding at 15% of the
source size; this carries that out for the whole dataset.

CPU-only. On Kaggle it belongs in a session with the accelerator switched off,
where it costs no GPU quota.

Resumable: a clip already present with the right frame count and geometry is
skipped, so an interrupted run continues rather than restarting.

The mirror is a drop-in substitute. Relative paths and frame counts are
preserved, so switching to it means changing --dataset-root and nothing else.

Examples:

    python scripts/build_video_mirror.py \\
        --dataset-root "$ASL_DATASET_ROOT" \\
        --mirror-root /kaggle/working/asl_citizen_256

    # Verify an existing mirror without re-encoding anything
    python scripts/build_video_mirror.py ... --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_training.data import (
    SHORT_SIDE,
    Manifest,
    MirrorError,
    VideoDecodeError,
    count_frames,
    detect_fps_flag,
    encode_clip,
    ffmpeg_available,
    verify_clip,
)

logger = logging.getLogger("build_video_mirror")

SPLITS = ("train", "validation", "test")


@dataclass
class Outcome:
    """What happened to one clip."""

    sample_id: str
    status: str  # encoded | skipped | failed
    error: str | None = None


@dataclass
class BuildReport:
    outcomes: list[Outcome] = field(default_factory=list)
    crf: int = 20
    fps_flag: str = ""
    seconds: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    def summary(self, mirror_root: Path) -> str:
        size = sum(f.stat().st_size for f in mirror_root.rglob("*.mp4")) / 1024**3
        return "\n".join(
            [
                "Mirror build",
                f"  short side        {SHORT_SIDE}",
                f"  crf               {self.crf}",
                f"  clips             {len(self.outcomes):,}",
                f"  encoded           {self.count('encoded'):,}",
                f"  skipped           {self.count('skipped'):,}  (already present)",
                f"  failed            {self.count('failed'):,}",
                f"  size              {size:.1f} GB",
                f"  elapsed           {self.seconds / 3600:.2f} h",
            ]
        )

    def to_dict(self) -> dict:
        return {
            "short_side": SHORT_SIDE,
            "crf": self.crf,
            "fps_flag": self.fps_flag,
            "clips": len(self.outcomes),
            "encoded": self.count("encoded"),
            "skipped": self.count("skipped"),
            "failed": self.count("failed"),
            "elapsed_hours": round(self.seconds / 3600, 3),
            "failures": [{"sample_id": o.sample_id, "error": o.error} for o in self.failures],
        }


def source_frame_count(record, source: Path) -> int:
    """Frames in the source clip.

    Prefers the audited count, and probes when manifests were regenerated with
    --probe-limit 0, which is the normal Kaggle path.
    """
    if record.frame_count:
        return record.frame_count
    return count_frames(source)


def process(
    record,
    dataset_root: Path,
    mirror_root: Path,
    crf: int,
    fps_flag: str,
    verify_only: bool,
) -> Outcome:
    source = Path(record.resolve_path(dataset_root))
    target = mirror_root / record.video_path

    try:
        expected = source_frame_count(record, source)
    except VideoDecodeError as exc:
        return Outcome(record.sample_id, "failed", f"source unreadable: {exc}")

    # Resume: an existing clip that already verifies needs no work. Verifying is
    # far cheaper than re-encoding, so this is safe to run repeatedly.
    if target.exists():
        try:
            verify_clip(target, expected)
            return Outcome(record.sample_id, "skipped")
        except MirrorError:
            if verify_only:
                return Outcome(record.sample_id, "failed", "existing clip failed verification")
            target.unlink(missing_ok=True)

    if verify_only:
        return Outcome(record.sample_id, "failed", "missing from the mirror")

    try:
        encode_clip(source, target, crf, fps_flag)
        verify_clip(target, expected)
    except (MirrorError, VideoDecodeError) as exc:
        target.unlink(missing_ok=True)
        return Outcome(record.sample_id, "failed", str(exc)[:300])

    return Outcome(record.sample_id, "encoded")


def copy_splits(dataset_root: Path, mirror_root: Path) -> None:
    """Copy the dataset's own split files verbatim.

    They are the authority on which signer belongs to which split. Regenerating
    them would put a derived artifact where the source of truth should be.
    """
    source = dataset_root / "splits"
    if not source.is_dir():
        logger.warning("no splits/ directory at %s; the mirror will not be self-contained", source)
        return

    target = mirror_root / "splits"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    logger.info("copied splits/ verbatim")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ASL_DATASET_ROOT"))
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="x264 quality, lower is better. Must match what calibration measured.",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=SPLITS,
        help="splits to mirror. All of them, unless you are resuming a partial build.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many clips, for a trial run"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check an existing mirror without encoding anything",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset_root is None:
        logger.error("no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT.")
        return 2
    if not args.verify_only and not ffmpeg_available():
        logger.error("ffmpeg not found on PATH.")
        return 2

    records = []
    for split in args.splits:
        path = args.artifacts_dir / "manifests" / f"asl_citizen_{split}.csv"
        if not path.exists():
            logger.error(
                "missing %s manifest at %s. Run scripts/audit_dataset.py --write-manifests first.",
                split,
                path,
            )
            return 1
        records.extend(Manifest.from_csv(path).records)

    if args.limit:
        records = records[: args.limit]

    args.mirror_root.mkdir(parents=True, exist_ok=True)

    fps_flag = ""
    if not args.verify_only:
        fps_flag = detect_fps_flag(
            Path(records[0].resolve_path(args.dataset_root)), args.mirror_root / ".scratch"
        )
        logger.info("frame-rate flag: %s", fps_flag)

    report = BuildReport(crf=args.crf, fps_flag=fps_flag)

    verb = "Verifying" if args.verify_only else "Encoding"
    logger.info(
        "%s %d clip(s) from %s at short side %d, %d job(s)",
        verb,
        len(records),
        ", ".join(args.splits),
        SHORT_SIDE,
        args.jobs,
    )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                process,
                record,
                args.dataset_root,
                args.mirror_root,
                args.crf,
                fps_flag,
                args.verify_only,
            )
            for record in records
        ]
        for done, future in enumerate(futures, start=1):
            outcome = future.result()
            report.outcomes.append(outcome)
            if outcome.status == "failed":
                logger.warning("%s: %s", outcome.sample_id, outcome.error)
            if done % 2000 == 0:
                rate = done / (time.perf_counter() - started)
                remaining = (len(records) - done) / rate / 3600
                logger.info(
                    "%d/%d  %.1f clips/s  ~%.1f h remaining", done, len(records), rate, remaining
                )
    report.seconds = time.perf_counter() - started

    if not args.verify_only:
        copy_splits(args.dataset_root, args.mirror_root)
        shutil.rmtree(args.mirror_root / ".scratch", ignore_errors=True)

    print()
    print(report.summary(args.mirror_root))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nReport written to {args.output}")

    if report.failures:
        logger.error(
            "%d clip(s) failed. The mirror is incomplete and must not be trained against.",
            len(report.failures),
        )
        return 1

    if not args.verify_only:
        print(
            "\nNext: run scripts/audit_dataset.py against the mirror and confirm the "
            "manifest identity matches the source."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
