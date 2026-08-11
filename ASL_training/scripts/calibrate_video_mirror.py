#!/usr/bin/env python3
"""Measure whether a downscaled mirror of the dataset is worth building.

Re-encodes a sample of clips at short side 256, then measures the three numbers
that decide it: how much smaller the result is, how long encoding the full
dataset would take, and how much faster the sample decodes through the real
loader path.

This is the gate before building a mirror of all 83,399 videos. If the decode
speedup is small, nothing else about the idea matters.

Writes only into --work-dir and leaves the source untouched.

Examples:

    python scripts/calibrate_video_mirror.py \\
        --dataset-root "$ASL_DATASET_ROOT" \\
        --work-dir /kaggle/working/mirror-calibration

    # Compare quality settings before committing
    python scripts/calibrate_video_mirror.py ... --crf 18 --samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_training.data import (
    SCALE_FILTER,
    SHORT_SIDE,
    Manifest,
    TemporalSampler,
    VideoDecodeError,
    count_frames,
    decode_clip,
    detect_fps_flag,
    encode_command,
    ffmpeg_available,
)

logger = logging.getLogger("calibrate_video_mirror")

# The full dataset, from the committed audit. Used only to extrapolate.
TOTAL_VIDEOS = 83_399

# Encoding settings live in asl_training.data.mirror, shared with the build, so
# that what is measured here is what gets built.
__all__ = ["SCALE_FILTER", "SHORT_SIDE"]


@dataclass
class ClipResult:
    """One clip's encode outcome."""

    sample_id: str
    source_bytes: int
    output_bytes: int
    encode_seconds: float
    source_frames: int
    output_frames: int
    resolution: str
    ok: bool = True
    error: str | None = None

    @property
    def frames_match(self) -> bool:
        return self.source_frames == self.output_frames


@dataclass
class DecodeResult:
    """Decode timings for one clip, source against mirror."""

    sample_id: str
    source_seconds: float
    mirror_seconds: float


@dataclass
class Report:
    clips: list[ClipResult] = field(default_factory=list)
    decodes: list[DecodeResult] = field(default_factory=list)
    crf: int = 20
    jobs: int = 4
    fps_flag: str = ""

    # Reporting ---------------------------------------------------------------

    @property
    def encoded(self) -> list[ClipResult]:
        return [c for c in self.clips if c.ok]

    @property
    def mismatched(self) -> list[ClipResult]:
        return [c for c in self.encoded if not c.frames_match]

    @property
    def failed(self) -> list[ClipResult]:
        return [c for c in self.clips if not c.ok]

    @property
    def size_ratio(self) -> float:
        source = sum(c.source_bytes for c in self.encoded)
        output = sum(c.output_bytes for c in self.encoded)
        return output / source if source else 0.0

    @property
    def projected_gb(self) -> float:
        if not self.encoded:
            return 0.0
        mean_bytes = statistics.mean(c.output_bytes for c in self.encoded)
        return mean_bytes * TOTAL_VIDEOS / 1024**3

    @property
    def projected_encode_hours(self) -> float:
        """Wall time to encode everything at --jobs parallelism."""
        if not self.encoded:
            return 0.0
        mean_seconds = statistics.mean(c.encode_seconds for c in self.encoded)
        return mean_seconds * TOTAL_VIDEOS / self.jobs / 3600

    @property
    def decode_speedup(self) -> float:
        if not self.decodes:
            return 0.0
        source = sum(d.source_seconds for d in self.decodes)
        mirror = sum(d.mirror_seconds for d in self.decodes)
        return source / mirror if mirror else 0.0

    def summary(self) -> str:
        lines = [
            "Mirror calibration",
            f"  sampled           {len(self.clips)} clip(s)",
            f"  encoded           {len(self.encoded)}",
            f"  crf               {self.crf}",
            f"  parallel jobs     {self.jobs}",
            "",
        ]

        if self.encoded:
            lines += [
                f"  size ratio        {self.size_ratio:.3f} of source",
                f"  projected size    {self.projected_gb:.1f} GB for {TOTAL_VIDEOS:,} videos",
                f"  projected encode  {self.projected_encode_hours:.1f} h at {self.jobs} job(s)",
                "",
            ]

        if self.decodes:
            source = statistics.mean(d.source_seconds for d in self.decodes) * 1000
            mirror = statistics.mean(d.mirror_seconds for d in self.decodes) * 1000
            lines += [
                f"  decode source     {source:.1f} ms per clip",
                f"  decode mirror     {mirror:.1f} ms per clip",
                f"  DECODE SPEEDUP    {self.decode_speedup:.2f}x   ({len(self.decodes)} clip(s))",
                "",
            ]

        lines.append(f"  frame mismatches  {len(self.mismatched)}")
        lines.append(f"  encode failures   {len(self.failed)}")

        resolutions = Counter(c.resolution for c in self.clips)
        lines.append(
            "  resolutions       " + ", ".join(f"{r} x{n}" for r, n in sorted(resolutions.items()))
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "short_side": SHORT_SIDE,
            "crf": self.crf,
            "jobs": self.jobs,
            "fps_flag": self.fps_flag,
            "sampled": len(self.clips),
            "encoded": len(self.encoded),
            "size_ratio": round(self.size_ratio, 4),
            "projected_size_gb": round(self.projected_gb, 2),
            "projected_encode_hours": round(self.projected_encode_hours, 2),
            "decode_speedup": round(self.decode_speedup, 3),
            "decode_sample": len(self.decodes),
            "frame_mismatches": [c.sample_id for c in self.mismatched],
            "encode_failures": [{"sample_id": c.sample_id, "error": c.error} for c in self.failed],
            "resolutions": dict(Counter(c.resolution for c in self.clips)),
        }


def encode_one(
    record, dataset_root: Path, mirror_root: Path, crf: int, fps_flag: str
) -> ClipResult:
    source = Path(record.resolve_path(dataset_root))
    target = mirror_root / record.video_path
    target.parent.mkdir(parents=True, exist_ok=True)

    resolution = f"{record.width}x{record.height}" if record.width else "unknown"

    started = time.perf_counter()
    result = subprocess.run(
        encode_command(source, target, crf, fps_flag), capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        return ClipResult(
            sample_id=record.sample_id,
            source_bytes=source.stat().st_size if source.exists() else 0,
            output_bytes=0,
            encode_seconds=elapsed,
            source_frames=0,
            output_frames=0,
            resolution=resolution,
            ok=False,
            error=(result.stderr or "").strip()[:200] or "ffmpeg failed",
        )

    try:
        source_frames = record.frame_count or count_frames(source)
        output_frames = count_frames(target)
    except VideoDecodeError as exc:
        return ClipResult(
            sample_id=record.sample_id,
            source_bytes=source.stat().st_size,
            output_bytes=target.stat().st_size if target.exists() else 0,
            encode_seconds=elapsed,
            source_frames=0,
            output_frames=0,
            resolution=resolution,
            ok=False,
            error=str(exc)[:200],
        )

    return ClipResult(
        sample_id=record.sample_id,
        source_bytes=source.stat().st_size,
        output_bytes=target.stat().st_size,
        encode_seconds=elapsed,
        source_frames=source_frames,
        output_frames=output_frames,
        resolution=resolution,
    )


def time_decode(path: Path, indices: list[int], frames: int) -> float:
    started = time.perf_counter()
    decode_clip(path, indices, expected_frames=frames)
    return time.perf_counter() - started


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ASL_DATASET_ROOT"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="scratch directory for the encoded sample; nothing else is written",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=120,
        help="clips timed through the real decode path (sequential, so kept smaller)",
    )
    parser.add_argument("--crf", type=int, default=20, help="x264 quality; lower is better")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--frames", type=int, default=16, help="frames the sampler selects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the encoded sample instead of deleting it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset_root is None:
        logger.error("no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT.")
        return 2
    if not ffmpeg_available():
        logger.error("ffmpeg not found on PATH.")
        return 2

    manifest_path = args.artifacts_dir / "manifests" / "asl_citizen_train.csv"
    if not manifest_path.exists():
        logger.error(
            "missing train manifest at %s. Run scripts/audit_dataset.py --write-manifests first.",
            manifest_path,
        )
        return 1

    # Train split only. Calibration needs representative video files, not the
    # evaluation splits, and there is no reason to touch test here.
    #
    # Frame counts may be absent: the Kaggle session regenerates manifests with
    # --probe-limit 0, which skips the probing that fills them in. Counting the
    # sampled clips directly costs a few seconds and keeps this independent of
    # how the manifests were produced.
    manifest = Manifest.from_csv(manifest_path)
    records = manifest.records
    probed = sum(1 for r in records if r.frame_count)
    if not probed:
        logger.info("manifest carries no frame counts; probing the sample directly")

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.samples, len(records)))

    mirror_root = args.work_dir / "mirror"
    mirror_root.mkdir(parents=True, exist_ok=True)

    fps_flag = detect_fps_flag(Path(sample[0].resolve_path(args.dataset_root)), args.work_dir)
    report = Report(crf=args.crf, jobs=args.jobs, fps_flag=fps_flag)

    logger.info(
        "Encoding %d clip(s) at short side %d, crf %d, %d job(s)",
        len(sample),
        SHORT_SIDE,
        args.crf,
        args.jobs,
    )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(encode_one, record, args.dataset_root, mirror_root, args.crf, fps_flag)
            for record in sample
        ]
        for done, future in enumerate(futures, start=1):
            report.clips.append(future.result())
            if done % 50 == 0:
                logger.info("encoded %d/%d", done, len(sample))

    # Decode timing runs sequentially and single-threaded, so the two paths are
    # measured under identical conditions. Both use the same frame indices.
    sampler = TemporalSampler(num_frames=args.frames, strategy="random_segment")
    timed = [c for c in report.encoded if c.frames_match][: args.decode_samples]
    by_id = {r.sample_id: r for r in sample}

    logger.info("Timing decode on %d clip(s)", len(timed))
    for clip in timed:
        record = by_id[clip.sample_id]
        indices = sampler.indices(clip.source_frames, random.Random(args.seed))
        source = Path(record.resolve_path(args.dataset_root))
        mirror = mirror_root / record.video_path
        try:
            report.decodes.append(
                DecodeResult(
                    sample_id=clip.sample_id,
                    source_seconds=time_decode(source, indices, args.frames),
                    mirror_seconds=time_decode(mirror, indices, args.frames),
                )
            )
        except VideoDecodeError as exc:
            logger.warning("decode timing skipped for %s: %s", clip.sample_id, exc)

    print()
    print(report.summary())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nReport written to {args.output}")

    if not args.keep:
        shutil.rmtree(mirror_root, ignore_errors=True)

    # Frame drift is disqualifying: the manifests index against these counts.
    if report.mismatched:
        logger.error(
            "%d clip(s) changed frame count. The mirror must not be built until "
            "this is understood.",
            len(report.mismatched),
        )
        return 1
    if report.failed:
        logger.error("%d clip(s) failed to encode.", len(report.failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
