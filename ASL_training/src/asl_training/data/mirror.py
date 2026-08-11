"""Re-encoding the dataset at training resolution.

Video decoding, not the GPU, limits training on a machine with few cores: the
source is 640x480 and a median 75 frames are decoded to keep 16. Re-encoding at
short side 256 — the resolution the transform resizes to anyway — measured 2.61x
cheaper to decode.

This module owns the encoding settings so the calibration that measures the idea
and the build that carries it out cannot drift apart. A mirror encoded on
different settings than the ones measured is not the thing that was approved.

Two invariants make a mirror a drop-in substitute for the source:

    identical relative paths     manifest identity is unchanged
    identical frame counts       the temporal sampler indexes against them

Both are verified per file. See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .decode import VideoDecodeError, count_frames

# A mirror declares itself here, in the dataset root, so a run can record what it
# actually read. Manifest identity cannot serve this purpose: it is identical
# between source and mirror by design, which is what makes the mirror a valid
# substitute and also what makes it invisible in a run record.
SUBSTRATE_FILE = "substrate.json"

# What a dataset root with no declaration is called. The original download has
# no marker file and never will.
SOURCE_SUBSTRATE = "source"

# The training transform resizes to 256 and then crops 224. Encoding below this
# would leave the random crop nothing to move within, removing the spatial
# augmentation rather than merely shrinking the file.
SHORT_SIDE = 256

# Landscape keeps height at SHORT_SIDE, portrait keeps width. "-2" lets the
# encoder choose the other dimension while keeping it even, which h264 requires.
# A fixed -2:SHORT_SIDE would scale the dataset's portrait clips by their long
# side, silently changing their scale relative to everything else.
SCALE_FILTER = f"scale='if(gt(iw,ih),-2,{SHORT_SIDE})':'if(gt(iw,ih),{SHORT_SIDE},-2)'"

# Frame-rate passthrough. Newer ffmpeg builds use -fps_mode, older ones -vsync.
# Both keep every source frame; anything else changes the frame count.
FPS_FLAGS = ("-fps_mode passthrough", "-vsync 0")


class MirrorError(RuntimeError):
    """Raised when a clip cannot be re-encoded, or the result is not usable."""


def substrate_identity(dataset_root: Path) -> str:
    """Name the substrate a dataset root represents.

    Returns SOURCE_SUBSTRATE for the original dataset, or an identity describing
    the re-encoding for a mirror. Every real run records this: the re-encode is
    lossy, so results from different substrates are not comparable, and nothing
    else in the run record distinguishes them.
    """
    marker = Path(dataset_root) / SUBSTRATE_FILE
    if not marker.exists():
        return SOURCE_SUBSTRATE

    try:
        declared = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MirrorError(f"unreadable substrate declaration at {marker}: {exc}") from exc

    identity = declared.get("identity")
    if not identity:
        raise MirrorError(f"{marker} declares no identity")
    return str(identity)


def write_substrate(dataset_root: Path, *, short_side: int, crf: int, source_identity: str) -> str:
    """Declare a built mirror, and return its identity.

    Encoding settings are part of the identity because a mirror rebuilt at
    different settings is a different substrate, and results do not carry over.
    """
    identity = f"mirror:short_side={short_side}:crf={crf}"
    payload = {
        "identity": identity,
        "short_side": short_side,
        "crf": crf,
        "codec": "h264",
        "source_manifest_identity": source_identity,
        "lossy": True,
    }
    (Path(dataset_root) / SUBSTRATE_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    return identity


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_command(source: Path, target: Path, crf: int, fps_flag: str) -> list[str]:
    """Build the re-encode command.

    Frame rate is passed through rather than normalized: manifests record a frame
    count per clip and the temporal sampler indexes against it, so a changed
    count would corrupt sampling rather than fail.
    """
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        SCALE_FILTER,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        *fps_flag.split(),
        "-threads",
        "1",
        "-an",
        str(target),
    ]


def detect_fps_flag(source: Path, scratch: Path) -> str:
    """Pick the frame-rate passthrough flag this ffmpeg build understands.

    Guessing wrong changes the frame count, which is the one thing that must not
    move, so this probes rather than assuming.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / ".fps-probe.mp4"
    try:
        for flag in FPS_FLAGS:
            probe.unlink(missing_ok=True)
            result = subprocess.run(
                encode_command(source, probe, 30, flag), capture_output=True, text=True
            )
            if result.returncode == 0:
                return flag
    finally:
        probe.unlink(missing_ok=True)

    raise MirrorError("neither -fps_mode nor -vsync worked; check the ffmpeg build")


def encode_clip(source: Path, target: Path, crf: int, fps_flag: str) -> None:
    """Re-encode one clip, raising rather than leaving a partial file behind."""
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        encode_command(source, target, crf, fps_flag), capture_output=True, text=True
    )
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        detail = (result.stderr or "").strip().splitlines()
        raise MirrorError(
            f"ffmpeg failed on {source.name}: {detail[-1] if detail else 'no output'}"
        )


def probe_dimensions(path: Path) -> tuple[int, int]:
    """Width and height of a video's first stream."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("video probing requires 'av'.") from exc

    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if not streams:
                raise MirrorError(f"no video stream in {path}")
            stream = streams[0]
            return int(stream.codec_context.width), int(stream.codec_context.height)
    except MirrorError:
        raise
    except Exception as exc:
        raise MirrorError(f"could not probe {path}: {type(exc).__name__}: {exc}") from exc


def verify_clip(target: Path, expected_frames: int) -> None:
    """Check the two properties that make a mirrored clip substitutable.

    Frame count must match the source exactly, and the short side must be
    SHORT_SIDE. A clip that decodes but has the wrong geometry would train
    without error at the wrong scale.
    """
    if not target.exists():
        raise MirrorError(f"{target} was not written")

    try:
        frames = count_frames(target)
    except VideoDecodeError as exc:
        raise MirrorError(f"{target.name} does not decode: {exc}") from exc

    if frames != expected_frames:
        raise MirrorError(
            f"{target.name} has {frames} frame(s), source has {expected_frames}. "
            "The manifests index against this count."
        )

    width, height = probe_dimensions(target)
    if min(width, height) != SHORT_SIDE:
        raise MirrorError(f"{target.name} is {width}x{height}; short side should be {SHORT_SIDE}")
