"""Video decoding.

Converts a source video into an ordered RGB frame sequence. Two properties are
contractual and are enforced here rather than assumed:

* frames come out in chronological order
* pixels come out as RGB

PyAV's ``to_ndarray(format="rgb24")`` performs an explicit conversion rather than
trusting whatever the decoder happened to produce.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DECODER_BACKEND = "pyav"
COLOR_SPACE = "rgb24"


class VideoDecodeError(Exception):
    """Raised when a video cannot be decoded into usable frames."""


@dataclass
class DecodedClip:
    """An ordered RGB frame sequence.

    Attributes:
        frames: ``[frames, channels, height, width]``, uint8, RGB.
        source_frame_count: Frames available in the source, before sampling.
        selected_indices: Source indices these frames came from.
        fps: Source frame rate, when the container reports one.
    """

    frames: torch.Tensor
    source_frame_count: int
    selected_indices: list[int]
    fps: float | None = None

    def __post_init__(self) -> None:
        if self.frames.ndim != 4:
            raise ValueError(
                f"decoded frames must be [frames, channels, height, width], got "
                f"{tuple(self.frames.shape)}"
            )
        if self.frames.shape[1] != 3:
            raise ValueError(f"decoded frames must have 3 RGB channels, got {self.frames.shape[1]}")
        if self.frames.shape[0] != len(self.selected_indices):
            raise ValueError(
                f"{self.frames.shape[0]} frames but {len(self.selected_indices)} selected indices"
            )


def decode_clip(
    path: str | Path,
    indices: list[int] | None = None,
    *,
    expected_frames: int | None = None,
) -> DecodedClip:
    """Decode a video, optionally keeping only the requested frame indices.

    Decodes sequentially and keeps the requested indices as they pass. Seeking
    would be faster but is unreliable on the variable-frame-rate webcam
    recordings this dataset contains, and an optimization must never change which
    frames are selected.

    Args:
        path: Video file.
        indices: Ordered source indices to keep. ``None`` decodes every frame.
            Repeated indices are honoured, which is how the short-video policy
            produces a full-length clip.
        expected_frames: When given, the result must contain exactly this many
            frames. A mismatch raises rather than yielding a short batch.

    Returns:
        The decoded clip.

    Raises:
        VideoDecodeError: If the file is missing, unreadable, has no video
            stream, yields no frames, or produces the wrong frame count.
    """
    path = Path(path)
    if not path.exists():
        raise VideoDecodeError(f"video not found: {path}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("video decoding requires 'av'.") from exc

    wanted = sorted(set(indices)) if indices is not None else None

    collected: dict[int, torch.Tensor] = {}
    total = 0
    fps: float | None = None

    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if not streams:
                raise VideoDecodeError(f"no video stream in {path}")

            stream = streams[0]
            stream.thread_type = "AUTO"
            if stream.average_rate:
                fps = float(stream.average_rate)

            highest = wanted[-1] if wanted else None

            for position, frame in enumerate(container.decode(stream)):
                total = position + 1

                if wanted is None or position in _as_set(wanted):
                    # Explicit RGB conversion. The decoder's native format is
                    # typically YUV, and assuming otherwise would silently
                    # swap colour channels.
                    array = frame.to_ndarray(format=COLOR_SPACE)
                    collected[position] = torch.from_numpy(array).permute(2, 0, 1).contiguous()

                if highest is not None and position >= highest:
                    # Everything requested has been seen; decoding further would
                    # only cost time. Frame selection is unaffected.
                    break

    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"could not decode {path}: {type(exc).__name__}: {exc}") from exc

    if not collected:
        raise VideoDecodeError(f"{path} yielded no frames")

    if indices is None:
        ordered_indices = sorted(collected)
    else:
        missing = [i for i in set(indices) if i not in collected]
        if missing:
            raise VideoDecodeError(
                f"{path}: requested frame index/indices {sorted(missing)[:5]} beyond the "
                f"{total} frame(s) actually decoded. The manifest's frame count "
                f"disagrees with the file."
            )
        ordered_indices = list(indices)

    frames = torch.stack([collected[i] for i in ordered_indices])

    if expected_frames is not None and frames.shape[0] != expected_frames:
        raise VideoDecodeError(
            f"{path}: decoded {frames.shape[0]} frames, expected {expected_frames}"
        )

    return DecodedClip(
        frames=frames,
        source_frame_count=total,
        selected_indices=ordered_indices,
        fps=fps,
    )


def _as_set(values: list[int]) -> set[int]:
    return set(values)


def count_frames(path: str | Path) -> int:
    """Count decodable frames.

    Prefers the container's reported count and falls back to demuxing, because
    webcam recordings frequently report zero in the header.
    """
    path = Path(path)
    if not path.exists():
        raise VideoDecodeError(f"video not found: {path}")

    try:
        import av

        with av.open(str(path)) as container:
            streams = container.streams.video
            if not streams:
                raise VideoDecodeError(f"no video stream in {path}")

            stream = streams[0]
            if stream.frames:
                return stream.frames

            return sum(1 for _ in container.decode(stream))
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"could not read {path}: {type(exc).__name__}: {exc}") from exc
