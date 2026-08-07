"""Temporal sampling: choosing which frames represent a clip.

Every model input contains exactly the configured number of frames, drawn in
chronological order. Evaluation sampling is deterministic; training sampling may
be random but must still preserve order.

Time reversal is not available. Reversed motion changes what a sign means, so it
is not an ordinary augmentation.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

# Deterministic strategies are the only ones permitted for validation and test.
DETERMINISTIC_STRATEGIES = ("uniform",)
RANDOM_STRATEGIES = ("random_segment", "random_window")
STRATEGIES = DETERMINISTIC_STRATEGIES + RANDOM_STRATEGIES

# Short-video policy. Uniform sampling repeats indices when a clip is shorter
# than the requested frame count, which preserves chronology and invents no
# motion. Recorded in the preprocessing identity.
SHORT_VIDEO_POLICY = "repeat-via-uniform-sampling-v1"


def uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    """Evenly spaced frame indices spanning the whole clip.

    Samples the centre of each of ``num_frames`` equal segments, so coverage is
    symmetric rather than biased toward the start.

    When the clip is shorter than ``num_frames``, indices repeat. That is the
    short-video policy: chronology is preserved and no frame is fabricated.

    Deterministic. Given the same inputs it always returns the same indices.
    """
    _validate(total_frames, num_frames)
    return [
        min(int((i + 0.5) * total_frames / num_frames), total_frames - 1) for i in range(num_frames)
    ]


def random_segment_indices(
    total_frames: int,
    num_frames: int,
    rng: random.Random,
) -> list[int]:
    """One random frame from each of ``num_frames`` equal segments.

    Covers the whole clip while varying which frame represents each segment.
    Order is preserved because segment ``i`` always precedes segment ``i + 1``.
    """
    _validate(total_frames, num_frames)

    indices = []
    for i in range(num_frames):
        start = i * total_frames / num_frames
        end = (i + 1) * total_frames / num_frames
        low = min(int(start), total_frames - 1)
        high = min(max(int(end) - 1, low), total_frames - 1)
        indices.append(rng.randint(low, high))
    return indices


def random_window_indices(
    total_frames: int,
    num_frames: int,
    rng: random.Random,
    *,
    min_coverage: float = 0.5,
) -> list[int]:
    """A random contiguous window, then uniform sampling within it.

    Simulates a clip whose boundaries were cut slightly differently. The window
    covers at least ``min_coverage`` of the clip, so a sign is not truncated to
    an unrecognizable fragment.

    Falls back to whole-clip uniform sampling when the clip is already at or
    below the requested frame count.
    """
    _validate(total_frames, num_frames)

    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")

    if total_frames <= num_frames:
        return uniform_indices(total_frames, num_frames)

    min_window = max(num_frames, int(total_frames * min_coverage))
    if min_window >= total_frames:
        return uniform_indices(total_frames, num_frames)

    window = rng.randint(min_window, total_frames)
    start = rng.randint(0, total_frames - window)

    return [start + i for i in uniform_indices(window, num_frames)]


def _validate(total_frames: int, num_frames: int) -> None:
    if total_frames < 1:
        raise ValueError(
            f"cannot sample from a clip with {total_frames} frames; a zero-frame "
            f"video is a corruption finding, not a sampling case"
        )
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")


@dataclass(frozen=True)
class TemporalSampler:
    """Selects frame indices according to a configured strategy.

    Attributes:
        num_frames: Frames per model input.
        strategy: One of ``STRATEGIES``.
        min_coverage: Minimum clip fraction for ``random_window``.
    """

    num_frames: int = 16
    strategy: str = "uniform"
    min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"unknown temporal sampling strategy {self.strategy!r}; "
                f"supported: {', '.join(STRATEGIES)}"
            )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")

    @property
    def is_deterministic(self) -> bool:
        """Whether this sampler may be used for validation and test."""
        return self.strategy in DETERMINISTIC_STRATEGIES

    def indices(self, total_frames: int, rng: random.Random | None = None) -> list[int]:
        """Choose ``num_frames`` ordered indices from a clip.

        Args:
            total_frames: Frames available in the source clip.
            rng: Required for random strategies. Passing one to a deterministic
                strategy is harmless and ignored.

        Returns:
            Ordered indices, exactly ``num_frames`` long.
        """
        if self.strategy == "uniform":
            selected = uniform_indices(total_frames, self.num_frames)
        else:
            if rng is None:
                raise ValueError(
                    f"strategy {self.strategy!r} is random and requires an rng. "
                    f"Evaluation must use a deterministic strategy instead; see "
                    f"docs/DATA_CONTRACT.md."
                )
            if self.strategy == "random_segment":
                selected = random_segment_indices(total_frames, self.num_frames, rng)
            else:
                selected = random_window_indices(
                    total_frames, self.num_frames, rng, min_coverage=self.min_coverage
                )

        # Ordering is a contract, not an implementation detail: a sampler that
        # returned unordered indices would silently scramble motion.
        if any(b < a for a, b in itertools.pairwise(selected)):
            raise RuntimeError(
                f"{self.strategy!r} produced out-of-order indices {selected}; "
                f"chronological order must be preserved"
            )
        if len(selected) != self.num_frames:
            raise RuntimeError(
                f"{self.strategy!r} produced {len(selected)} indices, expected {self.num_frames}"
            )
        return selected

    def to_dict(self) -> dict[str, object]:
        return {
            "num_frames": self.num_frames,
            "strategy": self.strategy,
            "min_coverage": self.min_coverage if self.strategy == "random_window" else None,
            "short_video_policy": SHORT_VIDEO_POLICY,
            "deterministic": self.is_deterministic,
        }
