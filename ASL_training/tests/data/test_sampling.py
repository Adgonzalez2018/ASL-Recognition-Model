"""Temporal sampling.

Frame count, chronological order, and evaluation determinism are contractual.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import random

import pytest

from asl_training.data.sampling import (
    TemporalSampler,
    random_segment_indices,
    random_window_indices,
    uniform_indices,
)

# Frame count and ordering -----------------------------------------------------


@pytest.mark.parametrize("total", [1, 3, 8, 15, 16, 17, 40, 300])
@pytest.mark.parametrize("num_frames", [8, 16, 32])
def test_uniform_returns_exactly_the_requested_count(total, num_frames):
    indices = uniform_indices(total, num_frames)
    assert len(indices) == num_frames
    assert all(0 <= i < total for i in indices)


@pytest.mark.parametrize("total", [1, 5, 16, 100])
@pytest.mark.parametrize("strategy", ["uniform", "random_segment", "random_window"])
def test_every_strategy_preserves_chronological_order(total, strategy):
    sampler = TemporalSampler(num_frames=16, strategy=strategy)
    indices = sampler.indices(total, random.Random(0))
    assert indices == sorted(indices)


@pytest.mark.parametrize("strategy", ["uniform", "random_segment", "random_window"])
def test_every_strategy_returns_the_configured_count(strategy):
    sampler = TemporalSampler(num_frames=16, strategy=strategy)
    for total in (1, 4, 16, 17, 250):
        assert len(sampler.indices(total, random.Random(total))) == 16


def test_uniform_spans_the_whole_clip():
    """Sampling must not be biased toward the start."""
    indices = uniform_indices(100, 16)
    assert indices[0] < 10
    assert indices[-1] > 90


def test_uniform_is_centred_not_front_loaded():
    indices = uniform_indices(16, 16)
    assert indices == list(range(16))


# Short-video policy -----------------------------------------------------------


def test_short_clip_repeats_frames_rather_than_failing():
    indices = uniform_indices(4, 16)
    assert len(indices) == 16
    assert set(indices) <= {0, 1, 2, 3}


def test_short_clip_preserves_chronology_without_inventing_motion():
    """Repetition is the policy; reversal would change what a sign means."""
    indices = uniform_indices(3, 12)
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 2


def test_single_frame_clip_is_handled():
    assert uniform_indices(1, 16) == [0] * 16


def test_zero_frame_clip_raises_as_a_corruption_finding():
    with pytest.raises(ValueError, match="corruption finding"):
        uniform_indices(0, 16)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_num_frames_raises(bad):
    with pytest.raises(ValueError, match="num_frames must be positive"):
        uniform_indices(30, bad)


# Determinism ------------------------------------------------------------------


def test_uniform_is_deterministic_across_calls():
    assert uniform_indices(137, 16) == uniform_indices(137, 16)


def test_uniform_sampler_ignores_rng():
    """Evaluation must not vary with random state."""
    sampler = TemporalSampler(num_frames=16, strategy="uniform")
    assert sampler.indices(100, random.Random(1)) == sampler.indices(100, random.Random(999))


def test_uniform_sampler_works_without_an_rng():
    sampler = TemporalSampler(num_frames=16, strategy="uniform")
    assert len(sampler.indices(100)) == 16


def test_random_strategies_require_an_rng():
    """A random strategy silently defaulting its seed would be unreproducible."""
    for strategy in ("random_segment", "random_window"):
        sampler = TemporalSampler(num_frames=16, strategy=strategy)
        with pytest.raises(ValueError, match="requires an rng"):
            sampler.indices(100)


def test_random_strategies_reproduce_from_the_same_seed():
    for strategy in ("random_segment", "random_window"):
        sampler = TemporalSampler(num_frames=16, strategy=strategy)
        first = sampler.indices(200, random.Random(42))
        second = sampler.indices(200, random.Random(42))
        assert first == second


def test_random_strategies_vary_across_seeds():
    sampler = TemporalSampler(num_frames=16, strategy="random_segment")
    a = sampler.indices(200, random.Random(1))
    b = sampler.indices(200, random.Random(2))
    assert a != b


def test_deterministic_flag_matches_behavior():
    assert TemporalSampler(strategy="uniform").is_deterministic
    assert not TemporalSampler(strategy="random_segment").is_deterministic
    assert not TemporalSampler(strategy="random_window").is_deterministic


# Random segment ---------------------------------------------------------------


def test_random_segment_covers_the_whole_clip():
    """Each segment contributes one frame, so coverage stays spread out."""
    indices = random_segment_indices(160, 16, random.Random(0))
    assert indices[0] < 20
    assert indices[-1] > 140


def test_random_segment_stays_in_range():
    for total in (1, 7, 16, 500):
        indices = random_segment_indices(total, 16, random.Random(total))
        assert all(0 <= i < total for i in indices)


# Random window ----------------------------------------------------------------


def test_random_window_respects_minimum_coverage():
    """A window must not shrink a sign to an unrecognizable fragment."""
    total = 200
    for seed in range(20):
        indices = random_window_indices(total, 16, random.Random(seed), min_coverage=0.5)
        span = indices[-1] - indices[0]
        assert span >= total * 0.4


def test_random_window_falls_back_to_uniform_for_short_clips():
    indices = random_window_indices(10, 16, random.Random(0))
    assert indices == uniform_indices(10, 16)


def test_random_window_rejects_invalid_coverage():
    with pytest.raises(ValueError, match="min_coverage"):
        random_window_indices(100, 16, random.Random(0), min_coverage=0)


# Configuration ----------------------------------------------------------------


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown temporal sampling strategy"):
        TemporalSampler(strategy="reverse")


def test_time_reversal_is_not_available():
    """Reversed motion changes meaning; it is not an ordinary augmentation."""
    from asl_training.data.sampling import STRATEGIES

    assert not any("revers" in s for s in STRATEGIES)


def test_spec_records_the_short_video_policy():
    spec = TemporalSampler(num_frames=16, strategy="uniform").to_dict()
    assert spec["short_video_policy"] == "repeat-via-uniform-sampling-v1"
    assert spec["deterministic"] is True
    assert spec["num_frames"] == 16


def test_spec_records_min_coverage_only_when_relevant():
    assert TemporalSampler(strategy="uniform").to_dict()["min_coverage"] is None
    assert TemporalSampler(strategy="random_window").to_dict()["min_coverage"] == 0.5
