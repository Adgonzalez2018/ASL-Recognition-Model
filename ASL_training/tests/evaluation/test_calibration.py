"""Confidence calibration.

The invariant that matters most: temperature scaling is monotonic, so it must
never change a top-1 prediction. If it did, reported accuracy would silently
stop describing the model.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from asl_training.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    max_softmax_confidence,
    reliability_bins,
)
from asl_training.evaluation.metrics import predicted_classes, top1_accuracy


def overconfident_logits(samples=400, num_classes=10, accuracy=0.7, scale=6.0, seed=0):
    """Logits that are right ``accuracy`` of the time but far too confident.

    This is the realistic case: modern classifiers are systematically
    overconfident, which is why calibration exists.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (samples,), generator=generator)

    logits = torch.randn(samples, num_classes, generator=generator) * 0.5
    for index in range(samples):
        correct = torch.rand(1, generator=generator).item() < accuracy
        target = (
            labels[index]
            if correct
            else (
                labels[index]
                + 1
                + int(torch.randint(0, num_classes - 1, (1,), generator=generator))
            )
            % num_classes
        )
        logits[index, target] += scale
    return logits, labels


# Temperature application ------------------------------------------------------


def test_temperature_scales_logits():
    logits = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(apply_temperature(logits, 2.0), torch.tensor([[1.0, 2.0]]))


def test_temperature_above_one_softens_confidence():
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    assert max_softmax_confidence(apply_temperature(logits, 3.0)) < max_softmax_confidence(logits)


def test_temperature_below_one_sharpens_confidence():
    logits = torch.tensor([[0.0, 2.0, 0.0]])
    assert max_softmax_confidence(apply_temperature(logits, 0.5)) > max_softmax_confidence(logits)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(bad):
    with pytest.raises(ValueError, match="temperature must be positive"):
        apply_temperature(torch.randn(2, 3), bad)


def test_temperature_never_changes_predictions():
    """Monotonicity, checked directly across a wide range."""
    torch.manual_seed(0)
    logits = torch.randn(200, 12)
    baseline = predicted_classes(logits)

    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
        assert torch.equal(predicted_classes(apply_temperature(logits, temperature)), baseline)


# Fitting ----------------------------------------------------------------------


def test_fitting_reduces_nll_on_overconfident_logits():
    logits, labels = overconfident_logits()
    result = fit_temperature(logits, labels)

    assert result.nll_after <= result.nll_before
    assert result.temperature > 0


def test_overconfident_logits_yield_temperature_above_one():
    logits, labels = overconfident_logits(scale=8.0)
    assert fit_temperature(logits, labels).temperature > 1.0


def test_fitting_reduces_calibration_error():
    logits, labels = overconfident_logits()
    result = fit_temperature(logits, labels)
    assert result.ece_after < result.ece_before


def test_fitting_preserves_accuracy_exactly():
    """The headline number must not move when only confidence is recalibrated."""
    logits, labels = overconfident_logits()
    result = fit_temperature(logits, labels)

    assert result.accuracy_after == result.accuracy_before
    assert result.accuracy_before == pytest.approx(top1_accuracy(logits, labels))


def test_fitting_lowers_mean_confidence_for_an_overconfident_model():
    logits, labels = overconfident_logits()
    result = fit_temperature(logits, labels)
    assert result.mean_confidence_after < result.mean_confidence_before


def test_well_calibrated_logits_yield_temperature_near_one():
    torch.manual_seed(5)
    logits = torch.randn(500, 6) * 1.0
    labels = torch.softmax(logits, dim=1).multinomial(1).squeeze(1)

    assert fit_temperature(logits, labels).temperature == pytest.approx(1.0, abs=0.35)


def test_result_records_the_binning_scheme():
    """ECE is only comparable across runs using identical binning."""
    logits, labels = overconfident_logits(samples=200)
    result = fit_temperature(logits, labels, bins=20, scheme="equal_mass")

    payload = result.to_dict()
    assert payload["bins"] == 20
    assert payload["binning_scheme"] == "equal_mass"
    assert payload["fit_on"] == "validation"


def test_result_reports_before_and_after():
    logits, labels = overconfident_logits(samples=200)
    payload = fit_temperature(logits, labels).to_dict()

    for section in ("before", "after"):
        for key in ("nll", "ece", "accuracy", "mean_confidence"):
            assert key in payload[section], f"{section}.{key} missing"


def test_unknown_binning_scheme_is_rejected():
    logits, labels = overconfident_logits(samples=50)
    with pytest.raises(ValueError, match="unknown binning scheme"):
        fit_temperature(logits, labels, scheme="adaptive")


# Expected calibration error ---------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_case():
    """Hand-built: confidence 1.0 and always correct."""
    logits = torch.full((10, 2), -20.0)
    logits[:, 0] = 20.0
    labels = torch.zeros(10, dtype=torch.long)

    assert expected_calibration_error(logits, labels) == pytest.approx(0.0, abs=1e-5)


def test_ece_is_large_when_confidently_wrong():
    logits = torch.full((10, 2), -20.0)
    logits[:, 0] = 20.0
    labels = torch.ones(10, dtype=torch.long)  # always wrong, always confident

    assert expected_calibration_error(logits, labels) == pytest.approx(1.0, abs=1e-4)


def test_ece_on_a_hand_computed_binning():
    """Half the samples at ~100% confidence and right, half wrong.

    Confidence ~1.0 with accuracy 0.5 gives a gap of ~0.5 in one bin.
    """
    logits = torch.full((10, 2), -20.0)
    logits[:, 0] = 20.0
    labels = torch.tensor([0] * 5 + [1] * 5)

    assert expected_calibration_error(logits, labels) == pytest.approx(0.5, abs=1e-4)


def test_ece_depends_on_the_binning_scheme():
    """Which is exactly why the scheme is recorded with every value."""
    logits, labels = overconfident_logits(samples=300)

    equal_width = expected_calibration_error(logits, labels, bins=15, scheme="equal_width")
    equal_mass = expected_calibration_error(logits, labels, bins=15, scheme="equal_mass")
    assert equal_width != equal_mass


def test_invalid_bin_count_is_rejected():
    logits, labels = overconfident_logits(samples=50)
    with pytest.raises(ValueError, match="bins must be at least 1"):
        expected_calibration_error(logits, labels, bins=0)


# Reliability ------------------------------------------------------------------


def test_reliability_bins_partition_all_samples():
    logits, labels = overconfident_logits(samples=200)
    bins = reliability_bins(logits, labels, bins=10)
    assert sum(entry["count"] for entry in bins) == 200


def test_reliability_reports_empty_bins_rather_than_omitting_them():
    logits = torch.full((10, 2), -20.0)
    logits[:, 0] = 20.0
    labels = torch.zeros(10, dtype=torch.long)

    bins = reliability_bins(logits, labels, bins=10)
    assert any(entry["count"] == 0 for entry in bins)
    assert all(entry["mean_confidence"] is None for entry in bins if entry["count"] == 0)


def test_reliability_includes_confidence_of_exactly_one():
    """A closed final bin, so maximum-confidence samples are not dropped."""
    logits = torch.tensor([[100.0, -100.0]])
    labels = torch.tensor([0])

    bins = reliability_bins(logits, labels, bins=10)
    assert sum(entry["count"] for entry in bins) == 1


def test_reliability_gap_is_confidence_minus_accuracy():
    logits = torch.full((10, 2), -20.0)
    logits[:, 0] = 20.0
    labels = torch.tensor([0] * 5 + [1] * 5)

    populated = [entry for entry in reliability_bins(logits, labels) if entry["count"]]
    assert len(populated) == 1
    entry = populated[0]
    assert entry["gap"] == pytest.approx(entry["mean_confidence"] - entry["accuracy"])


# Confidence -------------------------------------------------------------------


def test_max_softmax_is_bounded():
    torch.manual_seed(0)
    confidence = max_softmax_confidence(torch.randn(100, 8))
    assert float(confidence.min()) >= 1 / 8 - 1e-6
    assert float(confidence.max()) <= 1.0 + 1e-6


def test_uniform_logits_give_chance_confidence():
    confidence = max_softmax_confidence(torch.zeros(1, 4))
    assert float(confidence[0]) == pytest.approx(0.25)


def test_confidence_is_stable_with_extreme_logits():
    confidence = max_softmax_confidence(torch.tensor([[1000.0, -1000.0]]))
    assert float(confidence[0]) == pytest.approx(1.0)
    assert not math.isnan(float(confidence[0]))
