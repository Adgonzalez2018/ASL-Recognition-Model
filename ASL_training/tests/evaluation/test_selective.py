"""Selective prediction.

Coverage, selective accuracy, and threshold selection. The test-set isolation
property is structural: thresholds are chosen on validation and applied once
elsewhere, and the two operating points are reported separately.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.evaluation.selective import (
    accuracy_coverage_curve,
    apply_threshold,
    operating_point,
    select_threshold,
    selective_report,
)


def graded_logits(correct_high: int, wrong_low: int, num_classes: int = 4):
    """Confident correct predictions plus unconfident wrong ones.

    Rejecting by confidence should therefore raise accuracy sharply, which makes
    the threshold behaviour checkable by hand.
    """
    total = correct_high + wrong_low
    logits = torch.zeros(total, num_classes)
    labels = torch.zeros(total, dtype=torch.long)

    for index in range(correct_high):
        logits[index, 0] = 10.0  # confident and correct
        labels[index] = 0

    for offset in range(wrong_low):
        index = correct_high + offset
        logits[index, 1] = 0.15  # barely above chance, and wrong
        labels[index] = 0

    return logits, labels


# Operating points -------------------------------------------------------------


def test_zero_threshold_accepts_everything():
    logits, labels = graded_logits(6, 4)
    point = operating_point(logits, labels, 0.0)

    assert point.coverage == 1.0
    assert point.accepted == 10
    assert point.rejected == 0
    assert point.selective_accuracy == pytest.approx(0.6)


def test_high_threshold_rejects_the_unconfident():
    logits, labels = graded_logits(6, 4)
    point = operating_point(logits, labels, 0.9)

    assert point.accepted == 6
    assert point.coverage == pytest.approx(0.6)
    assert point.selective_accuracy == 1.0


def test_selective_accuracy_is_undefined_when_nothing_is_accepted():
    """No predictions were made, so there is no accuracy to report."""
    logits, labels = graded_logits(4, 4)
    point = operating_point(logits, labels, 1.1)

    assert point.accepted == 0
    assert point.coverage == 0.0
    assert point.selective_accuracy is None


def test_rejected_but_correct_is_reported():
    """The cost of the rejection policy, not just its benefit."""
    logits = torch.zeros(4, 3)
    logits[0, 0] = 10.0  # confident, correct
    logits[1, 0] = 0.05  # unconfident, correct
    logits[2, 1] = 10.0  # confident, wrong
    logits[3, 1] = 0.05  # unconfident, wrong
    labels = torch.tensor([0, 0, 0, 0])

    point = operating_point(logits, labels, 0.9)
    assert point.rejected_but_correct == 1
    assert point.rejected_but_correct_rate == pytest.approx(0.25)


def test_rejecting_correct_answers_appears_in_the_report():
    logits, labels = graded_logits(6, 4)
    payload = operating_point(logits, labels, 0.9).to_dict()
    assert "rejected_but_correct" in payload
    assert "rejected_but_correct_rate" in payload


# Curve ------------------------------------------------------------------------


def test_coverage_decreases_as_the_threshold_rises():
    torch.manual_seed(0)
    logits = torch.randn(200, 6) * 3
    labels = torch.randint(0, 6, (200,))

    curve = accuracy_coverage_curve(logits, labels)
    coverages = [point.coverage for point in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_curve_spans_the_threshold_range():
    logits, labels = graded_logits(10, 10)
    curve = accuracy_coverage_curve(logits, labels, thresholds=[0.0, 0.5, 0.99])

    assert [point.threshold for point in curve] == [0.0, 0.5, 0.99]
    assert curve[0].coverage == 1.0


def test_selective_accuracy_rises_with_the_threshold_when_confidence_is_informative():
    logits, labels = graded_logits(10, 10)
    curve = accuracy_coverage_curve(logits, labels, thresholds=[0.0, 0.9])

    assert curve[0].selective_accuracy == pytest.approx(0.5)
    assert curve[1].selective_accuracy == 1.0


# Threshold selection ----------------------------------------------------------


def test_selects_the_lowest_qualifying_threshold():
    """Among thresholds clearing the accuracy bar, decline the fewest answers."""
    logits, labels = graded_logits(8, 2)
    selection = select_threshold(logits, labels, target_selective_accuracy=0.9)

    assert selection.threshold is not None
    assert selection.achieved.selective_accuracy >= 0.9

    lower = operating_point(logits, labels, max(selection.threshold - 0.01, 0.0))
    assert lower.selective_accuracy is None or lower.selective_accuracy < 0.9


def test_reports_when_the_coverage_target_is_missed():
    """An honest 'cannot do both' is the useful answer."""
    logits, labels = graded_logits(2, 8)
    selection = select_threshold(
        logits, labels, target_selective_accuracy=0.95, target_coverage=0.5
    )

    assert selection.threshold is not None
    assert selection.achieved.coverage < 0.5
    assert selection.satisfied is False


def test_reports_when_no_threshold_reaches_the_target():
    torch.manual_seed(0)
    logits = torch.randn(100, 20) * 0.01  # uninformative
    labels = torch.randint(0, 20, (100,))

    selection = select_threshold(logits, labels, target_selective_accuracy=0.99)
    assert selection.threshold is None
    assert selection.satisfied is False
    assert selection.achieved is not None  # the best achievable is still reported


def test_selection_records_its_rule_and_provenance():
    logits, labels = graded_logits(8, 2)
    payload = select_threshold(logits, labels).to_dict()

    assert payload["selected_on"] == "validation"
    assert "lowest threshold" in payload["rule"]
    assert payload["target_selective_accuracy"] == 0.90
    assert "not" in payload["note"]


def test_invalid_accuracy_target_is_rejected():
    logits, labels = graded_logits(4, 4)
    with pytest.raises(ValueError, match="target_selective_accuracy"):
        select_threshold(logits, labels, target_selective_accuracy=1.5)


# Applying a threshold ---------------------------------------------------------


def test_applying_reports_both_operating_points():
    """The validation figure must not be presented as expected test behaviour."""
    validation_logits, validation_labels = graded_logits(8, 2)
    selection = select_threshold(validation_logits, validation_labels)

    test_logits, test_labels = graded_logits(6, 4)
    applied = apply_threshold(test_logits, test_labels, selection)

    assert applied["applied"] is not None
    assert applied["validation"] is not None
    assert "selective_accuracy_shift" in applied
    assert "coverage_shift" in applied


def test_applying_uses_the_validation_threshold_unchanged():
    validation_logits, validation_labels = graded_logits(8, 2)
    selection = select_threshold(validation_logits, validation_labels)

    test_logits, test_labels = graded_logits(6, 4)
    applied = apply_threshold(test_logits, test_labels, selection)

    assert applied["threshold"] == selection.threshold
    assert applied["applied"]["threshold"] == pytest.approx(selection.threshold)


def test_applying_nothing_when_no_threshold_qualified():
    torch.manual_seed(0)
    logits = torch.randn(80, 20) * 0.01
    labels = torch.randint(0, 20, (80,))
    selection = select_threshold(logits, labels, target_selective_accuracy=0.99)

    applied = apply_threshold(logits, labels, selection)
    assert applied["applied"] is None
    assert "nothing applied" in applied["note"]


# Report -----------------------------------------------------------------------


def test_report_records_which_confidence_was_used():
    logits, labels = graded_logits(8, 2)
    report = selective_report(logits, labels)

    assert report["confidence_source"] == "calibrated_max_softmax"
    assert report["samples"] == 10
    assert "selection" in report
    assert len(report["curve"]) == 100


def test_report_curve_entries_are_serializable():
    logits, labels = graded_logits(5, 5)
    entry = selective_report(logits, labels)["curve"][0]

    for key in ("threshold", "accepted", "rejected", "coverage", "selective_accuracy"):
        assert key in entry
