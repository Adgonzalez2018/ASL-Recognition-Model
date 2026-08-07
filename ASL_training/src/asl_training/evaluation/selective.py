"""Selective prediction.

A deployed system should be able to decline rather than return a confident wrong
sign. This module measures what declining buys and what it costs.

Thresholds are selected on validation data and applied once to test. Choosing a
threshold by looking at test results turns the reported operating point into a
number that describes nothing.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .calibration import max_softmax_confidence
from .metrics import predicted_classes

logger = logging.getLogger(__name__)

# The project's initial serving-candidate benchmark. A target, not a guarantee.
TARGET_SELECTIVE_ACCURACY = 0.90
TARGET_COVERAGE = 0.50


@dataclass
class OperatingPoint:
    """Behaviour at one confidence threshold."""

    threshold: float
    accepted: int
    rejected: int
    total: int
    coverage: float
    selective_accuracy: float | None
    rejected_but_correct: int

    @property
    def rejected_but_correct_rate(self) -> float:
        """Fraction of all samples that were declined despite being right.

        The cost of the rejection policy. A threshold with excellent selective
        accuracy that throws away many correct answers is not obviously a good
        trade, and this is what makes that visible.
        """
        return self.rejected_but_correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "coverage": round(self.coverage, 6),
            "selective_accuracy": (
                round(self.selective_accuracy, 6) if self.selective_accuracy is not None else None
            ),
            "rejected_but_correct": self.rejected_but_correct,
            "rejected_but_correct_rate": round(self.rejected_but_correct_rate, 6),
        }


def operating_point(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    *,
    confidence: torch.Tensor | None = None,
) -> OperatingPoint:
    """Behaviour when accepting predictions at or above ``threshold``.

    Args:
        logits: Calibrated logits, normally. The caller decides; the report
            records which confidence was used.
        labels: True class IDs.
        threshold: Minimum confidence to accept.
        confidence: Precomputed confidence, to avoid recomputing across a sweep.
    """
    if confidence is None:
        confidence = max_softmax_confidence(logits)

    correct = predicted_classes(logits) == labels
    accepted = confidence >= threshold

    accepted_count = int(accepted.sum())
    total = len(labels)

    return OperatingPoint(
        threshold=threshold,
        accepted=accepted_count,
        rejected=total - accepted_count,
        total=total,
        coverage=accepted_count / total if total else 0.0,
        # Undefined rather than zero when nothing is accepted: no predictions
        # were made, so there is no accuracy to report.
        selective_accuracy=(float(correct[accepted].float().mean()) if accepted_count else None),
        rejected_but_correct=int((correct & ~accepted).sum()),
    )


def accuracy_coverage_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    thresholds: list[float] | None = None,
) -> list[OperatingPoint]:
    """Selective accuracy against coverage across a threshold sweep."""
    confidence = max_softmax_confidence(logits)
    if thresholds is None:
        thresholds = [index / 100 for index in range(0, 100)]

    return [
        operating_point(logits, labels, threshold, confidence=confidence)
        for threshold in sorted(thresholds)
    ]


@dataclass
class ThresholdSelection:
    """A threshold chosen on validation data, and why."""

    threshold: float | None
    rule: str
    target_selective_accuracy: float
    target_coverage: float
    achieved: OperatingPoint | None
    satisfied: bool
    selected_on: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
            "rule": self.rule,
            "target_selective_accuracy": self.target_selective_accuracy,
            "target_coverage": self.target_coverage,
            "satisfied": self.satisfied,
            "selected_on": self.selected_on,
            "achieved": self.achieved.to_dict() if self.achieved else None,
            "note": (
                "Selected on validation data. Applying it to test is correct; "
                "re-selecting against test results is not. Test behaviour will "
                "differ from what is recorded here."
            ),
        }


def select_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_selective_accuracy: float = TARGET_SELECTIVE_ACCURACY,
    target_coverage: float = TARGET_COVERAGE,
    thresholds: list[float] | None = None,
) -> ThresholdSelection:
    """Choose the lowest threshold meeting the accuracy target.

    Lowest, because among thresholds that clear the accuracy bar, the one that
    declines fewest answers is preferable.

    Reports whether the coverage target was met as well, but does not fail when
    it is not: an honest "this model cannot do both" is the useful answer.
    """
    if not 0 < target_selective_accuracy <= 1:
        raise ValueError(
            f"target_selective_accuracy must be in (0, 1], got {target_selective_accuracy}"
        )

    curve = accuracy_coverage_curve(logits, labels, thresholds=thresholds)
    rule = f"lowest threshold with selective accuracy >= {target_selective_accuracy} on validation"

    qualifying = [
        point
        for point in curve
        if point.selective_accuracy is not None
        and point.selective_accuracy >= target_selective_accuracy
        and point.accepted > 0
    ]

    if not qualifying:
        best = max(
            (p for p in curve if p.selective_accuracy is not None),
            key=lambda p: p.selective_accuracy,
            default=None,
        )
        logger.warning(
            "no threshold reaches %.0f%% selective accuracy; the best achievable is "
            "%.4f at coverage %.4f",
            target_selective_accuracy * 100,
            best.selective_accuracy if best else float("nan"),
            best.coverage if best else float("nan"),
        )
        return ThresholdSelection(
            threshold=None,
            rule=rule,
            target_selective_accuracy=target_selective_accuracy,
            target_coverage=target_coverage,
            achieved=best,
            satisfied=False,
        )

    chosen = min(qualifying, key=lambda p: p.threshold)
    satisfied = chosen.coverage >= target_coverage

    if not satisfied:
        logger.warning(
            "threshold %.2f reaches %.4f selective accuracy but only %.4f coverage, "
            "below the %.2f target. The model can be accurate or available, not both, "
            "at this operating point.",
            chosen.threshold,
            chosen.selective_accuracy,
            chosen.coverage,
            target_coverage,
        )

    return ThresholdSelection(
        threshold=chosen.threshold,
        rule=rule,
        target_selective_accuracy=target_selective_accuracy,
        target_coverage=target_coverage,
        achieved=chosen,
        satisfied=satisfied,
    )


def apply_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    selection: ThresholdSelection,
) -> dict[str, Any]:
    """Apply a validation-selected threshold once to another split.

    Presents the validation and applied operating points side by side, because
    they will differ and the validation figure must not be presented as the
    expected behaviour elsewhere.
    """
    if selection.threshold is None:
        return {
            "applied": None,
            "validation": selection.to_dict(),
            "note": "no threshold satisfied the target on validation; nothing applied",
        }

    applied = operating_point(logits, labels, selection.threshold)

    result = {
        "threshold": selection.threshold,
        "applied": applied.to_dict(),
        "validation": selection.achieved.to_dict() if selection.achieved else None,
        "rule": selection.rule,
    }

    if (
        applied.selective_accuracy is not None
        and selection.achieved is not None
        and selection.achieved.selective_accuracy is not None
    ):
        result["selective_accuracy_shift"] = round(
            applied.selective_accuracy - selection.achieved.selective_accuracy, 6
        )
        result["coverage_shift"] = round(applied.coverage - selection.achieved.coverage, 6)
    return result


def selective_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    confidence_source: str = "calibrated_max_softmax",
    target_selective_accuracy: float = TARGET_SELECTIVE_ACCURACY,
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, Any]:
    """The full accuracy-versus-coverage picture for one split."""
    curve = accuracy_coverage_curve(logits, labels)
    selection = select_threshold(
        logits,
        labels,
        target_selective_accuracy=target_selective_accuracy,
        target_coverage=target_coverage,
    )

    return {
        "confidence_source": confidence_source,
        "samples": len(labels),
        "selection": selection.to_dict(),
        "curve": [point.to_dict() for point in curve],
    }
