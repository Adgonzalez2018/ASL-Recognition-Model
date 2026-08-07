"""Confidence calibration.

Raw maximum softmax is not a reliable probability: modern classifiers are
typically overconfident. Confidence is only useful if it supports a decision to
decline, so it must be calibrated before any threshold means anything.

Temperature is fit on validation logits only. Fitting on test data would make
the reported operating point a fiction.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .metrics import negative_log_likelihood, predicted_classes, top1_accuracy

logger = logging.getLogger(__name__)

BINNING_SCHEMES = ("equal_width", "equal_mass")
DEFAULT_BINS = 15
DEFAULT_SCHEME = "equal_width"


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by a temperature.

    Monotonic, so it cannot change the ranking within a sample. That property is
    verified rather than assumed; see :func:`fit_temperature`.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits / temperature


def max_softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Highest softmax probability per sample."""
    return torch.softmax(logits.double(), dim=1).max(dim=1).values.float()


@dataclass
class CalibrationResult:
    """A fitted temperature and the evidence for it."""

    temperature: float
    converged: bool
    iterations: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    accuracy_before: float
    accuracy_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    bins: int
    binning_scheme: str
    fit_on: str = "validation"
    optimizer: str = "lbfgs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": round(self.temperature, 6),
            "converged": self.converged,
            "iterations": self.iterations,
            "fit_on": self.fit_on,
            "optimizer": self.optimizer,
            "bins": self.bins,
            "binning_scheme": self.binning_scheme,
            "before": {
                "nll": round(self.nll_before, 6),
                "ece": round(self.ece_before, 6),
                "accuracy": round(self.accuracy_before, 6),
                "mean_confidence": round(self.mean_confidence_before, 6),
            },
            "after": {
                "nll": round(self.nll_after, 6),
                "ece": round(self.ece_after, 6),
                "accuracy": round(self.accuracy_after, 6),
                "mean_confidence": round(self.mean_confidence_after, 6),
            },
            "note": (
                "Temperature was fit on validation logits only. Applying it to test "
                "predictions is correct; refitting on test is not."
            ),
        }


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = DEFAULT_BINS,
    scheme: str = DEFAULT_SCHEME,
    max_iterations: int = 100,
) -> CalibrationResult:
    """Fit a scalar temperature by minimizing validation NLL.

    Args:
        logits: Raw validation logits, before any scaling.
        labels: True class IDs.
        bins: ECE bin count. ECE is only comparable across runs using the same
            binning, so it is recorded.
        scheme: One of ``BINNING_SCHEMES``.
        max_iterations: L-BFGS iteration cap.

    Returns:
        The fitted temperature and before/after metrics.

    Raises:
        ValueError: If the fit produces a non-positive temperature, or if
            calibration changes any top-1 prediction. The latter would mean the
            implementation is not monotonic, which makes the result meaningless.
    """
    if scheme not in BINNING_SCHEMES:
        raise ValueError(
            f"unknown binning scheme {scheme!r}; supported: {', '.join(BINNING_SCHEMES)}"
        )

    working = logits.detach().double()
    targets = labels.detach().long()

    log_temperature = torch.zeros(1, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iterations)
    criterion = torch.nn.CrossEntropyLoss()

    steps = {"count": 0}

    def closure():
        optimizer.zero_grad()
        # Optimize in log space so the temperature cannot go non-positive.
        loss = criterion(working / log_temperature.exp(), targets)
        loss.backward()
        steps["count"] += 1
        return loss

    optimizer.step(closure)

    temperature = float(log_temperature.detach().exp())
    if not (temperature > 0 and torch.isfinite(torch.tensor(temperature))):
        raise ValueError(
            f"temperature fitting produced {temperature}, which is not usable. "
            f"Check for non-finite logits or degenerate labels."
        )

    calibrated = apply_temperature(logits, temperature)

    # Monotonicity check, not an assumption. If predictions moved, the reported
    # accuracy would silently stop matching the uncalibrated model.
    before_predictions = predicted_classes(logits)
    after_predictions = predicted_classes(calibrated)
    changed = int((before_predictions != after_predictions).sum())
    if changed:
        raise ValueError(
            f"calibration changed {changed} top-1 prediction(s). Temperature "
            f"scaling is monotonic and must not; this indicates an implementation "
            f"error."
        )

    accuracy = top1_accuracy(logits, labels)
    result = CalibrationResult(
        temperature=temperature,
        converged=steps["count"] < max_iterations,
        iterations=steps["count"],
        nll_before=negative_log_likelihood(logits, labels),
        nll_after=negative_log_likelihood(calibrated, labels),
        ece_before=expected_calibration_error(logits, labels, bins=bins, scheme=scheme),
        ece_after=expected_calibration_error(calibrated, labels, bins=bins, scheme=scheme),
        accuracy_before=accuracy,
        accuracy_after=top1_accuracy(calibrated, labels),
        mean_confidence_before=float(max_softmax_confidence(logits).mean()),
        mean_confidence_after=float(max_softmax_confidence(calibrated).mean()),
        bins=bins,
        binning_scheme=scheme,
    )

    if temperature > 1:
        logger.info(
            "Fitted temperature %.4f (> 1): the model was overconfident, which is the usual case.",
            temperature,
        )
    else:
        logger.warning(
            "Fitted temperature %.4f (< 1): the model was underconfident, which is "
            "unusual and worth checking.",
            temperature,
        )

    if result.ece_after > result.ece_before:
        logger.warning(
            "ECE worsened after calibration (%.4f -> %.4f). The fit may be poor, or "
            "the validation set may be too small.",
            result.ece_before,
            result.ece_after,
        )

    return result


def reliability_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = DEFAULT_BINS,
    scheme: str = DEFAULT_SCHEME,
) -> list[dict[str, Any]]:
    """Per-bin confidence against accuracy.

    A well-calibrated model has mean confidence equal to accuracy in every bin.
    """
    confidence = max_softmax_confidence(logits)
    correct = (predicted_classes(logits) == labels).float()

    edges = _bin_edges(confidence, bins, scheme)

    result = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        # Include the upper edge in the final bin so confidence 1.0 is counted.
        in_bin = (
            (confidence >= low) & (confidence <= high)
            if index == len(edges) - 2
            else (confidence >= low) & (confidence < high)
        )
        count = int(in_bin.sum())

        if count == 0:
            result.append(
                {
                    "bin_lower": round(float(low), 6),
                    "bin_upper": round(float(high), 6),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue

        mean_confidence = float(confidence[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        result.append(
            {
                "bin_lower": round(float(low), 6),
                "bin_upper": round(float(high), 6),
                "count": count,
                "mean_confidence": round(mean_confidence, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(mean_confidence - accuracy, 6),
            }
        )
    return result


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = DEFAULT_BINS,
    scheme: str = DEFAULT_SCHEME,
) -> float:
    """Weighted mean gap between confidence and accuracy.

    ECE is sensitive to the binning scheme, so values are only comparable across
    experiments that used the same ``bins`` and ``scheme``. Both are recorded in
    every report.
    """
    total = len(labels)
    if total == 0:
        raise ValueError("no samples to evaluate")

    error = 0.0
    for entry in reliability_bins(logits, labels, bins=bins, scheme=scheme):
        if entry["count"]:
            error += (entry["count"] / total) * abs(entry["gap"])
    return error


def _bin_edges(confidence: torch.Tensor, bins: int, scheme: str) -> list[float]:
    """Bin boundaries for the chosen scheme."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    if scheme == "equal_width":
        return [index / bins for index in range(bins + 1)]

    # equal_mass: quantile edges, so every bin holds a similar sample count.
    # More robust when confidence clusters near 1.0, which it usually does.
    quantiles = torch.linspace(0, 1, bins + 1, dtype=torch.double)
    edges = torch.quantile(confidence.double(), quantiles).tolist()
    edges[0], edges[-1] = 0.0, 1.0

    # Collapse duplicate edges, which appear when confidence is highly
    # concentrated; a zero-width bin would otherwise skew the weighting.
    deduplicated = [edges[0]]
    for edge in edges[1:]:
        if edge > deduplicated[-1]:
            deduplicated.append(edge)
    return deduplicated
