"""Evaluation layer: metrics, calibration, and selective prediction.

    metrics      classification metrics, per-class and per-signer breakdowns
    calibration  temperature scaling, NLL, ECE, reliability
    selective    coverage, selective accuracy, threshold selection
    runner       evaluation execution, per-example and logit export

This layer measures without modifying. It never updates weights, and it never
lets test data influence a choice.

See docs/EVALUATION_CONTRACT.md.
"""

from .calibration import (
    BINNING_SCHEMES,
    CalibrationResult,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    max_softmax_confidence,
    reliability_bins,
)
from .metrics import (
    MACRO_AVERAGE_POLICIES,
    PerClassMetrics,
    PerSignerMetrics,
    SignerSummary,
    classification_report,
    confusion_matrix,
    confusion_pairs,
    macro_f1,
    mean_per_class_accuracy,
    negative_log_likelihood,
    per_class_metrics,
    per_signer_metrics,
    predicted_classes,
    top1_accuracy,
    top_k_accuracy,
    top_k_classes,
    weighted_f1,
)
from .runner import (
    MODES,
    EvaluationError,
    EvaluationOutput,
    build_report,
    evaluate,
    load_logits,
    per_example_records,
    save_logits,
    save_predictions,
    save_report,
)
from .selective import (
    TARGET_COVERAGE,
    TARGET_SELECTIVE_ACCURACY,
    OperatingPoint,
    ThresholdSelection,
    accuracy_coverage_curve,
    apply_threshold,
    operating_point,
    select_threshold,
    selective_report,
)

__all__ = [
    "BINNING_SCHEMES",
    "MACRO_AVERAGE_POLICIES",
    "MODES",
    "TARGET_COVERAGE",
    "TARGET_SELECTIVE_ACCURACY",
    "CalibrationResult",
    "EvaluationError",
    "EvaluationOutput",
    "OperatingPoint",
    "PerClassMetrics",
    "PerSignerMetrics",
    "SignerSummary",
    "ThresholdSelection",
    "accuracy_coverage_curve",
    "apply_temperature",
    "apply_threshold",
    "build_report",
    "classification_report",
    "confusion_matrix",
    "confusion_pairs",
    "evaluate",
    "expected_calibration_error",
    "fit_temperature",
    "load_logits",
    "macro_f1",
    "max_softmax_confidence",
    "mean_per_class_accuracy",
    "negative_log_likelihood",
    "operating_point",
    "per_class_metrics",
    "per_example_records",
    "per_signer_metrics",
    "predicted_classes",
    "reliability_bins",
    "save_logits",
    "save_predictions",
    "save_report",
    "select_threshold",
    "selective_report",
    "top1_accuracy",
    "top_k_accuracy",
    "top_k_classes",
    "weighted_f1",
]
