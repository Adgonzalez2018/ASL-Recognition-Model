"""Classification metrics.

Implemented directly rather than delegated, so the aggregation and label
alignment are the project's own and can be pinned by tests. The test suite
cross-checks these against an independent reference.

Two choices are made explicitly here because they change reported numbers and
must be identical across every compared experiment:

* argmax ties resolve to the lowest class ID
* macro averages are taken over classes with support, and the count is reported

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# How macro averages treat classes the split does not contain.
#   "support"   average over classes appearing in the true labels
#   "all"       average over every label-map class, scoring absent ones 0
MACRO_AVERAGE_POLICIES = ("support", "all")
DEFAULT_MACRO_POLICY = "support"


def predicted_classes(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 predictions, breaking ties toward the lowest class ID.

    Ties are rare in floating point but must not introduce run-to-run variation,
    so the rule is explicit rather than left to the backend.
    """
    _validate_logits(logits)
    maxima = logits.max(dim=1, keepdim=True).values
    is_max = logits == maxima
    # The first True along each row is the lowest class ID holding the maximum.
    return is_max.float().argmax(dim=1)


def top_k_classes(logits: torch.Tensor, k: int) -> torch.Tensor:
    """The ``k`` highest-scoring class IDs per sample, best first."""
    _validate_logits(logits)
    k = min(k, logits.shape[1])
    return logits.topk(k, dim=1).indices


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of samples whose highest-scoring class is correct."""
    _validate_pair(logits, labels)
    return float((predicted_classes(logits) == labels).float().mean())


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float | None:
    """Fraction of samples whose true class is among the ``k`` highest.

    Returns ``None`` when the vocabulary has fewer than ``k`` classes. Reporting
    1.0 for an undefined metric would be misleading.
    """
    _validate_pair(logits, labels)
    if logits.shape[1] < k:
        return None

    top_k = top_k_classes(logits, k)
    return float((top_k == labels.unsqueeze(1)).any(dim=1).float().mean())


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean negative log probability of the true class.

    Computed with log-softmax for numerical stability. Taking the log of a
    rounded probability loses precision exactly where confident predictions
    matter most.
    """
    _validate_pair(logits, labels)
    log_probs = torch.log_softmax(logits.double(), dim=1)
    return float(-log_probs[torch.arange(len(labels)), labels].mean())


@dataclass
class PerClassMetrics:
    """Precision, recall, F1, and counts for one class."""

    class_id: int
    gloss: str | None
    support: int
    predicted: int
    true_positives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "gloss": self.gloss,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


def per_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    glosses: list[str] | None = None,
) -> list[PerClassMetrics]:
    """Precision, recall, and F1 for every class in the vocabulary.

    Classes absent from the split are included with zero support, so a caller can
    see what was not evaluated rather than having it quietly omitted.
    """
    _validate_pair(logits, labels)
    predictions = predicted_classes(logits)

    # Counted with bincount rather than a per-class scan. This runs once per
    # validation epoch over the full vocabulary, and a 2731-iteration pass of
    # tensor comparisons there is a real cost.
    support = torch.bincount(labels, minlength=num_classes)
    predicted = torch.bincount(predictions, minlength=num_classes)
    hits = torch.bincount(labels[predictions == labels], minlength=num_classes)

    results = []
    for class_id in range(num_classes):
        support_count = int(support[class_id])
        predicted_count = int(predicted[class_id])
        true_positives = int(hits[class_id])

        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / support_count if support_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append(
            PerClassMetrics(
                class_id=class_id,
                gloss=glosses[class_id] if glosses else None,
                support=support_count,
                predicted=predicted_count,
                true_positives=true_positives,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return results


def macro_f1(
    per_class: list[PerClassMetrics],
    policy: str = DEFAULT_MACRO_POLICY,
) -> tuple[float, int]:
    """Unweighted mean F1, and how many classes it averaged over.

    The count is returned alongside the value because a macro average over 2000
    classes and one over 40 are not comparable, and the difference is invisible
    in the number itself.
    """
    if policy not in MACRO_AVERAGE_POLICIES:
        raise ValueError(
            f"unknown macro policy {policy!r}; supported: {', '.join(MACRO_AVERAGE_POLICIES)}"
        )

    selected = [m for m in per_class if m.support > 0] if policy == "support" else per_class
    if not selected:
        return 0.0, 0
    return sum(m.f1 for m in selected) / len(selected), len(selected)


def weighted_f1(per_class: list[PerClassMetrics]) -> float:
    """Support-weighted mean F1."""
    total = sum(m.support for m in per_class)
    if not total:
        return 0.0
    return sum(m.f1 * m.support for m in per_class) / total


def mean_per_class_accuracy(
    per_class: list[PerClassMetrics],
    policy: str = DEFAULT_MACRO_POLICY,
) -> tuple[float, int]:
    """Unweighted mean recall: the balanced-accuracy view.

    Required alongside top-1 because ASL Citizen's class support is uneven, and
    top-1 can look healthy while the tail is ignored entirely.
    """
    selected = [m for m in per_class if m.support > 0] if policy == "support" else per_class
    if not selected:
        return 0.0, 0
    return sum(m.recall for m in selected) / len(selected), len(selected)


@dataclass
class PerSignerMetrics:
    """Accuracy for one signer."""

    signer_id: str
    samples: int
    correct: int
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "samples": self.samples,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
        }


@dataclass
class SignerSummary:
    """Per-signer distribution, with a support floor applied to extremes.

    Worst-signer accuracy computed over a signer with three samples is noise, so
    signers below the floor are reported separately rather than being allowed to
    define the worst case.
    """

    per_signer: list[PerSignerMetrics]
    support_floor: int
    worst: PerSignerMetrics | None = None
    best: PerSignerMetrics | None = None
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    below_floor: list[PerSignerMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_floor": self.support_floor,
            "signers": len(self.per_signer),
            "signers_below_floor": len(self.below_floor),
            "worst": self.worst.to_dict() if self.worst else None,
            "best": self.best.to_dict() if self.best else None,
            "mean_accuracy": round(self.mean_accuracy, 6),
            "std_accuracy": round(self.std_accuracy, 6),
            "per_signer": [m.to_dict() for m in self.per_signer],
            "below_floor": [m.to_dict() for m in self.below_floor],
        }


def per_signer_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    signer_ids: list[str],
    *,
    support_floor: int = 10,
) -> SignerSummary:
    """Accuracy per signer, plus the worst and best above the support floor.

    Signer-independent generalization is a primary project goal, so this is
    required reporting rather than optional analysis.
    """
    _validate_pair(logits, labels)
    if len(signer_ids) != len(labels):
        raise ValueError(f"{len(signer_ids)} signer id(s) but {len(labels)} label(s)")

    correct = predicted_classes(logits) == labels

    grouped: dict[str, list[bool]] = {}
    for index, signer in enumerate(signer_ids):
        grouped.setdefault(signer, []).append(bool(correct[index]))

    per_signer = [
        PerSignerMetrics(
            signer_id=signer,
            samples=len(results),
            correct=sum(results),
            accuracy=sum(results) / len(results),
        )
        for signer, results in sorted(grouped.items())
    ]

    eligible = [m for m in per_signer if m.samples >= support_floor]
    below = [m for m in per_signer if m.samples < support_floor]

    accuracies = [m.accuracy for m in per_signer]
    mean = sum(accuracies) / len(accuracies) if accuracies else 0.0
    variance = sum((a - mean) ** 2 for a in accuracies) / len(accuracies) if accuracies else 0.0

    return SignerSummary(
        per_signer=per_signer,
        support_floor=support_floor,
        worst=min(eligible, key=lambda m: m.accuracy) if eligible else None,
        best=max(eligible, key=lambda m: m.accuracy) if eligible else None,
        mean_accuracy=mean,
        std_accuracy=variance**0.5,
        below_floor=below,
    )


def confusion_pairs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    glosses: list[str] | None = None,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """The most frequent confused class pairs.

    A full matrix over a large vocabulary is not directly interpretable, so the
    ranked pairs are what review actually uses. Glosses are retained, because a
    linguistically plausible confusion reads very differently from a random one.
    """
    _validate_pair(logits, labels)
    predictions = predicted_classes(logits)

    counts: dict[tuple[int, int], int] = {}
    for true_id, predicted_id in zip(labels.tolist(), predictions.tolist(), strict=True):
        if true_id != predicted_id:
            key = (true_id, predicted_id)
            counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "true_class_id": true_id,
            "predicted_class_id": predicted_id,
            "true_gloss": glosses[true_id] if glosses else None,
            "predicted_gloss": glosses[predicted_id] if glosses else None,
            "count": count,
        }
        for (true_id, predicted_id), count in ranked
    ]


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Counts indexed ``[true_class, predicted_class]``."""
    _validate_pair(logits, labels)
    predictions = predicted_classes(logits)

    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_id, predicted_id in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[true_id, predicted_id] += 1
    return matrix


def classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    glosses: list[str] | None = None,
    signer_ids: list[str] | None = None,
    macro_policy: str = DEFAULT_MACRO_POLICY,
    signer_support_floor: int = 10,
) -> dict[str, Any]:
    """The full clean metric set.

    Aggregate values, per-class and per-signer breakdowns, confusion analysis,
    and the policy choices that produced them.
    """
    _validate_pair(logits, labels)

    per_class = per_class_metrics(logits, labels, num_classes, glosses)
    macro, macro_count = macro_f1(per_class, macro_policy)
    mean_class_accuracy, balanced_count = mean_per_class_accuracy(per_class, macro_policy)

    report: dict[str, Any] = {
        "samples": len(labels),
        "num_classes": num_classes,
        "top1_accuracy": top1_accuracy(logits, labels),
        "top5_accuracy": top_k_accuracy(logits, labels, 5),
        "macro_f1": macro,
        "weighted_f1": weighted_f1(per_class),
        "mean_per_class_accuracy": mean_class_accuracy,
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
        "classes_in_macro_average": macro_count,
        "classes_in_balanced_accuracy": balanced_count,
        "classes_with_support": sum(1 for m in per_class if m.support > 0),
        "classes_never_predicted": sum(1 for m in per_class if m.predicted == 0),
        "policies": {
            "macro_average": macro_policy,
            "tie_breaking": "lowest_class_id",
            "signer_support_floor": signer_support_floor,
        },
        "per_class": [m.to_dict() for m in per_class],
        "confusion_pairs": confusion_pairs(logits, labels, glosses),
    }

    if report["top5_accuracy"] is None:
        report["top5_accuracy_unavailable_reason"] = (
            f"vocabulary has {num_classes} classes; top-5 is undefined below 5"
        )

    if signer_ids is not None:
        report["per_signer"] = per_signer_metrics(
            logits, labels, signer_ids, support_floor=signer_support_floor
        ).to_dict()

    return report


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be [samples, classes], got {tuple(logits.shape)}")
    if logits.shape[0] == 0:
        raise ValueError("no samples to evaluate")
    if not torch.isfinite(logits).all():
        raise ValueError(
            "logits contain non-finite values; the checkpoint or the evaluation "
            "pass is producing invalid output"
        )


def _validate_pair(logits: torch.Tensor, labels: torch.Tensor) -> None:
    _validate_logits(logits)
    if labels.ndim != 1:
        raise ValueError(f"labels must be [samples], got {tuple(labels.shape)}")
    if len(labels) != logits.shape[0]:
        raise ValueError(f"{logits.shape[0]} logit row(s) but {len(labels)} label(s)")
    if labels.numel():
        low, high = int(labels.min()), int(labels.max())
        if low < 0 or high >= logits.shape[1]:
            raise ValueError(
                f"labels span [{low}, {high}] but logits cover [0, {logits.shape[1] - 1}]"
            )
