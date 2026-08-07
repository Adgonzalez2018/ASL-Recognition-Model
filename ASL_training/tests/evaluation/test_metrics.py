"""Classification metrics.

Correctness is established two ways: against hand-computed values on small
fixtures with known answers, and against sklearn as an independent reference.
The second matters because the risk is not that a formula is wrong but that the
project's aggregation or label alignment differs from what a reader assumes.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.evaluation.metrics import (
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
    weighted_f1,
)


def one_hot_logits(labels: list[int], num_classes: int, magnitude: float = 10.0):
    """Logits that predict exactly the given classes."""
    logits = torch.zeros(len(labels), num_classes)
    for row, label in enumerate(labels):
        logits[row, label] = magnitude
    return logits


# Accuracy ---------------------------------------------------------------------


def test_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 3])
    assert top1_accuracy(one_hot_logits([0, 1, 2, 3], 4), labels) == 1.0


def test_completely_wrong_predictions():
    labels = torch.tensor([0, 1, 2, 3])
    assert top1_accuracy(one_hot_logits([1, 2, 3, 0], 4), labels) == 0.0


def test_partial_accuracy():
    labels = torch.tensor([0, 1, 2, 3])
    assert top1_accuracy(one_hot_logits([0, 1, 3, 2], 4), labels) == 0.5


def test_single_class_edge_case():
    logits = torch.tensor([[5.0, 1.0], [4.0, 2.0]])
    assert top1_accuracy(logits, torch.tensor([0, 0])) == 1.0


# Tie-breaking -----------------------------------------------------------------


def test_argmax_ties_resolve_to_the_lowest_class_id():
    """Deterministic, so results cannot vary between runs or backends."""
    logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 5.0, 5.0]])
    assert predicted_classes(logits).tolist() == [0, 1]


def test_tie_breaking_is_stable_across_calls():
    logits = torch.full((20, 5), 2.0)
    first = predicted_classes(logits)
    for _ in range(5):
        assert torch.equal(predicted_classes(logits), first)


# Top-k ------------------------------------------------------------------------


def test_top5_counts_a_true_class_ranked_third():
    logits = torch.zeros(1, 10)
    logits[0, 7] = 3.0  # first
    logits[0, 2] = 2.0  # second
    logits[0, 4] = 1.0  # third
    labels = torch.tensor([4])

    assert top_k_accuracy(logits, labels, 5) == 1.0
    assert top1_accuracy(logits, labels) == 0.0


def test_top5_is_unavailable_below_five_classes():
    """Returning 1.0 for an undefined metric would be misleading."""
    logits = torch.randn(4, 3)
    assert top_k_accuracy(logits, torch.tensor([0, 1, 2, 0]), 5) is None


def test_top5_is_defined_at_exactly_five_classes():
    logits = torch.randn(4, 5)
    assert top_k_accuracy(logits, torch.tensor([0, 1, 2, 3]), 5) == 1.0


def test_top5_is_at_least_top1():
    torch.manual_seed(0)
    logits = torch.randn(50, 10)
    labels = torch.randint(0, 10, (50,))
    assert top_k_accuracy(logits, labels, 5) >= top1_accuracy(logits, labels)


# Per-class --------------------------------------------------------------------


def test_per_class_counts_are_hand_checkable():
    # Class 0: 2 true, both predicted correctly.
    # Class 1: 2 true, one predicted as 0.
    # Class 2: 0 true, never predicted.
    labels = torch.tensor([0, 0, 1, 1])
    logits = one_hot_logits([0, 0, 1, 0], 3)

    metrics = {m.class_id: m for m in per_class_metrics(logits, labels, 3)}

    assert metrics[0].support == 2
    assert metrics[0].predicted == 3
    assert metrics[0].true_positives == 2
    assert metrics[0].precision == pytest.approx(2 / 3)
    assert metrics[0].recall == 1.0

    assert metrics[1].support == 2
    assert metrics[1].recall == 0.5
    assert metrics[1].precision == 1.0

    assert metrics[2].support == 0
    assert metrics[2].f1 == 0.0


def test_classes_absent_from_the_split_are_still_listed():
    """Omitting them would hide what was not evaluated."""
    labels = torch.tensor([0, 0])
    metrics = per_class_metrics(one_hot_logits([0, 0], 5), labels, 5)

    assert len(metrics) == 5
    assert sum(1 for m in metrics if m.support == 0) == 4


def test_gloss_labels_are_attached():
    metrics = per_class_metrics(
        one_hot_logits([0, 1], 2), torch.tensor([0, 1]), 2, ["APPLE", "BOOK"]
    )
    assert [m.gloss for m in metrics] == ["APPLE", "BOOK"]


# Macro averaging --------------------------------------------------------------


def test_macro_f1_reports_how_many_classes_it_averaged():
    """A macro average over 2000 classes and over 40 are not comparable."""
    labels = torch.tensor([0, 1])
    per_class = per_class_metrics(one_hot_logits([0, 1], 10), labels, 10)

    value, count = macro_f1(per_class, "support")
    assert value == 1.0
    assert count == 2

    value_all, count_all = macro_f1(per_class, "all")
    assert count_all == 10
    assert value_all == pytest.approx(0.2)


def test_unknown_macro_policy_is_rejected():
    per_class = per_class_metrics(one_hot_logits([0], 2), torch.tensor([0]), 2)
    with pytest.raises(ValueError, match="unknown macro policy"):
        macro_f1(per_class, "micro")


def test_weighted_f1_follows_support():
    # Class 0 has 3 samples all correct; class 1 has 1 sample, wrong.
    labels = torch.tensor([0, 0, 0, 1])
    per_class = per_class_metrics(one_hot_logits([0, 0, 0, 0], 2), labels, 2)

    macro, _ = macro_f1(per_class)
    weighted = weighted_f1(per_class)
    assert weighted > macro, "weighted should favour the well-served majority class"


def test_mean_per_class_accuracy_is_balanced_accuracy():
    """Top-1 can look healthy while the tail is ignored; this shows that."""
    labels = torch.tensor([0] * 9 + [1])
    logits = one_hot_logits([0] * 10, 2)

    per_class = per_class_metrics(logits, labels, 2)
    balanced, _ = mean_per_class_accuracy(per_class)

    assert top1_accuracy(logits, labels) == pytest.approx(0.9)
    assert balanced == pytest.approx(0.5)


# Reference cross-check --------------------------------------------------------


def test_matches_sklearn_on_random_data():
    """An independent reference for aggregation and label alignment."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    torch.manual_seed(7)
    num_classes = 8
    logits = torch.randn(200, num_classes)
    labels = torch.randint(0, num_classes, (200,))
    predictions = predicted_classes(logits)

    per_class = per_class_metrics(logits, labels, num_classes)

    assert top1_accuracy(logits, labels) == pytest.approx(
        sklearn_metrics.accuracy_score(labels.numpy(), predictions.numpy())
    )

    present = sorted(set(labels.tolist()))
    expected_macro = sklearn_metrics.f1_score(
        labels.numpy(), predictions.numpy(), labels=present, average="macro", zero_division=0
    )
    assert macro_f1(per_class, "support")[0] == pytest.approx(expected_macro)

    assert weighted_f1(per_class) == pytest.approx(
        sklearn_metrics.f1_score(
            labels.numpy(), predictions.numpy(), average="weighted", zero_division=0
        )
    )

    assert mean_per_class_accuracy(per_class, "support")[0] == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(labels.numpy(), predictions.numpy())
    )


def test_per_class_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    torch.manual_seed(3)
    logits = torch.randn(120, 5)
    labels = torch.randint(0, 5, (120,))
    predictions = predicted_classes(logits)

    precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(
        labels.numpy(), predictions.numpy(), labels=range(5), zero_division=0
    )
    ours = per_class_metrics(logits, labels, 5)

    for index, metric in enumerate(ours):
        assert metric.precision == pytest.approx(precision[index])
        assert metric.recall == pytest.approx(recall[index])
        assert metric.f1 == pytest.approx(f1[index])
        assert metric.support == support[index]


def test_confusion_matrix_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    torch.manual_seed(11)
    logits = torch.randn(80, 4)
    labels = torch.randint(0, 4, (80,))

    ours = confusion_matrix(logits, labels, 4).numpy()
    theirs = sklearn_metrics.confusion_matrix(
        labels.numpy(), predicted_classes(logits).numpy(), labels=range(4)
    )
    assert (ours == theirs).all()


# Negative log-likelihood ------------------------------------------------------


def test_nll_of_a_confident_correct_prediction_is_near_zero():
    logits = one_hot_logits([0], 4, magnitude=20.0)
    assert negative_log_likelihood(logits, torch.tensor([0])) == pytest.approx(0.0, abs=1e-6)


def test_nll_of_uniform_predictions_is_log_num_classes():
    import math

    logits = torch.zeros(4, 10)
    labels = torch.tensor([0, 1, 2, 3])
    assert negative_log_likelihood(logits, labels) == pytest.approx(math.log(10))


def test_nll_is_stable_with_extreme_logits():
    """Naive exp/log would overflow here."""
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    value = negative_log_likelihood(logits, torch.tensor([0, 1]))
    assert value == pytest.approx(0.0, abs=1e-6)

    wrong = negative_log_likelihood(logits, torch.tensor([1, 0]))
    assert wrong > 100
    assert wrong == wrong  # not NaN


# Per-signer -------------------------------------------------------------------


def test_per_signer_accuracy():
    labels = torch.tensor([0, 0, 1, 1])
    logits = one_hot_logits([0, 0, 1, 0], 2)
    signers = ["s1", "s1", "s2", "s2"]

    summary = per_signer_metrics(logits, labels, signers, support_floor=1)
    by_signer = {m.signer_id: m for m in summary.per_signer}

    assert by_signer["s1"].accuracy == 1.0
    assert by_signer["s2"].accuracy == 0.5
    assert summary.worst.signer_id == "s2"
    assert summary.best.signer_id == "s1"


def test_low_support_signers_do_not_define_the_worst_case():
    """Worst-signer accuracy over three samples is noise, not a finding."""
    labels = torch.tensor([0] * 20 + [1])
    predictions = [0] * 20 + [0]  # the lone sample from s_rare is wrong
    logits = one_hot_logits(predictions, 2)
    signers = ["s_main"] * 20 + ["s_rare"]

    summary = per_signer_metrics(logits, labels, signers, support_floor=10)

    assert summary.worst.signer_id == "s_main"
    assert [m.signer_id for m in summary.below_floor] == ["s_rare"]


def test_signer_distribution_statistics():
    labels = torch.tensor([0, 0, 0, 0])
    logits = one_hot_logits([0, 0, 0, 1], 2)
    summary = per_signer_metrics(logits, labels, ["a", "a", "b", "b"], support_floor=1)

    assert summary.mean_accuracy == pytest.approx(0.75)
    assert summary.std_accuracy == pytest.approx(0.25)


def test_signer_id_count_must_match():
    with pytest.raises(ValueError, match="signer id"):
        per_signer_metrics(one_hot_logits([0, 0], 2), torch.tensor([0, 0]), ["only_one"])


# Confusion --------------------------------------------------------------------


def test_confusion_pairs_rank_by_frequency():
    labels = torch.tensor([0, 0, 0, 1])
    logits = one_hot_logits([1, 1, 1, 0], 2)

    pairs = confusion_pairs(logits, labels, ["APPLE", "BOOK"])
    assert pairs[0]["true_class_id"] == 0
    assert pairs[0]["predicted_class_id"] == 1
    assert pairs[0]["count"] == 3
    assert pairs[0]["true_gloss"] == "APPLE"
    assert pairs[0]["predicted_gloss"] == "BOOK"


def test_confusion_pairs_exclude_correct_predictions():
    labels = torch.tensor([0, 1])
    assert confusion_pairs(one_hot_logits([0, 1], 2), labels) == []


# Full report ------------------------------------------------------------------


def test_report_contains_the_required_metric_set():
    torch.manual_seed(1)
    logits = torch.randn(60, 10)
    labels = torch.randint(0, 10, (60,))
    signers = [f"s{i % 4}" for i in range(60)]

    report = classification_report(
        logits, labels, 10, glosses=[f"G{i}" for i in range(10)], signer_ids=signers
    )

    for key in (
        "top1_accuracy",
        "top5_accuracy",
        "macro_f1",
        "weighted_f1",
        "mean_per_class_accuracy",
        "negative_log_likelihood",
        "per_class",
        "per_signer",
        "confusion_pairs",
        "policies",
    ):
        assert key in report, f"report is missing {key}"


def test_report_records_its_policy_choices():
    """ECE-style comparability: the policies must travel with the numbers."""
    logits = torch.randn(20, 6)
    labels = torch.randint(0, 6, (20,))
    report = classification_report(logits, labels, 6)

    assert report["policies"]["macro_average"] == "support"
    assert report["policies"]["tie_breaking"] == "lowest_class_id"
    assert "signer_support_floor" in report["policies"]


def test_report_explains_an_unavailable_top5():
    report = classification_report(torch.randn(10, 3), torch.randint(0, 3, (10,)), 3)
    assert report["top5_accuracy"] is None
    assert "undefined below 5" in report["top5_accuracy_unavailable_reason"]


def test_report_counts_classes_never_predicted():
    labels = torch.tensor([0, 1, 2])
    report = classification_report(one_hot_logits([0, 0, 0], 3), labels, 3)
    assert report["classes_never_predicted"] == 2


# Validation -------------------------------------------------------------------


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no samples"):
        top1_accuracy(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))


def test_non_finite_logits_are_rejected():
    logits = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    with pytest.raises(ValueError, match="non-finite"):
        top1_accuracy(logits, torch.tensor([0, 1]))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="logit row"):
        top1_accuracy(torch.randn(3, 4), torch.tensor([0, 1]))


def test_labels_outside_the_vocabulary_are_rejected():
    with pytest.raises(ValueError, match="labels span"):
        top1_accuracy(torch.randn(2, 3), torch.tensor([0, 5]))


def test_wrong_logit_rank_is_rejected():
    with pytest.raises(ValueError, match=r"\[samples, classes\]"):
        top1_accuracy(torch.randn(5), torch.tensor([0]))
