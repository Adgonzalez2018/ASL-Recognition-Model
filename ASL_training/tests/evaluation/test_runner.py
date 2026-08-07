"""Evaluation execution and export.

Two properties are structural rather than incidental: evaluation never modifies
weights, and every manifest sample is accounted for.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import csv

import pytest
import torch
from torch.utils.data import DataLoader

from asl_training.data import LabelMap, collate_clips
from asl_training.evaluation import (
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
from asl_training.models import ModelConfig, build_model

from ..training.conftest import SyntheticClips

GLOSSES = ["APPLE", "BOOK", "CAT", "DOG"]


@pytest.fixture
def label_map() -> LabelMap:
    return LabelMap.from_glosses(GLOSSES)


@pytest.fixture
def model():
    return build_model(
        ModelConfig(
            architecture="videomae_base",
            num_classes=len(GLOSSES),
            pretrained=False,
            num_frames=4,
            image_size=32,
            options={
                "hidden_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 96,
            },
        )
    )


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader(
        SyntheticClips(size=12, num_classes=len(GLOSSES)),
        batch_size=4,
        collate_fn=collate_clips,
        num_workers=0,
    )


@pytest.fixture
def output(model, loader) -> EvaluationOutput:
    return evaluate(model, loader, mode="validation")


# Weight immutability ----------------------------------------------------------


def test_evaluation_does_not_modify_weights(model, loader):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate(model, loader)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed during evaluation"


def test_evaluation_produces_no_gradients(model, loader):
    output = evaluate(model, loader)
    assert not output.logits.requires_grad

    for name, param in model.named_parameters():
        assert param.grad is None or float(param.grad.abs().sum()) == 0.0, name


def test_evaluation_restores_the_prior_training_mode(model, loader):
    model.train()
    evaluate(model, loader)
    assert model.training, "evaluation left the model in eval mode mid-training"

    model.eval()
    evaluate(model, loader)
    assert not model.training


# Sample accounting ------------------------------------------------------------


def test_every_manifest_sample_is_evaluated(output):
    assert output.evaluated == 12
    assert output.manifest_samples == 12
    assert output.complete


def test_metadata_lists_align_with_logits(output):
    assert len(output.sample_ids) == output.evaluated
    assert len(output.signer_ids) == output.evaluated
    assert len(output.glosses) == output.evaluated
    assert output.logits.shape[0] == output.evaluated


def test_dropping_samples_is_rejected(model):
    """drop_last would silently change the evaluated sample count."""
    loader = DataLoader(
        SyntheticClips(size=10, num_classes=len(GLOSSES)),
        batch_size=4,
        collate_fn=collate_clips,
        num_workers=0,
        drop_last=True,
    )
    with pytest.raises(EvaluationError, match="Silent sample loss"):
        evaluate(model, loader)


def test_mismatched_manifest_count_is_rejected(model, loader):
    with pytest.raises(EvaluationError, match="but the manifest has 99"):
        evaluate(model, loader, manifest_samples=99)


def test_unknown_mode_is_rejected(model, loader):
    with pytest.raises(ValueError, match="unknown mode"):
        evaluate(model, loader, mode="train")


def test_output_rejects_inconsistent_lengths():
    with pytest.raises(EvaluationError, match="lengths disagree"):
        EvaluationOutput(
            logits=torch.randn(3, 4),
            labels=torch.tensor([0, 1, 2]),
            sample_ids=["a", "b"],
            signer_ids=["s", "s", "s"],
            glosses=["g", "g", "g"],
            manifest_samples=3,
            mode="validation",
        )


# Per-example export -----------------------------------------------------------


def test_per_example_records_carry_everything_needed(output, label_map):
    records = per_example_records(output, label_map)
    assert len(records) == output.evaluated

    for key in (
        "sample_id",
        "signer_id",
        "gloss",
        "true_class_id",
        "predicted_class_id",
        "predicted_gloss",
        "correct",
        "confidence",
        "top_k_class_ids",
        "top_k_scores",
    ):
        assert key in records[0], f"record is missing {key}"


def test_records_allow_recomputing_accuracy_without_the_model(output, label_map):
    """Aggregate-only export would make this impossible."""
    from asl_training.evaluation import top1_accuracy

    records = per_example_records(output, label_map)
    recomputed = sum(r["correct"] for r in records) / len(records)
    assert recomputed == pytest.approx(top1_accuracy(output.logits, output.labels))


def test_top_k_is_ordered_by_score(output, label_map):
    record = per_example_records(output, label_map)[0]
    scores = record["top_k_scores"]
    assert scores == sorted(scores, reverse=True)


def test_predicted_gloss_matches_the_predicted_id(output, label_map):
    for record in per_example_records(output, label_map):
        assert record["predicted_gloss"] == label_map.to_gloss(record["predicted_class_id"])


def test_temperature_changes_confidence_but_not_predictions(output, label_map):
    raw = per_example_records(output, label_map)
    scaled = per_example_records(output, label_map, temperature=2.0)

    assert [r["predicted_class_id"] for r in raw] == [r["predicted_class_id"] for r in scaled]
    assert [r["confidence"] for r in raw] != [r["confidence"] for r in scaled]


def test_predictions_write_to_csv(output, label_map, tmp_path):
    path = save_predictions(per_example_records(output, label_map), tmp_path / "p.csv")

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == output.evaluated
    assert "sample_id" in rows[0]
    assert "confidence" in rows[0]


# Logit export -----------------------------------------------------------------


def test_logits_round_trip_raw(output, tmp_path):
    path = save_logits(output, tmp_path / "logits.pt", label_map_identity="x:4:sha256:abc")
    payload = load_logits(path)

    assert torch.allclose(payload["logits"], output.logits)
    assert torch.equal(payload["labels"], output.labels)
    assert payload["raw"] is True
    assert payload["label_map_identity"] == "x:4:sha256:abc"


def test_exported_logits_are_pre_temperature(output, tmp_path):
    """Calibration cannot be recovered from scaled or normalized values."""
    path = save_logits(output, tmp_path / "logits.pt")
    payload = load_logits(path)

    assert torch.allclose(payload["logits"], output.logits)
    # Not probabilities: rows must not sum to 1.
    sums = payload["logits"].sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_loading_non_raw_logits_is_rejected(tmp_path):
    path = tmp_path / "scaled.pt"
    torch.save({"logits": torch.randn(4, 3), "raw": False}, path)

    with pytest.raises(EvaluationError, match="not marked as raw"):
        load_logits(path)


def test_loading_missing_logits_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="logits not found"):
        load_logits(tmp_path / "absent.pt")


# Reports ----------------------------------------------------------------------


def test_report_includes_sample_accounting(output, label_map):
    report = build_report(output, label_map)

    accounting = report["sample_accounting"]
    assert accounting["manifest_samples"] == 12
    assert accounting["evaluated"] == 12
    assert accounting["skipped"] == 0
    assert accounting["complete"] is True


def test_report_records_whether_calibration_was_applied(output, label_map):
    assert build_report(output, label_map)["calibration_applied"] is False

    calibrated = build_report(output, label_map, temperature=1.5)
    assert calibrated["calibration_applied"] is True
    assert calibrated["temperature"] == 1.5


def test_report_carries_identities(output, label_map):
    report = build_report(
        output,
        label_map,
        identities={"checkpoint": "best.pt", "label_map_identity": label_map.identity},
    )
    assert report["identities"]["checkpoint"] == "best.pt"
    assert report["identities"]["label_map_identity"] == label_map.identity


def test_test_mode_report_carries_a_recording_notice(model, loader, label_map):
    """Repeated test reads erode the split even without formal tuning."""
    output = evaluate(model, loader, mode="test")
    report = build_report(output, label_map)

    assert "test_evaluation_notice" in report
    assert "Record this evaluation" in report["test_evaluation_notice"]


def test_validation_report_has_no_test_notice(output, label_map):
    assert "test_evaluation_notice" not in build_report(output, label_map)


def test_report_refuses_to_overwrite(output, label_map, tmp_path):
    path = tmp_path / "report.json"
    save_report(build_report(output, label_map), path)

    with pytest.raises(FileExistsError, match="already exists"):
        save_report(build_report(output, label_map), path)


def test_report_overwrite_is_possible_when_explicit(output, label_map, tmp_path):
    path = tmp_path / "report.json"
    save_report(build_report(output, label_map), path)
    save_report(build_report(output, label_map), path, overwrite=True)
    assert path.exists()


def test_report_is_serializable(output, label_map, tmp_path):
    import json

    path = save_report(build_report(output, label_map), tmp_path / "r.json")
    payload = json.loads(path.read_text())
    assert payload["metrics"]["samples"] == 12
