"""Evaluation execution and export.

Runs a checkpoint over a manifest without touching weights, and persists enough
per-example detail that metrics, calibration, and threshold analysis can all be
recomputed without running the model again.

Aggregate-only evaluation is insufficient: it cannot be re-binned, refit, or
traced back to individual failures.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.label_map import LabelMap
from ..models.base import BaseVideoClassifier
from .calibration import apply_temperature, max_softmax_confidence
from .metrics import classification_report, predicted_classes, top_k_classes

logger = logging.getLogger(__name__)

# Evaluation modes. They differ in what may influence what, so they are not
# interchangeable.
MODES = ("validation", "test", "robustness")


class EvaluationError(Exception):
    """Raised when an evaluation cannot be trusted."""


@dataclass
class EvaluationOutput:
    """Raw evaluation results, before aggregation.

    Logits are kept raw, before any temperature. Temperature scaling cannot be
    recovered from normalized probabilities, so storing softmax output instead
    would make calibration impossible after the fact.
    """

    logits: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]
    signer_ids: list[str]
    glosses: list[str]
    manifest_samples: int
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = {
            len(self.labels),
            len(self.sample_ids),
            len(self.signer_ids),
            len(self.glosses),
            self.logits.shape[0],
        }
        if len(counts) != 1:
            raise EvaluationError(
                f"evaluation output lengths disagree: logits={self.logits.shape[0]}, "
                f"labels={len(self.labels)}, sample_ids={len(self.sample_ids)}, "
                f"signer_ids={len(self.signer_ids)}, glosses={len(self.glosses)}"
            )

    @property
    def evaluated(self) -> int:
        return len(self.labels)

    @property
    def complete(self) -> bool:
        return self.evaluated == self.manifest_samples


@torch.no_grad()
def evaluate(
    model: BaseVideoClassifier,
    loader: DataLoader,
    *,
    mode: str = "validation",
    device: torch.device | str = "cpu",
    manifest_samples: int | None = None,
) -> EvaluationOutput:
    """Run a model over a loader, collecting raw logits.

    Weights are never modified. The model is placed in evaluation mode and its
    prior mode is restored, so calling this mid-training is safe.

    Raises:
        EvaluationError: If the evaluated sample count differs from the
            manifest, which would mean the reported metrics cover a dataset
            nobody defined.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; supported: {', '.join(MODES)}")

    device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(device)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    signer_ids: list[str] = []
    glosses: list[str] = []

    try:
        for batch in loader:
            logits = model(batch["pixel_values"].to(device)).logits

            all_logits.append(logits.float().cpu())
            all_labels.append(batch["labels"].cpu())
            sample_ids.extend(batch["sample_ids"])
            signer_ids.extend(batch["signer_ids"])
            glosses.extend(batch["glosses"])
    finally:
        if was_training:
            model.train()

    if not all_logits:
        raise EvaluationError("evaluation produced no samples; the loader was empty")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    expected = (
        manifest_samples if manifest_samples is not None else len(loader.dataset)  # type: ignore[arg-type]
    )
    if len(labels) != expected:
        raise EvaluationError(
            f"evaluated {len(labels)} sample(s) but the manifest has {expected}. "
            f"Silent sample loss during evaluation is prohibited; check for "
            f"drop_last on the loader or a skip policy on the dataset."
        )

    if not torch.isfinite(logits).all():
        raise EvaluationError("evaluation produced non-finite logits")

    return EvaluationOutput(
        logits=logits,
        labels=labels,
        sample_ids=sample_ids,
        signer_ids=signer_ids,
        glosses=glosses,
        manifest_samples=expected,
        mode=mode,
    )


def per_example_records(
    output: EvaluationOutput,
    label_map: LabelMap,
    *,
    temperature: float | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """One record per evaluated sample.

    Required so per-class and per-signer metrics, calibration, and threshold
    analysis can all be recomputed without rerunning the model, and so an
    individual failure can be traced back to its video.
    """
    scored = (
        apply_temperature(output.logits, temperature) if temperature is not None else output.logits
    )

    predictions = predicted_classes(scored)
    confidence = max_softmax_confidence(scored)
    k = min(top_k, scored.shape[1])
    top_classes = top_k_classes(scored, k)
    probabilities = torch.softmax(scored.double(), dim=1)

    records = []
    for index in range(output.evaluated):
        predicted_id = int(predictions[index])
        true_id = int(output.labels[index])
        top_ids = top_classes[index].tolist()

        records.append(
            {
                "sample_id": output.sample_ids[index],
                "signer_id": output.signer_ids[index],
                "gloss": output.glosses[index],
                "true_class_id": true_id,
                "predicted_class_id": predicted_id,
                "predicted_gloss": label_map.to_gloss(predicted_id),
                "correct": predicted_id == true_id,
                "confidence": round(float(confidence[index]), 6),
                "top_k_class_ids": top_ids,
                "top_k_glosses": [label_map.to_gloss(i) for i in top_ids],
                "top_k_scores": [round(float(probabilities[index, i]), 6) for i in top_ids],
                "split": output.mode,
            }
        )
    return records


def save_logits(output: EvaluationOutput, path: str | Path, **identities: Any) -> Path:
    """Persist raw logits for later calibration.

    Raw, pre-temperature. Calibration operates on logits and cannot be recovered
    from stored probabilities.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "logits": output.logits,
            "labels": output.labels,
            "sample_ids": output.sample_ids,
            "signer_ids": output.signer_ids,
            "mode": output.mode,
            "raw": True,
            "note": "Pre-temperature logits. Do not overwrite with scaled values.",
            **identities,
        },
        path,
    )
    return path


def load_logits(path: str | Path) -> dict[str, Any]:
    """Read exported logits, rejecting anything not stored raw."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"logits not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("raw", False):
        raise EvaluationError(
            f"{path} is not marked as raw logits. Calibration requires pre-temperature logits."
        )
    return payload


def save_predictions(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write per-example records as CSV for review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise EvaluationError("no records to write")

    columns = [
        "sample_id",
        "signer_id",
        "gloss",
        "true_class_id",
        "predicted_class_id",
        "predicted_gloss",
        "correct",
        "confidence",
        "split",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})
    return path


def build_report(
    output: EvaluationOutput,
    label_map: LabelMap,
    *,
    temperature: float | None = None,
    signer_support_floor: int = 10,
    identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the clean evaluation report for one split."""
    scored = (
        apply_temperature(output.logits, temperature) if temperature is not None else output.logits
    )

    report = {
        "mode": output.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_accounting": {
            "manifest_samples": output.manifest_samples,
            "evaluated": output.evaluated,
            "skipped": output.manifest_samples - output.evaluated,
            "complete": output.complete,
        },
        "calibration_applied": temperature is not None,
        "temperature": temperature,
        "metrics": classification_report(
            scored,
            output.labels,
            label_map.num_classes,
            glosses=list(label_map.glosses),
            signer_ids=output.signer_ids,
            signer_support_floor=signer_support_floor,
        ),
        "identities": identities or {},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }

    if output.mode == "test":
        report["test_evaluation_notice"] = (
            "The test split was read. Record this evaluation: the checkpoint, the "
            "calibration applied, the threshold applied, the date, and the reason. "
            "Repeated test reads erode the split's value even without formal tuning."
        )
    return report


def save_report(report: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Write a report, refusing to silently replace an existing one."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Existing evaluation output must not be "
            f"silently overwritten; write to a new directory or pass overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
