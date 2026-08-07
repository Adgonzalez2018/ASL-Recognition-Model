#!/usr/bin/env python3
"""Evaluate a checkpoint.

Runs a saved model over a split, exports per-example predictions and raw logits,
and produces classification, calibration, and selective-prediction reports.

Test-set isolation is enforced here rather than left to discipline:

* validation mode fits the temperature and selects the threshold
* test mode refuses to fit either, and requires the validation artifacts

Examples:

    # Validation: fit calibration and choose a threshold
    python scripts/evaluate.py --checkpoint outputs/exp-001/run/checkpoints/best.pt \\
        --split validation --output-dir outputs/exp-001/run/evaluation/validation

    # Test: apply what validation chose, once
    python scripts/evaluate.py --checkpoint outputs/exp-001/run/checkpoints/best.pt \\
        --split test --output-dir outputs/exp-001/run/evaluation/test \\
        --calibration outputs/exp-001/run/evaluation/validation/calibration.json \\
        --reason "final reporting for exp-001"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from asl_training.data import (
    EvalTransform,
    LabelMap,
    LoaderConfig,
    Manifest,
    TemporalSampler,
    VideoClipDataset,
)
from asl_training.evaluation import (
    apply_threshold,
    build_report,
    evaluate,
    fit_temperature,
    per_example_records,
    save_logits,
    save_predictions,
    save_report,
    select_threshold,
    selective_report,
)
from asl_training.evaluation.calibration import apply_temperature
from asl_training.evaluation.selective import ThresholdSelection
from asl_training.models import build_model_from_yaml, load_checkpoint_state
from asl_training.training.loop import resolve_device

logger = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="test requires --calibration and --reason",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ASL_DATASET_ROOT"))
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "calibration.json from a validation run. Required for test, where "
            "fitting is prohibited."
        ),
    )
    parser.add_argument(
        "--reason",
        default=None,
        help="why the test split is being read; recorded in the report",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--signer-support-floor", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.dataset_root is None:
        logger.error("no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT.")
        return 2

    # Test-set isolation, enforced rather than trusted.
    if args.split == "test":
        if args.calibration is None:
            logger.error(
                "test evaluation requires --calibration from a validation run. "
                "Fitting temperature or selecting a threshold on test would make "
                "the reported operating point meaningless."
            )
            return 2
        if not args.reason:
            logger.error(
                "test evaluation requires --reason. Every test read is recorded, "
                "because repeated reads erode the split even without formal tuning."
            )
            return 2

    label_map = LabelMap.load(args.artifacts_dir / "label_maps" / "asl_citizen.json")

    manifest_path = args.artifacts_dir / "manifests" / f"asl_citizen_{args.split}.csv"
    if not manifest_path.exists():
        logger.error("missing %s manifest at %s", args.split, manifest_path)
        return 1
    manifest = Manifest.from_csv(manifest_path)

    model = build_model_from_yaml(args.model_config, num_classes=label_map.num_classes)
    load_report = load_checkpoint_state(model, args.checkpoint)
    logger.info("Loaded checkpoint %s", args.checkpoint)

    frames = model.config.num_frames
    size = model.config.image_size

    dataset = VideoClipDataset(
        manifest,
        label_map,
        args.dataset_root,
        sampler=TemporalSampler(num_frames=frames, strategy="uniform"),
        transform=EvalTransform(crop_size=size, resize_size=int(size * 256 / 224)),
        split=args.split,
    )
    loader_config = LoaderConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    loader = DataLoader(dataset, **loader_config.for_split(args.split))

    device = resolve_device(args.device)
    output = evaluate(model, loader, mode=args.split, device=device)
    logger.info("Evaluated %d sample(s) on %s", output.evaluated, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Only the identity fields. The full checkpoint payload carries optimizer,
    # scheduler, and RNG state, which are tensors and belong nowhere near a
    # JSON report.
    recorded = load_report["checkpoint_metadata"].get("metadata", {})
    training_state = load_report["checkpoint_metadata"].get("training_state", {})

    identities = {
        "checkpoint": str(args.checkpoint),
        "label_map_identity": label_map.identity,
        "manifest_identity": manifest.identity,
        "preprocessing_identity": dataset.preprocessing.identity,
        "architecture": model.config.architecture,
        "checkpoint_recorded": {
            key: recorded.get(key)
            for key in (
                "architecture",
                "num_classes",
                "label_map_identity",
                "manifest_identity",
                "preprocessing_identity",
                "fine_tuning",
                "git_commit",
            )
        },
        "checkpoint_epoch": training_state.get("epoch"),
        "checkpoint_best_metric": training_state.get("best_metric"),
    }

    _verify_checkpoint_identities(identities)

    # Raw logits, before any temperature. Calibration cannot be recovered from
    # scaled values.
    save_logits(output, args.output_dir / "logits.pt", **identities)

    if args.split == "validation":
        calibration = fit_temperature(output.logits, output.labels)
        temperature = calibration.temperature
        logger.info(
            "Fitted temperature %.4f: NLL %.4f -> %.4f, ECE %.4f -> %.4f",
            temperature,
            calibration.nll_before,
            calibration.nll_after,
            calibration.ece_before,
            calibration.ece_after,
        )

        calibrated = apply_temperature(output.logits, temperature)
        selection = select_threshold(calibrated, output.labels)

        (args.output_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "calibration": calibration.to_dict(),
                    "threshold_selection": selection.to_dict(),
                    "identities": identities,
                },
                indent=2,
            )
            + "\n"
        )
        selective = selective_report(calibrated, output.labels)
        applied = None
    else:
        payload = json.loads(args.calibration.read_text())
        temperature = payload["calibration"]["temperature"]
        recorded = payload["threshold_selection"]

        _verify_calibration_matches(payload.get("identities", {}), identities)

        calibrated = apply_temperature(output.logits, temperature)
        selection = ThresholdSelection(
            threshold=recorded["threshold"],
            rule=recorded["rule"],
            target_selective_accuracy=recorded["target_selective_accuracy"],
            target_coverage=recorded["target_coverage"],
            achieved=None,
            satisfied=recorded["satisfied"],
        )
        # Reattach the validation operating point so both are reported together.
        if recorded.get("achieved"):
            from asl_training.evaluation.selective import OperatingPoint

            achieved = recorded["achieved"]
            selection.achieved = OperatingPoint(
                threshold=achieved["threshold"],
                accepted=achieved["accepted"],
                rejected=achieved["rejected"],
                total=achieved["accepted"] + achieved["rejected"],
                coverage=achieved["coverage"],
                selective_accuracy=achieved["selective_accuracy"],
                rejected_but_correct=achieved["rejected_but_correct"],
            )

        applied = apply_threshold(calibrated, output.labels, selection)
        selective = {"applied_from_validation": applied}
        logger.info("Applied validation threshold %s to test", selection.threshold)

    records = per_example_records(output, label_map, temperature=temperature)
    save_predictions(records, args.output_dir / "predictions.csv")

    report = build_report(
        output,
        label_map,
        temperature=temperature,
        signer_support_floor=args.signer_support_floor,
        identities=identities,
    )
    report["selective_prediction"] = selective
    if args.split == "test":
        report["test_read_record"] = {
            "reason": args.reason,
            "date": datetime.now(timezone.utc).isoformat(),
            "calibration_source": str(args.calibration),
            "temperature_refit": False,
            "threshold_reselected": False,
        }

    save_report(report, args.output_dir / "report.json", overwrite=args.overwrite)

    _print_summary(report, args.split, temperature, selection, applied)
    return 0


def _verify_checkpoint_identities(identities: dict) -> None:
    """Fail when the checkpoint's vocabulary is not the one being evaluated.

    A label-map mismatch means every class ID means something different from what
    the report will claim, and the metrics would look perfectly normal.

    Preprocessing identity is reported rather than enforced: a checkpoint records
    the *training* preprocessing, and evaluation legitimately uses the
    deterministic evaluation pipeline instead. The two are expected to differ.
    """
    recorded = identities.get("checkpoint_recorded", {})

    was = recorded.get("label_map_identity")
    now = identities.get("label_map_identity")
    if was and now and was != now:
        raise SystemExit(
            f"label-map mismatch: the checkpoint was trained against {was!r} but "
            f"this evaluation uses {now!r}. Every class ID would mean something "
            f"different from what the report claims."
        )

    architecture = recorded.get("architecture")
    if architecture and architecture != identities.get("architecture"):
        raise SystemExit(
            f"architecture mismatch: checkpoint {architecture!r}, evaluation "
            f"{identities.get('architecture')!r}"
        )


def _verify_calibration_matches(recorded: dict, current: dict) -> None:
    """Warn when calibration came from a different model or vocabulary."""
    for key in ("label_map_identity", "preprocessing_identity", "architecture"):
        was, now = recorded.get(key), current.get(key)
        if was and now and was != now:
            logger.warning(
                "calibration was fit under a different %s (%s) than this evaluation "
                "(%s); the applied temperature may not be valid",
                key,
                was,
                now,
            )


def _print_summary(report, split, temperature, selection, applied) -> None:
    metrics = report["metrics"]
    print(f"\nEvaluation: {split}")
    print(f"  samples          {report['sample_accounting']['evaluated']}")
    print(f"  top-1            {metrics['top1_accuracy']:.4f}")
    if metrics["top5_accuracy"] is not None:
        print(f"  top-5            {metrics['top5_accuracy']:.4f}")
    print(
        f"  macro F1         {metrics['macro_f1']:.4f} "
        f"over {metrics['classes_in_macro_average']} classes"
    )
    print(f"  mean class acc   {metrics['mean_per_class_accuracy']:.4f}")
    print(f"  NLL              {metrics['negative_log_likelihood']:.4f}")

    signer = metrics.get("per_signer")
    if signer and signer.get("worst"):
        print(
            f"  worst signer     {signer['worst']['accuracy']:.4f} "
            f"({signer['worst']['signer_id']}, n={signer['worst']['samples']})"
        )
        print(f"  signer spread    {signer['mean_accuracy']:.4f} +/- {signer['std_accuracy']:.4f}")

    print(f"  temperature      {temperature:.4f}")

    if selection and selection.threshold is not None:
        point = applied["applied"] if applied else selection.achieved.to_dict()
        print(f"  threshold        {selection.threshold:.2f}")
        print(f"  coverage         {point['coverage']:.4f}")
        print(f"  selective acc    {point['selective_accuracy']:.4f}")
        if not selection.satisfied:
            print("\n  The coverage target was not met at this operating point.")
    else:
        print("  threshold        none reached the accuracy target")

    if split == "test":
        print("\n  Test split read. Recorded in report.json.")


if __name__ == "__main__":
    raise SystemExit(main())
