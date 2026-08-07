#!/usr/bin/env python3
"""Train an ASL classifier.

Composes the model, data, and training layers into one run. The same command
serves both architectures; only the model config differs.

The test manifest is never loaded. Final test evaluation is a separate,
deliberate operation after model and threshold selection are fixed.

Examples:

    # Full baseline
    python scripts/train.py \\
        --model-config configs/models/videomae_base.yaml \\
        --dataset-config configs/datasets/asl_citizen.yaml \\
        --training-config configs/training/baseline.yaml \\
        --experiment exp-001-videomae-baseline \\
        --run-name videomae-baseline-seed42

    # Smoke run, clearly labeled as such
    python scripts/train.py ... --run-kind smoke --max-steps 20 --limit-samples 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from asl_training.data import (
    EvalTransform,
    LabelMap,
    LoaderConfig,
    Manifest,
    TemporalSampler,
    TrainTransform,
    VideoClipDataset,
    worker_init_fn,
)
from asl_training.models import build_model_from_yaml
from asl_training.training import (
    CheckpointMetadata,
    Trainer,
    TrainingConfig,
)
from asl_training.training.loop import environment_summary, git_commit, resolve_device

logger = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="where manifests and the label map live",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("ASL_DATASET_ROOT"),
        help="dataset root; defaults to $ASL_DATASET_ROOT",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("ASL_OUTPUT_ROOT", "outputs"),
        help="run directories are created beneath this; defaults to $ASL_OUTPUT_ROOT",
    )

    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-kind",
        choices=("full", "subset", "smoke", "preflight"),
        default=None,
        help="overrides the training config; a reduced run must say so",
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)

    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help=(
            "truncate the training split. Forces run_kind to 'subset' unless "
            "already reduced, because a truncated run is not a baseline."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "data-loader workers. Video decoding is CPU-bound, so this is commonly "
            "the bottleneck rather than the GPU. Use 0 to load in the main process."
        ),
    )
    parser.add_argument(
        "--allow-signer-overlap",
        action="store_true",
        help="permit signers shared across splits; requires a documented reason",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.dataset_root is None:
        logger.error("no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT.")
        return 2

    overrides = {
        "experiment": args.experiment,
        "run_name": args.run_name,
        **{
            key: value
            for key, value in (
                ("seed", args.seed),
                ("epochs", args.epochs),
                ("batch_size", args.batch_size),
                ("max_steps", args.max_steps),
                ("device", args.device),
                ("resume_from", args.resume_from),
                ("run_kind", args.run_kind),
            )
            if value is not None
        },
    }

    # A truncated run must never be reported as a full baseline.
    if args.limit_samples and overrides.get("run_kind", "full") == "full":
        logger.warning(
            "--limit-samples truncates the training split; labeling this run as "
            "'subset' rather than a full baseline"
        )
        overrides["run_kind"] = "subset"

    config = TrainingConfig.from_yaml(args.training_config, **overrides)

    label_map = LabelMap.load(args.artifacts_dir / "label_maps" / "asl_citizen.json")
    logger.info("Label map: %d classes, %s", label_map.num_classes, label_map.identity)

    manifests = {}
    for split in ("train", "validation"):
        path = args.artifacts_dir / "manifests" / f"asl_citizen_{split}.csv"
        if not path.exists():
            logger.error(
                "missing %s manifest at %s. Run scripts/audit_dataset.py --write-manifests first.",
                split,
                path,
            )
            return 1
        manifests[split] = Manifest.from_csv(path)

    # Split integrity is validated before training, not after it fails.
    combined = Manifest(records=[*manifests["train"].records, *manifests["validation"].records])
    report = combined.validate(label_map, allow_signer_overlap=args.allow_signer_overlap)
    for warning in report.warnings:
        logger.warning("manifest: %s", warning)
    report.raise_if_invalid()
    logger.info("Manifest validation passed: %s", report.counts)

    if args.limit_samples:
        manifests["train"] = Manifest(
            records=manifests["train"].records[: args.limit_samples],
            dataset_name=manifests["train"].dataset_name,
        )
        logger.warning("training split truncated to %d samples", len(manifests["train"]))

    model = build_model_from_yaml(args.model_config, num_classes=label_map.num_classes)
    frames = model.config.num_frames
    size = model.config.image_size

    datasets = {
        "train": VideoClipDataset(
            manifests["train"],
            label_map,
            args.dataset_root,
            sampler=TemporalSampler(num_frames=frames, strategy="random_segment"),
            transform=TrainTransform(crop_size=size, resize_size=int(size * 256 / 224)),
            split="train",
            seed=config.seed,
        ),
        "validation": VideoClipDataset(
            manifests["validation"],
            label_map,
            args.dataset_root,
            sampler=TemporalSampler(num_frames=frames, strategy="uniform"),
            transform=EvalTransform(crop_size=size, resize_size=int(size * 256 / 224)),
            split="validation",
            seed=config.seed,
        ),
    }

    loader_config = LoaderConfig(batch_size=config.batch_size, num_workers=args.num_workers)
    loaders = {
        split: DataLoader(
            dataset,
            worker_init_fn=worker_init_fn,
            **loader_config.for_split(split),
        )
        for split, dataset in datasets.items()
    }

    metadata = CheckpointMetadata(
        architecture=model.config.architecture,
        num_classes=model.num_classes,
        label_map_identity=label_map.identity,
        manifest_identity=combined.identity,
        preprocessing_identity=datasets["train"].preprocessing.identity,
        fine_tuning=model.config.fine_tuning,
        optimizer_name=config.optimizer.name,
        scheduler_name=config.scheduler.name,
        git_commit=git_commit(),
        environment=environment_summary(resolve_device(config.device)),
    )

    output_dir = args.output_root / config.experiment / config.run_name
    trainer = Trainer(
        model,
        config,
        loaders["train"],
        loaders["validation"],
        output_dir,
        metadata=metadata,
    )

    history = trainer.train()

    print(f"\nRun complete: {config.run_name} ({config.run_kind})")
    print(f"  output          {output_dir}")
    print(f"  epochs          {len(history)}")
    print(f"  optimizer steps {trainer.state.optimizer_step}")
    if trainer.state.best_metric is not None:
        print(
            f"  best {config.selection_metric}  {trainer.state.best_metric:.4f} "
            f"(epoch {trainer.state.best_epoch})"
        )
    if config.is_reduced:
        print(f"\n  This is a {config.run_kind} run. It is not a baseline.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
