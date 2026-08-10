#!/usr/bin/env python3
"""Measure what a training run will cost, before running one.

Uses the real dataset and the real configuration to report throughput, peak
memory, estimated epoch duration, and whether the data loader or the GPU is the
limiting factor. Takes a few minutes.

This is not an experiment. It writes no run directory and produces no result.

Run this before a full baseline, especially on Colab, where a wrong batch size
or worker count costs a whole session to discover.

Examples:

    python scripts/train_preflight.py \\
        --model-config configs/models/videomae_base.yaml \\
        --training-config configs/training/baseline.yaml \\
        --dataset-root "$ASL_DATASET_ROOT"

    # Compare batch sizes to find what fits
    python scripts/train_preflight.py ... --batch-size 4
    python scripts/train_preflight.py ... --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
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
from asl_training.training import CheckpointMetadata, TrainingConfig
from asl_training.training.loop import git_commit, resolve_device
from asl_training.training.preflight import run_preflight

logger = logging.getLogger("train_preflight")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ASL_DATASET_ROOT"))

    parser.add_argument(
        "--steps", type=int, default=20, help="measured optimizer steps (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the training config, to compare what fits",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-validation", action="store_true", help="skip the short validation pass"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the report as JSON, for comparing configurations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset_root is None:
        logger.error("no dataset root. Pass --dataset-root or set ASL_DATASET_ROOT.")
        return 2

    overrides: dict = {
        "experiment": "preflight",
        "run_name": "preflight",
        "run_kind": "preflight",
    }
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device

    config = TrainingConfig.from_yaml(args.training_config, **overrides)

    label_map = LabelMap.load(args.artifacts_dir / "label_maps" / "asl_citizen.json")

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

    model = build_model_from_yaml(args.model_config, num_classes=label_map.num_classes)
    frames = model.config.num_frames
    size = model.config.image_size
    resize = int(size * 256 / 224)

    train_dataset = VideoClipDataset(
        manifests["train"],
        label_map,
        args.dataset_root,
        sampler=TemporalSampler(num_frames=frames, strategy="random_segment"),
        transform=TrainTransform(crop_size=size, resize_size=resize),
        split="train",
        seed=config.seed,
    )
    loader_config = LoaderConfig(batch_size=config.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(
        train_dataset, worker_init_fn=worker_init_fn, **loader_config.for_split("train")
    )

    val_loader = None
    if not args.skip_validation:
        val_dataset = VideoClipDataset(
            manifests["validation"],
            label_map,
            args.dataset_root,
            sampler=TemporalSampler(num_frames=frames, strategy="uniform"),
            transform=EvalTransform(crop_size=size, resize_size=resize),
            split="validation",
            seed=config.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            worker_init_fn=worker_init_fn,
            **loader_config.for_split("validation"),
        )

    metadata = CheckpointMetadata(
        architecture=model.config.architecture,
        num_classes=model.num_classes,
        label_map_identity=label_map.identity,
        manifest_identity=manifests["train"].identity,
        preprocessing_identity=train_dataset.preprocessing.identity,
        fine_tuning=model.config.fine_tuning,
        optimizer_name=config.optimizer.name,
        scheduler_name=config.scheduler.name,
        git_commit=git_commit(),
    )

    logger.info(
        "Preflighting %s on %s: %d measured step(s) at batch %d x %d accumulation",
        model.config.architecture,
        resolve_device(config.device),
        args.steps,
        config.batch_size,
        config.gradient_accumulation_steps,
    )

    # The checkpoint round trip writes to a temporary directory: a preflight
    # must not leave anything that could later be mistaken for a run.
    with tempfile.TemporaryDirectory(prefix="asl-preflight-") as scratch:
        try:
            report = run_preflight(
                model,
                config,
                train_loader,
                val_loader,
                steps=args.steps,
                metadata=metadata,
                checkpoint_dir=Path(scratch) / "checkpoints",
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "out of memory at batch size %d. Lower --batch-size and raise "
                    "gradient_accumulation_steps by the same factor, so the "
                    "effective batch stays comparable across runs.",
                    config.batch_size,
                )
                return 1
            raise

    print()
    print(report.summary())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nReport written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
