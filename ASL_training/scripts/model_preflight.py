#!/usr/bin/env python3
"""Model-layer preflight.

Verifies that an architecture constructs, loads pretrained weights, accepts the
canonical input, and returns correctly shaped logits. docs/MODEL_CONTRACT.md
requires this to pass before any real training run; a failure blocks the run.

This is a structural check on synthetic tensors. It says nothing about model
quality and is not an experiment.

Examples:

    python scripts/model_preflight.py \\
        --config configs/models/videomae_base.yaml --num-classes 2731

    python scripts/model_preflight.py \\
        --config configs/models/video_swin_tiny.yaml \\
        --num-classes 100 --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Support running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_training.models import (
    ModelConfig,
    available_architectures,
    build_model,
)

logger = logging.getLogger("model_preflight")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="path to a model YAML config")
    source.add_argument(
        "--architecture",
        choices=available_architectures(),
        help="architecture name, instead of a config file",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="output class count; must equal the label-map size",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="skip the pretrained download for a fast offline structural check",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device to run on; 'cuda' verifies device movement",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser.parse_args(argv)


def run_preflight(
    config: ModelConfig,
    batch_size: int,
    device: str,
) -> dict:
    """Construct the model and exercise the full forward and backward path."""
    report: dict = {"config": config.to_dict(), "device": device, "checks": {}}
    checks = report["checks"]

    model = build_model(config)
    checks["construction"] = "ok"
    report["pretrained_load"] = getattr(model, "pretrained_load_report", None)

    params = model.parameter_report()
    report["parameters"] = params.to_dict()
    if params.trainable == 0:
        raise RuntimeError("no trainable parameters")
    checks["trainable_parameters"] = "ok"

    report["preprocessing"] = model.preprocessing().to_dict()

    torch_device = torch.device(device)
    model.to(torch_device)
    checks["device_movement"] = "ok"

    batch = torch.randn(
        batch_size,
        config.num_frames,
        3,
        config.image_size,
        config.image_size,
        device=torch_device,
    )
    labels = torch.randint(0, config.num_classes, (batch_size,), device=torch_device)

    model.eval()
    with torch.no_grad():
        eval_out = model(batch)
    expected = (batch_size, config.num_classes)
    actual = tuple(eval_out.logits.shape)
    if actual != expected:
        raise RuntimeError(f"expected logits {expected}, got {actual}")
    checks["forward_shape"] = "ok"
    report["logits_shape"] = list(actual)

    if eval_out.loss is not None:
        raise RuntimeError("loss returned without labels")
    checks["no_loss_without_labels"] = "ok"

    if not torch.isfinite(eval_out.logits).all():
        raise RuntimeError("forward pass produced non-finite logits")
    checks["finite_logits"] = "ok"

    model.train()
    train_out = model(batch, labels=labels)
    if train_out.loss is None:
        raise RuntimeError("no loss returned when labels were supplied")
    if not torch.isfinite(train_out.loss):
        raise RuntimeError("non-finite loss")
    report["initial_loss"] = float(train_out.loss.detach())
    checks["loss_computation"] = "ok"

    # A randomly initialized head over N classes should start near ln(N).
    # A wildly different value suggests a head or label misconfiguration.
    import math

    report["expected_initial_loss"] = round(math.log(config.num_classes), 4)

    train_out.loss.backward()

    # Compare by identity: head parameters are often named bare "weight"/"bias",
    # so name matching would misattribute backbone gradients to the head.
    head_ids = {id(p) for p in model.classification_head().parameters()}
    with_grad = [
        p
        for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    ]
    if not with_grad:
        raise RuntimeError("backward pass produced no gradients")
    checks["backward_pass"] = "ok"
    report["parameters_with_gradients"] = len(with_grad)

    if not any(id(p) in head_ids for p in with_grad):
        raise RuntimeError("classification head received no gradient")
    checks["head_trainable"] = "ok"

    if config.fine_tuning == "full" and not any(id(p) not in head_ids for p in with_grad):
        raise RuntimeError("backbone received no gradient under full fine-tuning")
    checks["backbone_trainable"] = "ok"

    if torch_device.type == "cuda":
        report["peak_memory_mb"] = round(
            torch.cuda.max_memory_allocated(torch_device) / (1024**2), 2
        )
        report["gpu"] = torch.cuda.get_device_name(torch_device)

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Configuration errors are preflight failures too, and must produce a
    # non-zero exit so a caller cannot proceed to a real run.
    try:
        overrides = {"num_classes": args.num_classes}
        if args.no_pretrained:
            overrides["pretrained"] = False

        if args.config:
            config = ModelConfig.from_yaml(args.config, **overrides)
        else:
            config = ModelConfig(architecture=args.architecture, **overrides)

        report = run_preflight(config, args.batch_size, args.device)
    except Exception as exc:
        logger.error("PREFLIGHT FAILED: %s", exc)
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1

    report["status"] = "passed"

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        params = report["parameters"]
        print(f"\nPreflight passed: {config.architecture}")
        print(f"  classes        {config.num_classes}")
        print(
            f"  input          [{args.batch_size}, {config.num_frames}, 3, "
            f"{config.image_size}, {config.image_size}]"
        )
        print(f"  logits         {report['logits_shape']}")
        print(f"  total params   {params['total']:,} ({params['approx_fp32_mb']} MB fp32)")
        print(f"  trainable      {params['trainable']:,}")
        print(f"  frozen         {params['frozen']:,}")
        print(f"  head params    {params['head']:,}")
        print(
            f"  initial loss   {report['initial_loss']:.4f} "
            f"(expected ~{report['expected_initial_loss']} for a random head)"
        )
        if "peak_memory_mb" in report:
            print(f"  peak memory    {report['peak_memory_mb']} MB on {report['gpu']}")
        print("\nThis is a structural check on synthetic data, not an experiment.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
