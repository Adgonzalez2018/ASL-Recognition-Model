"""End-to-end evaluation through the real command-line entry point.

The property under test is test-set isolation: the script must refuse to fit a
temperature or select a threshold on test data, and must require the validation
artifacts instead.

See docs/EVALUATION_CONTRACT.md.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
AUDIT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_dataset.py"

FRAMES = 4
SIZE = 32
GLOSSES = ["APPLE", "BOOK", "CAT", "DOG"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    return (
        _load(SCRIPT, "evaluate_script"),
        _load(TRAIN_SCRIPT, "train_script_eval"),
        _load(AUDIT_SCRIPT, "audit_script_eval"),
    )


def write_tiny_video(path: Path, frames: int = 8):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = SIZE
        stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            array[:, :, index % 3] = 200
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())


@pytest.fixture(scope="module")
def trained(tmp_path_factory, scripts):
    """A dataset, its artifacts, and a trained checkpoint."""
    evaluate_script, train_script, audit_script = scripts

    root = tmp_path_factory.mktemp("dataset")
    artifacts = tmp_path_factory.mktemp("artifacts")
    configs = tmp_path_factory.mktemp("configs")
    outputs = tmp_path_factory.mktemp("outputs")

    counter = 0
    for split, signers in (
        ("train", ["s01", "s02"]),
        ("val", ["s03", "s04"]),
        ("test", ["s05", "s06"]),
    ):
        rows = []
        for signer in signers:
            for gloss in GLOSSES:
                counter += 1
                name = f"clip{counter:03d}.mp4"
                write_tiny_video(root / "videos" / name)
                rows.append({"Participant ID": signer, "Video file": name, "Gloss": gloss})

        with (root / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Participant ID", "Video file", "Gloss"])
            writer.writeheader()
            writer.writerows(rows)

    assert (
        audit_script.main(
            ["--dataset-root", str(root), "--output-dir", str(artifacts), "--write-manifests"]
        )
        == 0
    )

    (configs / "model.yaml").write_text(
        "model:\n"
        "  architecture: videomae_base\n"
        "  pretrained: false\n"
        f"  num_frames: {FRAMES}\n"
        f"  image_size: {SIZE}\n"
        "  options:\n"
        "    hidden_size: 48\n"
        "    num_hidden_layers: 1\n"
        "    num_attention_heads: 2\n"
        "    intermediate_size: 96\n"
    )
    (configs / "training.yaml").write_text(
        "training:\n"
        "  run_kind: smoke\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  precision: fp32\n"
        "  device: cpu\n"
        "  checkpoint_every_minutes: null\n"
        "  log_every_steps: 1000\n"
        "  optimizer:\n"
        "    lr: 0.001\n"
        "  scheduler:\n"
        "    warmup_steps: 0\n"
    )

    assert (
        train_script.main(
            [
                "--model-config",
                str(configs / "model.yaml"),
                "--training-config",
                str(configs / "training.yaml"),
                "--artifacts-dir",
                str(artifacts),
                "--dataset-root",
                str(root),
                "--output-root",
                str(outputs),
                "--experiment",
                "eval-test",
                "--run-name",
                "run-001",
                "--num-workers",
                "0",
            ]
        )
        == 0
    )

    checkpoint = outputs / "eval-test" / "run-001" / "checkpoints" / "best.pt"
    assert checkpoint.exists()
    return evaluate_script, root, artifacts, configs, checkpoint


def base_args(trained, output_dir, split="validation", **extra):
    _, root, artifacts, configs, checkpoint = trained
    args = [
        "--checkpoint",
        str(checkpoint),
        "--model-config",
        str(configs / "model.yaml"),
        "--artifacts-dir",
        str(artifacts),
        "--dataset-root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--split",
        split,
        "--num-workers",
        "0",
        "--batch-size",
        "2",
        "--signer-support-floor",
        "1",
    ]
    for key, value in extra.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args


# Validation -------------------------------------------------------------------


def test_validation_produces_every_artifact(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"

    assert evaluate_script.main(base_args(trained, output)) == 0

    for name in ("report.json", "predictions.csv", "logits.pt", "calibration.json"):
        assert (output / name).exists(), f"missing {name}"


def test_validation_report_accounts_for_every_sample(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    report = json.loads((output / "report.json").read_text())
    accounting = report["sample_accounting"]

    assert accounting["evaluated"] == accounting["manifest_samples"] == 8
    assert accounting["skipped"] == 0
    assert accounting["complete"] is True


def test_validation_fits_calibration_and_selects_a_threshold(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    payload = json.loads((output / "calibration.json").read_text())
    assert payload["calibration"]["temperature"] > 0
    assert payload["calibration"]["fit_on"] == "validation"
    assert payload["threshold_selection"]["selected_on"] == "validation"


def test_calibration_preserves_accuracy(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    calibration = json.loads((output / "calibration.json").read_text())["calibration"]
    assert calibration["before"]["accuracy"] == calibration["after"]["accuracy"]


def test_report_carries_identities(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    identities = json.loads((output / "report.json").read_text())["identities"]
    assert identities["label_map_identity"].startswith("asl_citizen:")
    assert identities["preprocessing_identity"].startswith("preprocessing:eval:")
    assert identities["architecture"] == "videomae_base"


def test_predictions_are_per_example(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    with (output / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert {"sample_id", "signer_id", "confidence", "correct"} <= set(rows[0])


# Test-set isolation -----------------------------------------------------------


def test_test_split_without_calibration_is_refused(trained, tmp_path):
    """Fitting a temperature on test would make the operating point a fiction."""
    evaluate_script = trained[0]
    code = evaluate_script.main(
        base_args(trained, tmp_path / "test", split="test", reason="trying it on")
    )
    assert code == 2


def test_test_split_without_a_reason_is_refused(trained, tmp_path):
    """Every test read is recorded."""
    evaluate_script = trained[0]
    validation = tmp_path / "validation"
    evaluate_script.main(base_args(trained, validation))

    code = evaluate_script.main(
        base_args(
            trained,
            tmp_path / "test",
            split="test",
            calibration=validation / "calibration.json",
        )
    )
    assert code == 2


def test_test_split_applies_validation_artifacts(trained, tmp_path):
    evaluate_script = trained[0]
    validation = tmp_path / "validation"
    evaluate_script.main(base_args(trained, validation))

    test_output = tmp_path / "test"
    assert (
        evaluate_script.main(
            base_args(
                trained,
                test_output,
                split="test",
                calibration=validation / "calibration.json",
                reason="final reporting",
            )
        )
        == 0
    )

    report = json.loads((test_output / "report.json").read_text())
    record = report["test_read_record"]

    assert record["reason"] == "final reporting"
    assert record["temperature_refit"] is False
    assert record["threshold_reselected"] is False
    assert record["date"]


def test_test_run_uses_the_validation_temperature_unchanged(trained, tmp_path):
    evaluate_script = trained[0]
    validation = tmp_path / "validation"
    evaluate_script.main(base_args(trained, validation))

    fitted = json.loads((validation / "calibration.json").read_text())
    temperature = fitted["calibration"]["temperature"]

    test_output = tmp_path / "test"
    evaluate_script.main(
        base_args(
            trained,
            test_output,
            split="test",
            calibration=validation / "calibration.json",
            reason="final reporting",
        )
    )

    report = json.loads((test_output / "report.json").read_text())
    assert report["temperature"] == pytest.approx(temperature)


def test_test_report_shows_both_operating_points(trained, tmp_path):
    """Validation behaviour must not be presented as expected test behaviour."""
    evaluate_script = trained[0]
    validation = tmp_path / "validation"
    evaluate_script.main(base_args(trained, validation))

    test_output = tmp_path / "test"
    evaluate_script.main(
        base_args(
            trained,
            test_output,
            split="test",
            calibration=validation / "calibration.json",
            reason="final reporting",
        )
    )

    selective = json.loads((test_output / "report.json").read_text())["selective_prediction"]
    applied = selective["applied_from_validation"]

    if applied["applied"] is not None:
        assert applied["validation"] is not None
        assert "selective_accuracy_shift" in applied


def test_test_report_carries_the_recording_notice(trained, tmp_path):
    evaluate_script = trained[0]
    validation = tmp_path / "validation"
    evaluate_script.main(base_args(trained, validation))

    test_output = tmp_path / "test"
    evaluate_script.main(
        base_args(
            trained,
            test_output,
            split="test",
            calibration=validation / "calibration.json",
            reason="final reporting",
        )
    )

    report = json.loads((test_output / "report.json").read_text())
    assert "test_evaluation_notice" in report


# Output safety ----------------------------------------------------------------


def test_existing_report_is_not_silently_overwritten(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_script.main(base_args(trained, output))


def test_overwrite_is_possible_when_explicit(trained, tmp_path):
    evaluate_script = trained[0]
    output = tmp_path / "validation"
    evaluate_script.main(base_args(trained, output))

    args = base_args(trained, output)
    args.append("--overwrite")
    assert evaluate_script.main(args) == 0
