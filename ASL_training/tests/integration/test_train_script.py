"""End-to-end training through the real command-line entry point.

Exercises the whole path: annotations to manifests to decoded video to a
training run with checkpoint and resume. This is the "one complete end-to-end
path" the phase calls for, using the actual script rather than a reimplementation
of it.

See docs/TRAINING_CONTRACT.md.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
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
def train_script():
    return _load(SCRIPT, "train_script")


@pytest.fixture(scope="module")
def audit_script():
    return _load(AUDIT_SCRIPT, "audit_script_integration")


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
def prepared(tmp_path_factory, audit_script):
    """A tiny dataset plus the manifests and label map the audit produces."""
    root = tmp_path_factory.mktemp("dataset")
    artifacts = tmp_path_factory.mktemp("artifacts")

    counter = 0
    for split, signers in (
        ("train", ["s01", "s02"]),
        ("val", ["s03"]),
        ("test", ["s04"]),
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

    # Manifests come from the real audit path, not hand-built fixtures.
    code = audit_script.main(
        [
            "--dataset-root",
            str(root),
            "--output-dir",
            str(artifacts),
            "--write-manifests",
        ]
    )
    assert code == 0
    return root, artifacts


@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("configs")

    (directory / "model.yaml").write_text(
        "model:\n"
        "  architecture: videomae_base\n"
        "  pretrained: false\n"
        f"  num_frames: {FRAMES}\n"
        f"  image_size: {SIZE}\n"
        "  fine_tuning: full\n"
        "  options:\n"
        "    hidden_size: 48\n"
        "    num_hidden_layers: 1\n"
        "    num_attention_heads: 2\n"
        "    intermediate_size: 96\n"
    )
    (directory / "training.yaml").write_text(
        "training:\n"
        "  run_kind: smoke\n"
        "  seed: 42\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  precision: fp32\n"
        "  device: cpu\n"
        "  checkpoint_every_minutes: null\n"
        "  log_every_steps: 1000\n"
        "  optimizer:\n"
        "    name: adamw\n"
        "    lr: 0.001\n"
        "  scheduler:\n"
        "    name: cosine\n"
        "    warmup_steps: 0\n"
    )
    return directory


def base_args(prepared, configs, output, **extra):
    root, artifacts = prepared
    args = [
        "--model-config",
        str(configs / "model.yaml"),
        "--training-config",
        str(configs / "training.yaml"),
        "--artifacts-dir",
        str(artifacts),
        "--dataset-root",
        str(root),
        "--output-root",
        str(output),
        "--experiment",
        "integration",
        "--run-name",
        "smoke-001",
        # In-process loading. Spawned workers deadlock under pytest on macOS,
        # and worker behaviour is not what these tests are exercising.
        "--num-workers",
        "0",
    ]
    for key, value in extra.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args


# End to end -------------------------------------------------------------------


def test_full_path_from_annotations_to_a_trained_checkpoint(
    train_script, prepared, configs, tmp_path
):
    output = tmp_path / "outputs"
    assert train_script.main(base_args(prepared, configs, output)) == 0

    run = output / "integration" / "smoke-001"
    assert (run / "run_metadata.json").exists()
    assert (run / "history.json").exists()
    assert (run / "checkpoints" / "latest.pt").exists()
    assert (run / "checkpoints" / "best.pt").exists()


def test_run_metadata_records_every_identity(train_script, prepared, configs, tmp_path):
    """A run without these is not reproducible."""
    output = tmp_path / "outputs"
    train_script.main(base_args(prepared, configs, output))

    payload = json.loads((output / "integration" / "smoke-001" / "run_metadata.json").read_text())
    metadata = payload["metadata"]

    assert metadata["architecture"] == "videomae_base"
    assert metadata["num_classes"] == len(GLOSSES)
    assert metadata["label_map_identity"].startswith("asl_citizen:")
    assert metadata["manifest_identity"].startswith("asl_citizen:")
    assert metadata["preprocessing_identity"].startswith("preprocessing:")
    assert payload["environment"]["torch"]


def test_smoke_run_is_labeled_not_a_baseline(train_script, prepared, configs, tmp_path):
    output = tmp_path / "outputs"
    train_script.main(base_args(prepared, configs, output))

    payload = json.loads((output / "integration" / "smoke-001" / "run_metadata.json").read_text())
    assert payload["run_kind"] == "smoke"
    assert payload["is_reduced"] is True


def test_truncating_the_split_forces_a_subset_label(train_script, prepared, configs, tmp_path):
    """A truncated run must never be reported as a full baseline."""
    output = tmp_path / "outputs"
    args = base_args(prepared, configs, output, limit_samples=4)
    args.extend(["--run-kind", "full"])  # deliberately claim a full run

    assert train_script.main(args) == 0

    payload = json.loads((output / "integration" / "smoke-001" / "run_metadata.json").read_text())
    assert payload["run_kind"] == "subset"
    assert payload["is_reduced"] is True


def test_resumes_across_invocations(train_script, prepared, configs, tmp_path):
    """Two separate invocations, as two Colab sessions would be."""
    output = tmp_path / "outputs"

    assert train_script.main(base_args(prepared, configs, output, epochs=1)) == 0
    first = json.loads((output / "integration" / "smoke-001" / "history.json").read_text())
    assert len(first) == 1

    assert train_script.main(base_args(prepared, configs, output, epochs=3)) == 0
    second = json.loads((output / "integration" / "smoke-001" / "history.json").read_text())
    assert [entry["epoch"] for entry in second] == [1, 2]


def test_missing_dataset_root_exits_nonzero(train_script, prepared, configs, tmp_path):
    args = base_args(prepared, configs, tmp_path / "out")
    index = args.index("--dataset-root")
    args[index + 1] = ""
    args.pop(index)
    args.pop(index)
    assert train_script.main(args) == 2


def test_missing_manifests_exit_nonzero(train_script, prepared, configs, tmp_path):
    root, _ = prepared
    empty = tmp_path / "no-artifacts"
    (empty / "label_maps").mkdir(parents=True)

    from asl_training.data import LabelMap

    LabelMap.from_glosses(GLOSSES).save(empty / "label_maps" / "asl_citizen.json")

    args = [
        "--model-config",
        str(configs / "model.yaml"),
        "--training-config",
        str(configs / "training.yaml"),
        "--artifacts-dir",
        str(empty),
        "--dataset-root",
        str(root),
        "--output-root",
        str(tmp_path / "out"),
        "--experiment",
        "integration",
        "--run-name",
        "missing-manifests",
        "--num-workers",
        "0",
    ]
    assert train_script.main(args) == 1


def test_the_test_split_is_never_loaded(train_script, prepared, configs, tmp_path, monkeypatch):
    """The default training command must not touch the test manifest.

    Spies on manifest loading specifically rather than on file access generally,
    since the run legitimately opens many other files.
    """
    from asl_training.data import Manifest

    _, artifacts = prepared
    assert (artifacts / "manifests" / "asl_citizen_test.csv").exists()

    loaded: list[str] = []
    original = Manifest.from_csv.__func__

    def spy(cls, path, **kwargs):
        loaded.append(str(path))
        return original(cls, path, **kwargs)

    monkeypatch.setattr(Manifest, "from_csv", classmethod(spy))
    train_script.main(base_args(prepared, configs, tmp_path / "outputs"))

    assert loaded, "no manifests were loaded at all"
    assert not any("asl_citizen_test" in path for path in loaded), (
        f"the training command read the test manifest: {loaded}"
    )


def test_signer_leakage_blocks_training(train_script, prepared, configs, tmp_path):
    """Split integrity is validated before training, not after it fails."""
    import shutil

    from asl_training.data import Manifest
    from asl_training.data.manifest import ManifestValidationError

    root, artifacts = prepared
    leaky = tmp_path / "leaky-artifacts"
    shutil.copytree(artifacts, leaky)

    train = Manifest.from_csv(leaky / "manifests" / "asl_citizen_train.csv")
    validation = Manifest.from_csv(leaky / "manifests" / "asl_citizen_validation.csv")

    from dataclasses import replace

    leaked = replace(
        validation.records[0],
        sample_id="asl_citizen:leaked",
        signer_id=train.records[0].signer_id,
    )
    Manifest(records=[*validation.records, leaked]).to_csv(
        leaky / "manifests" / "asl_citizen_validation.csv"
    )

    args = [
        "--model-config",
        str(configs / "model.yaml"),
        "--training-config",
        str(configs / "training.yaml"),
        "--artifacts-dir",
        str(leaky),
        "--dataset-root",
        str(root),
        "--output-root",
        str(tmp_path / "out"),
        "--experiment",
        "integration",
        "--run-name",
        "leaky",
        "--num-workers",
        "0",
    ]
    with pytest.raises(ManifestValidationError, match="signer"):
        train_script.main(args)
