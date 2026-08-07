"""Dataset, collation, and the preprocessing identity.

The end of the data layer: manifest records in, canonical model batches out.
Verified against both model adapters, since a batch the model rejects is not a
working pipeline.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import DataLoader

from asl_training.data.asl_citizen import parse_annotations
from asl_training.data.dataset import (
    LoaderConfig,
    PreprocessingSpec,
    VideoClipDataset,
    collate_clips,
    worker_init_fn,
)
from asl_training.data.decode import VideoDecodeError
from asl_training.data.manifest import Manifest
from asl_training.data.sampling import TemporalSampler
from asl_training.data.transforms import EvalTransform, TrainTransform

CROP = 64  # small, so tests stay fast
FRAMES = 8


def make_dataset(root, split="train", **kwargs):
    parsed = parse_annotations(root)
    records = parsed.manifest.for_split(split)
    defaults = {
        "sampler": TemporalSampler(
            num_frames=FRAMES,
            strategy="random_segment" if split == "train" else "uniform",
        ),
        "transform": (
            TrainTransform(crop_size=CROP, resize_size=CROP + 16)
            if split == "train"
            else EvalTransform(crop_size=CROP, resize_size=CROP + 16)
        ),
    }
    defaults.update(kwargs)

    return VideoClipDataset(
        Manifest(records=records),
        parsed.label_map,
        root,
        split=split,
        **defaults,
    )


# Sample structure -------------------------------------------------------------


def test_produces_the_canonical_clip_shape(synthetic_root):
    dataset = make_dataset(synthetic_root)
    sample = dataset[0]
    assert sample["pixel_values"].shape == (FRAMES, 3, CROP, CROP)
    assert sample["pixel_values"].dtype == torch.float32


def test_sample_carries_evaluation_metadata(synthetic_root):
    """Per-class and per-signer analysis depend on these surviving."""
    sample = make_dataset(synthetic_root)[0]
    for key in ("label", "sample_id", "signer_id", "gloss", "split", "dataset_name"):
        assert key in sample, f"missing {key}"
    assert isinstance(sample["label"], int)


def test_length_matches_the_manifest_split(synthetic_root):
    assert len(make_dataset(synthetic_root, split="train")) == 6
    assert len(make_dataset(synthetic_root, split="test")) == 3


def test_labels_stay_inside_the_label_map(synthetic_root):
    dataset = make_dataset(synthetic_root)
    for index in range(len(dataset)):
        assert 0 <= dataset[index]["label"] < dataset.label_map.num_classes


def test_empty_manifest_is_rejected(synthetic_root):
    parsed = parse_annotations(synthetic_root)
    with pytest.raises(ValueError, match="dataset is empty"):
        VideoClipDataset(Manifest(records=[]), parsed.label_map, synthetic_root)


# Evaluation determinism -------------------------------------------------------


def test_evaluation_samples_are_reproducible(synthetic_root):
    dataset = make_dataset(synthetic_root, split="test")
    assert torch.allclose(dataset[0]["pixel_values"], dataset[0]["pixel_values"])


def test_evaluation_rejects_random_temporal_sampling(synthetic_root):
    """An unreproducible evaluation run must be impossible to configure."""
    with pytest.raises(ValueError, match="requires deterministic temporal sampling"):
        make_dataset(
            synthetic_root,
            split="test",
            sampler=TemporalSampler(num_frames=FRAMES, strategy="random_segment"),
        )


def test_evaluation_rejects_training_transforms(synthetic_root):
    with pytest.raises(ValueError, match="requires a deterministic transform"):
        make_dataset(
            synthetic_root,
            split="validation",
            transform=TrainTransform(crop_size=CROP, resize_size=CROP + 16),
        )


def test_training_permits_random_sampling(synthetic_root):
    dataset = make_dataset(synthetic_root, split="train")
    assert not dataset.sampler.is_deterministic


# Training reproducibility -----------------------------------------------------


def test_training_samples_reproduce_from_the_same_seed(synthetic_root):
    a = make_dataset(synthetic_root, split="train", seed=123)[0]["pixel_values"]
    b = make_dataset(synthetic_root, split="train", seed=123)[0]["pixel_values"]
    assert torch.allclose(a, b)


def test_training_samples_differ_across_seeds(synthetic_root):
    a = make_dataset(synthetic_root, split="train", seed=1)[0]["pixel_values"]
    b = make_dataset(synthetic_root, split="train", seed=2)[0]["pixel_values"]
    assert not torch.allclose(a, b)


def test_different_samples_get_independent_randomness(synthetic_root):
    """Every clip in a batch sharing one crop would be a seeding bug."""
    dataset = make_dataset(synthetic_root, split="train", seed=5)
    assert not torch.allclose(dataset[0]["pixel_values"], dataset[1]["pixel_values"])


# Failure policy ---------------------------------------------------------------


def test_missing_video_fails_loudly_by_default(synthetic_root, tmp_path):
    """Silent skipping would mean training on an undefined dataset."""
    import shutil

    root = tmp_path / "broken"
    shutil.copytree(synthetic_root, root)
    parsed = parse_annotations(root)
    target = parsed.manifest.for_split("train")[0]
    target.resolve_path(root).unlink()

    dataset = make_dataset(root, split="train")
    with pytest.raises(VideoDecodeError, match="not what was recorded"):
        for index in range(len(dataset)):
            if dataset.records[index].sample_id == target.sample_id:
                dataset[index]


def test_skip_policy_substitutes_and_counts(synthetic_root, tmp_path):
    """A skip must be recorded, never absorbed."""
    import shutil

    root = tmp_path / "skippable"
    shutil.copytree(synthetic_root, root)
    parsed = parse_annotations(root)
    target = parsed.manifest.for_split("train")[0]
    target.resolve_path(root).unlink()

    dataset = make_dataset(root, split="train", failure_policy="skip")
    index = next(i for i, r in enumerate(dataset.records) if r.sample_id == target.sample_id)

    sample = dataset[index]
    assert sample["pixel_values"].shape == (FRAMES, 3, CROP, CROP)
    assert len(dataset.failures) == 1
    assert dataset.failures[0].sample_id == target.sample_id
    assert dataset.failures[0].reason


def test_unknown_failure_policy_is_rejected(synthetic_root):
    with pytest.raises(ValueError, match="unknown failure_policy"):
        make_dataset(synthetic_root, failure_policy="ignore")


def test_class_id_outside_the_label_map_is_rejected(synthetic_root):
    from dataclasses import replace

    parsed = parse_annotations(synthetic_root)
    records = [replace(parsed.manifest.records[0], class_id=999)]
    with pytest.raises(ValueError, match="outside the label map"):
        VideoClipDataset(Manifest(records=records), parsed.label_map, synthetic_root)


# Collation --------------------------------------------------------------------


def test_collate_produces_the_canonical_batch(synthetic_root):
    dataset = make_dataset(synthetic_root)
    batch = collate_clips([dataset[i] for i in range(3)])

    assert batch["pixel_values"].shape == (3, FRAMES, 3, CROP, CROP)
    assert batch["labels"].shape == (3,)
    assert batch["labels"].dtype == torch.int64


def test_collate_preserves_metadata_order(synthetic_root):
    dataset = make_dataset(synthetic_root)
    samples = [dataset[i] for i in range(3)]
    batch = collate_clips(samples)

    assert batch["sample_ids"] == [s["sample_id"] for s in samples]
    assert batch["signer_ids"] == [s["signer_id"] for s in samples]
    assert batch["glosses"] == [s["gloss"] for s in samples]


def test_collate_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_clips([])


def test_collate_rejects_inconsistent_shapes(synthetic_root):
    dataset = make_dataset(synthetic_root)
    bad = dict(dataset[0])
    bad["pixel_values"] = torch.zeros(FRAMES, 3, 32, 32)
    with pytest.raises(ValueError, match="inconsistent clip shapes"):
        collate_clips([dataset[0], bad])


def test_collate_does_not_permute_for_any_architecture(synthetic_root):
    """Layout adaptation belongs to model adapters, not the data layer."""
    dataset = make_dataset(synthetic_root)
    batch = collate_clips([dataset[0]])
    assert batch["pixel_values"].shape[1] == FRAMES  # frames at dim 1
    assert batch["pixel_values"].shape[2] == 3  # channels at dim 2


# DataLoader -------------------------------------------------------------------


def test_loads_through_a_dataloader(synthetic_root):
    dataset = make_dataset(synthetic_root)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_clips)

    batch = next(iter(loader))
    assert batch["pixel_values"].shape == (2, FRAMES, 3, CROP, CROP)


def test_evaluation_loader_never_shuffles_or_drops():
    """Dropping would silently change the evaluated sample count."""
    config = LoaderConfig(batch_size=4, num_workers=0, drop_last_train=True)

    for split in ("validation", "test"):
        options = config.for_split(split)
        assert options["shuffle"] is False
        assert options["drop_last"] is False

    train = config.for_split("train")
    assert train["shuffle"] is True
    assert train["drop_last"] is True


def test_evaluation_loader_covers_every_sample(synthetic_root):
    dataset = make_dataset(synthetic_root, split="test")
    config = LoaderConfig(batch_size=2, num_workers=0)
    loader = DataLoader(dataset, **config.for_split("test"))

    seen = [sid for batch in loader for sid in batch["sample_ids"]]
    assert len(seen) == len(dataset)
    assert len(set(seen)) == len(dataset)


def test_evaluation_loader_order_is_stable(synthetic_root):
    dataset = make_dataset(synthetic_root, split="test")
    config = LoaderConfig(batch_size=2, num_workers=0)

    first = [s for b in DataLoader(dataset, **config.for_split("test")) for s in b["sample_ids"]]
    second = [s for b in DataLoader(dataset, **config.for_split("test")) for s in b["sample_ids"]]
    assert first == second


def test_worker_init_seeds_reproducibly():
    torch.manual_seed(0)
    worker_init_fn(0)
    first = random.random()

    torch.manual_seed(0)
    worker_init_fn(0)
    assert random.random() == first


def test_workers_get_different_seeds():
    torch.manual_seed(0)
    worker_init_fn(0)
    a = random.random()

    torch.manual_seed(0)
    worker_init_fn(1)
    assert random.random() != a


# Preprocessing identity -------------------------------------------------------


def test_preprocessing_spec_describes_the_whole_pipeline(synthetic_root):
    spec = make_dataset(synthetic_root, split="test").preprocessing.to_dict()

    assert spec["temporal_sampling"]["num_frames"] == FRAMES
    assert spec["temporal_sampling"]["deterministic"] is True
    assert spec["spatial_transform"]["crop_size"] == CROP
    assert spec["color_space"] == "rgb"
    assert spec["canonical_layout"] == "TCHW"
    assert spec["short_video_policy"]
    assert spec["decoder_backend"] == "pyav"


def test_preprocessing_identity_is_stable():
    a = PreprocessingSpec(TemporalSampler(16, "uniform"), EvalTransform())
    b = PreprocessingSpec(TemporalSampler(16, "uniform"), EvalTransform())
    assert a.identity == b.identity


@pytest.mark.parametrize(
    ("sampler", "transform"),
    [
        (TemporalSampler(32, "uniform"), EvalTransform()),
        (TemporalSampler(16, "uniform"), EvalTransform(crop_size=112, resize_size=128)),
        (TemporalSampler(16, "random_segment"), EvalTransform()),
    ],
)
def test_preprocessing_identity_changes_with_the_pipeline(sampler, transform):
    """A checkpoint's preprocessing must be distinguishable from another's."""
    baseline = PreprocessingSpec(TemporalSampler(16, "uniform"), EvalTransform())
    assert PreprocessingSpec(sampler, transform).identity != baseline.identity


def test_train_and_eval_preprocessing_are_distinguishable():
    train = PreprocessingSpec(TemporalSampler(16, "random_segment"), TrainTransform())
    evaluation = PreprocessingSpec(TemporalSampler(16, "uniform"), EvalTransform())
    assert train.identity != evaluation.identity
    assert "train" in train.identity
    assert "eval" in evaluation.identity


# Model-layer agreement --------------------------------------------------------


def test_batch_is_accepted_by_both_model_adapters(synthetic_root):
    """The acceptance criterion: a real batch passes through both models."""
    from asl_training.models import ModelConfig, build_model

    dataset = make_dataset(
        synthetic_root,
        split="test",
        sampler=TemporalSampler(num_frames=16, strategy="uniform"),
        transform=EvalTransform(crop_size=224, resize_size=256),
    )
    batch = collate_clips([dataset[i] for i in range(2)])
    num_classes = dataset.label_map.num_classes

    for architecture, options in (
        (
            "videomae_base",
            {
                "hidden_size": 96,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "intermediate_size": 192,
            },
        ),
        ("video_swin_tiny", {}),
    ):
        model = build_model(
            ModelConfig(
                architecture=architecture,
                num_classes=num_classes,
                pretrained=False,
                num_frames=16,
                image_size=224,
                options=options,
            )
        )
        model.eval()
        with torch.no_grad():
            out = model(batch["pixel_values"], labels=batch["labels"])

        assert out.logits.shape == (2, num_classes)
        assert torch.isfinite(out.logits).all()
        assert out.loss is not None and torch.isfinite(out.loss)


def test_dataset_frame_count_matches_the_model_contract(synthetic_root):
    """The data layer's frame count is what the model validates against."""
    from asl_training.models import ModelConfig, build_model

    dataset = make_dataset(
        synthetic_root,
        split="test",
        sampler=TemporalSampler(num_frames=16, strategy="uniform"),
        transform=EvalTransform(crop_size=224, resize_size=256),
    )
    batch = collate_clips([dataset[0]])

    model = build_model(
        ModelConfig(
            architecture="video_swin_tiny",
            num_classes=dataset.label_map.num_classes,
            pretrained=False,
            num_frames=16,
        )
    )
    model.validate_input(batch["pixel_values"])  # must not raise
