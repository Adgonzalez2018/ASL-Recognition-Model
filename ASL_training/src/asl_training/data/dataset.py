"""Dataset and batch collation.

Turns manifest records into the canonical batch the model layer accepts:

    [batch, frames, channels, height, width]

Runtime sample failures follow an explicit, counted policy. Silent skipping is
prohibited: a run that quietly drops a tenth of its data would report metrics
over a dataset nobody defined.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .decode import DecodedClip, VideoDecodeError, decode_clip
from .label_map import LabelMap
from .manifest import Manifest, ManifestRecord
from .sampling import SHORT_VIDEO_POLICY, TemporalSampler
from .transforms import EvalTransform, TrainTransform

logger = logging.getLogger(__name__)

# What to do when a sample fails at runtime.
#   "fail"  raise immediately. The default: a failure means the audit missed
#           something, and the run's dataset is not what was recorded.
#   "skip"  substitute another sample and count it. Only for pre-audited,
#           known-bad samples, and the count is reported.
FAILURE_POLICIES = ("fail", "skip")


@dataclass
class PreprocessingSpec:
    """The complete description of how a clip becomes a model tensor.

    Recorded with every checkpoint and exported for the future serving project,
    which must reproduce the evaluation pipeline rather than reimplement it from
    memory.
    """

    sampler: TemporalSampler
    transform: EvalTransform | TrainTransform
    decoder_backend: str = "pyav"
    color_space: str = "rgb"
    canonical_layout: str = "TCHW"

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_sampling": self.sampler.to_dict(),
            "spatial_transform": self.transform.to_dict(),
            "decoder_backend": self.decoder_backend,
            "color_space": self.color_space,
            "canonical_layout": self.canonical_layout,
            "short_video_policy": SHORT_VIDEO_POLICY,
        }

    @property
    def identity(self) -> str:
        """A fingerprint over everything that affects the produced tensor.

        A checkpoint records this. Evaluating it under different preprocessing
        is a different experiment, and the mismatch must be visible.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        kind = self.transform.to_dict()["kind"]
        return f"preprocessing:{kind}:sha256:{digest}"


@dataclass
class SampleFailure:
    """One runtime sample failure, recorded rather than swallowed."""

    sample_id: str
    video_path: str
    split: str
    reason: str


class VideoClipDataset(Dataset):
    """Manifest records as decoded, transformed video tensors.

    Args:
        manifest: Records for one split.
        label_map: Vocabulary. Class IDs are validated against it.
        dataset_root: Runtime root that record paths resolve against.
        sampler: Temporal sampling strategy.
        transform: Spatial transform. Must be deterministic for evaluation.
        split: Which split this dataset represents.
        seed: Base seed for reproducible training randomness.
        failure_policy: ``"fail"`` or ``"skip"``. See ``FAILURE_POLICIES``.
    """

    def __init__(
        self,
        manifest: Manifest,
        label_map: LabelMap,
        dataset_root: str | Path,
        *,
        sampler: TemporalSampler | None = None,
        transform: EvalTransform | TrainTransform | None = None,
        split: str = "train",
        seed: int = 0,
        failure_policy: str = "fail",
    ) -> None:
        self.records: list[ManifestRecord] = list(manifest.records)
        if not self.records:
            raise ValueError("dataset is empty; a manifest split with no records")

        self.label_map = label_map
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.seed = seed

        if failure_policy not in FAILURE_POLICIES:
            raise ValueError(
                f"unknown failure_policy {failure_policy!r}; "
                f"supported: {', '.join(FAILURE_POLICIES)}"
            )
        self.failure_policy = failure_policy

        is_training = split == "train"
        self.sampler = sampler or TemporalSampler(
            strategy="random_segment" if is_training else "uniform"
        )
        self.transform = transform or (TrainTransform() if is_training else EvalTransform())

        # Evaluation must be reproducible. Catching this here prevents a whole
        # evaluation run whose numbers cannot be reproduced.
        if not is_training:
            if not self.sampler.is_deterministic:
                raise ValueError(
                    f"split {split!r} requires deterministic temporal sampling, but "
                    f"strategy {self.sampler.strategy!r} is random. Evaluation must "
                    f"be reproducible; see docs/DATA_CONTRACT.md."
                )
            if not self.transform.is_deterministic:
                raise ValueError(
                    f"split {split!r} requires a deterministic transform, but "
                    f"{type(self.transform).__name__} applies training augmentation."
                )

        self._validate_labels()
        self.failures: list[SampleFailure] = []

    def _validate_labels(self) -> None:
        valid = set(self.label_map.class_ids)
        for record in self.records:
            if record.class_id not in valid:
                raise ValueError(
                    f"sample {record.sample_id}: class_id {record.class_id} is outside "
                    f"the label map (0 to {self.label_map.num_classes - 1})"
                )

    @property
    def preprocessing(self) -> PreprocessingSpec:
        return PreprocessingSpec(sampler=self.sampler, transform=self.transform)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int, epoch: int = 0) -> random.Random:
        """A per-sample generator derived from the experiment seed.

        Deriving from seed, index, and epoch means each sample gets an
        independent stream that is still reproducible, and that multiple data
        loader workers cannot accidentally share a sequence.
        """
        return random.Random((self.seed, index, epoch).__hash__())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        try:
            return self._load(record, index)
        except (VideoDecodeError, ValueError, RuntimeError) as exc:
            failure = SampleFailure(
                sample_id=record.sample_id,
                video_path=record.video_path,
                split=record.split,
                reason=f"{type(exc).__name__}: {exc}",
            )

            if self.failure_policy == "fail":
                raise VideoDecodeError(
                    f"sample {record.sample_id} failed to load: {exc}. The audit did "
                    f"not predict this failure, so the run's dataset is not what was "
                    f"recorded. Re-audit, or configure failure_policy='skip' with a "
                    f"documented exclusion list."
                ) from exc

            self.failures.append(failure)
            logger.warning(
                "skipping sample %s (%s): %s [%d skipped so far]",
                record.sample_id,
                record.video_path,
                exc,
                len(self.failures),
            )
            # Substitute a neighbour so the batch keeps its shape. The
            # substitution is counted and reported; it is not silent.
            return self._load(self.records[(index + 1) % len(self.records)], index)

    def _load(self, record: ManifestRecord, index: int) -> dict[str, Any]:
        path = record.resolve_path(self.dataset_root)
        rng = self._rng(index)

        total = record.frame_count
        if not total:
            # Manifest carries no audited frame count, so decode fully to learn it.
            probe = decode_clip(path)
            total = probe.source_frame_count
            clip = self._select(probe, total, rng)
        else:
            indices = self.sampler.indices(total, rng)
            try:
                decoded = decode_clip(path, indices, expected_frames=self.sampler.num_frames)
            except VideoDecodeError:
                # The audited frame count disagrees with the file. Fall back to
                # the file, which is authoritative, and report the discrepancy.
                probe = decode_clip(path)
                logger.warning(
                    "sample %s: manifest records %d frames but the file decoded %d; using the file",
                    record.sample_id,
                    total,
                    probe.source_frame_count,
                )
                clip = self._select(probe, probe.source_frame_count, rng)
            else:
                clip = decoded.frames

        pixel_values = self.transform(clip, rng)

        expected = (self.sampler.num_frames, 3, self.transform.crop_size, self.transform.crop_size)
        if tuple(pixel_values.shape) != expected:
            raise RuntimeError(
                f"sample {record.sample_id}: produced {tuple(pixel_values.shape)}, "
                f"expected {expected}"
            )

        return {
            "pixel_values": pixel_values,
            "label": record.class_id,
            "sample_id": record.sample_id,
            "signer_id": record.signer_id,
            "gloss": record.gloss,
            "split": record.split,
            "dataset_name": record.dataset_name,
        }

    def _select(self, decoded: DecodedClip, total: int, rng: random.Random) -> torch.Tensor:
        """Apply temporal sampling to an already fully decoded clip."""
        indices = self.sampler.indices(total, rng)
        return decoded.frames[torch.tensor(indices, dtype=torch.long)]


def collate_clips(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples into the canonical batch.

    Produces ``pixel_values`` of shape ``[batch, frames, channels, height,
    width]`` and int64 ``labels``, and preserves per-sample metadata in order so
    that per-class and per-signer evaluation remain possible.

    No architecture-specific permutation happens here. Adapters handle their own
    tensor layouts.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")

    shapes = {tuple(item["pixel_values"].shape) for item in batch}
    if len(shapes) > 1:
        raise ValueError(
            f"batch contains inconsistent clip shapes: {sorted(shapes)}. Every sample "
            f"must produce the same fixed-size tensor."
        )

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "signer_ids": [item["signer_id"] for item in batch],
        "glosses": [item["gloss"] for item in batch],
        "splits": [item["split"] for item in batch],
    }


@dataclass
class LoaderConfig:
    """Data loader settings, per split.

    Training shuffles and may drop the last partial batch. Evaluation never
    shuffles and never drops, because dropping would silently change the
    evaluated sample count.
    """

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last_train: bool = False

    def for_split(self, split: str) -> dict[str, Any]:
        is_training = split == "train"
        options: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": is_training,
            "drop_last": self.drop_last_train if is_training else False,
            "collate_fn": collate_clips,
        }
        if self.num_workers > 0:
            options["persistent_workers"] = self.persistent_workers
            options["prefetch_factor"] = self.prefetch_factor
        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "drop_last_train": self.drop_last_train,
        }


def worker_init_fn(worker_id: int) -> None:
    """Seed each data loader worker independently but reproducibly.

    Without this, workers can share a random sequence and produce correlated
    augmentation across a batch.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    try:
        import numpy as np

        np.random.seed((seed + worker_id) % (2**32))
    except ImportError:  # pragma: no cover
        pass
