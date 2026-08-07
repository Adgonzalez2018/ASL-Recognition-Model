"""Shared model contract.

Every test here runs against both architectures, because the point of the
contract is that the training and evaluation layers cannot tell them apart.

See docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import pytest
import torch

from asl_training.models import VideoClassifierOutput, build_model

from .conftest import NUM_CLASSES, make_batch, make_labels

# Input and output shape ------------------------------------------------------


def test_accepts_canonical_batch_and_returns_configured_logits(any_config):
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(any_config, batch_size=3))

    assert isinstance(out, VideoClassifierOutput)
    assert out.logits.shape == (3, NUM_CLASSES)


def test_single_sample_batch_is_accepted(any_config):
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(any_config, batch_size=1))
    assert out.logits.shape == (1, NUM_CLASSES)


def test_head_output_size_matches_configuration(any_config):
    model = build_model(any_config)
    head = model.classification_head()
    linear = head[-1] if isinstance(head, torch.nn.Sequential) else head
    assert linear.out_features == NUM_CLASSES


def test_logits_are_finite(any_config):
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(any_config))
    assert torch.isfinite(out.logits).all()


# Input validation ------------------------------------------------------------


def test_rejects_channels_first_layout(any_config):
    """The classic mistake: passing [B, C, T, H, W] instead of [B, T, C, H, W]."""
    model = build_model(any_config)
    transposed = make_batch(any_config).permute(0, 2, 1, 3, 4)
    with pytest.raises(ValueError, match="channels"):
        model(transposed)


def test_rejects_wrong_rank(any_config):
    model = build_model(any_config)
    with pytest.raises(ValueError, match="rank-5"):
        model(torch.randn(2, 3, any_config.image_size, any_config.image_size))


def test_rejects_wrong_frame_count(any_config):
    model = build_model(any_config)
    bad = torch.randn(2, any_config.num_frames + 2, 3, any_config.image_size, any_config.image_size)
    with pytest.raises(ValueError, match="frames"):
        model(bad)


def test_rejects_wrong_resolution(any_config):
    model = build_model(any_config)
    bad = torch.randn(2, any_config.num_frames, 3, 112, 112)
    with pytest.raises(ValueError, match="resolution"):
        model(bad)


def test_rejects_empty_batch(any_config):
    model = build_model(any_config)
    bad = torch.randn(0, any_config.num_frames, 3, any_config.image_size, any_config.image_size)
    with pytest.raises(ValueError, match="empty"):
        model(bad)


def test_rejects_integer_pixel_values(any_config):
    """Normalization is the data layer's job; raw uint8 must not slip through."""
    model = build_model(any_config)
    bad = torch.randint(
        0,
        255,
        (2, any_config.num_frames, 3, any_config.image_size, any_config.image_size),
        dtype=torch.uint8,
    )
    with pytest.raises(TypeError, match="floating-point"):
        model(bad)


def test_rejects_non_tensor_input(any_config):
    model = build_model(any_config)
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        model([[1, 2, 3]])


# Label validation ------------------------------------------------------------


def test_rejects_label_above_class_range(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config, batch_size=2)
    bad = torch.tensor([0, NUM_CLASSES], dtype=torch.long)
    with pytest.raises(ValueError, match="out of range"):
        model(batch, labels=bad)


def test_rejects_negative_label(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config, batch_size=2)
    with pytest.raises(ValueError, match="out of range"):
        model(batch, labels=torch.tensor([-1, 0], dtype=torch.long))


def test_rejects_float_labels(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config, batch_size=2)
    with pytest.raises(TypeError, match="integer dtype"):
        model(batch, labels=torch.tensor([0.0, 1.0]))


def test_rejects_label_batch_mismatch(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config, batch_size=3)
    with pytest.raises(ValueError, match="does not match batch size"):
        model(batch, labels=torch.tensor([0, 1], dtype=torch.long))


def test_rejects_one_hot_labels(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config, batch_size=2)
    one_hot = torch.zeros(2, NUM_CLASSES, dtype=torch.long)
    with pytest.raises(ValueError, match=r"shape \[batch\]"):
        model(batch, labels=one_hot)


# Loss ------------------------------------------------------------------------


def test_no_loss_without_labels(any_config):
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(any_config))
    assert out.loss is None


def test_loss_returned_with_labels(any_config):
    model = build_model(any_config)
    out = model(make_batch(any_config), labels=make_labels(any_config))
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_initial_loss_is_near_uniform_prior(any_config):
    """A random head over N classes should start near ln(N)."""
    import math

    model = build_model(any_config)
    model.eval()
    batch = make_batch(any_config, batch_size=8)
    labels = make_labels(any_config, batch_size=8)
    with torch.no_grad():
        loss = float(model(batch, labels=labels).loss)

    assert loss == pytest.approx(math.log(NUM_CLASSES), abs=1.0)


# Forward pass purity ---------------------------------------------------------


def test_forward_does_not_apply_softmax(any_config):
    """Logits are raw. Rows summing to 1 would mean a softmax leaked in."""
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        logits = model(make_batch(any_config, batch_size=4)).logits

    row_sums = logits.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (logits < 0).any(), "raw logits should include negative values"


def test_forward_does_not_mutate_labels(any_config):
    model = build_model(any_config)
    labels = make_labels(any_config)
    original = labels.clone()
    model(make_batch(any_config), labels=labels)
    assert torch.equal(labels, original)


def test_forward_does_not_mutate_input(any_config):
    model = build_model(any_config)
    batch = make_batch(any_config)
    original = batch.clone()
    model.eval()
    with torch.no_grad():
        model(batch)
    assert torch.equal(batch, original)


# Training and evaluation modes ----------------------------------------------


def test_eval_mode_is_deterministic(any_config):
    model = build_model(any_config)
    model.eval()
    batch = make_batch(any_config)
    with torch.no_grad():
        first = model(batch).logits
        second = model(batch).logits
    assert torch.allclose(first, second)


def test_no_gradients_under_no_grad(any_config):
    model = build_model(any_config)
    model.eval()
    with torch.no_grad():
        out = model(make_batch(any_config))
    assert not out.logits.requires_grad


def test_backward_reaches_backbone_and_head(any_config):
    """Full fine-tuning means gradients flow through the whole network."""
    model = build_model(any_config)
    model.train()
    out = model(make_batch(any_config), labels=make_labels(any_config))
    out.loss.backward()

    # Compare by identity: the head's parameters are named bare "weight"/"bias",
    # so suffix matching would classify every backbone weight as head.
    head_ids = {id(p) for p in model.classification_head().parameters()}
    got_head = False
    got_backbone = False
    for param in model.parameters():
        if param.grad is None or param.grad.abs().sum() == 0:
            continue
        if id(param) in head_ids:
            got_head = True
        else:
            got_backbone = True

    assert got_head, "classification head received no gradient"
    assert got_backbone, "backbone received no gradient under full fine-tuning"


# Fine-tuning strategy --------------------------------------------------------


def test_full_strategy_makes_everything_trainable(any_config):
    model = build_model(any_config)
    report = model.parameter_report()
    assert report.frozen == 0
    assert report.trainable == report.total
    assert not model.frozen_parameter_names()


def test_head_only_strategy_freezes_backbone(any_config):
    from asl_training.models import ModelConfig

    config = ModelConfig.from_dict({**any_config.to_dict(), "fine_tuning": "head_only"})
    model = build_model(config)
    report = model.parameter_report()

    assert report.trainable == report.head
    assert report.frozen == report.total - report.head
    assert report.frozen > 0

    head_ids = {id(p) for p in model.classification_head().parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable_ids == head_ids


# Parameter reporting ---------------------------------------------------------


def test_parameter_report_is_internally_consistent(any_config):
    model = build_model(any_config)
    report = model.parameter_report()

    assert report.total == report.trainable + report.frozen
    assert report.total > 0
    assert 0 < report.head < report.total
    assert report.approx_fp32_mb > 0
    assert set(report.to_dict()) == {"total", "trainable", "frozen", "head", "approx_fp32_mb"}


def test_head_parameter_count_matches_linear_dimensions(any_config):
    model = build_model(any_config)
    head = model.classification_head()
    linear = head[-1] if isinstance(head, torch.nn.Sequential) else head
    expected = linear.in_features * linear.out_features + linear.out_features
    assert model.parameter_report().head == expected


def test_describe_carries_metadata_for_run_records(any_config):
    model = build_model(any_config)
    described = model.describe()
    assert set(described) == {"config", "parameters", "preprocessing"}
    assert described["config"]["num_classes"] == NUM_CLASSES


# Preprocessing requirements --------------------------------------------------


def test_preprocessing_matches_configured_input(any_config):
    model = build_model(any_config)
    prep = model.preprocessing()
    assert prep.num_frames == any_config.num_frames
    assert prep.image_size == any_config.image_size
    assert prep.canonical_layout == "BTCHW"
    assert len(prep.mean) == 3
    assert len(prep.std) == 3
    assert all(s > 0 for s in prep.std)


def test_dropout_is_installed_when_configured(any_config):
    from asl_training.models import ModelConfig

    config = ModelConfig.from_dict({**any_config.to_dict(), "dropout": 0.5})
    model = build_model(config)
    head = model.classification_head()
    assert isinstance(head, torch.nn.Sequential)
    assert any(isinstance(m, torch.nn.Dropout) for m in head)
