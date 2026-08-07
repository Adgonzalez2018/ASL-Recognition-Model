"""Shared classification output.

Architecture-specific output objects must be converted into this type before
leaving the model layer, so that the training and evaluation layers never depend
on a third-party output class. See docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VideoClassifierOutput:
    """Result of a forward pass.

    Attributes:
        logits: Raw pre-softmax scores, shape ``[batch, num_classes]``.
        loss: Cross-entropy loss when labels were supplied, otherwise ``None``.

    The model layer never applies softmax, selects a class, or applies a
    temperature. Those belong to the evaluation layer.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
