"""Data layer: dataset parsing, manifests, decoding, sampling, and transforms.

Implemented so far (Phase 2B):

    label_map   the classifier's output vocabulary and its identity
    manifest    the record schema, split integrity, and signer-leakage checks

Decoding, temporal sampling, and transforms follow in Phase 2C.

See docs/DATA_CONTRACT.md.
"""

from .label_map import CONSTRUCTION_RULE, LabelMap, normalize_gloss
from .manifest import (
    REQUIRED_FIELDS,
    SPLITS,
    Manifest,
    ManifestRecord,
    ManifestValidationError,
    ValidationReport,
    normalize_split,
)

__all__ = [
    "CONSTRUCTION_RULE",
    "REQUIRED_FIELDS",
    "SPLITS",
    "LabelMap",
    "Manifest",
    "ManifestRecord",
    "ManifestValidationError",
    "ValidationReport",
    "normalize_gloss",
    "normalize_split",
]
