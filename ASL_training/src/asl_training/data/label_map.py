"""Label map: the classifier's output vocabulary.

The label map defines what each class ID means. Its size must equal the model's
output dimension, and its meaning must be identical across splits, architectures,
and experiments. A label map that changes silently invalidates every checkpoint
trained against it.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Construction rule, recorded so it can be reproduced and audited.
# Sorting is by normalized gloss, then by original gloss, so the result cannot
# depend on annotation-file row order, directory iteration order, or which split
# happened to be read first.
CONSTRUCTION_RULE = "sorted-by-normalized-gloss-v1"

_WHITESPACE = re.compile(r"\s+")


def normalize_gloss(gloss: str) -> str:
    """Normalize a gloss for ordering and comparison.

    Collapses whitespace and casefolds. Deliberately conservative: it does not
    strip punctuation, expand abbreviations, or merge variants, because those
    operations can silently fuse distinct signs. Any semantic merge requires an
    explicit reviewed mapping, per docs/DATA_CONTRACT.md.
    """
    return _WHITESPACE.sub(" ", gloss.strip()).casefold()


@dataclass(frozen=True)
class LabelMap:
    """A stable, contiguous, zero-indexed gloss vocabulary.

    Attributes:
        glosses: Original glosses, indexed by class ID.
        construction_rule: How the ordering was produced.
        dataset_name: The dataset this vocabulary belongs to. ASL Citizen and
            WLASL must never share one implicit label map.
        version: Optional caller-supplied version label.
    """

    glosses: tuple[str, ...]
    construction_rule: str = CONSTRUCTION_RULE
    dataset_name: str = "asl_citizen"
    version: str | None = None

    def __post_init__(self) -> None:
        if not self.glosses:
            raise ValueError("label map is empty")

        if len(self.glosses) < 2:
            raise ValueError(
                f"label map has {len(self.glosses)} class; multiclass classification "
                f"requires at least 2"
            )

        seen: dict[str, int] = {}
        for index, gloss in enumerate(self.glosses):
            if not isinstance(gloss, str) or not gloss.strip():
                raise ValueError(f"gloss at index {index} is empty or not a string")
            if gloss in seen:
                raise ValueError(f"duplicate gloss {gloss!r} at indices {seen[gloss]} and {index}")
            seen[gloss] = index

        # Distinct source glosses that normalize identically are a real hazard:
        # they are probably the same sign recorded inconsistently, but merging
        # them automatically could just as easily fuse two different signs.
        # Refuse, and require an explicit reviewed decision.
        collisions: dict[str, list[str]] = {}
        for gloss in self.glosses:
            collisions.setdefault(normalize_gloss(gloss), []).append(gloss)
        ambiguous = {k: v for k, v in collisions.items() if len(v) > 1}
        if ambiguous:
            examples = "; ".join(
                f"{key!r} <- {sorted(values)}" for key, values in sorted(ambiguous.items())[:5]
            )
            raise ValueError(
                f"{len(ambiguous)} normalized gloss collision(s): {examples}. "
                f"These may be the same sign recorded inconsistently, or genuinely "
                f"distinct signs. Resolve with an explicit reviewed mapping rather "
                f"than merging automatically."
            )

    # Construction -------------------------------------------------------------

    @classmethod
    def from_glosses(
        cls,
        glosses: list[str],
        *,
        dataset_name: str = "asl_citizen",
        version: str | None = None,
    ) -> LabelMap:
        """Build from an unordered collection, applying the deterministic rule.

        Duplicate occurrences of the same gloss are expected — one per sample —
        and are collapsed. The resulting order does not depend on input order.
        """
        unique = set(glosses)
        if not unique:
            raise ValueError("cannot build a label map from zero glosses")

        ordered = sorted(unique, key=lambda g: (normalize_gloss(g), g))
        return cls(
            glosses=tuple(ordered),
            construction_rule=CONSTRUCTION_RULE,
            dataset_name=dataset_name,
            version=version,
        )

    # Lookup -------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.glosses)

    @property
    def num_classes(self) -> int:
        """Size of the vocabulary, which must equal the model output dimension."""
        return len(self.glosses)

    @property
    def class_ids(self) -> range:
        """Valid class IDs: contiguous and zero-indexed."""
        return range(len(self.glosses))

    def to_id(self, gloss: str) -> int:
        """Map a gloss to its class ID.

        Raises:
            KeyError: If the gloss is not in the vocabulary. Unknown glosses are
                an error, never silently assigned a fallback class.
        """
        try:
            return self._gloss_to_id[gloss]
        except KeyError:
            raise KeyError(
                f"gloss {gloss!r} is not in the {self.dataset_name} label map "
                f"({self.num_classes} classes). The manifest and label map disagree."
            ) from None

    def to_gloss(self, class_id: int) -> str:
        """Map a class ID back to its gloss."""
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise TypeError(f"class_id must be an int, got {type(class_id).__name__}")
        if not 0 <= class_id < len(self.glosses):
            raise IndexError(
                f"class_id {class_id} is out of range; valid IDs are 0 to {len(self.glosses) - 1}"
            )
        return self.glosses[class_id]

    def __contains__(self, gloss: object) -> bool:
        return gloss in self._gloss_to_id

    @property
    def _gloss_to_id(self) -> dict[str, int]:
        # Rebuilt per access rather than cached, because the dataclass is frozen
        # and the vocabulary is small enough that this is not worth complicating.
        return {gloss: index for index, gloss in enumerate(self.glosses)}

    # Identity -----------------------------------------------------------------

    @property
    def identity(self) -> str:
        """A stable fingerprint of this vocabulary.

        Changes whenever the glosses or their order change, which is exactly when
        previously trained checkpoints stop being compatible. Recorded with every
        checkpoint and evaluation report.
        """
        payload = json.dumps(
            {
                "dataset_name": self.dataset_name,
                "construction_rule": self.construction_rule,
                "glosses": list(self.glosses),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.dataset_name}:{self.num_classes}:sha256:{digest}"

    def is_compatible_with(self, other: LabelMap) -> bool:
        """Whether two label maps assign identical meaning to every class ID."""
        return self.identity == other.identity

    # Serialization ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "construction_rule": self.construction_rule,
            "version": self.version,
            "identity": self.identity,
            "num_classes": self.num_classes,
            "glosses": list(self.glosses),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabelMap:
        """Rebuild from a serialized map, verifying its recorded identity."""
        for key in ("glosses", "dataset_name"):
            if key not in raw:
                raise ValueError(f"label map is missing required key {key!r}")

        label_map = cls(
            glosses=tuple(raw["glosses"]),
            construction_rule=raw.get("construction_rule", CONSTRUCTION_RULE),
            dataset_name=raw["dataset_name"],
            version=raw.get("version"),
        )

        recorded = raw.get("identity")
        if recorded and recorded != label_map.identity:
            raise ValueError(
                f"label map identity mismatch: file records {recorded!r} but the "
                f"loaded content produces {label_map.identity!r}. The file was "
                f"edited or written by a different construction rule."
            )

        recorded_count = raw.get("num_classes")
        if recorded_count is not None and recorded_count != label_map.num_classes:
            raise ValueError(
                f"label map records {recorded_count} classes but contains {label_map.num_classes}"
            )

        return label_map

    def save(self, path: str | Path) -> Path:
        """Write to JSON, refusing to overwrite silently."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Overwriting a label map would invalidate "
                f"every checkpoint trained against it. Write a new version instead."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> LabelMap:
        """Read from JSON, verifying the recorded identity."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"label map not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
