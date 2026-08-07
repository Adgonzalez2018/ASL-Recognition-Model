"""Label-map determinism, stability, and identity.

The label map is the contract that gives class IDs meaning. These tests exist to
catch the failure mode where it changes without anyone noticing, which silently
invalidates every checkpoint trained against it.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import json

import pytest

from asl_training.data.label_map import LabelMap, normalize_gloss

GLOSSES = ["BOOK", "CAT", "APPLE", "DOG", "elephant"]


# Determinism -----------------------------------------------------------------


def test_ordering_is_independent_of_input_order():
    """The map must not depend on row order, directory order, or split order."""
    a = LabelMap.from_glosses(GLOSSES)
    b = LabelMap.from_glosses(list(reversed(GLOSSES)))
    c = LabelMap.from_glosses([GLOSSES[i] for i in (2, 4, 0, 3, 1)])

    assert a.glosses == b.glosses == c.glosses
    assert a.identity == b.identity == c.identity


def test_duplicate_occurrences_are_collapsed():
    """One row per sample means the same gloss appears many times."""
    label_map = LabelMap.from_glosses(["CAT"] * 50 + ["DOG"] * 30)
    assert label_map.num_classes == 2
    assert label_map.glosses == ("CAT", "DOG")


def test_ordering_is_case_insensitive_but_preserves_original():
    label_map = LabelMap.from_glosses(["banana", "APPLE", "Cherry"])
    assert label_map.glosses == ("APPLE", "banana", "Cherry")


def test_construction_rule_is_recorded():
    label_map = LabelMap.from_glosses(GLOSSES)
    assert label_map.construction_rule == "sorted-by-normalized-gloss-v1"


# Contiguity and validity ------------------------------------------------------


def test_class_ids_are_contiguous_and_zero_indexed():
    label_map = LabelMap.from_glosses(GLOSSES)
    assert list(label_map.class_ids) == list(range(len(GLOSSES)))
    assert label_map.num_classes == len(GLOSSES)


def test_round_trip_id_and_gloss():
    label_map = LabelMap.from_glosses(GLOSSES)
    for class_id in label_map.class_ids:
        assert label_map.to_id(label_map.to_gloss(class_id)) == class_id


def test_rejects_empty_vocabulary():
    with pytest.raises(ValueError, match="zero glosses"):
        LabelMap.from_glosses([])


def test_rejects_single_class_vocabulary():
    with pytest.raises(ValueError, match="at least 2"):
        LabelMap.from_glosses(["ONLY"])


def test_rejects_duplicate_glosses_in_direct_construction():
    with pytest.raises(ValueError, match="duplicate gloss"):
        LabelMap(glosses=("CAT", "DOG", "CAT"))


@pytest.mark.parametrize("bad", ["", "   "])
def test_rejects_empty_gloss(bad):
    with pytest.raises(ValueError, match="empty or not a string"):
        LabelMap(glosses=("CAT", bad))


# Unsafe merging ---------------------------------------------------------------


def test_rejects_normalized_collisions_rather_than_merging():
    """Distinct source glosses that normalize alike must not fuse silently."""
    with pytest.raises(ValueError, match="normalized gloss collision"):
        LabelMap(glosses=("CAT", "cat"))


def test_collision_error_names_the_offenders():
    with pytest.raises(ValueError, match="CAT"):
        LabelMap(glosses=("CAT", "cat", "DOG"))


def test_whitespace_variants_are_treated_as_collisions():
    with pytest.raises(ValueError, match="normalized gloss collision"):
        LabelMap(glosses=("ICE CREAM", "ICE  CREAM"))


def test_normalize_is_conservative():
    """Normalization must not strip punctuation or merge variants."""
    assert normalize_gloss("  ICE   CREAM  ") == "ice cream"
    assert normalize_gloss("CAT") == "cat"
    # Distinct concepts must stay distinct.
    assert normalize_gloss("BANK-1") != normalize_gloss("BANK-2")
    assert normalize_gloss("RUN") != normalize_gloss("RUNNING")


# Lookup failures --------------------------------------------------------------


def test_unknown_gloss_raises_rather_than_defaulting():
    label_map = LabelMap.from_glosses(GLOSSES)
    with pytest.raises(KeyError, match="not in the asl_citizen label map"):
        label_map.to_id("NOTASIGN")


@pytest.mark.parametrize("bad_id", [-1, 5, 999])
def test_out_of_range_class_id_raises(bad_id):
    label_map = LabelMap.from_glosses(GLOSSES)
    with pytest.raises(IndexError, match="out of range"):
        label_map.to_gloss(bad_id)


def test_rejects_bool_class_id():
    label_map = LabelMap.from_glosses(GLOSSES)
    with pytest.raises(TypeError, match="must be an int"):
        label_map.to_gloss(True)


def test_membership_test():
    label_map = LabelMap.from_glosses(GLOSSES)
    assert "CAT" in label_map
    assert "NOTASIGN" not in label_map


# Identity ---------------------------------------------------------------------


def test_identity_is_stable_across_equivalent_construction():
    assert LabelMap.from_glosses(GLOSSES).identity == LabelMap.from_glosses(GLOSSES).identity


def test_identity_changes_when_a_class_is_added():
    base = LabelMap.from_glosses(GLOSSES)
    extended = LabelMap.from_glosses([*GLOSSES, "FISH"])
    assert base.identity != extended.identity


def test_identity_changes_when_order_changes():
    """Reordering changes what every class ID means, so it must change identity."""
    a = LabelMap(glosses=("CAT", "DOG"))
    b = LabelMap(glosses=("DOG", "CAT"))
    assert a.identity != b.identity


def test_identity_distinguishes_datasets():
    """ASL Citizen and WLASL must never share an implicit label map."""
    a = LabelMap.from_glosses(GLOSSES, dataset_name="asl_citizen")
    b = LabelMap.from_glosses(GLOSSES, dataset_name="wlasl")
    assert a.identity != b.identity
    assert not a.is_compatible_with(b)


def test_identity_encodes_class_count():
    label_map = LabelMap.from_glosses(GLOSSES)
    assert f":{len(GLOSSES)}:" in label_map.identity


def test_compatible_maps_agree():
    assert LabelMap.from_glosses(GLOSSES).is_compatible_with(LabelMap.from_glosses(GLOSSES))


# Serialization ----------------------------------------------------------------


def test_round_trips_through_dict():
    original = LabelMap.from_glosses(GLOSSES, version="v1")
    restored = LabelMap.from_dict(original.to_dict())
    assert restored.glosses == original.glosses
    assert restored.identity == original.identity
    assert restored.version == "v1"


def test_round_trips_through_file(tmp_path):
    original = LabelMap.from_glosses(GLOSSES)
    path = original.save(tmp_path / "label_map.json")
    assert LabelMap.load(path).identity == original.identity


def test_save_refuses_to_overwrite(tmp_path):
    """Overwriting would invalidate checkpoints trained against the old map."""
    path = tmp_path / "label_map.json"
    LabelMap.from_glosses(GLOSSES).save(path)
    with pytest.raises(FileExistsError, match="already exists"):
        LabelMap.from_glosses([*GLOSSES, "FISH"]).save(path)


def test_load_detects_tampering(tmp_path):
    """A hand-edited file must not load with a stale identity."""
    path = tmp_path / "label_map.json"
    LabelMap.from_glosses(GLOSSES).save(path)

    payload = json.loads(path.read_text())
    payload["glosses"].append("SNEAKY")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="identity mismatch"):
        LabelMap.load(path)


def test_load_detects_class_count_mismatch(tmp_path):
    path = tmp_path / "label_map.json"
    label_map = LabelMap.from_glosses(GLOSSES)
    payload = label_map.to_dict()
    payload["num_classes"] = 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="records 99 classes"):
        LabelMap.load(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="label map not found"):
        LabelMap.load(tmp_path / "absent.json")


def test_from_dict_requires_glosses():
    with pytest.raises(ValueError, match="missing required key 'glosses'"):
        LabelMap.from_dict({"dataset_name": "asl_citizen"})


def test_serialized_form_is_human_readable(tmp_path):
    """Reviewers must be able to read the vocabulary a model was trained on."""
    path = LabelMap.from_glosses(GLOSSES).save(tmp_path / "label_map.json")
    payload = json.loads(path.read_text())
    assert payload["glosses"] == sorted(GLOSSES, key=lambda g: (g.casefold(), g))
    assert set(payload) == {
        "dataset_name",
        "construction_rule",
        "version",
        "identity",
        "num_classes",
        "glosses",
    }


# Model-layer agreement --------------------------------------------------------


def test_num_classes_drives_the_model_output_dimension():
    """The model's class count comes from here, never from a batch."""
    from asl_training.models import ModelConfig

    label_map = LabelMap.from_glosses(GLOSSES)
    config = ModelConfig(architecture="videomae_base", num_classes=label_map.num_classes)
    assert config.num_classes == label_map.num_classes


def test_class_ids_are_valid_model_labels():
    """Every class ID must fall inside the model's accepted label range."""
    label_map = LabelMap.from_glosses(GLOSSES)
    assert min(label_map.class_ids) == 0
    assert max(label_map.class_ids) == label_map.num_classes - 1
