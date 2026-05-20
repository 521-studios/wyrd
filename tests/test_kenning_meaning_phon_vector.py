"""Tests for the Meaning.phonological_vector field (wyrd-ecjp.5 prereq).

The vector-scoring runtime path reads per-meaning phonological vectors
to compute the ``phon_score`` sub-score. This field is bundle-side
data; load_meanings reads ``phonological_vector`` from each bundle
word entry and constructs a PhonologicalVector instance via
``_parse_phon_vector``.

Legacy bundles (pre-kq7w.1 / pre-ecjp.8) emit no field — the meaning
loads with phonological_vector=None and the vector-scoring path
treats it as a default-empty vector (phon_score returns 0).
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.meaning import (
    Meaning,
    _parse_phon_vector,
    load_meanings,
)
from wyrd.generators.kenning.vectors.schemas import PhonologicalVector

# ---- _parse_phon_vector unit tests ----------------------------------------


def test_parse_returns_none_for_missing():
    assert _parse_phon_vector(None) is None


def test_parse_returns_none_for_non_dict_non_string():
    assert _parse_phon_vector(42) is None
    assert _parse_phon_vector([0.5, 0.3]) is None


def test_parse_parses_full_dict_shape():
    raw = {
        "cluster_density": 0.4,
        "final_fortition": 1.0,
        "final_cluster_rate": 0.5,
        "vowel_final_bias": 0.0,
        "soft_consonants": 0.6,
        "polysyllabic_bias": 0.3,
        "palatalization": 0.0,
        "sibilance": 0.2,
        "retroflexion": 0.0,
        "pharyngeal": 0.0,
        "vowel_height": -0.3,
        "vowel_backness": 0.5,
        "stop_vs_continuant": 0.0,
        "aspirated_voiceless": 0.0,
    }
    v = _parse_phon_vector(raw)
    assert isinstance(v, PhonologicalVector)
    assert v.cluster_density == 0.4
    assert v.final_fortition == 1.0
    assert v.soft_consonants == 0.6


def test_parse_parses_json_string():
    """A bundle that embeds the etymon.phonological_vector column
    verbatim hands the loader a string blob, not a dict. Loader
    json.loads it before constructing the dataclass."""
    raw = '{"cluster_density": 0.4, "final_fortition": 1.0, "vowel_final_bias": 0.0}'
    v = _parse_phon_vector(raw)
    assert isinstance(v, PhonologicalVector)
    assert v.cluster_density == 0.4


def test_parse_handles_malformed_json_string():
    """Bad JSON → None (graceful fallback; vector-scoring degrades)."""
    assert _parse_phon_vector("{not valid json") is None


def test_parse_handles_non_floatable_dimension_value():
    """Round-1 silent-failure-hunter caught: ``float("not-a-number")``
    raises ValueError. The 'Never raises' contract requires the
    whole-row fallback to None when any dimension's value can't be
    coerced."""
    raw = {"cluster_density": "not-a-number", "final_fortition": 0.5}
    assert _parse_phon_vector(raw) is None


def test_parse_handles_none_value_at_dimension():
    """Similar: ``float(None)`` raises TypeError. Fallback to None."""
    raw = {"cluster_density": None}
    assert _parse_phon_vector(raw) is None


def test_parse_handles_nested_dict_at_dimension():
    """``float({...})`` raises TypeError. Fallback to None."""
    raw = {"cluster_density": {"nested": 0.5}}
    assert _parse_phon_vector(raw) is None


def test_parse_handles_non_dict_extras_block():
    """``raw["extras"]`` may arrive as a non-dict (string, list, int) from
    a malformed bundle. The loader should treat it as absent rather
    than crashing or silently dropping other dims."""
    raw = {"cluster_density": 0.4, "extras": "not-a-dict"}
    v = _parse_phon_vector(raw)
    # Doesn't raise; the known dim still loads, extras drops cleanly.
    assert v is not None
    assert v.cluster_density == 0.4
    assert v.extras == {}


def test_parse_handles_non_floatable_extras_value():
    """A bad value INSIDE extras shouldn't crash — same fallback as
    a bad value in a named dim."""
    raw = {"cluster_density": 0.4, "extras": {"experimental": "nan-str"}}
    assert _parse_phon_vector(raw) is None


def test_parse_drops_unknown_dim_keys_to_extras():
    """Forward-compat: unknown top-level keys land in extras so the
    bundle can ship new dimensions before the schema's named-fields
    list catches up. dot() in PhonologicalVector reads extras for
    unknown weights."""
    raw = {
        "cluster_density": 0.4,
        "future_dim_v2": 0.7,  # Not in the known-dim-names set
    }
    v = _parse_phon_vector(raw)
    assert v.cluster_density == 0.4
    assert v.extras == {"future_dim_v2": 0.7}


def test_parse_preserves_explicit_extras_block():
    """An explicit extras block from the bundle (e.g. computed by a
    future enrichment pass and dumped as {extras: {...}}) round-trips."""
    raw = {
        "cluster_density": 0.4,
        "extras": {"experimental_dim": -0.3},
    }
    v = _parse_phon_vector(raw)
    assert v.extras["experimental_dim"] == -0.3


def test_parse_unifies_explicit_and_implicit_extras():
    """Both shapes coexist — the loader merges them so neither path
    drops data."""
    raw = {
        "cluster_density": 0.4,
        "another_future_dim": 0.5,  # implicit (unknown top-level)
        "extras": {"already_named_extra": 0.6},  # explicit
    }
    v = _parse_phon_vector(raw)
    assert v.extras == {"already_named_extra": 0.6, "another_future_dim": 0.5}


# ---- Meaning constructor + accessor ---------------------------------------


def test_meaning_phon_vector_defaults_to_none():
    """Legacy callers that don't pass phonological_vector see None
    on the field (back-compat with the pre-ecjp.5 Meaning ctor)."""
    m = Meaning(usage="wulf", tags=[], meanings=[], sources=[])
    assert m.phonological_vector is None


def test_meaning_phon_vector_round_trips():
    v = PhonologicalVector(cluster_density=0.5, soft_consonants=0.3)
    m = Meaning(
        usage="wulf",
        tags=[],
        meanings=[],
        sources=[],
        phonological_vector=v,
    )
    assert m.phonological_vector is v
    assert m.phonological_vector.cluster_density == 0.5


# ---- load_meanings integration -------------------------------------------


def test_load_meanings_reads_phon_vector_from_bundle():
    """Bundle subject -> word -> phonological_vector field round-trips
    into the Meaning instance. Validates the wyrd-ecjp.5 / wyrd-kq7w.1
    bundle-export wiring."""
    bundle = [
        {
            "meaning": ["wolf"],
            "modifier_tags": ["animal"],
            "modifier_type": "Animal",
            "words": [
                {
                    "modern_usage": "wolf-",
                    "old_english": ["wulf"],
                    "phonological_vector": {
                        "cluster_density": 0.0,
                        "final_fortition": 1.0,
                        "soft_consonants": 0.0,
                        "vowel_final_bias": 0.0,
                    },
                },
            ],
        },
    ]
    meaning_db, _tags_db = load_meanings(bundle)
    meanings = meaning_db["wolf-"]
    assert len(meanings) == 1
    m = meanings[0]
    assert m.phonological_vector is not None
    assert m.phonological_vector.final_fortition == 1.0
    assert m.phonological_vector.cluster_density == 0.0


def test_load_meanings_handles_missing_phon_vector():
    """Legacy bundle (no phonological_vector field) loads with the
    field at None — no crash, no surprise default."""
    bundle = [
        {
            "meaning": ["wolf"],
            "modifier_tags": ["animal"],
            "modifier_type": "Animal",
            "words": [
                {
                    "modern_usage": "wolf-",
                    "old_english": ["wulf"],
                    # no phonological_vector
                },
            ],
        },
    ]
    meaning_db, _tags_db = load_meanings(bundle)
    m = meaning_db["wolf-"][0]
    assert m.phonological_vector is None


def test_load_meanings_handles_phon_vector_as_string_blob():
    """If the bundle embeds the column verbatim (raw JSON string), the
    loader json.loads + constructs cleanly. Defensive against bundle
    builders that don't deserialize before emit."""
    bundle = [
        {
            "meaning": ["wolf"],
            "modifier_tags": ["animal"],
            "modifier_type": "Animal",
            "words": [
                {
                    "modern_usage": "wolf-",
                    "old_english": ["wulf"],
                    "phonological_vector": '{"final_fortition": 1.0}',
                },
            ],
        },
    ]
    meaning_db, _tags_db = load_meanings(bundle)
    m = meaning_db["wolf-"][0]
    assert m.phonological_vector is not None
    assert m.phonological_vector.final_fortition == 1.0
