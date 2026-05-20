"""Bundle-export + load round-trip tests for per-language
phonological_vector siblings (wyrd-ecjp.10a Phase 7).

Each etymon row in the lexicon DB carries a JSON phonological_vector
blob (kq7w.1 corpus enrichment). The bundle now emits these as
``<lang>_phonological_vector`` siblings per the D26 pattern. The
runtime loader parses them into Meaning.phonological_vectors (plural,
dict[lang_field, dict[canonical_form, PhonologicalVector]]) and
populates Meaning.phonological_vector (singular) with the first
non-empty entry for back-compat.

These tests pin the emit + load contract using synthetic fixtures —
no live LexiconDB needed.
"""

from __future__ import annotations

import json

from wyrd.generators.kenning.lexicon.bundle._emit import (
    _BucketAccumulator,
    _emit_phonological_vector_list,
)
from wyrd.generators.kenning.lexicon.bundle._subject import (
    _absorb_member_phonological_vector,
    _WordLanguageAccumulators,
)
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.vectors.schemas import PhonologicalVector

# ---- emit-side: BucketAccumulator + formatter -----------------------------


def test_bucket_accumulator_has_phonological_vector_field():
    bucket = _BucketAccumulator()
    assert hasattr(bucket, "phonological_vector")
    assert bucket.phonological_vector == {}


def test_emit_phonological_vector_list_sorts_by_form():
    vectors = {
        "ceaster": {"cluster_density": 0.7, "final_fortition": 0.5},
        "burh": {"cluster_density": 0.4},
    }
    out = _emit_phonological_vector_list(vectors)
    assert out == [
        {"form": "burh", "phonological_vector": {"cluster_density": 0.4}},
        {
            "form": "ceaster",
            "phonological_vector": {"cluster_density": 0.7, "final_fortition": 0.5},
        },
    ]


def test_emit_phonological_vector_list_empty():
    assert _emit_phonological_vector_list({}) == []


# ---- absorb-side: walks the member dict ----------------------------------


def test_absorb_member_phonological_vector_parses_json_blob():
    accs = _WordLanguageAccumulators()
    fam = {
        "member_phonological_vector_by_id": {
            42: json.dumps({"cluster_density": 0.8, "final_fortition": 0.3}),
        }
    }
    _absorb_member_phonological_vector(accs, fam, 42, "old-english", "ceaster")
    assert accs.phonological_vector == {
        "old-english": {"ceaster": {"cluster_density": 0.8, "final_fortition": 0.3}},
    }


def test_absorb_member_phonological_vector_skips_null_blob():
    accs = _WordLanguageAccumulators()
    fam = {"member_phonological_vector_by_id": {42: None}}
    _absorb_member_phonological_vector(accs, fam, 42, "old-english", "ceaster")
    assert accs.phonological_vector == {}


def test_absorb_member_phonological_vector_skips_member_not_in_map():
    """When member_phonological_vector_by_id has no entry for member_id
    (kq7w.1 enrichment hasn't reached this row), the absorb-helper is
    a no-op. Empty dict already, so this is the legacy DB / pre-
    enrichment case."""
    accs = _WordLanguageAccumulators()
    fam = {"member_phonological_vector_by_id": {}}
    _absorb_member_phonological_vector(accs, fam, 42, "old-english", "ceaster")
    assert accs.phonological_vector == {}


def test_absorb_member_phonological_vector_handles_malformed_json():
    """Per-row exception isolation: a malformed JSON blob (interrupted
    enrichment run, manual hand-edit gone wrong) is silently dropped
    rather than aborting the whole bundle export."""
    accs = _WordLanguageAccumulators()
    fam = {"member_phonological_vector_by_id": {42: "{not valid json"}}
    _absorb_member_phonological_vector(accs, fam, 42, "old-english", "ceaster")
    assert accs.phonological_vector == {}


def test_absorb_member_phonological_vector_handles_non_dict_json():
    """A JSON blob that parses to a non-dict (string, list, number)
    is dropped rather than crashing the emit step. The runtime
    expects dict shape."""
    accs = _WordLanguageAccumulators()
    fam = {"member_phonological_vector_by_id": {42: json.dumps([1, 2, 3])}}
    _absorb_member_phonological_vector(accs, fam, 42, "old-english", "ceaster")
    assert accs.phonological_vector == {}


# ---- load-side: load_meanings parses per-language siblings ---------------


def _bundle_with_word(word: dict) -> dict:
    """Minimal bundle JSON wrapping a single subject + word for the
    load_meanings round-trip."""
    return {
        "subjects": [
            {
                "meaning": ["Town"],
                "modifier_tags": ["settlement"],
                "modifier_type": "Topographical",
                "words": [word],
            }
        ]
    }


def test_load_meanings_parses_per_language_phon_vectors():
    bundle = _bundle_with_word(
        {
            "modern_usage": "-ceaster",
            "old_english": ["ceaster"],
            "old_english_phonological_vector": [
                {"form": "ceaster", "phonological_vector": {"cluster_density": 0.7}},
            ],
        }
    )
    meaning_db, _tags_db = load_meanings(bundle)
    meaning = meaning_db["-ceaster"][0]
    assert "old_english" in meaning.phonological_vectors
    assert "ceaster" in meaning.phonological_vectors["old_english"]
    assert meaning.phonological_vectors["old_english"]["ceaster"] == PhonologicalVector(
        cluster_density=0.7
    )


def test_load_meanings_populates_singular_phon_vector_from_first_nonempty():
    """Back-compat: Meaning.phonological_vector (singular) is populated
    from the first non-empty entry in the deterministic lang × form
    walk. Callers reading the singular field continue to get a vector
    even though the bundle now emits per-language siblings."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-ceaster",
            "old_english": ["ceaster"],
            "old_english_phonological_vector": [
                {"form": "ceaster", "phonological_vector": {"cluster_density": 0.7}},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-ceaster"][0]
    assert meaning.phonological_vector == PhonologicalVector(cluster_density=0.7)


def test_load_meanings_multi_lang_picks_first_lang_first_form_alphabetically():
    """When multiple languages carry vectors, the singular picks the
    first language alphabetically (deterministic). Same rule for
    multiple forms within a language."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["xeester"],
            "old_english_phonological_vector": [
                {"form": "xeester", "phonological_vector": {"cluster_density": 0.9}},
            ],
            "old_scandinavian": ["abc"],
            "old_scandinavian_phonological_vector": [
                {"form": "abc", "phonological_vector": {"cluster_density": 0.1}},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    # old_english < old_scandinavian alphabetically → old_english wins
    assert meaning.phonological_vector == PhonologicalVector(cluster_density=0.9)


def test_load_meanings_legacy_top_level_vector_preserved():
    """Bundles emitted before wyrd-ecjp.10a used a top-level
    'phonological_vector' key. We still read it (back-compat). When
    BOTH legacy and new siblings are present, legacy wins for the
    singular (deterministic dispatch precedence)."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["bar"],
            "phonological_vector": {"cluster_density": 0.5},
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    assert meaning.phonological_vector == PhonologicalVector(cluster_density=0.5)
    # No per-language siblings → phonological_vectors empty
    assert meaning.phonological_vectors == {}


def test_load_meanings_no_phon_vector_anywhere():
    """Legacy bundle with no phon-vector data: Meaning.phonological_vector
    is None, Meaning.phonological_vectors is {}."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["bar"],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    assert meaning.phonological_vector is None
    assert meaning.phonological_vectors == {}


def test_load_meanings_phon_vector_suffix_excluded_from_sources():
    """The <lang>_phonological_vector sibling must NOT show up in
    Meaning.sources (which holds per-language form arrays). Regression
    guard against the field-exclusion filter."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["bar"],
            "old_english_phonological_vector": [
                {"form": "bar", "phonological_vector": {"cluster_density": 0.5}},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    assert "old_english_phonological_vector" not in meaning.sources
    # old_english itself is still in sources (the form array)
    assert "old_english" in meaning.sources


def test_load_meanings_non_dict_entry_silently_skipped():
    """A bundle could in principle ship a non-dict entry in the
    sibling list (operator hand-edit gone wrong, third-party tooling).
    The load path skips rather than raising AttributeError —
    'best-effort parse' contract extends to surrounding entry shape."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["good"],
            "old_english_phonological_vector": [
                "not a dict",  # entry must be dict
                {"form": "good", "phonological_vector": {"cluster_density": 0.5}},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    # The valid entry still landed; the non-dict was silently dropped
    assert meaning.phonological_vectors["old_english"]["good"] == PhonologicalVector(
        cluster_density=0.5
    )


def test_load_meanings_entry_missing_form_key_silently_skipped():
    """Similar: an entry-shaped dict missing the 'form' key would
    KeyError on the per_form[entry['form']] assignment. Skip it."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["good"],
            "old_english_phonological_vector": [
                {"phonological_vector": {"cluster_density": 0.9}},  # missing form
                {"form": "good", "phonological_vector": {"cluster_density": 0.5}},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    assert meaning.phonological_vectors["old_english"]["good"] == PhonologicalVector(
        cluster_density=0.5
    )


def test_load_meanings_malformed_phon_vector_entry_silently_skipped():
    """A malformed sibling entry (non-dict phon_vector value, missing
    form key on entry, etc.) shouldn't crash load_meanings. Drop to
    None for that entry; other valid entries still land."""
    bundle = _bundle_with_word(
        {
            "modern_usage": "-foo",
            "old_english": ["good", "bad"],
            "old_english_phonological_vector": [
                {"form": "good", "phonological_vector": {"cluster_density": 0.5}},
                # Malformed: phon_vector value is a string, not dict
                {"form": "bad", "phonological_vector": "not a dict"},
            ],
        }
    )
    meaning_db, _ = load_meanings(bundle)
    meaning = meaning_db["-foo"][0]
    # Good entry parsed
    assert meaning.phonological_vectors["old_english"]["good"] == PhonologicalVector(
        cluster_density=0.5
    )
    # Bad entry dropped (would've crashed without the _parse_phon_vector
    # guard's "best-effort" contract)
    assert "bad" not in meaning.phonological_vectors["old_english"]
