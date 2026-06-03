"""wyrd-rogd.10 Phase 2 — runtime resolves the era-grid via the morpheme id.

The era-grid is resolved against the UNIFIED morpheme (all connective-form
fragments sharing a morpheme_id), not the per-surface Meaning. On today's data
this is byte-identical (the bundle already stamps each fragment with the family
root's era_reflexes), so these tests pin the MERGE behavior directly — the union
that becomes visible once rogd.9 enriches a morpheme's lineage span.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import (
    _merge_morpheme_meanings,
    _resolve_morpheme,
)


def _m(usage: str, morpheme_id, era_reflexes=None, glosses=None) -> Meaning:
    return Meaning(
        usage,
        [],
        [],
        {},
        era_reflexes=era_reflexes or {},
        era_reflex_glosses=glosses or {},
        morpheme_id=morpheme_id,
    )


def test_meaning_carries_morpheme_id():
    assert _m("-ing", "old-english:ing").morpheme_id == "old-english:ing"
    assert _m("-ing", None).morpheme_id is None


def test_merge_unions_era_reflexes_across_fragments():
    # Two fragments of one morpheme with DIFFERENT reflexes → merged carries the
    # union (this is the unfragmenting payoff rogd.9 will surface).
    a = _m("-ing", "old-english:ing", {"old-english": [("ing", "cluster")]})
    b = _m("-ing-", "old-english:ing", {"middle-english": [("inge", "descent")]})
    merged = _merge_morpheme_meanings([a, b])
    assert set(merged.era_reflexes) == {"old-english", "middle-english"}
    assert merged.era_reflex_for("old-english") == ["ing"]
    assert merged.era_reflex_for("middle-english") == ["inge"]


def test_merge_unions_glosses_and_dedupes_forms_first_seen():
    a = _m(
        "-ing",
        "old-english:ing",
        {"old-english": [("ing", "cluster")]},
        {"old-english": {"ing": "meadow"}},
    )
    b = _m(
        "Ing-",  # sorts after "-ing"; its duplicate "ing" form must NOT override
        "old-english:ing",
        {"old-english": [("ing", "descent")]},
        {"old-english": {"ing": "people-of"}},
    )
    merged = _merge_morpheme_meanings([a, b])
    assert merged.era_reflex_for("old-english") == ["ing"]  # deduped
    # first-seen (usage-sorted: "-ing" before "Ing-") wins the gloss
    assert merged.era_reflex_gloss_for("old-english") == {"ing": "meadow"}


def test_merge_returns_base_unchanged_when_no_gain():
    a = _m("-ing", "old-english:ing", {"old-english": [("ing", "cluster")]})
    b = _m("-ing-", "old-english:ing", {"old-english": [("ing", "cluster")]})
    merged = _merge_morpheme_meanings([a, b])
    assert merged is a or merged is b  # the unchanged base, no needless copy
    assert merged.era_reflex_for("old-english") == ["ing"]


def test_resolve_morpheme_lookup_and_fallback():
    a = _m("-ing", "old-english:ing", {"old-english": [("ing", "cluster")]})
    meaning_db = {"-ing": [a]}
    hit = _resolve_morpheme(meaning_db, "old-english:ing")
    assert hit.era_reflex_for("old-english") == ["ing"]
    # unset / unknown → None, so the caller falls back to the per-surface Meaning
    assert _resolve_morpheme(meaning_db, None) is None
    assert _resolve_morpheme(meaning_db, "old-english:missing") is None
