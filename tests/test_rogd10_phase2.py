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


def test_merge_genuine_reflexes_win_over_constructed_self_seeds():
    """wyrd-5olv: a morpheme_id groups every usage CONSTRUCTED with the
    morpheme as a head (every -ton town carries morpheme_id=old-english:tūn),
    and each self-seeds its OWN surface (source='self'). The merged morpheme
    must show the morpheme's ATTESTED spellings, not the constructed towns —
    so when a language has any genuine (non-self) reflex, the self-seeds for
    that language are dropped."""
    core = _m("-ton", "old-english:tūn", {"modern-english": [("-ton", "descent"), ("town", "cluster")]})
    bolton = _m(
        "-bolton",
        "old-english:tūn",
        {"modern-english": [("-bolton", "self"), ("-ton", "descent"), ("town", "cluster")]},
    )
    newton = _m(
        "-newton",
        "old-english:tūn",
        {"modern-english": [("-newton", "self"), ("-ton", "descent"), ("town", "cluster")]},
    )
    merged = _merge_morpheme_meanings([core, bolton, newton])
    modern = merged.era_reflex_for("modern-english")
    assert set(modern) == {"-ton", "town"}  # attested spellings only
    assert "-bolton" not in modern and "-newton" not in modern  # constructed towns dropped


def test_merge_keeps_self_seeds_when_no_genuine_reflex_preserves_ling():
    """wyrd-5olv guardrail: an all-self connective morpheme (no cluster/descent
    attestation, e.g. -ing) must NOT be stripped — its self-seeded variants are
    the morpheme's only surfaces. In particular -ling must survive. This is why
    the fix is genuine-wins-with-fallback, NOT a compound-detector (which can't
    tell the real suffix -ling from the constructed town -bolton)."""
    ing = _m("-ing", "old-english:ing", {"modern-english": [("-ing", "self"), ("-inge", "self")]})
    ling = _m("-ling", "old-english:ing", {"modern-english": [("-ling", "self")]})
    merged = _merge_morpheme_meanings([ing, ling])
    modern = merged.era_reflex_for("modern-english")
    assert "-ling" in modern  # the guardrail
    assert set(modern) == {"-ing", "-inge", "-ling"}


def test_merge_genuine_wins_is_per_language():
    """wyrd-5olv: the genuine-wins choice is independent per language. A
    language with only self-seeds keeps them even when a sibling language has
    genuine reflexes (so a morpheme isn't blanked in an era it's only
    self-attested in)."""
    a = _m(
        "-ton",
        "old-english:tūn",
        {"modern-english": [("town", "cluster")], "old-english": [("-ton", "self")]},
    )
    b = _m(
        "-bolton",
        "old-english:tūn",
        {"modern-english": [("-bolton", "self"), ("town", "cluster")]},
    )
    merged = _merge_morpheme_meanings([a, b])
    assert set(merged.era_reflex_for("modern-english")) == {"town"}  # genuine wins
    assert merged.era_reflex_for("old-english") == ["-ton"]  # self kept (no genuine there)


def test_merge_lone_meaning_keeps_its_self_seed():
    """wyrd-5olv: a single-usage morpheme is its own merge — its self-seed is
    its own surface, never a constructed-compound artifact, so it's kept
    verbatim (the genuine-wins filter only fires across a real merge)."""
    only = _m("-holme", "old-norse:holmr", {"old-norse": [("-holme", "self"), ("helm", "self")]})
    merged = _merge_morpheme_meanings([only])
    assert merged is only
    assert merged.era_reflex_for("old-norse") == ["-holme", "helm"]


def test_resolve_morpheme_lookup_and_fallback():
    a = _m("-ing", "old-english:ing", {"old-english": [("ing", "cluster")]})
    meaning_db = {"-ing": [a]}
    hit = _resolve_morpheme(meaning_db, "old-english:ing")
    assert hit.era_reflex_for("old-english") == ["ing"]
    # unset / unknown → None, so the caller falls back to the per-surface Meaning
    assert _resolve_morpheme(meaning_db, None) is None
    assert _resolve_morpheme(meaning_db, "old-english:missing") is None
