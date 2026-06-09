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
    _era_grid,
    _merge_morpheme_meanings,
    _resolve_morpheme,
)


def _modern_cell(grid, lang="modern-english"):
    for fam in grid:
        for stage in fam["stages"]:
            if stage["language"] == lang:
                return [c["form"] for c in stage["forms"]]
    return []


def test_era_grid_narrows_present_day_to_own_surface():
    """wyrd-phww: with the merge unioning self-seeds, _era_grid(own_surface=...)
    narrows the PRESENT-DAY stage (modern-english) to genuine reflexes + only the
    self-seed matching the name's own surface — so the generated/own surface
    shows ('-ton') without the constructed-town flood ('-bolton' dropped). The
    own-form self-seed in a NON-present-day stage (OE 'tūn') is untouched."""
    m = _m(
        "-ton",
        "old-english:tūn",
        {
            "modern-english": [("town", "cluster"), ("-ton", "self"), ("-bolton", "self")],
            "old-english": [("tūn", "self")],
        },
    )
    grid = _era_grid(m, {}, own_surface="-ton")
    assert set(_modern_cell(grid)) == {"town", "-ton"}  # genuine + matching self only
    assert "-bolton" not in _modern_cell(grid)  # non-matching self-seed dropped
    assert _modern_cell(grid, "old-english") == ["tūn"]  # OE own-form self-seed survives


def test_era_grid_none_keeps_full_union():
    """wyrd-phww: own_surface=None (no narrowing) keeps every self-seed — the
    full union, used when no specific surface applies."""
    m = _m(
        "-ton",
        "old-english:tūn",
        {"modern-english": [("town", "cluster"), ("-ton", "self"), ("-bolton", "self")]},
    )
    assert set(_modern_cell(_era_grid(m, {}, own_surface=None))) == {"town", "-ton", "-bolton"}


def test_era_grid_own_surface_matches_case_and_dashes_insensitively():
    """wyrd-phww: the own-surface match ignores case + affix dashes, so the
    bundle's 'Park-' self-seed matches a generated surface of 'park'."""
    m = _m(
        "park",
        "old-english:pearroc",
        {"modern-english": [("parrock", "cluster"), ("Park-", "self"), ("Paddock-", "self")]},
    )
    cell = _modern_cell(_era_grid(m, {}, own_surface="park"))
    assert set(cell) == {"parrock", "Park-"}  # 'park' folds to 'Park-'; 'Paddock-' dropped


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


def test_merge_unions_self_seeds_with_genuine_reflexes():
    """wyrd-phww (supersedes the wyrd-5olv genuine-WINS rule): the merge now
    UNIONS self-seeds with genuine reflexes (genuine first, source preserved).
    Self-seeds are no longer dropped HERE — instead _era_grid narrows the
    present-day stage to the name's OWN surface at render time (see
    test_era_grid_narrows_present_day_to_own_surface), so the generated/own
    surface stays available to the grid while the ~120-constructed-town flood
    is still avoided per-name."""
    core = _m(
        "-ton", "old-english:tūn", {"modern-english": [("-ton", "descent"), ("town", "cluster")]}
    )
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
    assert set(modern) == {"-ton", "town", "-bolton", "-newton"}  # union, nothing dropped
    assert modern[:2] == ["-ton", "town"]  # genuine listed first (deterministic)


def test_merge_keeps_self_seeds_when_no_genuine_reflex_preserves_ling():
    """An all-self connective morpheme (no cluster/descent attestation, e.g.
    -ing) keeps its self-seeded variants — in particular -ling must survive.
    Held under both the old genuine-wins-with-fallback rule and the wyrd-phww
    union (which keeps self-seeds unconditionally)."""
    ing = _m("-ing", "old-english:ing", {"modern-english": [("-ing", "self"), ("-inge", "self")]})
    ling = _m("-ling", "old-english:ing", {"modern-english": [("-ling", "self")]})
    merged = _merge_morpheme_meanings([ing, ling])
    modern = merged.era_reflex_for("modern-english")
    assert "-ling" in modern  # the guardrail
    assert set(modern) == {"-ing", "-inge", "-ling"}


def test_merge_unions_per_language():
    """wyrd-phww: the union is independent per language — each language keeps its
    genuine reflexes plus its self-seeds (a self-only language is unaffected)."""
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
    assert set(merged.era_reflex_for("modern-english")) == {"town", "-bolton"}  # genuine + self
    assert merged.era_reflex_for("old-english") == ["-ton"]  # self-only language kept


def test_merge_keeps_glosses_for_unioned_self_seed_forms():
    """wyrd-phww: the union keeps every form, so glosses stay in lockstep —
    both the genuine form's gloss and a self-seed's gloss survive (they're only
    narrowed later, per-name, at the grid)."""
    core = _m(
        "-ton",
        "old-english:tūn",
        {"modern-english": [("town", "cluster")]},
        {"modern-english": {"town": "settlement"}},
    )
    bolton = _m(
        "-bolton",
        "old-english:tūn",
        {"modern-english": [("-bolton", "self"), ("town", "cluster")]},
        {"modern-english": {"-bolton": "Bolton (a place)"}},
    )
    merged = _merge_morpheme_meanings([core, bolton])
    assert set(merged.era_reflex_for("modern-english")) == {"town", "-bolton"}
    assert merged.era_reflex_gloss_for("modern-english") == {
        "town": "settlement",
        "-bolton": "Bolton (a place)",
    }


def test_merge_genuine_source_wins_for_duplicate_form():
    """wyrd-5olv: when a form appears as a self-seed in one usage and a genuine
    reflex in another, the genuine entry wins — the surviving form keeps its
    attested source, not 'self'."""
    a = _m("-ton", "old-english:tūn", {"modern-english": [("-ton", "self")]})
    b = _m("-newton", "old-english:tūn", {"modern-english": [("-ton", "descent")]})
    merged = _merge_morpheme_meanings([a, b])
    assert merged.era_reflex_for("modern-english") == ["-ton"]
    assert merged.era_reflex_sources_for("modern-english") == {"-ton": "descent"}


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
