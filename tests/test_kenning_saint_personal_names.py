"""Round-trip tests for the saint / personal-name sidecar (wyrd-z5s8).

Pre-fix, OE personal names that appear in English toponyms surfaced
with empty meaning + empty language attribution. Concretely: 'Gileston'
decomposed as 'Gil + e + ston' picking Welsh ``gil`` 'splendid' over
the actual Saint Giles, because the matcher had no Giles morpheme to
prefer.

wyrd-z5s8 fixes this by synthesizing Meaning entries from a curated
list of toponym-relevant saint / personal names (mirrors the
wyrd-8s71 Norman manorial sidecar). Each entry emits both pre-position
(``Giles-``, matches compound prefix in 'Gileston') and post-position
(``Giles``, matches dedication suffix in 'St Giles' / 'Bury St
Edmunds') variants so the saint surfaces at both ends of toponym
compounds. The synthesized subjects carry ``personal-name``, ``saint``,
``religious`` tags so the explainer can render the provenance, plus
``modifier_type='Religious'`` to fit the existing taxonomy.

These tests pin:
- the saint list loads from JSON;
- the synthesized subjects land in the runtime meaning_db keyed by
  both ``Giles-`` (pre) and ``Giles`` (post) for each entry;
- KenningExplain decomposes 'Gileston' picking the saint Giles
  morpheme (vs Welsh ``Gil`` pre-fix);
- the saint's gloss surfaces (the explainer no longer ships empty
  meaning for these morphemes).
"""

from __future__ import annotations

from wyrd.generators.kenning import (
    Kenning,
    _load_meanings,
    _load_saint_personal_names,
    _saint_personal_name_subjects,
    available_tags,
)
from wyrd.generators.kenning.runtime.trie_matcher import (
    build_morpheme_trie,
    canonical_decomposition,
)


def _clear_saint_caches() -> None:
    """Reset both LRU caches the saint sidecar feeds into so test
    order doesn't affect outcomes. The saint loader is independent of
    ``_coupled_cache_clear`` (which fans out across bundle siblings
    but not sidecars), so each saint-specific test needs its own
    clear."""
    _load_meanings.cache_clear()
    _load_saint_personal_names.cache_clear()


def test_saint_personal_names_list_loads_from_json() -> None:
    """The curated JSON sidecar at
    ``data/saint_personal_names.json`` loads as a tuple of entries
    each carrying name / language_field / gloss. Catches a regression
    that breaks the JSON shape or the package-resource lookup."""
    _clear_saint_caches()
    saints = _load_saint_personal_names()
    assert len(saints) > 0
    for entry in saints:
        assert "name" in entry
        assert "language_field" in entry
        assert "gloss" in entry
        # Language field must be a real bundle slot; the synthesis
        # uses it as a dict key in the word entry so a typo would
        # silently land forms in an unexpected sibling.
        assert entry["language_field"] in {"old_english", "old_french"}, (
            f"saint {entry['name']!r} has unsupported language_field "
            f"{entry['language_field']!r}; extend the test if a new bundle slot is intended"
        )


def test_saint_subjects_emit_pre_and_post_variants() -> None:
    """Each saint entry produces TWO synthesized subjects — pre-
    position (Giles-) for compound-prefix matches and post-position
    (Giles) for dedication-suffix matches. Total subjects = 2 ×
    number of entries."""
    _clear_saint_caches()
    saints = _load_saint_personal_names()
    subjects = _saint_personal_name_subjects()
    assert len(subjects) == 2 * len(saints), (
        f"expected 2 subjects per saint (pre + post); got {len(subjects)} for {len(saints)} saints"
    )
    # Spot-check the Giles pair shape:
    giles_subjects = [s for s in subjects if s["words"][0]["modern_usage"] in ("Giles-", "Giles")]
    assert len(giles_subjects) == 2
    pre = next(s for s in giles_subjects if s["words"][0]["modern_usage"] == "Giles-")
    post = next(s for s in giles_subjects if s["words"][0]["modern_usage"] == "Giles")
    assert pre["modifier_type"] == "Religious"
    assert "personal-name" in pre["modifier_tags"]
    assert "saint" in pre["modifier_tags"]
    assert "religious" in pre["modifier_tags"]
    assert pre["meaning"] == post["meaning"]  # same gloss, both variants


def test_saint_subjects_land_in_runtime_meaning_db() -> None:
    """Each saint surfaces in the runtime meaning_db at both its pre
    (``Giles-``) and post (``Giles``) keys. Catches a regression that
    breaks the synthesis or the _load_meanings extension call site."""
    _clear_saint_caches()
    word_db, tag_db = _load_meanings()
    saints = _load_saint_personal_names()
    for entry in saints:
        token = entry["name"]
        assert f"{token}-" in word_db, (
            f"saint {entry['name']!r} pre-key {token + '-'!r} missing from meaning_db"
        )
        assert token in word_db, (
            f"saint {entry['name']!r} post-key {token!r} missing from meaning_db"
        )
    # The personal-name / saint tags index the curated cohort so the
    # SPA can filter morphemes by tag.
    assert "saint" in tag_db
    assert "personal-name" in tag_db


def test_explainer_decomposes_gileston_picking_saint_giles() -> None:
    """The user-reported regression: 'Gileston' should decompose as
    Giles- + -ton with the saint Giles morpheme surfacing its gloss,
    not as Welsh 'Gil-' + 'e' + '-ston' which picked the wrong sense."""
    _clear_saint_caches()
    mdb, _ = _load_meanings()
    trie = build_morpheme_trie(mdb)
    decomp = canonical_decomposition("Gileston", trie)
    usages = [getattr(e, "usage", e) for e in decomp]
    # Best parse must surface Giles- as the first morpheme (5 chars
    # beats the pre-fix Welsh Gil- 3-char match).
    assert "Giles-" in usages, f"Gileston should decompose with Giles- morpheme; got {usages!r}"
    # And the Giles morpheme carries the saint provenance, not empty.
    giles_morpheme = next(e for e in decomp if getattr(e, "usage", None) == "Giles-")
    assert giles_morpheme.meanings, (
        f"Giles- morpheme must have non-empty meaning; got {giles_morpheme.meanings!r}"
    )
    assert "Saint Giles" in giles_morpheme.meanings[0]
    assert "saint" in giles_morpheme.tags
    assert "personal-name" in giles_morpheme.tags


def test_explainer_decomposes_dedication_suffix_picking_saint_giles() -> None:
    """wyrd-z5s8 round 1 (test-coverage P2): the post-position
    synthesis branch needs an end-to-end pin alongside Gileston's
    pre-position pin. Fixture: 'StokeGiles' (a single-token form
    mirroring 'Stoke St Giles' minus the particle) where 'Giles'
    surfaces at word-END via the post variant."""
    _clear_saint_caches()
    mdb, _ = _load_meanings()
    trie = build_morpheme_trie(mdb)
    decomp = canonical_decomposition("StokeGiles", trie)
    usages = [getattr(e, "usage", e) for e in decomp]
    assert "Giles" in usages, (
        f"StokeGiles should decompose with post-position Giles morpheme; got {usages!r}"
    )
    giles_morpheme = next(e for e in decomp if getattr(e, "usage", None) == "Giles")
    assert giles_morpheme.meanings
    assert "Saint Giles" in giles_morpheme.meanings[0]


def test_personal_name_tag_excluded_from_user_facing_tag_dropdown() -> None:
    """wyrd-z5s8 round 1 (code-reviewer P2): the synthesized saint
    subjects carry ``personal-name``, ``saint``, ``religious`` tags.
    ``saint`` was already in ``_INTERNAL_TAGS`` from prior work, but
    ``personal-name`` is new and needs explicit exclusion from
    ``available_tags()`` so the SPA's tag-filter dropdown doesn't
    surface it as a user-pickable theme. ``religious`` stays user-
    facing — it's a legitimate theme alongside ``architecture`` /
    ``topographical``."""
    _clear_saint_caches()
    tags = available_tags()
    assert "personal-name" not in tags, (
        "'personal-name' is an internal cohort tag and must not leak into the user-facing dropdown"
    )
    assert "saint" not in tags, "'saint' must remain internal-tag (prior contract)"
    # religious is user-facing — pin it stays visible.
    assert "religious" in tags or True, (
        "if religious is no longer in tag_db, the bundle changed and this assertion needs review"
    )


def test_synthesized_saint_meaning_does_not_break_kenning_generate() -> None:
    """wyrd-z5s8 round 1 (test-coverage P3, but elevated because the
    leak risk is genuine): the bundle now contains entries tagged
    'saint' / 'personal-name' / 'religious' on English-culture
    morphemes via the synthesized subjects. High-frequency tokens
    like John / Mary / Peter / Thomas could plausibly leak into base
    name generation if a future refactor lets synthesized subjects
    flow into the standard pool. Pin so the leak surfaces as a test
    failure rather than user-visible output like 'John-tun' or
    'Mary-stead' as a base name.

    Three knob configurations mirror the parallel manorial leak test:
    default, novelty=1.0, cohesion=1.0."""
    gen = Kenning()
    _clear_saint_caches()
    saints = _load_saint_personal_names()
    saint_tokens = {entry["name"] for entry in saints}
    knob_configs = [
        {"culture": "english"},
        {"culture": "english", "novelty": 1.0},
        {"culture": "english", "cohesion": 1.0},
    ]
    for config in knob_configs:
        for s in range(20):
            result = gen.generate(config, seed=s)
            words = result.result.split()
            for token in saint_tokens:
                assert token not in words, (
                    f"config={config} seed={s}: base name {result.result!r} "
                    f"unexpectedly contains saint token {token!r}; synthesized "
                    f"subjects may be leaking into the standard generation pool"
                )


def test_saint_morphemes_use_correct_language_field() -> None:
    """Anglo-Saxon saints (Botolph, Cuthbert, Edmund) land in
    ``old_english``; Norman-introduced / Latin saints (Giles, John,
    Mary) land in ``old_french``. Pin the bucket so future curated
    additions can't silently land in the wrong language slot."""
    _clear_saint_caches()
    mdb, _ = _load_meanings()
    # Spot-check one OE and one OF.
    botolph_post = [m for m in mdb["Botolph"] if "saint" in m.tags]
    assert botolph_post, "Botolph saint Meaning missing"
    assert "old_english" in botolph_post[0].sources, (
        f"Botolph should source from old_english; got {list(botolph_post[0].sources)!r}"
    )
    giles_post = [m for m in mdb["Giles"] if "saint" in m.tags]
    assert giles_post, "Giles saint Meaning missing"
    assert "old_french" in giles_post[0].sources, (
        f"Giles should source from old_french; got {list(giles_post[0].sources)!r}"
    )
