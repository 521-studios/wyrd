"""Round-trip tests for the Norman manorial-affix explainer (wyrd-8s71).

PR #98 (wyrd-obu) wired the ``manorial_affix`` knob so Kenning emits
post-base Anglo-Norman family surnames. The explainer (KenningExplain)
decomposes input strings against the same meaning_db, but pre-wyrd-8s71
the meaning_db carried no entries for the family surnames — so a name
the SPA had just generated decomposed only partially when fed back
("Stoke Mandeville" → "Stoke" + [Mandeville] unaccounted).

wyrd-8s71 fixes this by synthesizing Meaning entries for every Norman
family at load time. Multi-word families ("La Zouche", "De Vere") get
keyed on the surname token; the particle stays unaccounted by design
(registering "La" / "De" as standalone manorial particles would match
in non-manorial contexts and pollute every other decomposition).

These tests pin:
- the synthesized subjects exist in the runtime meaning_db with the
  correct shape (FR language slot, manorial+norman tags);
- KenningExplain decomposes single-word manorial names cleanly;
- KenningExplain decomposes multi-word manorial names with the
  surname-token recognized and the particle accepted as unaccounted;
- the explainer surfaces the full family string (including the
  particle) in the gloss, so a SPA reader sees "La Zouche" even when
  only "Zouche" matched.
"""

from __future__ import annotations

from wyrd.generators.kenning import (
    Kenning,
    _load_meanings,
    _load_norman_manorial_families,
)
from wyrd.registry import discover, get


def test_norman_manorial_subjects_landed_in_runtime_meaning_db() -> None:
    """Each family's surname token must surface as a key in the
    runtime meaning_db. Catches a regression that breaks the
    _norman_manorial_subjects synthesis or the _load_meanings
    extension call site."""
    _load_meanings.cache_clear()
    word_db, tag_db = _load_meanings()
    families = _load_norman_manorial_families()
    for family in families:
        token = family.split()[-1]
        assert token in word_db, (
            f"manorial family {family!r} (token {token!r}) missing from runtime meaning_db"
        )
    # The manorial+norman tag indexes are also load-bearing — the
    # explainer / SPA can filter morphemes by tag.
    assert "manorial" in tag_db
    assert "norman" in tag_db


def test_explainer_decomposes_single_word_manorial_affix() -> None:
    """Round-trip: a name Kenning generated with manorial_affix=1
    must decompose cleanly via KenningExplain. Pin the canonical
    Domesday-shape names so the trip-back doesn't leak the manorial
    affix as an unaccounted fragment."""
    discover()
    exp = get("kenning-explain")
    for name in ("Stoke Mandeville", "Castle Cary", "Newton Lacy"):
        results = exp.generate_all({"name": name}, seed=0)
        assert results, f"explainer returned no decompositions for {name}"
        # The first (best) decomposition must include the manorial token
        # in the explanation.
        family_token = name.split()[-1]
        first = results[0]
        assert "Norman manorial family" in first.explanation, (
            f"{name}: explanation missing manorial provenance: {first.explanation[:200]}"
        )
        assert family_token in first.explanation


def test_explainer_decomposes_multi_word_manorial_affix() -> None:
    """For multi-word families ("La Zouche"), the surname token
    decomposes via the synthesized Meaning, and the explainer's gloss
    surfaces the full "La Zouche" because the synthesized Meaning's
    gloss carries the full family string.

    The leading particle "La" now resolves to the Old-French definite
    article ``la`` — the converged corpus (post-f1ss-tail) gained that
    etymon, so the particle is no longer an unaccounted ``[La]``
    fragment. The original docstring anticipated exactly this: "a future
    refactor that ALSO registers the particles ... trips the test"; here
    it was the corpus growth, not a refactor, that registered ``la``.
    The manorial synthesis still carries the surname + full family gloss
    regardless of whether the particle accounts."""
    discover()
    exp = get("kenning-explain")
    results = exp.generate_all({"name": "Ashby La Zouche"}, seed=0)
    first = results[0]
    # Surname token recognized + full family name in the gloss.
    assert "Zouche" in first.explanation
    assert "La Zouche" in first.explanation, (
        f"explainer should surface full family name in gloss: {first.explanation[:300]}"
    )
    # Particle "La" now decomposes via the French definite article
    # ``la`` rather than staying an unaccounted ``[La]`` fragment.
    assert "la " in first.explanation.lower()
    assert "[La]" not in first.explanation


def test_generate_then_explain_round_trip_recognizes_affix() -> None:
    """Full round-trip: Kenning.generate with manorial_affix=1 emits
    a name; feeding the same name string into KenningExplain
    decomposes the affix cleanly. This is the primary user-visible
    contract — a name the SPA just generated should explain
    consistently when copy-pasted into the explainer."""
    discover()
    gen = get("kenning")
    exp = get("kenning-explain")
    # Generate with affix=1, then explain.
    generated = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=42)
    affix_component = next(c for c in generated.components if c.get("location") == "manorial-affix")
    family = affix_component["usage"]
    family_token = family.split()[-1]

    explained = exp.generate_all({"name": generated.result}, seed=42)
    assert explained
    first = explained[0]
    # The family surname token is recognized as a manorial Meaning.
    assert family_token in first.explanation
    assert "Norman manorial family" in first.explanation


def test_synthesized_manorial_subjects_use_old_french_language_slot() -> None:
    """The synthesized subjects must use the ``old_french`` field so
    the explainer's ``_ROOT_CODES`` lookup renders them as "FR"
    rather than ``(?)``. Same regression class as wyrd-1cjg's
    celtic_mix routing — a wrong language slot would silently leave
    the morpheme un-attributed in the explainer output."""
    discover()
    exp = get("kenning-explain")
    results = exp.generate_all({"name": "Stoke Mandeville"}, seed=0)
    # The "(FR)" tag is part of the explainer's per-element rendering
    # for old_french-rooted morphemes.
    assert "(FR)" in results[0].explanation, (
        f"Mandeville should render as (FR) in the explainer; got: {results[0].explanation[:300]}"
    )


def test_manorial_tags_are_excluded_from_user_facing_tag_dropdown() -> None:
    """``available_tags()`` populates the SPA's tag-filter dropdown.
    Manorial / norman are classification markers on the synthesized
    affix Meanings — picking either in the dropdown would restrict
    generation to manorial families (none of which appear in any
    culture's proportions, so the result would be a no-op tag
    filter). The user-facing knob for manorial layering is
    ``manorial_affix`` (a probability field), not a tag selection.
    Pin so a future refactor that drops manorial/norman from
    ``_INTERNAL_TAGS`` would re-introduce the UX wart."""
    from wyrd.generators.kenning import available_tags

    tags = available_tags()
    assert "manorial" not in tags
    assert "norman" not in tags


def test_synthesized_manorial_tokens_are_disjoint_from_main_bundle() -> None:
    """Drift guard: the synthesized surname tokens (Mandeville, Lacy,
    Cary, Marshal, Percy, etc.) must not collide with any usage in
    the main runtime bundle or the irish_anglicizations.json sidecar.
    A future curator who adds 'Marshal' as a generic English
    topographic morpheme (or whatever) would silently shadow the
    manorial entry; pinning the disjointness surfaces the collision
    at curate-time."""
    import json
    from importlib import resources

    from wyrd.generators.kenning import _runtime_db_bundle_dict

    main = _runtime_db_bundle_dict()
    sidecar = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    main_subjects = main.get("subjects") or []
    sidecar_subjects = sidecar["subjects"] if isinstance(sidecar, dict) else sidecar
    pre_synthesis_usages: set[str] = set()
    for source in (main_subjects, sidecar_subjects):
        for subject in source:
            for word in subject["words"]:
                # Canonical-strip dashes so a subject like "-tun" can't
                # collide with a manorial token by surface coincidence.
                pre_synthesis_usages.add(word["modern_usage"].strip("-"))

    manorial_tokens = {family.split()[-1] for family in _load_norman_manorial_families()}
    collisions = manorial_tokens & pre_synthesis_usages
    assert collisions == set(), (
        f"Norman family surname tokens collide with main-bundle usages: {collisions}; "
        f"the synthesized Meaning would silently shadow the existing entry"
    )


def test_synthesized_manorial_meaning_does_not_break_kenning_generate() -> None:
    """The bundle now contains entries tagged 'manorial'/'norman' on
    English-culture morphemes (the synthesized subjects). Verify
    that Kenning.generate (without manorial_affix) doesn't get
    confused — base names should not contain Norman family surnames
    when the affix knob is off. Pin so a future refactor that lets
    the synthesized subjects flow into the standard generation pool
    doesn't silently emit 'Mandeville-tun' as a base name.

    Knob configurations exercised:
    - default knobs (the steady state most users hit)
    - cohesion=1.0 (tag-class-prior bias, the other path where a
      tag-tagged subject could leak in)
    - novelty=1.0 (uniform-marginal blend — the canary, restored by
      wyrd-57d8)

    The novelty=1.0 config is the deterministic canary. wyrd-fcub re-wired
    novelty onto the vector path and surfaced this PRE-EXISTING leak: at
    novelty=1 the uniform blend makes degenerate repeats (``corner corner``)
    likely, and the repeat-diversification re-pick
    (NewName._collect_repick_pools) read meaning_db raw — with no base-pool
    class exclusion — and grafted in a synthesized manorial subject (``Malet``)
    to vary the repeat. wyrd-57d8 fixed it by (a) keeping synthesized manorial
    subjects out of the base pool (Meaning.is_base_pool_excluded, mirroring D40
    saint subjects) and (b) making the diversification re-pick apply that same
    exclusion. seed=13 was the original repro.
    """
    gen = Kenning()
    families = set(_load_norman_manorial_families())
    family_tokens = {f.split()[-1] for f in families}
    knob_configs = [
        {"culture": "english"},
        {"culture": "english", "cohesion": 1.0},
        {"culture": "english", "novelty": 1.0},
    ]
    for config in knob_configs:
        for s in range(20):
            result = gen.generate(config, seed=s)
            words = result.result.split()
            for token in family_tokens:
                assert token not in words, (
                    f"config={config} seed={s}: base name {result.result!r} "
                    f"unexpectedly contains manorial token {token!r}; synthesized "
                    f"subjects may be leaking into the standard generation pool"
                )
