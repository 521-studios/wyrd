"""End-to-end tests for Kenning.generate with scoring_mode='vector'
(wyrd-ecjp.5 PR C).

The dispatch path: Kenning.generate reads scoring_mode, translates
the per-call knobs into a RequestVector via the adapter, optionally
loads priors from a JSON sidecar, calls
NameGenerator.select_via_vector, and returns a GenerationResult.

These tests cover the wiring end-to-end against the LIVE bundle —
unlike the vector_name_select unit tests which use synthetic
meaning_db fixtures, the dispatch needs the real bundle's
structures + meanings to round-trip the NewName output surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning
from wyrd.seed import MAX_SAFE_INTEGER, rng_for


@pytest.fixture(autouse=True)
def _permissive_structure_allowlist(monkeypatch):
    """wyrd-c6o1.5: this module builds SYNTHETIC structs directly into
    NameGenerator (and exercises the un-curated generation path); the operator
    allowlist (data/structures.yaml, which ships some lone-word structures
    disabled) is a bundle overlay irrelevant to these selection/rendering tests.
    Empty it so a synthetic fixture struct (e.g. ``(bare[name])``) isn't filtered
    out — restoring the pre-allowlist behavior these tests were written against.

    The coupled cache-clear on EXIT is load-bearing: a real-``Kenning()`` test in
    this module can build + cache a per-culture NameGenerator while the allowlist
    is empty; without dropping it, that unfiltered generator would leak into a
    later module (e.g. the rogd10 parity snapshot) and drift seeded output."""
    from wyrd.generators.kenning import _load_meanings
    from wyrd.generators.kenning.runtime import structure_allowlist

    monkeypatch.setattr(structure_allowlist, "_load_bundled_cached", lambda: {})
    yield
    _load_meanings.cache_clear()  # rebound to the coupled clear → also drops _load_culture


def test_kenning_generate_default_runs():
    """wyrd-rt2m: a bare generate (no scoring_mode — the interface is removed)
    runs the default vector path and returns a name without raising."""
    k = Kenning()
    result = k.generate({"culture": "english"}, seed=42)
    assert result.result  # non-empty surface string


def test_kenning_default_scoring_mode_is_vector():
    """wyrd-rt2m: with the scoring_mode interface removed, the implicit default
    is now VECTOR — explicit scoring_mode='vector' produces the same output as
    the bare default (bit-stable)."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42)
    b = k.generate({"culture": "english", "scoring_mode": "vector"}, seed=42)
    assert a.result == b.result


def test_kenning_scoring_mode_vector_smoke():
    """scoring_mode='vector' without priors — vector path runs but
    might return empty / raise depending on bundle coverage. We just
    check it doesn't crash with an unhandled exception; if it raises
    the documented ValueError, that's fine — operator's fault for
    requesting vector mode without setting up the data."""
    k = Kenning()
    # Pass a register-relevant tag so the sem axis has signal
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.5,
                "scoring_mode": "vector",
            },
            seed=42,
        )
        # If we got here, the vector path produced a non-empty pick
        assert result.result  # non-empty surface
    except ValueError as e:
        # Documented loud-failure path: vector returned None because
        # the bundle/data combo couldn't produce a non-zero score
        # for any slot. Acceptable in v1 (bundle doesn't yet ship
        # phonological_vector / priors via ecjp.8).
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_with_priors(tmp_path: Path):
    """scoring_mode='vector' with priors_path pointing at a JSON
    sidecar. The priors load + the baseline axis contributes."""
    # Build a minimal valid priors sidecar
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "native": [
                    {
                        "culture": "english",
                        "position": "pre",
                        "tag": "water",
                        "era_midpoint": 0,
                        "lemmas": {"bourne": 50, "brook": 30},
                    }
                ],
                "loan": [],
            }
        )
    )
    k = Kenning()
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.3,
                "scoring_mode": "vector",
                "priors_path": str(priors_file),
            },
            seed=42,
        )
        # Vector path produced something (priors + sem + phon all
        # contribute non-zero for water-tagged lemmas)
        assert result.result
    except ValueError as e:
        # Acceptable — even with priors, the bundle may not have
        # phonological_vector for water-tagged pre-slot lemmas, and
        # the gate might filter everything.
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_default_uses_uniform_tag_weights():
    """Vector mode with no explicit semantic signal (no tags, no
    mood, no harshness) now falls back to uniform-tag-weights via
    ``build_request_vector``'s no-opinion-default branch — so the
    request still expresses semantic interest (1.0 over the full
    tag universe) and ``baseline_score_native`` returns non-zero
    for any lemma the priors data covers. Pin that the dispatch
    produces a name rather than raising the legacy "no eligible
    name" error.

    Pre-fix contract was: empty register → no eligible name (vector
    mode only worked with a mood). That was the gate on adopting
    vector as a default scoring mode; this test pins the new
    works-by-default contract."""
    k = Kenning()
    result = k.generate(
        {
            "culture": "english",
            "scoring_mode": "vector",
        },
        seed=42,
    )
    assert result.result, "vector mode default-call must produce a non-empty name"


def test_kenning_seed_stable_within_mode():
    """Same seed + same mode → same result (bit-stable contract per
    GenerationResult)."""
    k = Kenning()
    r1 = k.generate({"culture": "english"}, seed=7)
    r2 = k.generate({"culture": "english"}, seed=7)
    assert r1.result == r2.result


def test_kenning_seed_stable_within_vector_mode():
    """Same seed + scoring_mode='vector' → same result. The vector
    path's bit-stability is critical for ecjp.6/7 drift measurement —
    if two runs at the same seed diverge, the drift signal is noise.
    Round-1 reviewer caught the gap that legacy was pinned but the
    vector path wasn't."""
    k = Kenning()
    params = {
        "culture": "english",
        "tags": ["water"],
        "harshness": 0.5,
        "scoring_mode": "vector",
    }
    # Either both raise OR both succeed with identical output. The
    # contract isn't "vector mode succeeds against the live bundle"
    # (that depends on data not yet shipped); it's "same inputs →
    # same outputs whatever those outputs are".
    try:
        r1 = k.generate(params, seed=7)
        r2 = k.generate(params, seed=7)
        assert r1.result == r2.result
    except ValueError:
        # If the first raised, the second must raise identically too.
        with pytest.raises(ValueError):
            k.generate(params, seed=7)


def test_kenning_scoring_mode_unknown_value_validated_by_schema():
    """Per the schema's enum, 'banana' is invalid. The generator
    itself doesn't validate (relies on the schema layer); but a
    typo'd value falls through the else branch and runs the legacy
    path — pinning that fallback behavior."""
    k = Kenning()
    # Unknown scoring_mode → falls through to else → legacy path runs
    result = k.generate(
        {"culture": "english", "scoring_mode": "banana"},
        seed=42,
    )
    assert result.result


# ---- wyrd-izcr: vector path honors structure qualifier flags ---------------


def test_vector_seed_1170_no_longer_produces_split_affix_port_all():
    """wyrd-izcr regression: pre-fix seed 1170 produced ``Port All``
    at sub-seed 5 — Port- + -all rendered as two separate words
    because the vector path dropped the name/saint flags from
    qualifier-flagged structures (e.g. [pre+saint, post+name]).
    Post-fix the same struct triggers the qualifier filter; if
    the saint/name pool is empty under the live request, the
    NameGenerator retries with a different struct.

    ``Kenning.generate`` produces one name per seed; the original
    bug surfaced via the CLI's count=N dispatch which derives
    sub-seeds from the master seed via
    ``seed_rng.randrange(MAX_SAFE_INTEGER + 1)`` (see
    ``wyrd.app._dispatch``). Replay the exact same sub-seed
    sequence here so the regression catches the actual buggy slot
    rather than just seed=1170 itself."""
    k = Kenning()
    master = rng_for(1170)
    sub_seeds = [master.randrange(MAX_SAFE_INTEGER + 1) for _ in range(10)]
    names: list[str] = []
    for sub in sub_seeds:
        result = k.generate(
            {"culture": "english", "scoring_mode": "vector"},
            seed=sub,
        )
        names.append(result.result)
    assert "Port All" not in names, f"Port All bug regressed: {names}"


def _build_synthetic_vector_name_gen(structs: dict, meaning_db: dict):
    """Build a NameGenerator with synthetic data for direct
    select_via_vector testing — bypasses the live bundle so the
    retry loop, qualifier filter, and exhaust path can be exercised
    deterministically."""
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    # wyrd-eyjk/D40: the part pool keeps the dash-variant forms (pre/post
    # buckets); the single pool records the BARE surface (lone words are
    # structurally bare), so it lands at the ("bare", …, "single") buckets a
    # single-word struct references.
    proportions = dict.fromkeys(meaning_db, 1)
    single_proportions = {k.replace("-", ""): 1 for k in meaning_db}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(single_proportions, "single")
    return NameGenerator(meaning_db, mg, structs)


def _vector_request(culture: str = "english"):
    """Build a RequestVector that produces non-zero scores against
    a synthetic Meaning tagged 'urban' — so eligibility, not
    scoring, determines whether the dispatch picks a meaning."""
    from wyrd.generators.kenning.vectors.schemas import (
        EligibilityGate,
        RegisterEffect,
        RequestVector,
        ScoringWeights,
    )

    return RequestVector(
        gate=EligibilityGate(culture=culture),
        register=RegisterEffect(name="test", semantic_tags={"urban": 1.0}),
        weights=ScoringWeights(),
    )


def test_select_via_vector_retry_exhausts_when_no_qualifier_pool():
    """wyrd-izcr + wyrd-tbke: when every struct in the pool requires a
    qualifier slot but the meaning_db has zero qualifier-flagged morphemes,
    NameGenerator.select_via_vector exhausts the retry budget AND the
    permissive fallback fills nothing (every slot is an unsatisfiable
    qualifier slot → all-None). That degenerate all-empty result returns
    None — caller (Kenning.generate) then raises the operator-visible
    ValueError. (The PARTIAL case — some slots fillable — degrades to a
    shorter name instead of None; see
    test_select_via_vector_degrades_instead_of_raising_on_partial_empty.)"""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # meaning_db has only bare morphemes — no is_name() True, no
    # literal Saint- usage. Any qualifier-flagged struct slot is
    # unsatisfiable.
    db = {
        "Port-": [Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])],
        "-all": [Meaning(usage="-all", tags=["urban"], meanings=[], sources=[])],
    }
    # Two structs, both with qualifier-flagged slots — exactly the
    # shape that produced "Port All" pre-fix.
    structs = {
        ((("pre", "saint", "single"),), (("post", "name", "single"),)): 4,
        ((("pre", "saint", "single"),), (("pre", "name", "single"),)): 11,
    }
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is None


def test_select_via_vector_degrades_instead_of_raising_on_partial_empty():
    """wyrd-tbke empty-pick PARITY: when no struct is FULLY satisfiable under
    the gates but SOME slots can fill, select_via_vector degrades to a shorter
    name (dropping the unsatisfiable slot) instead of returning None — matching
    the proportions ``_select_no_tag`` contract, so the dispatch never raises
    'no eligible name'."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # A fillable bare/single slot (urban-tagged morpheme, lands in the
    # ("bare","single") bucket via the synthetic single pool) + an
    # UNsatisfiable saint slot (no saint-usage morpheme in db). Both words are
    # grammatical (bare word + saint qualifier word) so the struct survives the
    # wyrd-zzli filter. Non-permissive can't complete the struct; the
    # permissive fallback fills the bare slot and drops the saint slot → a
    # non-empty degraded name rather than None.
    db = {"Port-": [Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])]}
    structs = {((("bare", "single"),), (("post", "saint", "single"),)): 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is not None, "should degrade gracefully, not return None (→ raise)"
    assert str(result), "degraded name must be non-empty (the pre slot filled)"
    assert "port" in str(result).lower()
    # The unsatisfiable saint slot was dropped → one rendered word, not two.
    assert len(str(result).split()) == 1


def test_select_via_vector_degrades_with_empty_slot_first():
    """wyrd-tbke: a dropped (None) slot at a NON-final reconstruction index —
    the unsatisfiable saint slot FIRST, the fillable bare slot second — still
    degrades to the 1-word name. Pins the reconstruction idx-walk against a
    mid-sequence None (an off-by-one would mis-map the trailing pick)."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    db = {"Port-": [Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])]}
    structs = {((("post", "saint", "single"),), (("bare", "single"),)): 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is not None
    assert "port" in str(result).lower()
    assert len(str(result).split()) == 1


def test_select_via_vector_retry_excludes_tried_structs():
    """wyrd-izcr: after an attempt fails (qualifier pool empty),
    that struct is excluded from subsequent retries. Without
    exclusion, weighted_choice could re-pick the same un-satisfiable
    struct and burn the retry budget without exploring alternatives.
    Pin via a synthetic 2-struct pool where one is unsatisfiable
    and one is — the satisfiable struct MUST be picked within the
    retry budget regardless of initial weighted-choice outcome."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # One usable bare morpheme + one name-flagged morpheme.
    smith = Meaning(usage="Smith-", tags=["family name", "urban"], meanings=[], sources=[])
    bare = Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])
    end = Meaning(usage="-end", tags=["urban"], meanings=[], sources=[])
    db = {"Smith-": [smith], "Port-": [bare], "-end": [end]}
    # Heavy weight on the unsatisfiable struct so it dominates the
    # initial pick; the satisfiable compound struct survives only
    # via retry-with-exclusion.
    unsatisfiable = ((("pre", "saint", "single"),), (("pre", "name", "single"),))
    satisfiable = ((("pre",), ("post",)),)
    structs = {unsatisfiable: 999, satisfiable: 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    # Must succeed: retry should drop the unsatisfiable struct and
    # land on the compound one.
    assert result is not None


def test_select_via_vector_retry_limit_bounds_attempts(monkeypatch):
    """wyrd-izcr: _VECTOR_STRUCT_RETRY_LIMIT bounds the retry budget.
    Pin the constant by counting weighted_choice calls when every
    struct is unsatisfiable — should be at most the limit."""
    import random

    from wyrd.generators.kenning.runtime import proportions as proportions_mod
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    db = {"-all": [Meaning(usage="-all", tags=["urban"], meanings=[], sources=[])]}
    structs = {
        ((("pre", "saint", "single"),), (("post", "name", "single"),)): 4,
        ((("pre", "saint", "single"),), (("pre", "name", "single"),)): 11,
    }
    name_gen = _build_synthetic_vector_name_gen(structs, db)

    call_count = 0
    real_weighted_choice = proportions_mod.weighted_choice

    def counting_weighted_choice(rng, items):
        nonlocal call_count
        call_count += 1
        return real_weighted_choice(rng, items)

    monkeypatch.setattr(proportions_mod, "weighted_choice", counting_weighted_choice)
    # Force the limit down so the test can pin the exact bound without
    # depending on how many structs survive exclusion.
    monkeypatch.setattr(proportions_mod, "_VECTOR_STRUCT_RETRY_LIMIT", 3)

    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is None
    # Each retry consumes one weighted_choice; after the per-struct
    # exclusion drains the 2-element pool, the retry loop's
    # ``if not remaining: return None`` exits early — so at most 2
    # weighted_choice calls were issued, less than the limit of 3.
    assert call_count <= 3


def test_select_via_vector_saint_filter_matches_legacy_literal_usage_only():
    """wyrd-izcr: saint-qualifier filter uses the legacy literal
    usage check (Meaning.usage.replace("-", "").lower() == "saint"),
    NOT m.is_saint() (the tag check). Andrew- is saint-tagged but
    its usage is "Andrew-", not "Saint-". Legacy bucket-keying
    routes it through (pre, "name") via Meaning.key()'s if/elif
    precedence; the vector qualifier filter must do the same to
    avoid behavioral divergence between modes."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # Andrew- is saint-tagged but is also is_name() True (male
    # name) — legacy routes it to the name bucket, not saint.
    andrew = Meaning(
        usage="Andrew-",
        tags=["male name", "saint", "urban"],
        meanings=[],
        sources=[],
    )
    # Literal Saint- — the only morpheme that fills the saint slot
    # in legacy mode.
    saint_literal = Meaning(
        usage="Saint-",
        tags=["saint", "urban"],
        meanings=[],
        sources=[],
    )
    db = {"Andrew-": [andrew], "Saint-": [saint_literal]}
    # Saint-flagged single-element 1-word struct. D39: a lone word is
    # structurally bare (the single pool buckets as "bare" regardless of
    # the Saint- prefix form), so the slot key is ("bare", "saint", "single").
    structs = {((("bare", "saint", "single"),),): 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    # Over many seeds, only the saint morpheme should ever fill the saint slot
    # — never Andrew-. NewName.name is list-of-words; the slot struct is a
    # 1-word 1-element compound so name[0][0] is the picked morpheme rendered
    # at its slot-derived position. wyrd-g1hj: the vector path now emits the
    # POSITION-FORM (not the stored ``Saint-`` variant), and a ("bare", …) slot
    # renders bare → ``Saint`` (Andrew would render ``Andrew``); we assert the
    # saint morpheme is the only one admitted.
    picked: set[str] = set()
    for seed in range(20):
        result = name_gen.select_via_vector(
            random.Random(seed),
            request=_vector_request(),
            priors=EmpiricalPriors(),
        )
        if result is None:
            continue
        picked.add(result.name[0][0])
    assert picked == {"Saint"}, f"saint slot admitted non-literal usage: {picked}"


# ---- wyrd-bol9: NameGenerator usage_frequency_by_bucket build -------------


def test_name_generator_usage_frequency_by_bucket_snapshots_generators():
    """wyrd-bol9: NameGenerator._build_usage_frequency_by_bucket
    snapshots MeaningGenerator's per-bucket weight tables into a
    flat ``bucket_key → {usage: frequency}`` map. Single-element and
    multi-element bucket variants land at DISTINCT keys (``("pre",)``
    vs ``("bare", "single")`` — D39: lone words bucket as bare) so the
    vector path can consult them separately — proportions samples them
    as separate buckets and the per-usage frequency distributions can
    diverge."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    common = Meaning(usage="Common-", tags=["urban"], meanings=[], sources=[])
    rare = Meaning(usage="Rare-", tags=["urban"], meanings=[], sources=[])
    meaning_db = {"Common-": [common], "Rare-": [rare]}

    multi_word_proportions = {"Common-": 6, "Rare-": 2}
    # wyrd-eyjk/D40: lone-word occurrences record the BARE surface form, so the
    # single pool's usages are dash-less (and land at the ("bare", "single")
    # bucket by their derived position, not via the retired add_bare hack).
    single_word_proportions = {"common": 3, "rare": 1}
    mg = MeaningGenerator(meaning_db, {}, multi_word_proportions)
    mg.load_parts(single_word_proportions, "single")
    # 2-element compound word — passes is_structurally_grammatical
    # (single-element bare-pre/post/inner words get filtered).
    structs = {((("pre",), ("post",)),): 1}

    name_gen = NameGenerator(meaning_db, mg, structs)
    assert name_gen.usage_frequency_by_bucket[("pre",)] == {"Common-": 6, "Rare-": 2}
    assert name_gen.usage_frequency_by_bucket[("bare", "single")] == {
        "common": 3,
        "rare": 1,
    }


def test_name_generator_usage_frequency_by_bucket_separates_qualifier_buckets():
    """Qualifier-flagged buckets land at distinct lookup keys —
    ``("pre", "name")`` and ``("pre",)`` must NOT collapse into each
    other, even though both contain pre-position morphemes."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    bare = Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])
    family = Meaning(
        usage="Smith-",
        tags=["family name", "urban"],
        meanings=[],
        sources=[],
    )
    meaning_db = {"Port-": [bare], "Smith-": [family]}
    mg = MeaningGenerator(meaning_db, {}, {"Port-": 5, "Smith-": 3})
    structs = {((("pre",), ("post",)),): 1}

    name_gen = NameGenerator(meaning_db, mg, structs)
    assert name_gen.usage_frequency_by_bucket[("pre",)] == {"Port-": 5}
    assert name_gen.usage_frequency_by_bucket[("pre", "name")] == {"Smith-": 3}


def test_select_via_vector_renders_picks_at_slot_positions():
    """wyrd-g1hj: the vector reconstruction renders each pick at its SLOT-derived
    position, not the stored dash-variant — the regression behind `Gōstōn` →
    `gōs-`/`tōn-` (two pre) and `-tre-` inner-at-word-start. For a (pre, post)
    struct, slot 0 emits a pre-form (`X-`) and slot 1 a post-form (`-x`), even
    though both morphemes are STORED pre-shaped (`Ton-` at the post slot must
    render `-ton`)."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    db = {
        "Gos-": [Meaning("Gos-", ["urban"], ["g"], {})],
        "Ton-": [Meaning("Ton-", ["urban"], ["t"], {})],
    }
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts({"Gos-": 1})  # surface 'gos' → pre bucket
    mg.load_parts({"-ton": 1})  # surface 'ton' → post bucket (stored variant is still 'Ton-')
    structs = {((("pre",), ("post",)),): 1}
    ng = NameGenerator(db, mg, structs)

    seen = False
    for seed in range(30):
        r = ng.select_via_vector(
            random.Random(seed), request=_vector_request(), priors=EmpiricalPriors()
        )
        if r is None:
            continue
        seen = True
        word = r.name[0]  # one word, two morphemes
        assert len(word) == 2
        assert word[0].endswith("-") and not word[0].startswith("-"), (
            f"pre slot must render a pre-form: {word}"
        )
        assert word[1].startswith("-"), (
            f"post slot must render a slot-derived post-form, not the stored variant: {word}"
        )
    assert seen, "expected at least one generated name across seeds"


def test_select_via_vector_applies_d8_d18_rendering():
    """wyrd-nbpw: vector mode threads inflection_density / spelling_variety
    through the SAME _render_substitutions the legacy proportions path uses, so
    D8 inflection (form + grammatical-case label) and D18 spelling variants
    render identically across both scoring modes. Default (both knobs 0) skips
    the substitution pass entirely → rendered/inflection_labels stay None
    (bit-stable). Mirrors test_select_populates_inflection_labels_at_high_density
    on the proportions side."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # 'family name' flips is_name() True so the single bare-name slot fills;
    # 'urban' gives the request a non-zero score. old_english carries a variant
    # pool (cot/cotum) + an inflection (cotum/dative_or_pl).
    m = Meaning(
        "-cot",
        ["family name", "urban"],
        [],
        {"old_english": ["cot", "cotum"]},
        inflections={"old_english": [("cotum", "dative_or_pl")]},
    )
    db = {"-cot": [m]}
    structs = {((("bare", "name", "single"),),): 1}
    ng = _build_synthetic_vector_name_gen(structs, db)

    inflected = ng.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
        inflection_density=1.0,
    )
    assert inflected.rendered == [["cotum"]]
    assert inflected.inflection_labels == [["dative_or_pl"]]

    # wyrd-24s6 (D41): default knobs off → no D8/D18 substitution pass, but the
    # NATIVE render now runs (era="" → as-selected). This synthetic Meaning has
    # no morpheme_id, so _native_form_for_meanings returns None → the native
    # layer is [[None]] (each slot falls back to the modern usage at __str__
    # time, so the displayed name is unchanged). inflection_labels stays None.
    plain = ng.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert plain.rendered == [[None]]  # native layer ran; no morpheme_id → None
    assert plain.inflection_labels is None


def test_vector_mode_knobs_reach_render_path():
    """wyrd-nbpw: scoring_mode='vector' threads inflection_density /
    spelling_variety through generate → _generate_via_vector →
    select_via_vector to the render pass. A dropped keyword anywhere in that
    chain would leave the knobs inert; this pins the end-to-end wiring (mirror
    of the proportions test_high_inflection_density_changes_output_when_pool_present)."""
    k = Kenning()
    canonical = set()
    inflected = set()
    varied = set()
    for seed in range(20):
        canonical.add(
            k.generate({"culture": "english", "scoring_mode": "vector"}, seed=seed).result
        )
        inflected.add(
            k.generate(
                {"culture": "english", "scoring_mode": "vector", "inflection_density": 1.0},
                seed=seed,
            ).result
        )
        varied.add(
            k.generate(
                {"culture": "english", "scoring_mode": "vector", "spelling_variety": 1.0},
                seed=seed,
            ).result
        )
    assert inflected != canonical, "inflection_density must reach the vector render path"
    assert varied != canonical, "spelling_variety must reach the vector render path"


def test_select_via_vector_applies_d18_variant_without_label():
    """wyrd-nbpw: the D18 spelling-variant branch (distinct from D8) — at
    spelling_variety=1.0 the rendered surface is the attested variant and the
    inflection label is None (variants carry no grammatical case). Mirrors the
    proportions test_render_substitutions_substitutes_variant_with_case_mimic."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    m = Meaning(
        "-cot",
        ["family name", "urban"],
        [],
        {"old_english": ["cot"]},
        variants={"old_english": [("cotte", 10)]},
    )
    structs = {((("bare", "name", "single"),),): 1}
    ng = _build_synthetic_vector_name_gen(structs, {"-cot": [m]})
    out = ng.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
        spelling_variety=1.0,
    )
    assert out.rendered is not None
    assert out.rendered[0][0].lower() == "cotte", f"variant not applied: {out.rendered}"
    assert out.inflection_labels == [[None]], "a D18 variant carries no inflection label"
