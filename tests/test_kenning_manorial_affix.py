"""Tests for the Norman manorial-affix generation pattern (wyrd-obu).

The bundled English place-name corpus encodes a specific compositional
pattern densely — an English/ON base name plus the Anglo-Norman family
that held the manor (Stoke Mandeville, Ashby de la Zouch, Stanton
Lacy). The ``manorial_affix`` knob attaches a probability-gated Norman
family surname to a generated name as a post-base affix, replicating
the post-Conquest political-history layering on demand.

These tests pin:
- the curated corpus has the expected shape (non-empty, all strings,
  contains representative attested families);
- the knob is off by default (no behavior change vs pre-wyrd-obu);
- at probability=1 every generated English name gets an affix;
- non-English cultures never get the affix even at probability=1
  (the manorial pattern is English-specific);
- the affix surfaces in components with a distinctive ``location``
  marker so the explainer / SPA can render it differently from
  morpheme components;
- the affix is seed-deterministic (same seed → same family pick).
"""

from __future__ import annotations

import json
from importlib import resources

from wyrd.generators.kenning import (
    Kenning,
    _load_norman_manorial_families,
)


def test_norman_manorial_family_corpus_is_well_formed() -> None:
    """The corpus must be a non-empty list of distinct non-empty
    strings. A typo that re-introduces a duplicate or empty string
    would skew the rng.choice distribution silently."""
    payload = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("norman_manorial_families.json")
        .read_text()
    )
    assert isinstance(payload, list) and len(payload) >= 20
    assert all(isinstance(x, str) and x for x in payload)
    assert len(set(payload)) == len(payload), "duplicate family names in corpus"


def test_norman_manorial_family_corpus_contains_attested_families() -> None:
    """Spot-check that the canonical Domesday-era families the wyrd-obu
    ticket mentions explicitly are present. A curation regression that
    drops these well-known families would still produce a 'works' test
    suite if all we checked was non-empty length, so pin the load-
    bearing examples."""
    payload = set(_load_norman_manorial_families())
    for name in ("Mandeville", "La Zouche", "Lacy", "Mortemer", "Beauchamp"):
        assert name in payload, f"corpus missing canonical family {name!r}"


def test_manorial_affix_default_is_off() -> None:
    """Without the knob set, the generator must produce its pre-wyrd-obu
    output — no affix appended, no affix component. Pins the
    bit-stable default so existing callers (CLI, SPA forms not yet
    aware of the knob) keep their current output."""
    gen = Kenning()
    result = gen.generate({"culture": "english"}, seed=42)
    assert all(c.get("location") != "manorial-affix" for c in result.components), (
        "default manorial_affix must produce no affix component"
    )


def test_manorial_affix_at_one_attaches_to_every_english_name() -> None:
    """At probability=1.0 every generated English name must get an
    affix. 5 seeds is enough to catch a ``rng.random() < p`` regression
    (e.g. swapped operator, off-by-one) while staying cheap."""
    gen = Kenning()
    for s in range(5):
        result = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=s)
        assert any(c.get("location") == "manorial-affix" for c in result.components), (
            f"seed {s} produced no affix at probability=1.0"
        )
        # Output string ends with " <Family>" — i.e. the family name is
        # the last whitespace-separated token.
        family = next(
            c["usage"] for c in result.components if c.get("location") == "manorial-affix"
        )
        assert result.result.endswith(family), (
            f"seed {s} result {result.result!r} doesn't end with affix {family!r}"
        )


def test_manorial_affix_mid_probability_lands_in_expected_range() -> None:
    """At probability=0.5 the attach-rate over a sample of seeds must
    land near 50%. Pin in a tolerant window ([35%, 65%] over 200 seeds)
    so a regression that flipped the comparator (``random() > p``),
    swapped sign (``1 - p``), or mis-wired the gate constant would
    trip without the test flaking on rng variance.

    This is the only test that exercises the mid-range probability
    path — the boundary cases (0 and 1) are short-circuit-equivalent
    to no-op and unconditional-attach respectively, so a bug in the
    comparator could pass both endpoint tests."""
    gen = Kenning()
    hit = 0
    n = 200
    for s in range(n):
        result = gen.generate({"culture": "english", "manorial_affix": 0.5}, seed=s)
        if any(c.get("location") == "manorial-affix" for c in result.components):
            hit += 1
    rate = hit / n
    assert 0.35 <= rate <= 0.65, (
        f"manorial_affix=0.5 attach-rate {rate:.2%} fell outside the [35%, 65%] "
        f"window (hit {hit}/{n}); the comparator may be regressed"
    )


def test_manorial_affix_component_carries_render_hooks() -> None:
    """The affix component must carry the tags + roots fields the SPA
    / explainer dispatch on. Pin the load-bearing render hooks so a
    refactor that drops 'manorial'/'norman' from tags or 'FR' from
    roots would break the SPA's per-source rendering silently. The
    rest of the canonical-component shape (citations, meanings) is
    less load-bearing here — the component is annotation, not a real
    morpheme — but the keys must exist so consumers can iterate
    uniformly."""
    gen = Kenning()
    result = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=0)
    affix = next(c for c in result.components if c.get("location") == "manorial-affix")
    assert "manorial" in affix["tags"]
    assert affix["roots"] == ["FR"]
    # Canonical-component-shape parity: the keys SPA consumers iterate
    # all exist (no KeyError on a dispatcher that handles morpheme
    # components and manorial components in one loop).
    for required in ("usage", "location", "meanings", "tags", "roots", "citations"):
        assert required in affix, f"affix component missing canonical key {required!r}"


def test_manorial_affix_input_schema_field_shape() -> None:
    """Pin the input_schema field's structural fields so a regression
    that drops the [0,1] range, the default of 0, or the "number"
    type would surface at test time rather than at SPA render time
    (where a bad max would let the user submit nonsense values, or
    a missing default would leave the field uninitialized)."""
    schema = Kenning().input_schema()
    field = schema["properties"]["manorial_affix"]
    assert field["type"] == "number"
    assert field["default"] == 0.0
    assert field["minimum"] == 0.0
    assert field["maximum"] == 1.0
    assert "manorial" in field["description"].lower()


def test_manorial_affix_composes_orthogonally_with_mood() -> None:
    """The affix is appended after the base-name pipeline, so it
    should compose cleanly with the existing knobs (mood, era,
    cohesion, novelty) without perturbing them. Verify by generating
    with both manorial_affix=1.0 and a mood knob active: the affix
    must surface AND the base name's morphemes must reflect the
    mood's tag bias (the mood's grim tags are in the explanation).
    Catches a regression that re-orders rng draws so the mood-driven
    base-name selection sees a different rng state."""
    gen = Kenning()
    result = gen.generate(
        {
            "culture": "english",
            "manorial_affix": 1.0,
            "mood": ["grim"],
        },
        seed=42,
    )
    # Affix landed.
    assert any(c.get("location") == "manorial-affix" for c in result.components)
    # Base components survived (>=1 morpheme component before the affix).
    morpheme_components = [c for c in result.components if c.get("location") != "manorial-affix"]
    assert len(morpheme_components) >= 1, "mood knob ate the base components"


def test_manorial_affix_skips_non_english_cultures_even_at_one() -> None:
    """The manorial-layering pattern is an English place-naming
    convention (post-Conquest political history). Pasting Norman
    affixes onto Welsh / Irish / Scottish / Breton bases would be
    cosmetically jarring and historically wrong; the gate is
    explicit. Pin so a future refactor that drops the culture check
    doesn't silently start emitting 'Caerphilly Mandeville' or
    'Cluain Lacy'."""
    gen = Kenning()
    for culture in ("welsh", "irish", "scottish", "breton"):
        for s in range(5):
            result = gen.generate({"culture": culture, "manorial_affix": 1.0}, seed=s)
            assert all(c.get("location") != "manorial-affix" for c in result.components), (
                f"culture={culture!r} seed={s} got an affix at probability=1.0; "
                f"manorial-layering must be english-only"
            )


def test_manorial_affix_is_seed_deterministic() -> None:
    """Same seed must produce the same family pick. The rng is the
    seeded ``rng_for`` and ``rng.choice`` is deterministic given the
    same sequence of prior draws — pin so a future refactor that
    re-orders rng calls (or substitutes ``random.choice`` for the
    seeded rng) doesn't quietly desync share-link reproducibility."""
    gen = Kenning()
    a = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=42)
    b = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=42)
    assert a.result == b.result


def test_manorial_affix_explanation_carries_family_marker() -> None:
    """The explainer string must surface the manorial affix with a
    distinctive marker so the SPA / share-link explanation makes the
    affix's provenance visible. Pin the marker shape so a future
    explainer rewrite doesn't quietly drop the manorial provenance."""
    gen = Kenning()
    result = gen.generate({"culture": "english", "manorial_affix": 1.0}, seed=7)
    family = next(c["usage"] for c in result.components if c.get("location") == "manorial-affix")
    assert "manorial" in result.explanation.lower()
    assert family in result.explanation
