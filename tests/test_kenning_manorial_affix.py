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
    affix. Sample multiple seeds because the rng-driven affix pick
    could be hidden behind a single-seed coincidence with
    probability=0.x."""
    gen = Kenning()
    for s in range(30):
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
