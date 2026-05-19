"""Tests for inflection-form resolution at match time (wyrd-lx8).

Each lemma's known inflected forms (D8 metadata, ``<lang>_inflections``
entries in the bundle) get registered as shadow ``Meaning`` entries in
the ``meaning_db`` returned by ``load_meanings``. Effect: the matcher
recognizes ``cotum`` / ``brycgan`` / ``hamum`` as surface forms and
resolves them to the lemma without leaving the inflection suffix as
'unaccounted'.

This shrinks the unaccounted-fragment list in two ways:
1. Names whose only gap was an inflected form of a known lemma
   become ``perfect=true``.
2. Names with multiple gaps lose one fragment per resolved inflection.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import Name


def test_inflected_form_registers_in_meaning_db() -> None:
    """Loading a bundle with inflection metadata adds a shadow Meaning
    per inflected form keyed by its dashed surface."""
    word_db, _ = load_meanings(
        [
            {
                "meaning": ["Cottage"],
                "modifier_tags": ["architecture"],
                "modifier_type": "Architectural",
                "words": [
                    {
                        "modern_usage": "-cot",
                        "old_english": ["cot"],
                        "old_english_inflections": [
                            {"form": "cotum", "inflection": "dative_or_pl"},
                            {"form": "cotan", "inflection": "weak_oblique"},
                        ],
                    }
                ],
            }
        ]
    )
    # Canonical entry registered.
    assert "-cot" in word_db
    # Shadow entries for each inflection — same dash markers as the
    # canonical (post-suffix here) so position constraints carry over.
    assert "-cotum" in word_db
    assert "-cotan" in word_db
    # All three meanings share the lemma's gloss + tags.
    for usage in ("-cot", "-cotum", "-cotan"):
        m = word_db[usage][0]
        assert "Cottage" in m.meanings
        assert "architecture" in m.tags


def test_dash_pattern_preserved_for_pre_suffix() -> None:
    """A pre-suffix lemma's inflections are also pre-suffix; a post-
    suffix lemma's are post; an inner is inner. Mirrors the canonical
    so the legacy matcher's position filter applies the same way."""
    for canonical_usage, expected_shadow in [
        ("Brycg-", "Brycgan-"),  # pre stays pre
        ("-cot", "-cotum"),  # post stays post
        ("-en-", "-enan-"),  # inner stays inner
    ]:
        word_db, _ = load_meanings(
            [
                {
                    "meaning": ["X"],
                    "modifier_tags": ["topography"],
                    "modifier_type": "Topographical",
                    "words": [
                        {
                            "modern_usage": canonical_usage,
                            "old_english": [canonical_usage.replace("-", "").lower() or "x"],
                            "old_english_inflections": [
                                {
                                    "form": expected_shadow.replace("-", ""),
                                    "inflection": "test",
                                }
                            ],
                        }
                    ],
                }
            ]
        )
        assert expected_shadow in word_db, (
            f"canonical {canonical_usage!r} → expected shadow {expected_shadow!r}; "
            f"got keys {[k for k in word_db if expected_shadow.replace('-', '') in k.lower()]}"
        )


def test_inflected_form_resolves_in_place_name() -> None:
    """Concrete worked example: ``Bradcotum`` (Broad + cot-dative-pl)
    parses cleanly when the inflection is registered. Without the
    inflection registration the legacy matcher would leave ``um`` as
    unaccounted residue (or with wyrd-4hx7's matcher might leave
    other fragments)."""
    word_db, _ = load_meanings(
        [
            {
                "meaning": ["Broad"],
                "modifier_tags": ["descriptive"],
                "modifier_type": "Descriptive",
                "words": [{"modern_usage": "Brad-", "old_english": ["brad"]}],
            },
            {
                "meaning": ["Cottage"],
                "modifier_tags": ["architecture"],
                "modifier_type": "Architectural",
                "words": [
                    {
                        "modern_usage": "-cot",
                        "old_english": ["cot"],
                        "old_english_inflections": [
                            {"form": "cotum", "inflection": "dative_or_pl"},
                        ],
                    }
                ],
            },
        ]
    )
    n = Name("Bradcotum")
    n.find_meaning(word_db)
    assert n.count_unaccounted() == 0, f"expected perfect parse via Brad- + -cotum; got {n.words!r}"


def test_no_duplicate_when_inflected_form_equals_lemma() -> None:
    """Some lemmas have an inflected form whose surface is identical
    to the lemma (e.g. nominative singular for invariant nouns). Don't
    register a duplicate at the same meaning_db key."""
    word_db, _ = load_meanings(
        [
            {
                "meaning": ["X"],
                "modifier_tags": ["topography"],
                "modifier_type": "Topographical",
                "words": [
                    {
                        "modern_usage": "-cot",
                        "old_english": ["cot"],
                        "old_english_inflections": [
                            # Same surface as the lemma — should not double-register.
                            {"form": "cot", "inflection": "nominative"},
                            {"form": "cotum", "inflection": "dative_or_pl"},
                        ],
                    }
                ],
            }
        ]
    )
    # Single entry under the canonical key.
    assert len(word_db["-cot"]) == 1
    # The non-duplicate inflection still lands.
    assert "-cotum" in word_db


def test_inflected_form_does_not_break_existing_matches() -> None:
    """Existing canonical-form matches still work; the new shadow
    entries are purely additive. A name that perfectly parses against
    the lemma alone should still perfectly parse with inflections
    registered."""
    word_db, _ = load_meanings(
        [
            {
                "meaning": ["Broad"],
                "modifier_tags": ["descriptive"],
                "modifier_type": "Descriptive",
                "words": [
                    {
                        "modern_usage": "Brad-",
                        "old_english": ["brad"],
                        "old_english_inflections": [
                            {"form": "bradan", "inflection": "weak_oblique"},
                        ],
                    }
                ],
            },
            {
                "meaning": ["Homestead"],
                "modifier_tags": ["architecture"],
                "modifier_type": "Architectural",
                "words": [{"modern_usage": "-ham", "old_english": ["ham"]}],
            },
        ]
    )
    n = Name("Bradham")
    n.find_meaning(word_db)
    assert n.count_unaccounted() == 0


def test_shadow_meanings_carry_lemma_metadata() -> None:
    """Post-match consumers (proportions, explainer) need the shadow
    Meaning to carry the lemma's tags + meanings + sources so they
    can't tell the difference. Avoid having to special-case
    'this match was via an inflection' downstream."""
    word_db, _ = load_meanings(
        [
            {
                "meaning": ["Cottage", "Hut"],
                "modifier_tags": ["architecture", "social"],
                "modifier_type": "Architectural",
                "words": [
                    {
                        "modern_usage": "-cot",
                        "old_english": ["cot"],
                        "old_english_inflections": [
                            {"form": "cotum", "inflection": "dative_or_pl"},
                        ],
                    }
                ],
            }
        ]
    )
    canonical = word_db["-cot"][0]
    shadow = word_db["-cotum"][0]
    assert shadow.meanings == canonical.meanings
    assert shadow.tags == canonical.tags
    assert shadow.sources == canonical.sources
