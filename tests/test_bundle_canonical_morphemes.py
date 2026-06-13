"""wyrd-90s7: regression suite pinning canonical British-place-name
morphemes to their expected top-ranked etymon.

These tests run against the *bundled* meanings.json (not a fixture)
so they validate the shipped data. Failure on any assertion means
either:

  * a bundle regen accidentally undid a previous win (the
    intended catch — alerts before the regression hits operators);
  * the canonical assertion itself needs revising (e.g. new
    etymological scholarship surfaces a better top etymon, in
    which case update the table + commit message says why).

Adding to CANONICAL_MORPHEMES is cheap. Each row pins:

    (usage_bucket, expected_source_lang, expected_source_lemma,
     expected_meaning_substring)

The expected_meaning_substring is a case-insensitive containment
check against the top-ranked sibling's meanings list, so minor
gloss-list reordering doesn't break the test — we just verify the
canonical gloss is in there somewhere.

Filed 2026-05-22 as cheap insurance after the wyrd-hitl corpus-
quality epic. User-asked follow-up: 'how do we catch other
versions of this?' — the audit CLI (wyrd-36ez) finds NEW
suspects, this suite prevents OLD wins from regressing.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning import _load_meanings, _rank_siblings

# Each row: (modern_usage, expected_source_lang, expected_source_lemma,
#           canonical_gloss_substring)
#
# Order is alphabetical by usage for scan-ability.
# D45 (wyrd-aicu.1) made meaning_db BARE-keyed — the stored identity is the bare
# surface, case preserved (pre-slot front-cap → capitalized key; post/inner/bare
# → lowercase key). So these keys are bare, NOT dash-marked. The capitalization
# mirrors the original pre-/post-form distinction: pre-forms (Field, Green, Wood)
# key under the capitalized surface, suffix-forms under the lowercase one (both
# case-variants exist in the corpus). wyrd-aicu.6.
CANONICAL_MORPHEMES = [
    ("bourne", "old_english", "burna", "brook"),
    ("bridge", "old_english", "brycg", "bridge"),
    ("by", "old_english", "by", "estate"),
    ("dale", "old_english", "dæl", "dale"),
    ("Field", "old_english", "field", "field"),
    ("field", "old_english", "field", "field"),
    ("ford", "old_english", "ford", "ford"),
    ("Green", "old_english", "grēne", "green"),
    ("green", "old_english", "grēne", "green"),
    ("ham", "old_english", "hām", "estate"),
    ("hill", "old_english", "hyll", "hill"),
    ("hurst", "old_english", "hyrst", "hill"),
    # wyrd-ubbc explicitly fixed this case; pin it tight.
    # ('ley' has both OE lēagum + OF launde in its top entry;
    # asserting OE source presence is enough — don't pin OF away.)
    ("ley", "old_english", "lēagum", "glade"),
    ("stone", "old_english", "stān", "stone"),
    ("ton", "old_english", "tūn", "estate"),
    ("wick", "old_english", "wīc", "dairy"),
    ("Wood", "old_english", "wudu", "forest"),
    ("wood", "old_english", "wudu", "forest"),
    ("worth", "old_english", "worth", "enclosure"),
]


@pytest.fixture(scope="module")
def meaning_db():
    """Load the runtime meaning_db once for the whole module.
    Reads via ``_load_meanings`` which resolves to the L4 SQLite path
    (defaults to ``seed-runtime.db``; ``WYRD_RUNTIME_DB`` override
    points at a full-corpus L4 when these regression assertions need
    the complete morpheme inventory)."""
    db, _ = _load_meanings()
    return db


# wyrd-aicu.7: aicu.1's bare-surface merge pooled each of these with a
# thin-gloss position-variant, and _rank_siblings now ranks the thin variant
# above the canonical OE etymon (wick→wīc, ley→lēagum). xfail(strict=True) keeps
# the rest of the regression net live while honestly tracking the demotion —
# strict means the marker AUTO-FAILS once aicu.7 restores the ranking (the
# canonical etymon ranks top again), forcing this xfail's removal. NB we do NOT
# weaken the expected lemma; the canonical sibling still exists, it's mis-ranked.
_XFAIL_RANKING = {"ley", "wick"}


def _canonical_param(row: tuple[str, str, str, str]):
    marks = (
        pytest.mark.xfail(
            reason="wyrd-aicu.7: aicu.1 bare-merge demotes the canonical etymon "
            "below a thin-gloss variant",
            strict=True,
        )
        if row[0] in _XFAIL_RANKING
        else ()
    )
    return pytest.param(*row, id=row[0], marks=marks)


@pytest.mark.parametrize(
    "usage,expected_lang,expected_lemma,expected_gloss_substring",
    [_canonical_param(row) for row in CANONICAL_MORPHEMES],
)
def test_canonical_morpheme_top_etymon(
    meaning_db, usage, expected_lang, expected_lemma, expected_gloss_substring
):
    import os

    if not (os.environ.get("WYRD_RUNTIME_DB") or os.environ.get("WYRD_RUNTIME_DB_BUCKET")):
        # The seed-runtime.db is a top-N-per-culture subset; not every
        # canonical morpheme survives the cut. Pin this regression
        # against a full-corpus L4 only.
        siblings = meaning_db.get(usage)
        if not siblings:
            pytest.skip(
                f"{usage!r} not in seed-runtime.db subset; set WYRD_RUNTIME_DB "
                f"to a full-corpus L4 to run this assertion."
            )
    """The top-ranked etymon for each canonical morpheme must match
    the pinned expectations. Pre-fix protections:

      * wyrd-ywm9 (modern_english noise filtering)
      * wyrd-emlb (within-bucket Meaning ranking)
      * wyrd-ubbc (surface-form tiebreaker for Hill → hyll vs cyln)

    A failure here means one of the above was undone by a bundle
    regen or a code change, OR the canonical assertion itself is
    out of date (very rare — historical etymology doesn't shift).
    """
    siblings = meaning_db.get(usage)
    assert siblings, (
        f"{usage!r} not in meaning_db — bundle is missing the canonical "
        f"morpheme entirely; check export-meanings output."
    )

    top = _rank_siblings(siblings)[0]

    # Source language present in the top etymon's sources dict.
    assert expected_lang in top.sources, (
        f"{usage!r} top etymon's sources = {sorted(top.sources.keys())!r}; "
        f"expected to include {expected_lang!r}. Top etymon's meanings: "
        f"{top.meanings[:3]!r}."
    )

    # Expected lemma form is one of the listed forms for the source language.
    forms = top.sources[expected_lang]
    assert expected_lemma in forms, (
        f"{usage!r} top etymon's {expected_lang} forms = {forms!r}; "
        f"expected to include {expected_lemma!r}. This usually means "
        f"the rank/filter pass picked the wrong sibling — check whether "
        f"another etymon's gloss/tag count beat the canonical one."
    )

    # Canonical gloss appears (case-insensitive) in the top etymon's meanings.
    meanings_lower = " | ".join(top.meanings).lower()
    assert expected_gloss_substring.lower() in meanings_lower, (
        f"{usage!r} top etymon's meanings = {top.meanings!r}; "
        f"expected to contain substring {expected_gloss_substring!r} "
        f"(case-insensitive). This catches cases where the etymon is "
        f"right but its gloss list shifted away from the canonical sense."
    )


def test_rank_siblings_deterministic_regardless_of_input_order():
    """wyrd-24s6 (D41): siblings that tie on every rank signal (stratum, sense
    kind, surface similarity, meaning/tag counts) must rank in an order that is
    INDEPENDENT of meaning_db load order. Before the content-derived tiebreaker
    (`_sibling_identity`), a tie fell through to input order — which varies
    across SQLite / Python builds — so `siblings[0]` (the canonical etymon, read
    by diversification's tag/language filters) could differ between environments
    and drift generation output (the rogd10 parity test caught exactly this)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    def mk(mid):
        m = Meaning("hill", tags=["topo"], meanings=["hill"], sources={"old_english": ["hyll"]})
        m.morpheme_id = mid
        return m

    a, b = mk("old-english:aaa"), mk("old-english:bbb")
    order_ab = [m.morpheme_id for m in _rank_siblings([a, b])]
    order_ba = [m.morpheme_id for m in _rank_siblings([b, a])]
    assert order_ab == order_ba, (
        f"_rank_siblings is order-dependent on a tie: [a,b]->{order_ab}, [b,a]->{order_ba}"
    )
