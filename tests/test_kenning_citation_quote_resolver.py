"""Unit tests for ``_resolve_quote_page`` — the pure quote->page resolver
extracted from ``backfill_citation_pages`` (wyrd-azv/wyrd-8st/wyrd-3yu).

These pin the ``(page, status, ambiguous)`` contract directly, including the
property that ``ambiguous`` is independent of ``status`` (a quote can resolve
to a page *and* be ambiguous, or be ambiguous yet land before the first page).
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.citations import _resolve_quote_page

# A normalized body where "quick" occurs twice (offsets 4 and 20) so it is
# ambiguous, while "brown fox" occurs once. Identity offset map for simplicity.
_NORM_TEXT = "the quick brown fox quick ends here today"
_NORM_TO_ORIG = list(range(len(_NORM_TEXT)))
# Running headers as (offset, page): page 5 from offset 4, page 6 from offset 20.
_HEADERS = [(4, 5), (20, 6)]


def _resolve(quote, *, headers=_HEADERS):
    return _resolve_quote_page(
        quote, norm_text=_NORM_TEXT, norm_to_orig=_NORM_TO_ORIG, headers=headers
    )


def test_none_quote_is_no_quote():
    assert _resolve(None) == (None, "no_quote", False)


def test_empty_body_after_attribution_prefix_is_no_quote():
    # _quote_body_excerpt drops everything before the first " | "; nothing left.
    assert _resolve("PROVIDER | ") == (None, "no_quote", False)


def test_quote_not_in_text():
    assert _resolve("PROV | not present anywhere") == (None, "quote_not_in_text", False)


def test_unique_match_resolves_to_page_unambiguously():
    # "brown fox" occurs once, at offset 10 -> page 5, not ambiguous.
    assert _resolve("PROV | brown fox") == (5, "ok", False)


def test_ambiguous_match_still_resolves_but_flags_ambiguous():
    # "quick" occurs at offsets 4 and 20; leftmost (4) -> page 5, flagged.
    assert _resolve("PROV | quick") == (5, "ok", True)


def test_ambiguous_is_independent_of_before_first_page_status():
    # Headers begin only at offset 30: the leftmost "quick" (offset 4) precedes
    # the first page, so status is before_first_page -- yet it is still ambiguous.
    page, status, ambiguous = _resolve("PROV | quick", headers=[(30, 7)])
    assert (page, status, ambiguous) == (None, "before_first_page", True)


def test_whitespace_runs_in_quote_are_normalized_before_matching():
    # The stored quote may carry OCR multi-spaces; matching is on normalized text.
    assert _resolve("PROV |   brown    fox  ") == (5, "ok", False)
