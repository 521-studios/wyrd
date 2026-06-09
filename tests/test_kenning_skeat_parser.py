"""Tests for the Skeat-format etymology parser.

The fixtures here are synthetic, hand-crafted to exercise specific patterns
the parser handles. We keep them short and structurally typical of Skeat
without reproducing real entries verbatim.
"""

from __future__ import annotations

from wyrd.generators.kenning.parsers.skeat import (
    _split_form_by_suffix,
    parse_entry,
    parse_skeat_text,
    parse_toc,
)

# Minimal synthetic fixture in Skeat's structural style. Two TOC sections,
# two body sections, four entries with different etymology patterns.
SYNTHETIC_SKEAT = """\
THE PLACE-NAMES OF NOWHERESHIRE.

CONTENTS.

§  1.    Prefatory Remarks 1
§  2.    The suffix -ton  : — Alpha, Beta, Gamma
§  3.    The suffix -ham  : — Delta, Epsilon


§  1.    Prefatory Remarks.

Some prose introduction we don't care about.


§  2.    The Suffix -ton.

Alpha. This is the A.S. fake-tun, lit. 'fake town'; from fake, an invented thing,
and tun, an enclosure.

Beta is spelt Betatone in Domesday. Here Beta represents A.S. Betan, gen. case
of Beta, a personal name.

Gamma. Spelt Gammatun in 1066. This is the A.S. gammatun.


§  3.    The Suffix -ham.

Delta. This is the A.S. delt-ham; from delt, a clearing, and ham, home.

Epsilon. Spelt Epsiloham in old charters. Possibly from a personal name Epsila.
"""


def test_parse_toc_extracts_sections_and_toponyms() -> None:
    toc = parse_toc(SYNTHETIC_SKEAT)
    assert toc == {
        "-ton": ["Alpha", "Beta", "Gamma"],
        "-ham": ["Delta", "Epsilon"],
    }


def test_parse_skeat_finds_all_toc_entries() -> None:
    entries = parse_skeat_text(SYNTHETIC_SKEAT)
    by_name = {e.toponym: e for e in entries}
    # Every TOC entry should be detected in the body.
    assert set(by_name) == {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}


def test_hyphenated_form_yields_high_confidence() -> None:
    entries = {e.toponym: e for e in parse_skeat_text(SYNTHETIC_SKEAT)}
    alpha = entries["Alpha"]
    assert alpha.confidence == "high"
    assert alpha.historical_form == "fake-tun"
    assert [(el.form, el.position) for el in alpha.elements] == [
        ("fake", "pre"),
        ("tun", "post"),
    ]
    # "from fake, an invented thing, and tun, an enclosure" — glosses attach.
    glosses = {el.form: el.gloss for el in alpha.elements}
    assert glosses["fake"] == "an invented thing"
    assert glosses["tun"] == "an enclosure"


def test_personal_name_pattern_yields_medium() -> None:
    entries = {e.toponym: e for e in parse_skeat_text(SYNTHETIC_SKEAT)}
    beta = entries["Beta"]
    assert beta.confidence == "medium"
    assert beta.section_suffix == "-ton"
    forms = [(el.form, el.position, el.inflection) for el in beta.elements]
    assert forms == [
        ("beta", "pre", "genitive"),
        ("ton", "post", None),
    ]


def test_unhyphenated_form_split_by_section_suffix() -> None:
    entries = {e.toponym: e for e in parse_skeat_text(SYNTHETIC_SKEAT)}
    gamma = entries["Gamma"]
    # 'gammatun' ends in 'tun' (the section suffix root) → split happens.
    assert gamma.confidence == "high"
    assert gamma.historical_form == "gammat-un" or gamma.historical_form == "gamma-tun"
    forms = [el.form for el in gamma.elements]
    assert forms == ["gamma", "tun"]


def test_split_form_by_suffix_helper() -> None:
    assert _split_form_by_suffix("ceaster-tun", "tun") == ("ceaster-", "tun")
    assert _split_form_by_suffix("dictun", "tun") == ("dic", "tun")
    assert _split_form_by_suffix("ham", "ham") is None  # too short
    assert _split_form_by_suffix("foo", "bar") is None
    assert _split_form_by_suffix("", "ton") is None


def test_parse_entry_low_confidence_when_no_etymology() -> None:
    body = "Foo. Some discursive prose with no A.S. forms or personal names."
    entry = parse_entry("Foo", body, suffix_hint="-ton")
    assert entry.confidence == "low"
    assert entry.elements == []
    assert entry.section_suffix == "-ton"


def test_lit_gloss_attaches_to_first_element_only() -> None:
    """Pattern D: a 'lit. ...' overall gloss with no 'from X ... and Y' clause
    lands on the FIRST recovered element and no other."""
    entry = parse_entry("Acton", "Acton. A.S. Ac-tun; lit. 'oak town'.", suffix_hint="-ton")
    assert [el.form for el in entry.elements] == ["ac", "tun"]
    assert entry.elements[0].gloss == "oak town"
    assert entry.elements[1].gloss is None


def test_personal_name_does_not_clobber_recovered_elements() -> None:
    """Pattern C only fires when nothing was recovered yet. A hyphenated A.S.
    form (Pattern A) already filled the elements, so the trailing 'gen. of X,
    a personal name' must NOT overwrite them (guards the `not entry.elements`)."""
    entry = parse_entry(
        "Acton", "Acton. A.S. Ac-tun, gen. of Acca, a personal name.", suffix_hint="-ton"
    )
    assert [el.form for el in entry.elements] == ["ac", "tun"]
    assert entry.confidence == "high"  # from Pattern A, not Pattern C's "medium"
    assert all(el.form != "acca" for el in entry.elements)


def test_oe_and_on_forms_set_their_language() -> None:
    """The O.E. and O.N. detection branches set the element language (the
    fixtures elsewhere only exercise A.S.)."""
    oe = parse_entry("Brackley", "Brackley. O.E. Bracca-leah.", suffix_hint="-ley")
    assert [el.form for el in oe.elements] == ["bracca", "leah"]
    assert {el.language for el in oe.elements} == {"old-english"}

    on = parse_entry("Kirkby", "Kirkby. O.N. Kirkju-byr.", suffix_hint="-by")
    assert [el.form for el in on.elements] == ["kirkju", "byr"]
    assert {el.language for el in on.elements} == {"old-norse"}


def test_historical_form_priority_as_beats_on() -> None:
    """A.S. is tried before O.N.; when both appear, the A.S. form wins."""
    entry = parse_entry("Mixed", "Mixed. A.S. Aac-tun and O.N. Foo-bar.", suffix_hint="-ton")
    assert [el.form for el in entry.elements] == ["aac", "tun"]
    assert {el.language for el in entry.elements} == {"old-english"}


def test_skeat_parser_source_quote_budget_matches_llm_extractor() -> None:
    """wyrd-9v1: the regex parser and the LLM extractor must share one
    source_quote budget so the SPA citation view sees consistent snippet
    sizes regardless of which parser produced the row.

    Lives in parsers.skeat (where ParsedEntry lives); llm_extractor
    re-exports it. This test pins the symbol-equality so a future
    divergence (someone redefines SOURCE_QUOTE_BUDGET in either module)
    gets caught."""
    from wyrd.generators.kenning.extractors.llm import (
        SOURCE_QUOTE_BUDGET as EXTRACTOR_BUDGET,
    )
    from wyrd.generators.kenning.parsers.skeat import SOURCE_QUOTE_BUDGET

    assert SOURCE_QUOTE_BUDGET == EXTRACTOR_BUDGET
    assert SOURCE_QUOTE_BUDGET == 500


def test_skeat_parser_shorten_truncates_at_budget() -> None:
    """wyrd-9v1: _shorten's default cap is the shared SOURCE_QUOTE_BUDGET.
    A body longer than the budget gets truncated with an ellipsis; the
    truncation point sits at a word boundary, never mid-word."""
    from wyrd.generators.kenning.parsers.skeat import (
        SOURCE_QUOTE_BUDGET,
        _shorten,
    )

    long_body = "alpha " * 200  # 1200 chars, plenty over the 500 budget
    out = _shorten(long_body)
    # Truncated to ≤ budget, ends with ellipsis (the trailing-space-then-
    # rsplit(' ', 1)[0] ensures a word-boundary cut).
    assert len(out) <= SOURCE_QUOTE_BUDGET + 1  # +1 for the ellipsis char
    assert out.endswith("…")
    assert "alpha" in out
    # Short body passes through unchanged (under the cap).
    short = "tiny body"
    assert _shorten(short) == short
