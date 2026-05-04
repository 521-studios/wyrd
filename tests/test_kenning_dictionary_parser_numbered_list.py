"""Tests for the numbered-list parser variant (wyrd-5af).

Targets ordinal-prefixed treatise sections like Longnon's "Les Noms de
lieu de la France" vol 2 saint-names, where each etymology entry begins
with a 3- or 4-digit ordinal followed by a period:

    1659.  Caradocus :  Saint-Caradu  (Côtes-du-Nord, ...)
    1660.  Garaunus, martyr (...) :  Saint-Chéron (Eure, ...)

The alphabetical-headword parser misses these (capital-letter anchoring
on the WRONG token); the numbered-list parser splits at ordinal
boundaries and yields one ParsedEntry per item.
"""

from __future__ import annotations

import textwrap

from wyrd.generators.kenning.dictionary_parser import parse_numbered_list_text


def test_parse_numbered_list_segments_each_ordinal():
    """Each `<n>. <body>` line starts a fresh entry; multi-line bodies
    join into one body_text per ordinal."""
    text = textwrap.dedent("""
        1659.  Caradocus :  Saint-Caradu  (Côtes-du-Nord, Morbihan),
        Saint-Cadreuc  (Côtes-du-Nord), Saint-Carreuc  (Côtes-du-Nord).

        1660.  Garaunus, martyr (Chartrain du Vᵉ siècle) :  Saint-Chéron
        (Eure, Eure-et-Loir, Marne, Sarthe, Seine-et-Oise).

        1661.  Carilefus :  Saint-Calais (Eure, Eure-et-Loir, Maine-
        et-Loire, Mayenne, Sarthe).
    """).strip()
    entries = parse_numbered_list_text(text)
    assert len(entries) == 3
    assert entries[0].toponym == "Saint-Caradu"
    assert "Caradocus" in entries[0].body_text
    assert "Saint-Cadreuc" in entries[0].body_text
    assert "Saint-Carreuc" in entries[0].body_text
    # Multi-line continuation joined into one body.
    assert entries[1].toponym == "Saint-Chéron"
    assert "Eure" in entries[1].body_text
    assert entries[2].toponym == "Saint-Calais"


def test_parse_numbered_list_min_entry_number_skips_front_matter():
    """`min_entry_number` lets the caller skip ordinals below a threshold,
    e.g. when the saint-names section opens at ordinal 1659 in Longnon
    vol 2 and the earlier numbered text is general-topographic, not the
    saint format we want. Without the filter, both sections would be
    parsed and the LLM would have to figure out the difference."""
    text = textwrap.dedent("""
        1.  General entry one body.

        2.  General entry two body.

        1659.  Caradocus :  Saint-Caradu (Côtes-du-Nord).

        1660.  Garaunus :  Saint-Chéron (Eure).
    """).strip()
    entries = parse_numbered_list_text(text, min_entry_number=1659)
    assert len(entries) == 2
    assert entries[0].toponym == "Saint-Caradu"
    assert entries[1].toponym == "Saint-Chéron"


def test_parse_numbered_list_max_entries_caps_output():
    """`max_entries` caps the result for smoke testing — only process
    the first N entries before stopping."""
    text = textwrap.dedent("""
        1.  Alpha :  Saint-Alpha (Loire).
        2.  Beta :  Saint-Beta (Loire).
        3.  Gamma :  Saint-Gamma (Loire).
        4.  Delta :  Saint-Delta (Loire).
    """).strip()
    entries = parse_numbered_list_text(text, max_entries=2)
    assert len(entries) == 2
    assert entries[0].toponym == "Saint-Alpha"
    assert entries[1].toponym == "Saint-Beta"


def test_parse_numbered_list_empty_toponym_when_no_saint_match():
    """Entries whose body has no detectable Saint-X form get
    `toponym=""`. The LLM validator will reject these anyway, but the
    parser doesn't filter them out — it's a parser, not a curator. Pin
    the empty-toponym shape so a future curator pass knows to skip
    them."""
    text = "1.  Some prose without any detectable place name token."
    entries = parse_numbered_list_text(text)
    assert len(entries) == 1
    assert entries[0].toponym == ""
    assert "prose" in entries[0].body_text


def test_parse_numbered_list_handles_sainte_and_st_variants():
    """Toponym detection covers feminine (Sainte-X) and the abbreviated
    'St' / 'St.' forms that occasionally surface in OCR'd French
    dictionaries."""
    text = textwrap.dedent("""
        1.  Catharina :  Sainte-Catherine (Île-de-France).
        2.  Petrus :  St-Pierre (Calvados).
        3.  Maria :  St. Marie-aux-Mines (Haut-Rhin).
    """).strip()
    entries = parse_numbered_list_text(text)
    assert len(entries) == 3
    assert entries[0].toponym == "Sainte-Catherine"
    assert entries[1].toponym == "St-Pierre"
    assert entries[2].toponym.startswith("St")


def test_parse_numbered_list_ignores_intra_paragraph_numbers():
    """Numbers that appear MID-LINE (page references, dates, century
    notation like 'Vᵉ siècle') don't start a new entry. Only ordinals
    at the START of a line do. Pin to prevent the regex from being
    inadvertently widened."""
    text = textwrap.dedent("""
        1.  Caradocus, attested in 1086. Cf. n° 1659 below for the
        derivative chain. The Saint-Caradu (Côtes-du-Nord) form
        is the modern reflex.
    """).strip()
    entries = parse_numbered_list_text(text)
    # Only one entry — the "1086" and "1659" inside the body don't
    # restart parsing.
    assert len(entries) == 1
    assert "1086" in entries[0].body_text
    assert "1659" in entries[0].body_text


def test_parse_numbered_list_empty_input_returns_empty():
    """Edge case: empty / whitespace-only input → empty list."""
    assert parse_numbered_list_text("") == []
    assert parse_numbered_list_text("   \n\n  \n") == []


def test_parse_numbered_list_no_numbered_lines_returns_empty():
    """Input with no ordinal-prefixed lines (e.g. pure prose) → empty
    list. The parser doesn't fall back to any other segmentation; the
    caller picked the wrong parser if this happens."""
    text = textwrap.dedent("""
        This is a treatise on place-names. It contains many sentences
        but no numbered list whatsoever. Saint-Pierre might be
        mentioned, but the parser doesn't extract it.
    """).strip()
    assert parse_numbered_list_text(text) == []


def test_parse_numbered_list_handles_longnon_ocr_artifacts():
    """Real OCR from Longnon vol 2 has scattered spaces, dropped
    diacritics, and noise tokens like 'C<Me.s-du-Nord' for
    'Côtes-du-Nord'. The parser should still segment by ordinal even
    when the body text is messy."""
    text = textwrap.dedent("""
        1659.  Caradocus  ;  Saint-Caradu  (C<Me.s-du-Nord,  Mo!])iliaii),
        Saint-Cadreuc  (Cùles-du-Nord),  Saint-Carreuc  (Cùtcs-du-
        Nord).

        1660.  r.ai-aunus.  luarlvr  (.'liartraiii  du  v''  siOcdo  :  Saillt-Ché-
        ron  (luire.   luiie-el-Loir,    Miirm",  SafLlu-,  Seiiie-et-Oise).
    """).strip()
    entries = parse_numbered_list_text(text)
    assert len(entries) == 2
    # Even with mangled OCR around Saint-Cadreuc, the first clean
    # Saint-Caradu match wins for the toponym field.
    assert entries[0].toponym == "Saint-Caradu"
    # Multi-line continuation joins despite OCR mid-word breaks.
    assert "Saint-Carreuc" in entries[0].body_text


def test_parse_numbered_list_source_quote_truncation():
    """Long entries get their `source_quote` truncated via the existing
    `_shorten` helper (SOURCE_QUOTE_BUDGET = 500 chars). The full body
    stays in `body_text` for the LLM extractor; the truncated quote is
    for the citation row."""
    long_body = "Caradocus : Saint-Caradu " + ("padding text " * 200)
    text = f"1659.  {long_body}"
    entries = parse_numbered_list_text(text)
    assert len(entries) == 1
    assert len(entries[0].source_quote) <= 503  # _shorten leaves slack for "..."
    # body_text is full-fidelity.
    assert len(entries[0].body_text) > 500


def test_parse_numbered_list_max_entries_zero_returns_empty():
    """Edge case: `max_entries=0` is a valid no-op cap."""
    text = "1. Caradocus : Saint-Caradu."
    assert parse_numbered_list_text(text, max_entries=0) == []


def test_parse_numbered_list_max_entry_number_caps_at_section_boundary():
    """`max_entry_number` skips ordinals ABOVE a threshold — used to
    drop the index-appendix at the tail of a numbered section. In
    Longnon vol 2 the saint-names body runs 1659-2138; ordinals past
    2138 are cross-reference index entries that the LLM will reject
    for $0.005 each. Pin so a regression that drops the cap surfaces
    here rather than as a surprise mining bill."""
    text = textwrap.dedent("""
        1659.  Caradocus :  Saint-Caradu (Côtes-du-Nord).
        2138.  Last :  Saint-Last (Final).
        2347.  IndexEntry.  See 2105 above for the actual etymology.
        2400.  AnotherIndex.  Cross-ref to 1659.
    """).strip()
    entries = parse_numbered_list_text(text, min_entry_number=1659, max_entry_number=2138)
    assert len(entries) == 2
    assert entries[0].toponym == "Saint-Caradu"
    assert entries[1].toponym == "Saint-Last"


def test_parse_numbered_list_min_and_max_entry_number_combine():
    """Both bounds compose: only entries in [min, max] survive."""
    text = textwrap.dedent("""
        100. Pre-section.
        500. Saint-Foo (Loire).
        1000. Saint-Bar (Sarthe).
        1500. Saint-Baz (Calvados).
        2000. Post-section.
    """).strip()
    entries = parse_numbered_list_text(text, min_entry_number=500, max_entry_number=1500)
    assert len(entries) == 3
    assert {e.toponym for e in entries} == {
        "Saint-Foo",
        "Saint-Bar",
        "Saint-Baz",
    }
