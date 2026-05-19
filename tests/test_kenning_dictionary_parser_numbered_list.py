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
    """Toponym detection covers every saint-prefix shape that shows up
    in real OCR: Saint/Sainte (full canonical), St/St. (abbreviated
    masculine), Ste/Ste. (abbreviated feminine), Sts/Sts. (plural).
    All three trailing-token shapes (hyphen, space-period-space, plain
    space) must resolve."""
    text = textwrap.dedent("""
        1.  Catharina :  Sainte-Catherine (Île-de-France).
        2.  Petrus :  St-Pierre (Calvados).
        3.  Maria :  St. Marie-aux-Mines (Haut-Rhin).
        4.  Maria :  Ste-Marie (Charente).
        5.  Fides :  Ste. Foy (Aveyron).
        6.  Innocentes :  Sts-Innocents (Paris).
    """).strip()
    entries = parse_numbered_list_text(text)
    assert len(entries) == 6
    assert entries[0].toponym == "Sainte-Catherine"
    assert entries[1].toponym == "St-Pierre"
    # "St. Marie-aux-Mines" — period-space variant; toponym is the
    # space-separated form. Pin the actual extracted shape rather than
    # a loose startswith() so a regression that drops the period
    # support surfaces here.
    assert entries[2].toponym == "St. Marie-aux-Mines"
    assert entries[3].toponym == "Ste-Marie"
    assert entries[4].toponym == "Ste. Foy"
    assert entries[5].toponym == "Sts-Innocents"


def test_parse_numbered_list_diacritic_class_covers_french_capitals():
    """The toponym regex's first-letter class must accept every accented
    capital that shows up in French place-names. Pin the rare ones
    (Ë, Ï, Ô, Œ, Æ) so a regression that drops them breaks here. Real
    examples: Île-de-France (Î), Œuvre (Œ), Évreux (É)."""
    text = textwrap.dedent("""
        1. A : Saint-Évreux.
        2. B : Saint-Île-Aux-Moines.
        3. C : Saint-Œuvre.
        4. D : Saint-Ëpiphanie.
        5. E : Saint-Œuf.
    """).strip()
    entries = parse_numbered_list_text(text)
    assert len(entries) == 5
    assert entries[0].toponym == "Saint-Évreux"
    assert entries[1].toponym == "Saint-Île-Aux-Moines"
    assert entries[2].toponym == "Saint-Œuvre"
    assert entries[3].toponym == "Saint-Ëpiphanie"
    assert entries[4].toponym == "Saint-Œuf"


def test_parse_numbered_list_out_of_range_ordinal_in_body_is_continuation():
    """Bounds-aware boundary detection (round-2 fix): a continuation
    line that happens to start with a year-like token like '1900. ...'
    must NOT flush the in-progress entry when it falls outside
    [min_entry_number, max_entry_number]. Without this guard, body
    prose containing year citations or page references would shred
    the in-progress entry across two ParsedEntry rows.

    Regression case: pre-fix this test would yield 3 entries (1659,
    1900, 2000) with the 1659 body cut off mid-sentence; post-fix it
    yields 2 entries (1659, 2000) with 1900 absorbed into 1659's body."""
    text = textwrap.dedent("""
        1659.  Caradocus : Saint-Caradu. The form is first attested in
        1900.  This 1900-prefixed line should NOT start a new entry.

        2000.  Out-of-range upper boundary, would fall outside the
        saint-names section but here we test the bounds work.
    """).strip()
    # Saint-names range; 1900 is INSIDE [1659, 2138], so without the
    # bounds shrinking, the body's "1900." would split. Tighten max to
    # 1700 to push 1900 outside the range.
    entries = parse_numbered_list_text(text, min_entry_number=1659, max_entry_number=1700)
    # Only the 1659 entry should remain — 2000 is also out-of-range.
    assert len(entries) == 1
    assert entries[0].toponym == "Saint-Caradu"
    # The "1900." continuation got absorbed into the body, not flushed.
    assert "1900" in entries[0].body_text
    assert "should NOT start a new entry" in entries[0].body_text


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


# --- CLI dispatch + flag flow-through (round-2 test-coverage) -------------


def test_select_parser_and_run_routes_numbered_list_explicitly():
    """`_select_parser_and_run(text, 'numbered-list')` must dispatch to
    the numbered-list parser. Without this test, a regression that
    forgot the new `if parser == 'numbered-list'` branch would silently
    fall through to the auto path (skeat → alphabetical) and yield the
    pre-PR 67-entry parse instead of the 475-entry numbered parse."""
    from wyrd.generators.kenning.cli import _select_parser_and_run

    text = textwrap.dedent("""
        1659. Caradocus : Saint-Caradu (Côtes-du-Nord).
        1660. Garaunus : Saint-Chéron (Eure).
    """).strip()
    entries = _select_parser_and_run(text, "numbered-list")
    assert len(entries) == 2
    assert entries[0].toponym == "Saint-Caradu"


def test_select_parser_and_run_auto_does_not_pick_numbered_list():
    """`auto` mode must NOT pick the numbered-list parser even when the
    text has ordinal-shaped lines mixed with prose. wyrd-5af design
    intent: auto stays as skeat → alphabetical so a few stray
    ordinals in body prose can't flip a Mawer-shaped book onto the
    wrong parser. Pin so a future 'auto includes numbered-list' tweak
    has to think about this regression risk first."""
    from wyrd.generators.kenning.cli import _select_parser_and_run

    # Text with numbered-list shape — if `auto` were greedy it would
    # pick numbered-list and yield 2 entries; instead it should fall
    # through to skeat (which returns 0) → alphabetical (which returns
    # 0 for this stub). Either way, NOT 2.
    text = textwrap.dedent("""
        1659. Caradocus : Saint-Caradu (Côtes-du-Nord).
        1660. Garaunus : Saint-Chéron (Eure).
    """).strip()
    auto_entries = _select_parser_and_run(text, "auto")
    nl_entries = _select_parser_and_run(text, "numbered-list")
    # The numbered-list parser produces 2 from this text. If auto picked
    # it, auto would also return 2 — and the assertion below would fail.
    assert len(nl_entries) == 2
    assert len(auto_entries) != 2 or len(auto_entries) == 0


def test_select_parser_and_run_threads_min_max_entry_to_numbered_list():
    """`_select_parser_and_run` plumbs the `min_entry_number` and
    `max_entry_number` kwargs through to the numbered-list parser. A
    regression that dropped one of the kwargs at the dispatcher would
    leave the bounds story unreachable from the CLI surface — exactly
    the round-1 reviewer concern. Pin both bounds composing correctly."""
    from wyrd.generators.kenning.cli import _select_parser_and_run

    text = textwrap.dedent("""
        100. Pre-section.
        500. Saint-Foo.
        1000. Saint-Bar.
        1500. Saint-Baz.
        2000. Post-section.
    """).strip()
    entries = _select_parser_and_run(
        text, "numbered-list", min_entry_number=500, max_entry_number=1500
    )
    assert {e.toponym for e in entries} == {"Saint-Foo", "Saint-Bar", "Saint-Baz"}


def test_cli_min_entry_max_entry_flags_flow_through_to_parser(tmp_path, monkeypatch):
    """End-to-end CLI wiring (round-2 reviewer gap): `wyrd kenning
    lexicon mine-llm ... --parser numbered-list --min-entry 1659
    --max-entry 2138` must thread both bounds down to the parser. A
    regression that hardcoded the bounds, dropped a kwarg in the
    dispatcher chain, or renamed an option would slip past the unit
    tests above because they call `_select_parser_and_run` directly."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import llm_extractor
    from wyrd.generators.kenning.lexicon import init_schema

    captured: dict[str, object] = {}

    def stub_select(text, parser, *, min_entry_number=1, max_entry_number=None):
        captured["parser"] = parser
        captured["min_entry_number"] = min_entry_number
        captured["max_entry_number"] = max_entry_number
        return []

    # mine-llm imports `_select_parser_and_run` from cli.utils into its own
    # module namespace (post-wyrd-g143 slice 5). Patch the reference the
    # mine_llm module actually resolves to at call time, not the cli/utils
    # re-export — local-bound imports don't follow monkeypatch on a different
    # namespace.
    from wyrd.generators.kenning.cli.lexicon import mine_llm as _mine_llm_mod

    monkeypatch.setattr(_mine_llm_mod, "_select_parser_and_run", stub_select)

    class FakeClient:
        model = "fake"
        base_url = "http://localhost:0"

        def __init__(self, **_):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    src_path = tmp_path / "test_book.txt"
    src_path.write_text("dummy body\n")
    db_path = tmp_path / "test.db"
    init_schema(db_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-llm",
            str(src_path),
            "--db",
            str(db_path),
            "--provider",
            "ollama",
            "--parser",
            "numbered-list",
            "--min-entry",
            "1659",
            "--max-entry",
            "2138",
        ],
    )
    assert result.exit_code == 0, (result.output or "") + (result.stderr or "")
    assert captured.get("parser") == "numbered-list"
    assert captured.get("min_entry_number") == 1659
    assert captured.get("max_entry_number") == 2138


def test_cli_min_entry_default_is_one_max_entry_default_is_none(tmp_path, monkeypatch):
    """Default values for the new --min-entry / --max-entry flags must
    preserve the historic 'no filter' behavior. Pin so a future default
    change doesn't silently start filtering existing mining runs."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import llm_extractor
    from wyrd.generators.kenning.lexicon import init_schema

    captured: dict[str, object] = {}

    def stub_select(text, parser, *, min_entry_number=1, max_entry_number=None):
        captured["min_entry_number"] = min_entry_number
        captured["max_entry_number"] = max_entry_number
        return []

    # mine-llm imports `_select_parser_and_run` from cli.utils into its own
    # module namespace (post-wyrd-g143 slice 5). Patch the reference the
    # mine_llm module actually resolves to at call time, not the cli/utils
    # re-export — local-bound imports don't follow monkeypatch on a different
    # namespace.
    from wyrd.generators.kenning.cli.lexicon import mine_llm as _mine_llm_mod

    monkeypatch.setattr(_mine_llm_mod, "_select_parser_and_run", stub_select)

    class FakeClient:
        model = "fake"
        base_url = "http://localhost:0"

        def __init__(self, **_):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    src_path = tmp_path / "test_book.txt"
    src_path.write_text("dummy body\n")
    db_path = tmp_path / "test.db"
    init_schema(db_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-llm",
            str(src_path),
            "--db",
            str(db_path),
            "--provider",
            "ollama",
        ],
    )
    assert result.exit_code == 0, (result.output or "") + (result.stderr or "")
    assert captured.get("min_entry_number") == 1
    assert captured.get("max_entry_number") is None


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
