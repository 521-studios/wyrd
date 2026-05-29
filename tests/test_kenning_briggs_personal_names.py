"""Tests for the Briggs EPNS personal-names ingester (wyrd-uzoh / wyrd-jac1).

Covers:
- Text-shaping primitives: ``_normalize_form``, ``_strip_uncertainty``,
  ``_expand_bracket_variants``.
- Column reconstruction: ``_split_pages``, ``_column_reconstruct``.
- Entry parsing: ``_entry_blocks``, ``_parse_entry``,
  ``_parse_attestations``.
- Schema: UNIQUE indexes + cascade delete via ``personal_name`` /
  ``personal_name_toponym_attestation``.
- End-to-end ingest: fresh DB + fixture .txt → expected row counts;
  idempotent re-run produces zero net inserts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning import briggs_personal_names_ingester as briggs
from wyrd.generators.kenning.briggs_personal_names_ingester import (
    SOURCE_DOC,
    AttestationRecord,
    IngestStats,
    PersonalNameRecord,
    _column_reconstruct,
    _entry_blocks,
    _expand_bracket_variants,
    _normalize_form,
    _parse_attestations,
    _parse_entry,
    _split_pages,
    _strip_uncertainty,
    emit_briggs_jsonl,
    ingest_briggs_index,
    parse_briggs_index,
)
from wyrd.generators.kenning.cli.lexicon.ingest_briggs_personal_names import (
    lexicon_ingest_briggs_personal_names,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

# ---------------------------------------------------------------------
# _normalize_form: diacritic-strip + UTF-8 fold
# ---------------------------------------------------------------------


def test_normalize_form_strips_combining_diacritics() -> None:
    """NFD-decomposable diacritics drop and the base letter survives:
    Ēadwulf → 'eadwulf', Ælfgār → 'aelgar' fragments composed in fold."""
    assert _normalize_form("Ēadwulf") == "eadwulf"
    assert _normalize_form("Cēolwulf") == "ceolwulf"
    assert _normalize_form("Wīgmund") == "wigmund"


def test_normalize_form_maps_germanic_letters_via_fold() -> None:
    """Letters with no NFD decomposition (æ, þ, ð, ƿ, ø, ł) ride the
    explicit fold map. ð → 'd' is the Briggs-convention fold (not 'dh')
    — pins the chosen rule against accidental reversal."""
    assert _normalize_form("Ælfwine") == "aelfwine"
    assert _normalize_form("Þorgrim") == "thorgrim"
    assert _normalize_form("ðōr") == "dor"
    assert _normalize_form("Œdipa") == "oedipa"
    assert _normalize_form("Wƿulf") == "wwulf"


def test_normalize_form_lowercases_and_strips() -> None:
    assert _normalize_form("  Æthelred  ") == "aethelred"
    assert _normalize_form("EADGAR") == "eadgar"


# ---------------------------------------------------------------------
# _strip_uncertainty: ? / ?? prefix peeling
# ---------------------------------------------------------------------


def test_strip_uncertainty_handles_three_states() -> None:
    """No prefix → not uncertain. ? → uncertain. ?? → uncertain AND
    serious doubt (?? is a strict refinement of ?)."""
    assert _strip_uncertainty("Abel") == ("Abel", False, False)
    assert _strip_uncertainty("?Abel") == ("Abel", True, False)
    assert _strip_uncertainty("??Abel") == ("Abel", True, True)


def test_strip_uncertainty_does_not_strip_internal_question_marks() -> None:
    """Only leading ? / ?? peel — internal '?' inside a token stays."""
    cleaned, _, _ = _strip_uncertainty("Wha?t")
    assert cleaned == "Wha?t"


# ---------------------------------------------------------------------
# _expand_bracket_variants: Ab(b)a → ['Aba', 'Abba']
# ---------------------------------------------------------------------


def test_expand_bracket_variants_returns_two_forms_when_paren_present() -> None:
    """`Ab(b)a` is Briggs's compact form for Aba + Abba — both share
    the same attestations downstream."""
    assert _expand_bracket_variants("Ab(b)a") == ["Aba", "Abba"]


def test_expand_bracket_variants_handles_trailing_paren_letter() -> None:
    """`Gǣg(a)` → [`Gǣg`, `Gǣga`] (the parenthesized letter is optional
    at the end of the name)."""
    assert _expand_bracket_variants("Gǣg(a)") == ["Gǣg", "Gǣga"]


def test_expand_bracket_variants_returns_single_form_when_no_paren() -> None:
    assert _expand_bracket_variants("Abel") == ["Abel"]
    assert _expand_bracket_variants("Ēadwulf") == ["Ēadwulf"]


# ---------------------------------------------------------------------
# _split_pages: detect page-number-line boundaries
# ---------------------------------------------------------------------


def test_split_pages_breaks_on_pure_decimal_line() -> None:
    """A line whose stripped content is a pure decimal integer (<= 4
    digits) is dropped and acts as a page boundary."""
    lines = [
        "Aalfra DLV.",
        "Abel PASE1.",
        "  3  ",  # page marker (centered)
        "Brand fem.",
        "Briggs DLV.",
    ]
    pages = _split_pages(lines)
    assert len(pages) == 2
    assert pages[0] == ["Aalfra DLV.", "Abel PASE1."]
    assert pages[1] == ["Brand fem.", "Briggs DLV."]


def test_split_pages_does_not_break_on_intra_body_numbers() -> None:
    """A date qualifier like ``1601`` appears inside an entry body, NOT
    as a standalone page marker. The line containing it has surrounding
    text, so .strip() is non-pure-digit and the line stays."""
    lines = [
        "Abel PASE1 Ablishmare 1601 in Benson (O).",
        "  3  ",
        "Brand fem.",
    ]
    pages = _split_pages(lines)
    assert len(pages) == 2
    assert pages[0] == ["Abel PASE1 Ablishmare 1601 in Benson (O)."]


def test_split_pages_drops_trailing_empty_bucket() -> None:
    """File ending with a page marker doesn't leave an empty trailing
    page list."""
    lines = ["Abel PASE1.", "3"]
    pages = _split_pages(lines)
    assert len(pages) == 1


# ---------------------------------------------------------------------
# _column_reconstruct: two-column layout → linear stream
# ---------------------------------------------------------------------


def test_column_reconstruct_emits_left_then_right() -> None:
    """The two-column flow is left-top-to-bottom then right-top-to-
    bottom — each column read independently."""
    # Lines are exactly 80 chars wide (left:0-37, right:38-79).
    page = [
        "Aalfra DLV.                           Abba PASE3 Abbington (Bk).",
        "Abel PASE1 (OE) Ablishmare (Le).      Aca DLV; Acton (ND).",
    ]
    out = _column_reconstruct(page)
    parts = out.split("\n")
    # Left column first
    assert parts[0].startswith("Aalfra DLV.")
    assert parts[1].startswith("Abel PASE1")
    # Then right column
    right_start = next((i for i, p in enumerate(parts) if p.startswith("Abba PASE3")), None)
    assert right_start is not None, "right-column marker 'Abba PASE3' missing from output"
    assert right_start > 1
    assert parts[right_start + 1].startswith("Aca DLV")


def test_column_reconstruct_treats_short_lines_as_left_only() -> None:
    """A line shorter than the column boundary contributes only to the
    left column; the corresponding right slot stays blank."""
    page = [
        "Short left only.",
        "Longer line with right side here.       Right column content here.",
    ]
    out = _column_reconstruct(page)
    # The short-left line is present
    assert "Short left only." in out
    # And the right-side content is present (in the right column block)
    assert "Right column content" in out


# ---------------------------------------------------------------------
# _entry_blocks: split a stream into one entry per blob
# ---------------------------------------------------------------------


def test_entry_blocks_splits_on_sentence_terminator() -> None:
    """Period-followed-by-whitespace-then-capital terminates an entry.
    Intra-body periods (``n.d.``, ``pp.61–4``) don't split."""
    stream = "Abel PASE1 Ablishmare (Le). Brand fem Brandford (D). Carl PASE2."
    blobs = list(_entry_blocks(stream))
    assert len(blobs) == 3
    assert blobs[0].startswith("Abel")
    assert blobs[1].startswith("Brand")
    assert blobs[2].startswith("Carl")


def test_entry_blocks_drops_section_markers() -> None:
    """Alphabet section headers (—A—, —B—) are typographic, not entries,
    and must not appear as blobs."""
    stream = "—A— Abel PASE1. —B— Brand fem."
    blobs = list(_entry_blocks(stream))
    assert len(blobs) == 2
    assert all("—" not in b for b in blobs)


def test_entry_blocks_drops_index_header_echo() -> None:
    """The literal ``The index`` running-head appears at the top of every
    page in the source PDF; it's not an entry."""
    stream = "The index Abel PASE1."
    blobs = list(_entry_blocks(stream))
    assert len(blobs) == 1
    assert "The index" not in blobs[0]


def test_entry_blocks_does_not_split_on_nd_abbreviation() -> None:
    """``n.d.`` (no date) is an intra-body abbreviation, not an entry
    boundary — followed by lowercase, so the terminator regex (which
    requires a capital after the period) doesn't fire."""
    stream = "Abel PASE1 Ablishmare n.d. in Benson (O). Brand."
    blobs = list(_entry_blocks(stream))
    assert len(blobs) == 2
    assert "n.d." in blobs[0]


# ---------------------------------------------------------------------
# _parse_entry: full pipeline on golden entries
# ---------------------------------------------------------------------


def test_parse_entry_minimal_dlv_only() -> None:
    """Bare 'Headform DLV' produces a personal-name with no attestations
    + has_dlv set."""
    entry = _parse_entry("Aalfra DLV")
    assert entry is not None
    assert entry.name.headform == "Aalfra"
    assert entry.name.has_dlv is True
    assert entry.name.pase_count is None
    assert entry.attestations == []


def test_parse_entry_pase_count_extraction() -> None:
    """PASE1, PASE17, PASE350 all parse to the integer count."""
    for raw, expected in (("Abel PASE1", 1), ("Abel PASE17", 17), ("Abel PASE350", 350)):
        entry = _parse_entry(raw)
        assert entry is not None and entry.name.pase_count == expected


def test_parse_entry_distinguishes_lang_hint_from_county_tag() -> None:
    """A leading ``(OE)`` paren after the citation block is a language
    hint, NOT a county code. The disambiguator: every comma-token inside
    the paren must be in LANGUAGE_HINTS."""
    entry = _parse_entry("Abel PASE1 (Bib, OE) Ablishmare (Le)")
    assert entry is not None
    assert entry.name.language_hints == ["Bib", "OE"]
    # And the (Le) county tag survives — it's the trailing attestation.
    assert len(entry.attestations) == 1
    assert entry.attestations[0].county_code == "Le"


def test_parse_entry_does_not_consume_county_as_lang_hint() -> None:
    """If the first paren group contains an unknown-to-LANGUAGE_HINTS
    token it's a county tag, NOT a lang hint. The entry below has just
    Ablishmare (Le) — no language hints."""
    entry = _parse_entry("Abel PASE1 Ablishmare (Le)")
    assert entry is not None
    assert entry.name.language_hints == []
    assert len(entry.attestations) == 1


def test_parse_entry_handles_fem_marker_after_headform() -> None:
    """``Abarhilda fem PASE3`` — the bare 'fem' token after the headform
    sets is_feminine, then the citation block continues."""
    entry = _parse_entry("Abarhilda fem PASE3")
    assert entry is not None
    assert entry.name.is_feminine is True
    assert entry.name.pase_count == 3


def test_parse_entry_handles_fem_inside_lang_hint_paren() -> None:
    """Some ME-form names express fem inside the lang-hint paren:
    ``Mary PASE1 (fem, OE) Maryhall (D)``. The ingester promotes it
    to is_feminine and drops it from language_hints."""
    entry = _parse_entry("Mary PASE1 (fem, OE) Maryhall (D)")
    assert entry is not None
    assert entry.name.is_feminine is True
    assert entry.name.language_hints == ["OE"]


def test_parse_entry_ascharter_refs_accumulate() -> None:
    """Multiple ASCh tokens in the head accumulate to a list."""
    entry = _parse_entry("Aetheric ASCh7–8.72 ASCh12.5 Aetherstone (O)")
    assert entry is not None
    assert entry.name.ascharter_refs == ["ASCh7–8.72", "ASCh12.5"]


def test_parse_entry_returns_none_for_unparseable_blob() -> None:
    """Garbage with no leading capital headform returns None."""
    assert _parse_entry("") is None
    assert _parse_entry("(just a paren tag)") is None


# ---------------------------------------------------------------------
# _parse_attestations: county-tag detection, "X in Y" extraction
# ---------------------------------------------------------------------


def test_parse_attestations_county_inherited_within_group() -> None:
    """Multiple toponyms in one semicolon group share the trailing
    county tag: ``Abel, Cain, Daniel (Bk)`` → 3 attestations all (Bk)."""
    atts = list(_parse_attestations("Abel, Cain, Daniel (Bk)"))
    assert len(atts) == 3
    assert all(a.county_code == "Bk" for a in atts)
    assert [a.toponym_form for a in atts] == ["Abel", "Cain", "Daniel"]


def test_parse_attestations_separate_county_per_semicolon_group() -> None:
    """Groups separated by ``;`` each carry their own county tag."""
    atts = list(_parse_attestations("Abingdon (Bk); Acton (ND)"))
    assert len(atts) == 2
    assert atts[0].county_code == "Bk"
    assert atts[1].county_code == "ND"


def test_parse_attestations_skips_lang_hint_paren_groups() -> None:
    """A group whose tag is in LANGUAGE_HINTS (e.g. ``(OE)``) is a stray
    language hint, not an attestation."""
    atts = list(_parse_attestations("(OE); Abingdon (Bk)"))
    assert len(atts) == 1
    assert atts[0].county_code == "Bk"


def test_parse_attestations_skips_group_without_county_tag() -> None:
    """Group lacking a trailing (CC) paren is malformed for our purposes
    and dropped (catches mid-body citation residue like ``ASCh7–8.72``)."""
    atts = list(_parse_attestations("ASCh7–8.72; Abingdon (Bk)"))
    assert len(atts) == 1


def test_parse_attestations_x_in_y_no_date() -> None:
    """``Aelfeh in Aughton (We)`` — X is the attested variant of the PN
    as it appeared in toponym Y. Y is the modern toponym."""
    atts = list(_parse_attestations("Aelfeh in Aughton (We)"))
    assert len(atts) == 1
    assert atts[0].toponym_form == "Aughton"
    assert atts[0].attested_variant == "Aelfeh"
    assert atts[0].date_qualifier is None


def test_parse_attestations_x_date_in_y() -> None:
    """``Abban wylle 996 in Benson (O)`` — the date sits between X and
    'in Y'. X stays attested_variant, Y stays toponym_form."""
    atts = list(_parse_attestations("Abban wylle 996 in Benson (O)"))
    assert len(atts) == 1
    assert atts[0].toponym_form == "Benson"
    assert atts[0].attested_variant == "Abban wylle"
    assert atts[0].date_qualifier == "996"


def test_parse_attestations_date_no_in_clause() -> None:
    """``Abingdon 1257 (Bk)`` — date qualifier present, no "in Y" → the
    pre-date form IS the toponym."""
    atts = list(_parse_attestations("Abingdon 1257 (Bk)"))
    assert len(atts) == 1
    assert atts[0].toponym_form == "Abingdon"
    assert atts[0].attested_variant is None
    assert atts[0].date_qualifier == "1257"


def test_parse_attestations_uncertainty_prefixes_propagate() -> None:
    """? → is_uncertain; ?? → is_uncertain AND is_serious_doubt."""
    atts = list(_parse_attestations("?Abel, ??Brand (Bk)"))
    assert len(atts) == 2
    assert atts[0].is_uncertain is True and atts[0].is_serious_doubt is False
    assert atts[1].is_uncertain is True and atts[1].is_serious_doubt is True


def test_parse_attestations_unknown_county_preserved_for_visibility() -> None:
    """Unknown county codes are NOT filtered at parse time — the parser
    yields the row so the ingester can count + drop. Single seam."""
    atts = list(_parse_attestations("Brett (Zz)"))
    assert len(atts) == 1
    assert atts[0].county_code == "Zz"
    assert atts[0].county_canonical == ""


# ---------------------------------------------------------------------
# AttestationRecord: ?? promotes is_uncertain via __post_init__
# ---------------------------------------------------------------------


def test_attestation_record_serious_doubt_implies_uncertain() -> None:
    """``is_serious_doubt=True`` with ``is_uncertain=False`` is a parser
    bug — silently promote rather than reject."""
    rec = AttestationRecord(
        toponym_form="X",
        attested_variant=None,
        county_code="Bk",
        date_qualifier=None,
        is_uncertain=False,
        is_serious_doubt=True,
        raw_text="X",
    )
    assert rec.is_uncertain is True


def test_attestation_record_county_canonical_lookup() -> None:
    """Known county code → canonical name; unknown → empty string."""
    known = AttestationRecord(
        toponym_form="X",
        attested_variant=None,
        county_code="Bk",
        date_qualifier=None,
        is_uncertain=False,
        is_serious_doubt=False,
        raw_text="X",
    )
    unknown = AttestationRecord(
        toponym_form="X",
        attested_variant=None,
        county_code="Zz",
        date_qualifier=None,
        is_uncertain=False,
        is_serious_doubt=False,
        raw_text="X",
    )
    assert known.county_canonical == "Buckinghamshire"
    assert unknown.county_canonical == ""


# ---------------------------------------------------------------------
# PersonalNameRecord: __post_init__ derives normalized_form
# ---------------------------------------------------------------------


def test_personal_name_record_post_init_re_derives_normalized_form() -> None:
    """A caller that passes a stale normalized_form gets it silently
    overwritten — pins the no-drift invariant."""
    rec = PersonalNameRecord(headform="Ēadwulf", normalized_form="stale-value")
    assert rec.normalized_form == "eadwulf"


# ---------------------------------------------------------------------
# Fixture builder + end-to-end ingest
# ---------------------------------------------------------------------


# A self-contained Briggs-shaped fixture exercising every behavior the
# integration tests pin. Every line stays ≤38 chars so the column
# reconstructor treats them as left-column-only; intra-entry line wraps
# (Abel's body spans 3 lines) test the within-column wrap-join path.
# Two-column-pdftotext layout has its own focused unit tests above.
_FIXTURE_TEXT = """\
—A—
Aalfra DLV.
Ab(b)a PASE1 Abbey (Bk).
Abel PASE2 (Bib, OE) Ablishmare
1601 in Benson (O); Abelhall (Sf).
Abarhilda fem PASE3.
Ælfheah PASE4 Aelfeh
in Aughton (We).
3
—B—
?Brand fem (OE)
??Brandford (D); Brett (Zz).
"""


def _write_fixture(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the fixture .txt and monkeypatch the front-matter line
    cutoff to zero (the real source's 794-line front matter is irrelevant
    to behavior under test)."""
    path.write_text(_FIXTURE_TEXT, encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    return path


def test_parse_briggs_index_yields_expected_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline on the fixture: 5 source entries + Ab(b)a expansion
    = 6 yielded ParsedEntry rows."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    entries = list(parse_briggs_index(src))
    headforms = [e.name.headform for e in entries]
    assert headforms == ["Aalfra", "Aba", "Abba", "Abel", "Abarhilda", "Ælfheah", "Brand"]
    # Bracket-variant clones share the same attestations.
    aba = next(e for e in entries if e.name.headform == "Aba")
    abba = next(e for e in entries if e.name.headform == "Abba")
    assert aba.attestations[0].toponym_form == "Abbey"
    assert abba.attestations[0].toponym_form == "Abbey"


def test_emit_briggs_jsonl_drops_unknown_county_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Brett (Zz)`` is yielded by the parser but NOT emitted to JSONL
    — the emit-side filter keeps the rebuild path from materializing
    unknown-county rows. Counter bumps for visibility."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    out = tmp_path / "briggs.jsonl"
    stats = emit_briggs_jsonl(src, out)
    assert stats.attestations_unknown_county == 1
    # No attestation row carries Brett as toponym_form (Brett is the
    # unknown-county entry the emit filter drops). The Brand entry's
    # raw_entry payload retains the original text — that's a PN-side
    # provenance field, not an attestation, so we scope this check to
    # attestation events only.
    att_topo_forms = {
        parsed["toponym_form"]
        for line in out.read_text(encoding="utf-8").splitlines()
        if (parsed := json.loads(line)).get("_type") == "personal_name_toponym_attestation"
    }
    assert "Brett" not in att_topo_forms
    assert "Brandford" in att_topo_forms


def test_ingest_briggs_index_populates_personal_name_and_attestation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh DB → fixture in → expected row counts.

    The fixture exercises: bracket-variant expansion (Ab(b)a → 2 rows),
    DLV-only entry (Aalfra, no attestations), fem-marker (Abarhilda),
    "X in Y" extraction (Aelfeh in Aughton), diacritic-bearing headform
    (Ælfheah → normalized 'aelfheah'), unknown-county filter (Brett(Zz)
    dropped)."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    jsonl = tmp_path / "briggs.jsonl"
    with LexiconDB(fresh_db_for(tmp_path)) as db:
        stats = ingest_briggs_index(db, src, jsonl_path=jsonl)
        pns = list(
            db.conn.execute(
                "SELECT headform, normalized_form, is_feminine, pase_count, "
                "       has_dlv, source_doc "
                "  FROM personal_name ORDER BY headform"
            )
        )
        atts = list(
            db.conn.execute(
                "SELECT pn.headform, a.toponym_form, a.attested_variant, "
                "       a.county_code, a.county_canonical, a.date_qualifier, "
                "       a.is_uncertain, a.is_serious_doubt "
                "  FROM personal_name_toponym_attestation a "
                "  JOIN personal_name pn ON pn.id = a.personal_name_id "
                " ORDER BY pn.headform, a.toponym_form"
            )
        )
    # 7 PN rows: Aalfra + Aba + Abba (bracket-expanded) + Abel + Abarhilda
    # + Ælfheah + Brand.
    assert {row["headform"] for row in pns} == {
        "Aalfra",
        "Aba",
        "Abba",
        "Abel",
        "Abarhilda",
        "Ælfheah",
        "Brand",
    }
    assert stats.pn_rows_emitted == 7

    # Diacritic-fold pin: Ælfheah → aelfheah.
    aelfheah = next(row for row in pns if row["headform"] == "Ælfheah")
    assert aelfheah["normalized_form"] == "aelfheah"

    # PASE counts + DLV markers.
    aalfra = next(row for row in pns if row["headform"] == "Aalfra")
    assert aalfra["has_dlv"] == 1 and aalfra["pase_count"] is None
    abarhilda = next(row for row in pns if row["headform"] == "Abarhilda")
    assert abarhilda["is_feminine"] == 1 and abarhilda["pase_count"] == 3
    brand = next(row for row in pns if row["headform"] == "Brand")
    assert brand["is_feminine"] == 1

    # Attestation counts:
    # - Aba(Bk), Abba(Bk) — 2 from bracket-clone
    # - Abel: Ablishmare (with 'in Benson' + 1601 date), Abelhall
    # - Ælfheah: Aughton (with 'Aelfeh' attested_variant)
    # - Brand: Brandford (D); Brett(Zz) dropped at emit
    # Total = 2 + 2 + 1 + 1 = 6
    assert stats.attestation_rows_emitted == 6
    assert stats.attestations_unknown_county == 1
    assert len(atts) == 6

    # "Ablishmare 1601 in Benson" pins both attested_variant and date.
    abel_in_benson = next(
        row for row in atts if row["headform"] == "Abel" and row["toponym_form"] == "Benson"
    )
    assert abel_in_benson["attested_variant"] == "Ablishmare"
    assert abel_in_benson["date_qualifier"] == "1601"

    # "Aelfeh in Aughton" pins the no-date X-in-Y branch.
    aelfeh = next(row for row in atts if row["headform"] == "Ælfheah")
    assert aelfeh["toponym_form"] == "Aughton"
    assert aelfeh["attested_variant"] == "Aelfeh"
    assert aelfeh["date_qualifier"] is None

    # "??Brandford" pins the serious-doubt + uncertain pair.
    brandford = next(row for row in atts if row["headform"] == "Brand")
    assert brandford["is_uncertain"] == 1 and brandford["is_serious_doubt"] == 1


def test_ingest_briggs_index_is_idempotent_on_re_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running on a populated DB inserts zero net new rows. Pins the
    UNIQUE(headform, source_doc) and COALESCE-padded attestation dedup."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    db_path = fresh_db_for(tmp_path)
    jsonl = tmp_path / "briggs.jsonl"
    with LexiconDB(db_path) as db:
        first = ingest_briggs_index(db, src, jsonl_path=jsonl)
        pn_count_after_first = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name"
        ).fetchone()["c"]
        att_count_after_first = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation"
        ).fetchone()["c"]
    with LexiconDB(db_path) as db:
        ingest_briggs_index(db, src, jsonl_path=jsonl)
        pn_count_after_second = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name"
        ).fetchone()["c"]
        att_count_after_second = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation"
        ).fetchone()["c"]
    # personal_names_inserted counts every row that ran through
    # _insert_personal_name (INSERT OR IGNORE + SELECT-back), so it
    # equals the input row count regardless of dedup. The row-count
    # delta below is the actual idempotency proof.
    assert first.personal_names_inserted > 0
    assert pn_count_after_first == pn_count_after_second
    assert att_count_after_first == att_count_after_second


def test_personal_name_cascade_delete_removes_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``personal_name_toponym_attestation`` has ON DELETE CASCADE on
    its FK to ``personal_name``. Pins the schema declaration against
    a future migration that drops the cascade."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    db_path = fresh_db_for(tmp_path)
    jsonl = tmp_path / "briggs.jsonl"
    with LexiconDB(db_path) as db:
        ingest_briggs_index(db, src, jsonl_path=jsonl)
        # SQLite requires PRAGMA foreign_keys = ON for cascades to fire.
        db.conn.execute("PRAGMA foreign_keys = ON")
        abel_id = db.conn.execute(
            "SELECT id FROM personal_name WHERE headform = 'Abel'"
        ).fetchone()["id"]
        before = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation "
            "WHERE personal_name_id = ?",
            (abel_id,),
        ).fetchone()["c"]
        db.conn.execute("DELETE FROM personal_name WHERE id = ?", (abel_id,))
        db.commit()
        after = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation "
            "WHERE personal_name_id = ?",
            (abel_id,),
        ).fetchone()["c"]
    assert before >= 2
    assert after == 0


def test_emit_briggs_jsonl_writes_source_doc_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every L2 JSONL must lead with exactly one ``_type: source`` row
    whose ``ref`` matches the SOURCE_DOC constant — the build pipeline
    keys file-level source resolution on this anchor."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    out = tmp_path / "briggs.jsonl"
    emit_briggs_jsonl(src, out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    parsed = json.loads(first_line)
    assert parsed["_type"] == "source"
    assert parsed["ref"] == SOURCE_DOC


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def fresh_db_for(tmp_path: Path) -> Path:
    """A fresh init_schema'd DB at a unique path under tmp_path."""
    p = tmp_path / "lexicon.db"
    init_schema(p)
    return p


# ---------------------------------------------------------------------
# Silent-skip counter coverage (wyrd-jac1 scope expansion)
# ---------------------------------------------------------------------
#
# Most counters pair with a `continue` statement somewhere in the parser
# (attestation_groups_skipped_no_county / _lang_only, entries_unparsed);
# entries_with_zero_attestations and entries_citation_only are
# observational counters that bump on entries that still yield. The
# tests below pin each behavior against a fixture that exercises the
# matching trigger. Without these, a future refactor that silently
# stops bumping a counter (or wires it to the wrong site) would leave
# operators blind to dropped data.


def test_parse_attestations_skipped_no_county_bumps_counter() -> None:
    """A group without a trailing ``(CC)`` tag is dropped; the counter
    bumps once per dropped group."""
    stats = IngestStats()
    list(_parse_attestations("ASCh7–8.72; Abingdon (Bk)", stats=stats))
    assert stats.attestation_groups_skipped_no_county == 1
    assert stats.attestation_groups_skipped_lang_only == 0


def test_parse_attestations_skipped_lang_only_bumps_counter() -> None:
    """A group whose ``(CC)`` slot is a language hint (``(OE)``) is
    dropped via the dedicated counter, not the no-county one."""
    stats = IngestStats()
    list(_parse_attestations("(OE); Abingdon (Bk)", stats=stats))
    assert stats.attestation_groups_skipped_lang_only == 1
    assert stats.attestation_groups_skipped_no_county == 0


def test_parse_attestations_stats_none_is_pure_call() -> None:
    """Passing ``stats=None`` (the default) is a no-op — the function
    still yields rows but doesn't try to mutate anything. Pins the
    pure-call contract used by unit tests."""
    out = list(_parse_attestations("Abingdon (Bk)"))
    assert len(out) == 1


def test_parse_briggs_index_entries_unparsed_bumps_on_none_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blob that ``_parse_entry`` rejects (no leading capital
    headform) bumps ``entries_unparsed``. Setup uses a lowercase-start
    chunk as the FIRST blob — it survives the split because
    RE_ENTRY_TERMINATOR only fires at period+capital boundaries, so a
    leading lowercase blob is preserved as its own segment until the
    first capital-start entry."""
    text = "—A—\nlowercase noise prefix. Abel PASE1 Abbey (Bk).\n"
    src = tmp_path / "briggs.txt"
    src.write_text(text, encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    stats = IngestStats()
    entries = list(parse_briggs_index(src, stats=stats))
    assert {e.name.headform for e in entries} == {"Abel"}
    assert stats.entries_unparsed == 1


def test_parse_briggs_index_zero_attestation_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry with no toponym attestations bumps
    ``entries_with_zero_attestations``. If the entry also has at least
    one citation reference (PASE / DLV / ASCh), it also bumps
    ``entries_citation_only`` — a strict subset relationship pinned
    here so future refactors can't widen one without the other."""
    text = (
        "—A—\n"
        "Aalfra DLV.\n"  # zero atts + citation-only
        "Abel PASE2 Ablishmare (Le).\n"  # has atts
        "Abarhilda fem.\n"  # zero atts + NO citation refs
    )
    src = tmp_path / "briggs.txt"
    src.write_text(text, encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    stats = IngestStats()
    entries = list(parse_briggs_index(src, stats=stats))
    # 3 entries total.
    assert len(entries) == 3
    # Aalfra + Abarhilda both have zero attestations.
    assert stats.entries_with_zero_attestations == 2
    # Only Aalfra (has DLV) qualifies as citation-only.
    assert stats.entries_citation_only == 1


def test_emit_briggs_jsonl_threads_stats_through_parse_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the IngestStats produced by ``emit_briggs_jsonl``
    accumulates the silent-skip counters from the parser pass. Pins the
    threading: a regression that disconnected stats from
    ``parse_briggs_index`` would leave these at zero."""
    text = (
        "—A—\n"
        "lowercase prefix.\n"  # entries_unparsed
        "Aalfra DLV.\n"  # zero atts + citation-only
        "Abel PASE2 Foo (Bk); (OE); Bar (D).\n"
        # ↑ "(OE)" is a standalone attestation group whose paren tag is
        # a language hint → bumps attestation_groups_skipped_lang_only.
    )
    src = tmp_path / "briggs.txt"
    src.write_text(text, encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    out = tmp_path / "briggs.jsonl"
    stats = emit_briggs_jsonl(src, out)
    assert stats.entries_unparsed >= 1
    assert stats.entries_with_zero_attestations >= 1
    assert stats.entries_citation_only >= 1
    assert stats.attestation_groups_skipped_lang_only >= 1


# ---------------------------------------------------------------------
# ascharter_refs JSONL serialization (wyrd-jac1: emit-side round-trip)
# ---------------------------------------------------------------------


def test_emit_briggs_jsonl_dedupes_repeated_headform_first_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two entries sharing a headform emit only ONE personal_name row — the
    first (richer) one — while BOTH entries' attestations still emit. Pins
    the ``seen_headforms`` first-wins invariant in ``emit_briggs_jsonl``: a
    switch to last-wins or per-entry emit would silently corrupt PN metadata
    and duplicate rows on rebuild."""
    src = tmp_path / "briggs.txt"
    # Same headform "Aba" in two entries; first carries PASE1, second PASE9.
    src.write_text("—A—\nAba PASE1 Foo (Bk).\nAba PASE9 Bar (D).\n", encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    out = tmp_path / "briggs.jsonl"
    emit_briggs_jsonl(src, out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    aba_pn = [r for r in rows if r.get("_type") == "personal_name" and r["headform"] == "Aba"]
    assert len(aba_pn) == 1  # deduped to a single PN row
    assert aba_pn[0]["pase_count"] == 1  # first wins (PASE1), not the later PASE9
    att_toponyms = {
        r["toponym_form"] for r in rows if r.get("_type") == "personal_name_toponym_attestation"
    }
    assert {"Foo", "Bar"} <= att_toponyms  # both entries' attestations still emit


def test_emit_briggs_jsonl_truncates_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``emit_briggs_jsonl`` opens the artifact in ``"w"`` mode — re-emitting
    to the same path REWRITES rather than appends, so line count and the
    single ``_type: source`` anchor stay stable. Guards the canonical-artifact
    contract that the DB-level idempotency tests can't see (INSERT OR IGNORE
    keeps DB counts stable even if emit silently flipped to append-mode)."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    out = tmp_path / "briggs.jsonl"
    emit_briggs_jsonl(src, out)
    first = out.read_text(encoding="utf-8").splitlines()
    emit_briggs_jsonl(src, out)
    second = out.read_text(encoding="utf-8").splitlines()
    assert len(first) == len(second)  # rewrite, not append
    source_anchors = [line for line in second if json.loads(line).get("_type") == "source"]
    assert len(source_anchors) == 1  # exactly one anchor after re-run


@pytest.mark.parametrize(
    ("body", "expected_date"),
    [
        ("Abington Hy3 (Bk)", "Hy3"),
        ("Acton Edw1 (D)", "Edw1"),
        ("Barton Ric2 (O)", "Ric2"),
        ("Carlton John (We)", "John"),
        ("Dalton Steph (Sf)", "Steph"),
    ],
)
def test_parse_attestations_regnal_year_date_qualifiers(body: str, expected_date: str) -> None:
    """``RE_DATE_PREFIX`` recognizes regnal-year date forms
    (``Hy[1-8] | Edw[1-3] | Ric[1-3] | John | Steph``), not just numeric
    years — a break in that alternation would misparse the date/toponym
    split. The toponym preceding the regnal year is kept; the year lands in
    ``date_qualifier``."""
    atts = list(_parse_attestations(body))
    assert len(atts) == 1
    assert atts[0].date_qualifier == expected_date
    assert atts[0].toponym_form == body.split()[0]


def test_emit_briggs_jsonl_serializes_ascharter_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry carrying an ASCh charter reference round-trips the
    ``ascharter_refs`` list into the personal_name JSONL row as a
    JSON-encoded payload (the emit-side counterpart to the parse-side
    ``test_parse_entry_ascharter_refs_accumulate``)."""
    src = tmp_path / "briggs.txt"
    src.write_text("—A—\nAetheric ASCh7–8.72 Aetherstone (O).\n", encoding="utf-8")
    monkeypatch.setattr(briggs, "INDEX_FIRST_LINE_NUM", 0)
    out = tmp_path / "briggs.jsonl"
    emit_briggs_jsonl(src, out)
    pn_rows = [
        parsed
        for line in out.read_text(encoding="utf-8").splitlines()
        if (parsed := json.loads(line)).get("_type") == "personal_name"
    ]
    aetheric = next(r for r in pn_rows if r["headform"] == "Aetheric")
    assert json.loads(aetheric["ascharter_refs"]) == ["ASCh7–8.72"]


# ---------------------------------------------------------------------
# _parse_attestations: malformed-group silent-skip guards (wyrd-jac1)
# ---------------------------------------------------------------------
#
# Each guard pairs with a `continue` that drops malformed text rather
# than emitting a bogus attestation. Pinning them keeps a refactor from
# turning a trailing punctuation artifact into a spurious row.


def test_parse_attestations_skips_empty_trailing_semicolon_group() -> None:
    """A trailing ``;`` produces an empty semicolon group that is
    skipped, not turned into an attestation."""
    atts = list(_parse_attestations("Abingdon (Bk);"))
    assert [a.toponym_form for a in atts] == ["Abingdon"]


def test_parse_attestations_skips_empty_comma_item() -> None:
    """An empty item inside a comma list (``Abel,, Cain (Bk)``) is
    skipped; the surrounding real toponyms still emit."""
    atts = list(_parse_attestations("Abel,, Cain (Bk)"))
    assert [a.toponym_form for a in atts] == ["Abel", "Cain"]
    assert all(a.county_code == "Bk" for a in atts)


def test_parse_attestations_skips_group_with_empty_head() -> None:
    """A group that is only a county tag with no toponym head (``(Bk)``)
    is skipped via the empty-head guard — no county-only attestation."""
    atts = list(_parse_attestations("(Bk); Acton (D)"))
    assert [a.toponym_form for a in atts] == ["Acton"]


def test_entry_blocks_skips_sub_two_char_fragment() -> None:
    """A terminator-delimited fragment shorter than 2 chars (an OCR
    speck like a stray ``X.``) is dropped, not yielded as an entry."""
    blocks = list(_entry_blocks("Aalfra DLV. X. Abel PASE1 Foo (Bk)."))
    assert all(len(b) >= 2 for b in blocks)
    assert not any(b.strip() == "X" for b in blocks)
    assert any(b.startswith("Aalfra") for b in blocks)
    assert any(b.startswith("Abel") for b in blocks)


def test_column_reconstruct_trims_leading_and_trailing_blank_rows() -> None:
    """Blank first/last page lines must not survive as spurious entry
    boundaries — both columns get their leading AND trailing blank slots
    trimmed before the linear join."""
    # Left field is exactly COLUMN_BOUNDARY wide; the right field begins
    # at that column. Derive the pad from the source constant so the test
    # tracks the boundary instead of silently desyncing if it changes.
    wide = "Aalfra DLV.".ljust(briggs.COLUMN_BOUNDARY) + "Abba PASE3 Abbington (Bk)."
    out = _column_reconstruct(["", wide, ""])
    lines = out.split("\n")
    assert lines[0].startswith("Aalfra DLV.")
    assert lines[-1].startswith("Abba PASE3")
    assert "" not in lines  # leading/trailing blanks trimmed from both columns


# ---------------------------------------------------------------------
# CLI command: end-to-end invocation (wyrd-jac1 — was 46% covered)
# ---------------------------------------------------------------------


def test_cli_ingest_briggs_personal_names_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``ingest-briggs-personal-names`` command, driven through Click's
    CliRunner: parses the fixture, writes the JSONL artifact at the
    requested path, populates both DB tables, and emits the operator
    summary + the wyrd-jac1 silent-skip visibility line — all on stderr."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    db_path = fresh_db_for(tmp_path)
    jsonl = tmp_path / "briggs.jsonl"

    result = CliRunner().invoke(
        lexicon_ingest_briggs_personal_names,
        [str(src), "--db", str(db_path), "--jsonl-out", str(jsonl)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert jsonl.exists()

    with LexiconDB(db_path) as db:
        pn = db.conn.execute("SELECT COUNT(*) AS c FROM personal_name").fetchone()["c"]
        att = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation"
        ).fetchone()["c"]
    assert pn == 7
    assert att == 6

    # All operator-facing output is emitted with err=True.
    err = result.stderr
    assert "Done in" in err
    assert "pn_emitted=7" in err
    assert "Silent-skip counters:" in err
    assert f"JSONL artifact: {jsonl}" in err


def test_cli_ingest_briggs_personal_names_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-invoking the command on the same DB is a no-op on row counts
    (the ingest's UNIQUE/dedup contract holds through the CLI path)."""
    src = _write_fixture(tmp_path / "briggs.txt", monkeypatch)
    db_path = fresh_db_for(tmp_path)
    jsonl = tmp_path / "briggs.jsonl"
    args = [str(src), "--db", str(db_path), "--jsonl-out", str(jsonl)]

    first = CliRunner().invoke(lexicon_ingest_briggs_personal_names, args, catch_exceptions=False)
    assert first.exit_code == 0
    with LexiconDB(db_path) as db:
        pn1 = db.conn.execute("SELECT COUNT(*) AS c FROM personal_name").fetchone()["c"]
        att1 = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation"
        ).fetchone()["c"]

    second = CliRunner().invoke(lexicon_ingest_briggs_personal_names, args, catch_exceptions=False)
    assert second.exit_code == 0
    with LexiconDB(db_path) as db:
        pn2 = db.conn.execute("SELECT COUNT(*) AS c FROM personal_name").fetchone()["c"]
        att2 = db.conn.execute(
            "SELECT COUNT(*) AS c FROM personal_name_toponym_attestation"
        ).fetchone()["c"]

    assert (pn1, att1) == (pn2, att2)
