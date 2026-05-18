"""Tests for the Wikipedia place-name backfill report (wyrd-4453).

Builds small in-memory lexicon DBs and small synthetic
``*_place_names.json`` fixtures, then exercises the categorization
logic plus the CLI shim. Hermetic — no live DB or filesystem
dependencies outside ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.wikipedia_backfill_report import (
    CountyReport,
    FileReport,
    RegionReport,
    compute_backfill_report,
    format_report,
    report_to_dict,
)

# ---------- fixtures -----------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh lexicon DB with schema applied."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    # Seed a source row so toponym_attestation rows have a citation
    # path (source_doc references the source.id implicitly).
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test Source')")
    db.commit()
    yield db


def _add_toponym(db: LexiconDB, modern_name: str) -> int:
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
    db.commit()
    return cur.lastrowid


def _add_attestation(db: LexiconDB, toponym_id: int, form: str = "Old Form") -> None:
    db.conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, source_doc) VALUES (?, ?, ?)",
        (toponym_id, form, "test_src"),
    )
    db.commit()


def _write_place_names(
    data_dir: Path, filename: str, payload: dict[str, dict[str, list[str]]]
) -> Path:
    """Write one ``*_place_names.json`` fixture."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------- core categorization ----------------------------------------


def test_three_categories_count_correctly(db, tmp_path):
    """The three categories — not_in_db / in_db_no_attestation /
    in_db_with_attestation — each map to the right column. One name
    per category, single region/county for clarity."""
    not_in_db = "Nowhereville"
    in_db_no_att = "Halfthere"
    in_db_with_att = "Fullythere"

    # not_in_db: no toponym row at all.
    # in_db_no_att: toponym row, but no attestation.
    half_id = _add_toponym(db, in_db_no_att)
    # silence the unused-id warning; we don't add an attestation.
    assert half_id > 0
    # in_db_with_att: toponym row + ≥1 attestation.
    full_id = _add_toponym(db, in_db_with_att)
    _add_attestation(db, full_id)

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"TestCounty": [not_in_db, in_db_no_att, in_db_with_att]}},
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    assert len(reports) == 1
    fr = reports[0]
    assert fr.filename == "english_place_names.json"
    assert fr.total == 3
    assert fr.in_db == 2  # halfthere + fullythere
    assert fr.attested == 1  # only fullythere
    assert fr.in_db_no_attestation == 1
    assert fr.gap == 2  # not_in_db + in_db_no_att


def test_multiple_regions_in_one_file(db, tmp_path):
    """England and The Isle of Man both in english_place_names.json
    (the real file's shape). Each region aggregates its own counties;
    the file totals sum the regions."""
    _add_toponym(db, "Mantown")
    _add_toponym(db, "Berkshirely")

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {
            "England": {"Berkshire": ["Berkshirely", "Berkshire-Unknown"]},
            "The Isle of Man": {"The Isle of Man": ["Mantown"]},
        },
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    assert len(reports) == 1
    fr = reports[0]
    region_by_name = {r.name: r for r in fr.regions}
    assert set(region_by_name) == {"England", "The Isle of Man"}
    assert region_by_name["England"].total == 2
    assert region_by_name["England"].in_db == 1
    assert region_by_name["The Isle of Man"].total == 1
    assert region_by_name["The Isle of Man"].in_db == 1
    assert fr.total == 3
    assert fr.in_db == 2


def test_multiple_files(db, tmp_path):
    """All 5 files in the same data_dir; the report walks all 5 in
    canonical order."""
    _add_toponym(db, "EnglishPlace")
    _add_toponym(db, "ScottishPlace")

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["EnglishPlace"]}},
    )
    _write_place_names(
        data_dir,
        "scottish_place_names.json",
        {"Scotland": {"Aberdeen": ["ScottishPlace"]}},
    )
    _write_place_names(data_dir, "welsh_place_names.json", {"Wales": {"Anglesey": ["WelshPlace"]}})
    _write_place_names(
        data_dir,
        "irish_place_names.json",
        {"Republic of Ireland": {"Clare": ["IrishPlace"]}},
    )
    _write_place_names(
        data_dir, "breton_place_names.json", {"Brittany": {"Finistère": ["BretonPlace"]}}
    )

    reports = compute_backfill_report(db, data_dir=data_dir)
    assert [r.filename for r in reports] == [
        "english_place_names.json",
        "scottish_place_names.json",
        "welsh_place_names.json",
        "irish_place_names.json",
        "breton_place_names.json",
    ]
    by_name = {r.filename: r for r in reports}
    assert by_name["english_place_names.json"].in_db == 1
    assert by_name["scottish_place_names.json"].in_db == 1
    assert by_name["welsh_place_names.json"].in_db == 0
    assert by_name["irish_place_names.json"].in_db == 0
    assert by_name["breton_place_names.json"].in_db == 0


def test_language_filter_restricts_to_one_file(db, tmp_path):
    """--language english picks ONLY english_place_names.json even
    when other files exist."""
    data_dir = tmp_path / "data"
    _write_place_names(data_dir, "english_place_names.json", {"England": {"A": ["X"]}})
    _write_place_names(data_dir, "scottish_place_names.json", {"Scotland": {"B": ["Y"]}})

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    assert len(reports) == 1
    assert reports[0].filename == "english_place_names.json"


def test_language_filter_case_insensitive(db, tmp_path):
    """Operators may type --language English; the lookup should not
    care about case."""
    data_dir = tmp_path / "data"
    _write_place_names(data_dir, "english_place_names.json", {"England": {"A": ["X"]}})

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("English",))
    assert len(reports) == 1


def test_language_filter_multiple(db, tmp_path):
    """Multiple --language flags accumulate."""
    data_dir = tmp_path / "data"
    _write_place_names(data_dir, "english_place_names.json", {"England": {"A": ["X"]}})
    _write_place_names(data_dir, "scottish_place_names.json", {"Scotland": {"B": ["Y"]}})
    _write_place_names(data_dir, "welsh_place_names.json", {"Wales": {"C": ["Z"]}})

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english", "scottish"))
    assert {r.filename for r in reports} == {
        "english_place_names.json",
        "scottish_place_names.json",
    }


def test_empty_db_all_not_in_db(db, tmp_path):
    """Empty DB → every entry is not_in_db (in_db=0, attested=0,
    gap=total)."""
    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["A", "B", "C"]}},
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    assert fr.total == 3
    assert fr.in_db == 0
    assert fr.attested == 0
    assert fr.gap == 3


def test_duplicate_name_counts_each_occurrence(db, tmp_path):
    """When the same modern_name appears in multiple regions/counties
    (Wikipedia happily has a Springfield in N places), each occurrence
    counts. The DB row may dedup-collapse to one toponym; that doesn't
    matter — the report counts Wikipedia entries, not toponym rows."""
    full_id = _add_toponym(db, "Springfield")
    _add_attestation(db, full_id)

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {
            "England": {
                "Bedfordshire": ["Springfield", "Springfield"],
                "Berkshire": ["Springfield"],
            },
            "The Isle of Man": {"The Isle of Man": ["Springfield"]},
        },
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    # 4 Wikipedia entries (2 in Bedfordshire, 1 in Berkshire, 1 in IoM)
    assert fr.total == 4
    # All 4 resolve to the same DB row, but counted per occurrence.
    assert fr.in_db == 4
    assert fr.attested == 4


def test_missing_file_produces_empty_report(db, tmp_path):
    """A data_dir without all 5 files → missing files produce empty
    file-reports (no crash). The CLI surfaces them as zeros."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # No files written.

    reports = compute_backfill_report(db, data_dir=data_dir)
    # All 5 files reported, all empty.
    assert len(reports) == 5
    assert all(r.total == 0 for r in reports)
    assert all(r.regions == () for r in reports)


def test_bogus_language_filter_raises(db, tmp_path):
    """--language bogus is now a loud error. R1 silent-failure-hunter
    MEDIUM: previously silently produced an empty report, so an
    operator typing ``--language welch`` would get zero output and
    exit 0 with no diagnostic. Now raises ValueError (the CLI
    converts to click.BadParameter)."""
    data_dir = tmp_path / "data"
    _write_place_names(data_dir, "english_place_names.json", {"England": {"A": ["X"]}})

    with pytest.raises(ValueError, match="unknown --language"):
        compute_backfill_report(db, data_dir=data_dir, languages=("bogus",))


# ---------- shape validation -------------------------------------------


def test_invalid_top_level_raises(db, tmp_path):
    """A *_place_names.json that's a list instead of dict raises."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "english_place_names.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level must be dict"):
        compute_backfill_report(db, data_dir=data_dir, languages=("english",))


def test_invalid_subregion_value_raises(db, tmp_path):
    """A country whose value is a list (not dict-of-list) raises."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "english_place_names.json").write_text(
        json.dumps({"England": ["not a dict"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must map to dict"):
        compute_backfill_report(db, data_dir=data_dir, languages=("english",))


def test_invalid_names_value_raises(db, tmp_path):
    """A county whose value is a string (not list) raises."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "english_place_names.json").write_text(
        json.dumps({"England": {"Bedfordshire": "Aley Green"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must map to list"):
        compute_backfill_report(db, data_dir=data_dir, languages=("english",))


# ---------- aggregation -------------------------------------------------


def test_aggregates_match_county_sum(db, tmp_path):
    """File.total == sum(region.total); region.total == sum(county.total)."""
    _add_toponym(db, "A")
    _add_toponym(db, "C")
    full_id = _add_toponym(db, "E")
    _add_attestation(db, full_id)

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {
            "England": {
                "Bedfordshire": ["A", "B"],
                "Berkshire": ["C", "D"],
            },
            "The Isle of Man": {"The Isle of Man": ["E"]},
        },
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    assert fr.total == 5
    assert fr.in_db == 3  # A, C, E
    assert fr.attested == 1  # E
    region_by_name = {r.name: r for r in fr.regions}
    england = region_by_name["England"]
    assert england.total == 4
    assert england.in_db == 2  # A, C
    assert england.attested == 0
    # county-level sums == region totals
    assert sum(c.total for c in england.counties) == england.total
    assert sum(c.in_db for c in england.counties) == england.in_db


def test_pct_attested_zero_when_no_total():
    """Degenerate guard: pct_attested on an empty CountyReport
    shouldn't divide-by-zero."""
    cr = CountyReport(name="Empty", total=0, in_db=0, attested=0)
    assert cr.pct_attested == 0.0
    rr = RegionReport(name="Empty", counties=())
    assert rr.pct_attested == 0.0
    fr = FileReport(filename="empty.json", regions=())
    assert fr.pct_attested == 0.0


def test_county_report_rejects_invalid_invariant():
    """CountyReport's __post_init__ enforces
    ``0 <= attested <= in_db <= total``. A producer bug that built
    ``CountyReport(total=0, in_db=5, attested=10)`` would yield a
    negative gap and a confusing report. R1 type-design MEDIUM."""
    # Valid construction works.
    CountyReport(name="A", total=10, in_db=5, attested=3)
    # Negative violations
    with pytest.raises(ValueError, match="violates"):
        CountyReport(name="A", total=10, in_db=15, attested=5)  # in_db > total
    with pytest.raises(ValueError, match="violates"):
        CountyReport(name="A", total=10, in_db=5, attested=8)  # attested > in_db
    with pytest.raises(ValueError, match="violates"):
        CountyReport(name="A", total=10, in_db=5, attested=-1)  # attested < 0


def test_attestation_from_non_scholar_source_does_not_count(db, tmp_path):
    """An attestation row with source_doc='rando-port' (or any other
    non-scholar source per ``_NON_SCHOLAR_SOURCES``) MUST NOT count
    as scholar-attested. The entire retirement gate depends on this —
    a name attested only by the legacy seed isn't evidence the
    Wikipedia seed can be retired. R1 silent-failure-hunter HIGH."""
    name = "RandoOnlyPlace"
    toponym_id = _add_toponym(db, name)
    # Attest only from rando-port (non-scholar source).
    db.conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, source_doc) "
        "VALUES (?, ?, 'rando-port')",
        (toponym_id, name),
    )
    db.commit()

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"TestCounty": [name]}},
    )
    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    assert fr.total == 1
    assert fr.in_db == 1  # toponym exists
    assert fr.attested == 0  # but rando-port doesn't count as scholar


def test_attestation_with_null_source_doc_counts(db, tmp_path):
    """An attestation with NULL source_doc (legacy/unknown provenance)
    is counted as scholar — only the explicit non-scholar set is
    excluded. Conservative default: if we don't KNOW it's
    non-scholar, treat as scholar."""
    name = "LegacyAttestedPlace"
    toponym_id = _add_toponym(db, name)
    db.conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, source_doc) VALUES (?, ?, NULL)",
        (toponym_id, name),
    )
    db.commit()

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"TestCounty": [name]}},
    )
    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    assert fr.attested == 1


def test_nfc_normalization_matches_canonical_diacritic(db, tmp_path):
    """The same character can be encoded as NFC (precomposed) or NFD
    (decomposed). Wikipedia and the DB may differ. After NFC
    normalization on both sides, the match should succeed. R1
    silent-failure-hunter MEDIUM."""
    # NFC: U+00E9 (precomposed é)
    nfc_name = "Café"
    # NFD: U+0065 U+0301 (e + combining acute)
    nfd_name = "Café"
    assert nfc_name != nfd_name  # they're different code-point sequences
    import unicodedata as _u

    assert _u.normalize("NFC", nfd_name) == nfc_name

    # DB stores NFC form
    toponym_id = _add_toponym(db, nfc_name)
    _add_attestation(db, toponym_id)

    # Wikipedia file stores the NFD form
    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"TestCounty": [nfd_name]}},
    )
    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    fr = reports[0]
    # NFC normalization should match: name found, attested.
    assert fr.in_db == 1
    assert fr.attested == 1


def test_cli_unknown_language_is_bad_parameter(db, tmp_path):
    """--language welch (typo) must error loudly, not silently produce
    empty output. R1 silent-failure-hunter MEDIUM."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli as cli_root

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runner = CliRunner()
    # Find the live lexicon path used by the CLI default; pass a
    # synthetic empty one to avoid leaning on it.
    db_path = tmp_path / "lexicon.db"
    from wyrd.generators.kenning.lexicon import init_schema

    init_schema(db_path)
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "report-wikipedia-backfill",
            "--db",
            str(db_path),
            "--data-dir",
            str(data_dir),
            "--language",
            "welch",  # typo
        ],
    )
    assert result.exit_code != 0
    assert "unknown --language" in result.output


# ---------- JSON output -------------------------------------------------


def test_as_json_shape(db, tmp_path):
    """report_to_dict produces the documented nested-dict shape."""
    full_id = _add_toponym(db, "FullPlace")
    _add_attestation(db, full_id)

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["FullPlace", "MissingPlace"]}},
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    payload = report_to_dict(reports)
    assert set(payload.keys()) == {"files", "totals"}
    assert "english_place_names.json" in payload["files"]
    eng = payload["files"]["english_place_names.json"]
    assert set(eng.keys()) == {"total", "in_db", "attested", "gap", "regions"}
    assert eng["total"] == 2
    assert eng["in_db"] == 1
    assert eng["attested"] == 1
    assert eng["gap"] == 1
    assert set(eng["regions"].keys()) == {"England"}
    region = eng["regions"]["England"]
    assert set(region.keys()) == {"total", "in_db", "attested", "gap", "counties"}
    assert "Bedfordshire" in region["counties"]
    county = region["counties"]["Bedfordshire"]
    assert county == {"total": 2, "in_db": 1, "attested": 1, "gap": 1}
    assert payload["totals"] == {"total": 2, "in_db": 1, "attested": 1, "gap": 1}


# ---------- formatter ---------------------------------------------------


def test_format_report_includes_header_and_total(db, tmp_path):
    """The human table has the header row and a TOTAL footer."""
    _add_toponym(db, "A")
    data_dir = tmp_path / "data"
    _write_place_names(data_dir, "english_place_names.json", {"England": {"Bedfordshire": ["A"]}})

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    text = format_report(reports)
    assert "Wikipedia backfill report" in text
    assert "total" in text
    assert "%_attested" in text
    assert "english_place_names.json" in text
    assert "England" in text
    assert "TOTAL" in text


def test_format_report_verbose_includes_counties(db, tmp_path):
    """--verbose expands each region into per-county rows; non-verbose
    suppresses them."""
    _add_toponym(db, "A")
    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["A"], "Berkshire": ["B"]}},
    )

    reports = compute_backfill_report(db, data_dir=data_dir, languages=("english",))
    non_verbose = format_report(reports, verbose=False)
    verbose = format_report(reports, verbose=True)

    assert "Bedfordshire" not in non_verbose
    assert "Berkshire" not in non_verbose
    assert "Bedfordshire" in verbose
    assert "Berkshire" in verbose


# ---------- CLI smoke ---------------------------------------------------


def test_cli_smoke_empty_db_no_crash(db, tmp_path):
    """Invoke the CLI against an empty DB + minimal data dir; exit 0."""
    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["A", "B"]}},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "report-wikipedia-backfill",
            "--db",
            str(db.path),
            "--data-dir",
            str(data_dir),
            "--language",
            "english",
        ],
    )
    assert result.exit_code == 0, result.output
    # Table goes to stderr; CliRunner mixes streams by default.
    assert "Wikipedia backfill report" in result.output
    assert "english_place_names.json" in result.output


def test_cli_as_json_stdout(db, tmp_path):
    """--as-json emits parseable JSON on stdout. CliRunner in click 8.3
    separates stdout from stderr by default."""
    full_id = _add_toponym(db, "FullPlace")
    _add_attestation(db, full_id)

    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["FullPlace"]}},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "report-wikipedia-backfill",
            "--db",
            str(db.path),
            "--data-dir",
            str(data_dir),
            "--language",
            "english",
            "--as-json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["totals"]["total"] == 1
    assert payload["totals"]["attested"] == 1


def test_cli_verbose_includes_counties(db, tmp_path):
    """--verbose expands per-county rows in the CLI output."""
    _add_toponym(db, "A")
    data_dir = tmp_path / "data"
    _write_place_names(
        data_dir,
        "english_place_names.json",
        {"England": {"Bedfordshire": ["A"]}},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "report-wikipedia-backfill",
            "--db",
            str(db.path),
            "--data-dir",
            str(data_dir),
            "--language",
            "english",
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Bedfordshire" in result.output
