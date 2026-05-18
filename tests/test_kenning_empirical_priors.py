"""Tests for wyrd-ecjp.2: empirical-baseline priors extraction.

Builds small in-memory lexicon DBs and exercises ``extract_priors``,
``mine_empirical_baselines``, and ``dump_empirical_priors_to_json``
against synthetic toponym + etymology + element fixtures.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.empirical_priors import (
    _OPEN_BOUND_OFFSET,
    _era_midpoint,
    _LoanKey,
    _NativeKey,
    _position_label,
    dump_empirical_priors_to_json,
    era_midpoint_for,
    extract_priors,
    known_era_midpoints,
    mine_empirical_baselines,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

# ---------- fixtures -----------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh lexicon DB with schema applied. Yields the LexiconDB."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test Source')")
    db.commit()
    yield db


def _add_toponym(
    db: LexiconDB,
    toponym_id: int,
    modern_name: str,
    country: str | None = "England",
) -> None:
    db.conn.execute(
        "INSERT INTO toponym (id, modern_name, country) VALUES (?, ?, ?)",
        (toponym_id, modern_name, country),
    )


def _add_etymon(
    db: LexiconDB,
    etymon_id: int,
    canonical_form: str,
    language: str,
    tags: tuple[str, ...] = (),
) -> None:
    db.conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (?, ?, ?)",
        (etymon_id, canonical_form, language),
    )
    for tag in tags:
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (etymon_id, tag),
        )


def _add_etymology(
    db: LexiconDB,
    *,
    toponym_id: int,
    elements: list[int],  # [etymon_id, …] in ordinal order
    attested_year: int | None = 950,
    source: str = "test_src",
) -> int:
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology "
        "(toponym_id, source_id, attested_year, confidence) "
        "VALUES (?, ?, ?, 'high')",
        (toponym_id, source, attested_year),
    )
    ety_id = cur.lastrowid
    for ordinal, etymon_id in enumerate(elements):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (ety_id, ordinal, etymon_id),
        )
    db.commit()
    return ety_id


# ---------- _position_label ---------------------------------------------


def test_position_label_single_element_is_pre():
    """Solo morpheme acts as the head — matches legacy
    english_proportions.structures convention."""
    assert _position_label(0, 1) == "pre"


def test_position_label_binary_compound():
    assert _position_label(0, 2) == "pre"
    assert _position_label(1, 2) == "post"


def test_position_label_ternary_compound():
    assert _position_label(0, 3) == "pre"
    assert _position_label(1, 3) == "inner"
    assert _position_label(2, 3) == "post"


def test_position_label_long_compound_middle_is_inner():
    """A 5-element name: 0 -> pre, 4 -> post, 1/2/3 -> inner."""
    assert _position_label(0, 5) == "pre"
    assert _position_label(4, 5) == "post"
    for inner_idx in (1, 2, 3):
        assert _position_label(inner_idx, 5) == "inner"


# ---------- era_midpoint -------------------------------------------------


def test_era_midpoint_closed_cell_is_arithmetic_midpoint():
    assert _era_midpoint(800, 1100) == 950
    assert _era_midpoint(1100, 1500) == 1300


def test_era_midpoint_open_low_uses_offset_below_end():
    assert _era_midpoint(None, 800) == 800 - _OPEN_BOUND_OFFSET


def test_era_midpoint_open_high_uses_offset_above_start():
    assert _era_midpoint(1700, None) == 1700 + _OPEN_BOUND_OFFSET


def test_era_midpoint_rejects_double_unbounded():
    with pytest.raises(ValueError, match="both start and end as None"):
        _era_midpoint(None, None)


def test_era_midpoint_for_old_english_attested_year_resolves():
    assert era_midpoint_for("old-english", 950) == 950
    # Year falls into oe-early (None, 800) — uses offset below end.
    assert era_midpoint_for("old-english", 700) == 800 - _OPEN_BOUND_OFFSET


def test_era_midpoint_for_returns_none_on_untracked_language():
    # Greek isn't in LANGUAGE_TO_FAMILY (per era.py).
    assert era_midpoint_for("ancient-greek", 500) is None


def test_known_era_midpoints_covers_every_cell():
    midpoints = known_era_midpoints()
    # Sanity: every family has at least one cell.
    families = {family for (family, _label) in midpoints}
    assert families == {
        "english",
        "norse",
        "brythonic",
        "goidelic",
        "latin",
        "norman-french",
    }
    # Known checks against the era.py table.
    assert midpoints[("english", "oe-late")] == 950
    assert midpoints[("english", "me")] == 1300


# ---------- extract_priors ----------------------------------------------


def test_extract_priors_empty_db_returns_empty(db):
    native, loan, summary = extract_priors(db)
    assert native == {}
    assert loan == {}
    assert summary.rows_scanned == 0


def test_extract_priors_binary_compound_emits_pre_and_post(db):
    """One OE 2-element toponym attested 950: each element contributes
    to (english, pre|post, tag, 950, lemma_ref) under both native and
    loan tables."""
    _add_toponym(db, 1, "Edwinstone")
    _add_etymon(db, 10, "Eadwine", "old-english", tags=("name",))
    _add_etymon(db, 11, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=1, elements=[10, 11], attested_year=950)

    native, loan, summary = extract_priors(db)

    expected_native = {
        _NativeKey("english", "pre", "name", 950, "old-english:Eadwine"): 1,
        _NativeKey("english", "post", "architecture", 950, "old-english:tūn"): 1,
    }
    expected_loan = {
        _LoanKey("old-english", "english", "pre", "name", 950, "old-english:Eadwine"): 1,
        _LoanKey("old-english", "english", "post", "architecture", 950, "old-english:tūn"): 1,
    }
    assert native == expected_native
    assert loan == expected_loan
    assert summary.rows_scanned == 2
    assert summary.rows_with_era == 2
    assert summary.native_cells_written == 2
    assert summary.loan_cells_written == 2


def test_extract_priors_multi_tag_lemma_emits_one_cell_per_tag(db):
    """An etymon tagged [plant, tree] contributes to TWO native cells
    per occurrence — the request-side tag weights pick which one
    dominates at scoring time."""
    _add_toponym(db, 1, "Oakville")
    _add_etymon(db, 10, "āc", "old-english", tags=("plant", "tree"))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    native, _loan, summary = extract_priors(db)

    assert native[_NativeKey("english", "pre", "plant", 950, "old-english:āc")] == 1
    assert native[_NativeKey("english", "pre", "tree", 950, "old-english:āc")] == 1
    # Two rows in the SQL result because of the etymon_tag join, but the
    # element only appears once — the position/era are shared.
    assert summary.rows_scanned == 2


def test_extract_priors_aggregates_across_toponyms(db):
    """Two toponyms both contributing 'old-english:tūn' in the post
    position at era=950 — counts accumulate."""
    _add_toponym(db, 1, "X")
    _add_toponym(db, 2, "Y")
    _add_etymon(db, 10, "Eadwine", "old-english", tags=("name",))
    _add_etymon(db, 11, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=1, elements=[10, 11], attested_year=950)
    _add_etymology(db, toponym_id=2, elements=[10, 11], attested_year=950)

    native, _loan, _summary = extract_priors(db)

    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 2


def test_extract_priors_skips_no_country(db):
    """Toponym with country=None — extractor skips and bumps the
    counter so operators can see the gap."""
    _add_toponym(db, 1, "Mysteryville", country=None)
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.skipped_no_country == 1


def test_extract_priors_skips_no_era(db):
    """Etymology with attested_year=None — no era_midpoint, row
    skipped."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=None)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.skipped_no_era == 1


def test_extract_priors_skips_untagged_etymon(db):
    """SQL inner-joins on etymon_tag — untagged etymons don't appear.
    rows_scanned reflects only the joined rows."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=())
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.rows_scanned == 0


def test_extract_priors_loan_partitions_by_donor_language(db):
    """An ON loan in an English toponym surfaces under
    donor='old-norse' in the loan table; a sibling OE element under
    donor='old-english'. The native view aggregates both."""
    _add_toponym(db, 1, "Carlton")
    _add_etymon(db, 10, "karl", "old-norse", tags=("social",))
    _add_etymon(db, 11, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=1, elements=[10, 11], attested_year=950)

    native, loan, _summary = extract_priors(db)

    # Native: both lemmas land under culture=english regardless of donor.
    assert native[_NativeKey("english", "pre", "social", 950, "old-norse:karl")] == 1
    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 1
    # Loan: per-donor partition.
    assert loan[_LoanKey("old-norse", "english", "pre", "social", 950, "old-norse:karl")] == 1
    assert (
        loan[_LoanKey("old-english", "english", "post", "architecture", 950, "old-english:tūn")]
        == 1
    )


# ---------- mine_empirical_baselines ------------------------------------


def test_mine_empirical_baselines_dry_run_does_not_write(db):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    result = mine_empirical_baselines(db, apply=False)

    assert result["native_cells_written"] == 1
    assert _table_count(db, "empirical_priors_native") == 0
    assert _table_count(db, "empirical_priors_loan") == 0


def test_mine_empirical_baselines_apply_writes(db):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    mine_empirical_baselines(db, apply=True)

    assert _table_count(db, "empirical_priors_native") == 1
    assert _table_count(db, "empirical_priors_loan") == 1


def test_mine_empirical_baselines_apply_is_replace_not_merge(db):
    """A second apply with NEW input data must replace, not accumulate."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    assert _table_count(db, "empirical_priors_native") == 1

    # Mutation: delete the etymology and re-apply. Table should empty.
    db.conn.execute("DELETE FROM toponym_etymology_element")
    db.conn.execute("DELETE FROM toponym_etymology")
    db.commit()
    mine_empirical_baselines(db, apply=True)
    assert _table_count(db, "empirical_priors_native") == 0
    assert _table_count(db, "empirical_priors_loan") == 0


def test_mine_empirical_baselines_apply_is_idempotent(db):
    """Two consecutive applies on unchanged data produce identical
    table state."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)

    mine_empirical_baselines(db, apply=True)
    first_rows = sorted(db.conn.execute("SELECT * FROM empirical_priors_native").fetchall())
    mine_empirical_baselines(db, apply=True)
    second_rows = sorted(db.conn.execute("SELECT * FROM empirical_priors_native").fetchall())
    assert first_rows == second_rows


# ---------- dump_empirical_priors_to_json --------------------------------


def test_dump_json_structure(db, tmp_path):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    out = tmp_path / "priors.json"

    dump_empirical_priors_to_json(db, out, version="test-v1")

    artifact = json.loads(out.read_text())
    assert artifact["version"] == "test-v1"
    assert len(artifact["native"]) == 1
    cell = artifact["native"][0]
    assert cell["culture"] == "english"
    assert cell["position"] == "pre"
    assert cell["tag"] == "plant"
    assert cell["era_midpoint"] == 950
    assert cell["lemmas"] == {"old-english:x": 1}
    assert len(artifact["loan"]) == 1


def test_dump_json_byte_stable_across_runs(db, tmp_path):
    _add_toponym(db, 1, "X")
    _add_toponym(db, 2, "Y")
    _add_etymon(db, 10, "a", "old-english", tags=("plant", "tree"))
    _add_etymon(db, 11, "b", "old-norse", tags=("water",))
    _add_etymology(db, toponym_id=1, elements=[10, 11], attested_year=950)
    _add_etymology(db, toponym_id=2, elements=[10], attested_year=950)
    mine_empirical_baselines(db, apply=True)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    dump_empirical_priors_to_json(db, out_a, version="v1")
    dump_empirical_priors_to_json(db, out_b, version="v1")

    assert out_a.read_bytes() == out_b.read_bytes()


def test_dump_json_sorts_lemmas_in_cell(db, tmp_path):
    """Per-cell lemma dict sorted by lemma_ref so the JSON diff is
    review-friendly when new lemmas appear."""
    _add_toponym(db, 1, "X")
    _add_toponym(db, 2, "Y")
    _add_etymon(db, 10, "zeta", "old-english", tags=("plant",))
    _add_etymon(db, 11, "alpha", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    _add_etymology(db, toponym_id=2, elements=[11], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    out = tmp_path / "priors.json"
    dump_empirical_priors_to_json(db, out, version="v1")

    artifact = json.loads(out.read_text())
    cell = artifact["native"][0]
    lemma_keys = list(cell["lemmas"].keys())
    assert lemma_keys == sorted(lemma_keys)


# ---------- CLI ----------------------------------------------------------


def test_cli_mine_empirical_baselines_dry_run(db, tmp_path):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    runner = CliRunner()

    result = runner.invoke(
        cli_root,
        ["lexicon", "mine-empirical-baselines", "--db", str(db.path)],
    )

    assert result.exit_code == 0, result.output
    assert "dry-run" in (result.output + (result.stderr or ""))
    assert _table_count(db, "empirical_priors_native") == 0


def test_cli_mine_empirical_baselines_apply(db, tmp_path):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    runner = CliRunner()

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-empirical-baselines",
            "--db",
            str(db.path),
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _table_count(db, "empirical_priors_native") == 1


def test_cli_dump_empirical_priors(db, tmp_path):
    _add_toponym(db, 1, "X")
    _add_etymon(db, 10, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=1, elements=[10], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    out = tmp_path / "priors.json"
    runner = CliRunner()

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "dump-empirical-priors",
            "--db",
            str(db.path),
            "--output",
            str(out),
            "--version",
            "cli-test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    artifact = json.loads(out.read_text())
    assert artifact["version"] == "cli-test"
    assert len(artifact["native"]) == 1


# ---------- helpers ------------------------------------------------------


def _table_count(db: LexiconDB, table: str) -> int:
    cur = db.conn.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]
