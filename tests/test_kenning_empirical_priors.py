"""Tests for wyrd-ecjp.2: empirical-baseline priors extraction.

Builds small in-memory lexicon DBs and exercises ``extract_priors``,
``mine_empirical_baselines``, and ``dump_empirical_priors_to_json``
against synthetic toponym + etymology + element fixtures.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.lexicon.empirical_priors import (
    _CELTIC_LEMMA_LANGUAGES,
    _CELTIC_LEMMA_REPRESENTATIVE,
    _COUNTRY_TO_CULTURE,
    _CULTURE_TO_FAMILY,
    _OPEN_BOUND_OFFSET,
    ExtractionResult,
    _era_midpoint,
    _LoanKey,
    _NativeKey,
    _position_label,
    dump_empirical_priors_to_json,
    era_midpoint_for_culture,
    extract_priors,
    known_era_midpoints,
    mine_empirical_baselines,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.vector_name_select import (
    _BUNDLE_FIELD_TO_L3_LANG,
    _lemma_ref_for,
)

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
    modern_name: str,
    country: str | None = "England",
) -> int:
    """Insert a toponym and return its assigned id (lastrowid).

    Mirrors production: the AUTOINCREMENT PK is assigned by SQLite, not
    by the caller. Returning lastrowid lets tests chain etymology
    insertion without hardcoding fragile integer IDs.
    """
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES (?, ?)",
        (modern_name, country),
    )
    return cur.lastrowid


def _add_etymon(
    db: LexiconDB,
    canonical_form: str,
    language: str,
    tags: tuple[str, ...] = (),
) -> int:
    """Insert an etymon (+ optional tags) and return its lastrowid."""
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    etymon_id = cur.lastrowid
    for tag in tags:
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (etymon_id, tag),
        )
    return etymon_id


def _add_etymology(
    db: LexiconDB,
    *,
    toponym_id: int,
    elements: list[int],
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


def test_position_label_single_element_is_bare():
    """Solo morpheme → 'bare' (D40 vocabulary; wyrd-nkiq).

    Must match the runtime lookup label
    (``_slot_position_label`` returns 'bare' for a no-dash slot), else
    single-word priors are stranded in an unqueried cell.
    """
    assert _position_label(0, 1) == "bare"


def test_position_label_single_element_matches_runtime_lookup_label():
    """wyrd-nkiq regression: the bake-side single-element label MUST equal
    the runtime lookup label for a no-dash (single-morpheme) slot. If these
    drift again, single-word priors go unreachable silently."""
    from wyrd.generators.kenning.runtime.vector_name_select import (
        _slot_position_label,
    )

    # A bare (dash-free) structural element is a single-morpheme word.
    assert _slot_position_label("Barechester") == "bare"
    assert _position_label(0, 1) == _slot_position_label("Barechester")


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


def test_era_midpoint_for_culture_resolves_english():
    assert era_midpoint_for_culture("english", 950) == 950


def test_era_midpoint_for_culture_unknown_culture_raises():
    """Unknown culture string -> ValueError (loud config-or-caller
    error). Production callers route via _COUNTRY_TO_CULTURE, whose
    values are by contract also in _CULTURE_TO_FAMILY; an unknown
    string here would silently route every row of that culture into
    skipped_year_*. Raise instead."""
    with pytest.raises(ValueError, match="missing from _CULTURE_TO_FAMILY"):
        era_midpoint_for_culture("klingon", 950)


def test_era_midpoint_for_culture_year_past_all_cells_returns_none():
    """Latin's last cell ends at 1800. Year=2000 has no matching cell —
    the loop exhausts and returns None."""
    _CULTURE_TO_FAMILY["__test_latin"] = "latin"
    try:
        assert era_midpoint_for_culture("__test_latin", 2000) is None
    finally:
        del _CULTURE_TO_FAMILY["__test_latin"]


def test_era_midpoint_for_culture_raises_on_missing_family_cells():
    """Config drift: _CULTURE_TO_FAMILY entry pointing at a family
    that isn't in ERA_CELLS must raise loudly, not silently route
    every row of that culture into skipped_year_*."""
    _CULTURE_TO_FAMILY["__test_orphan"] = "__no_such_family"
    try:
        with pytest.raises(ValueError, match="not in era.ERA_CELLS"):
            era_midpoint_for_culture("__test_orphan", 950)
    finally:
        del _CULTURE_TO_FAMILY["__test_orphan"]


def test_era_midpoint_for_culture_boundary_year_lower_inclusive():
    """oe-late starts at 800 (inclusive). 800 -> 950 (oe-late)."""
    assert era_midpoint_for_culture("english", 800) == 950


def test_era_midpoint_for_culture_boundary_year_upper_exclusive():
    """oe-late ends at 1100 (exclusive). 1100 -> 1300 (me)."""
    assert era_midpoint_for_culture("english", 1100) == 1300
    # 1099 is still oe-late.
    assert era_midpoint_for_culture("english", 1099) == 950


def test_era_midpoint_for_culture_open_low_boundary():
    """oe-early has start=None (open-low) — every year below 800 lands
    there."""
    assert era_midpoint_for_culture("english", 1) == _era_midpoint(None, 800)
    assert era_midpoint_for_culture("english", 799) == _era_midpoint(None, 800)


def test_known_era_midpoints_covers_every_cell():
    midpoints = known_era_midpoints()
    families = {family for (family, _label) in midpoints}
    assert families == {
        "english",
        "norse",
        "brythonic",
        "goidelic",
        "latin",
        "norman-french",
    }
    # English checks against the era/cells.py table.
    assert midpoints[("english", "oe-late")] == 950
    assert midpoints[("english", "me")] == 1300


def test_known_era_midpoints_value_checks_non_english_families():
    """Pin the midpoint convention for every family — closes the gap
    where only english was previously exercised."""
    m = known_era_midpoints()
    # Norse: on-classical (None, 1100) -> 1000.
    assert m[("norse", "on-classical")] == 1000
    # Norse: on-late (1100, 1300) -> 1200.
    assert m[("norse", "on-late")] == 1200
    # Brythonic: old (None, 1100) -> 1000.
    assert m[("brythonic", "old")] == 1000
    # Brythonic: middle (1100, 1500) -> 1300.
    assert m[("brythonic", "middle")] == 1300
    # Goidelic: old-irish (None, 900) -> 800.
    assert m[("goidelic", "old-irish")] == 800
    # Goidelic: middle-irish (900, 1200) -> 1050.
    assert m[("goidelic", "middle-irish")] == 1050
    # Latin: classical (None, 200) -> 100.
    assert m[("latin", "classical")] == 100
    # Latin: medieval (700, 1500) -> 1100.
    assert m[("latin", "medieval")] == 1100
    # Norman-french: anglo-norman (1066, 1500) -> 1283.
    assert m[("norman-french", "anglo-norman")] == 1283


# ---------- ExtractionResult ---------------------------------------------


def test_extraction_result_partition_invariant_enforced():
    """Mismatched skipped vs rows_scanned should raise loudly."""
    with pytest.raises(ValueError, match="partition invariant violated"):
        ExtractionResult(
            rows_scanned=10,
            rows_emitted=3,
            native_cells_written=3,
            loan_cells_written=3,
            skipped_country_unknown=1,
            skipped_country_unmapped=0,
            skipped_year_unknown=1,
            skipped_year_out_of_range=0,
            skipped_no_tag=1,
            # 3 + 1 + 0 + 1 + 0 + 1 = 6, not 10
        )


def test_extraction_result_partition_invariant_passes_when_balanced():
    """Sum of buckets equals rows_scanned -> no raise."""
    er = ExtractionResult(
        rows_scanned=10,
        rows_emitted=4,
        native_cells_written=3,
        loan_cells_written=3,
        skipped_country_unknown=1,
        skipped_country_unmapped=1,
        skipped_year_unknown=2,
        skipped_year_out_of_range=1,
        skipped_no_tag=1,
    )
    assert er.rows_scanned == 10


# ---------- extract_priors ----------------------------------------------


def test_classify_prior_row_each_gate_returns_its_skip_status() -> None:
    """The pure gate chain extracted from extract_priors, called directly on dict
    rows. Each gate returns its own skip-status name (both keys None); a fully-
    valid row returns "ok" with both the native + loan keys built.

    ``skipped_year_out_of_range`` can't be hit with an english country (english
    era cells are open-ended, so every english year resolves), so it's exercised
    via a temporarily-injected closed-bound culture — latin's cells end at 1800,
    mirroring test_extract_priors_skips_year_out_of_range_when_year_past_cells.
    """
    from wyrd.generators.kenning.lexicon.empirical_priors import _classify_prior_row

    base = {
        "country": "England",
        "attested_year": 950,
        "tag": "settlement",
        "ordinal": 0,
        "element_count": 2,
        "language": "old-english",
        "canonical_form": "tun",
    }

    def row(**overrides):
        return {**base, **overrides}

    assert _classify_prior_row(row(country=None))[0] == "skipped_country_unknown"
    assert _classify_prior_row(row(country=""))[0] == "skipped_country_unknown"
    assert _classify_prior_row(row(country="Atlantis"))[0] == "skipped_country_unmapped"
    assert _classify_prior_row(row(attested_year=None))[0] == "skipped_year_unknown"
    assert _classify_prior_row(row(tag=None))[0] == "skipped_no_tag"

    # year-out-of-range: a closed-bound culture (latin ends at 1800) + a year past it.
    _CULTURE_TO_FAMILY["__test_latin"] = "latin"
    _COUNTRY_TO_CULTURE["__TestCountry"] = "__test_latin"
    try:
        out_of_range = _classify_prior_row(
            row(country="__TestCountry", attested_year=2000, language="latin")
        )
        assert out_of_range[0] == "skipped_year_out_of_range"
    finally:
        del _CULTURE_TO_FAMILY["__test_latin"]
        del _COUNTRY_TO_CULTURE["__TestCountry"]

    status, native_key, loan_key = _classify_prior_row(row())
    assert status == "ok"
    assert native_key is not None and loan_key is not None
    assert loan_key.donor == "old-english"  # loan key keys on the donor language


def test_extract_priors_empty_db_returns_empty(db):
    native, loan, summary = extract_priors(db)
    assert native == {}
    assert loan == {}
    assert summary.rows_scanned == 0
    assert summary.rows_emitted == 0


def test_extract_priors_binary_compound_emits_pre_and_post(db):
    tid = _add_toponym(db, "Edwinstone")
    e_eadwine = _add_etymon(db, "Eadwine", "old-english", tags=("name",))
    e_tun = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e_eadwine, e_tun], attested_year=950)

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
    assert summary.rows_emitted == 2
    assert summary.native_cells_written == 2
    assert summary.loan_cells_written == 2


def test_extract_priors_ternary_compound_emits_pre_inner_post(db):
    """Verifies the element_count subquery + _position_label end-to-end
    for 3-element toponyms — the 'inner' position only appears in
    compounds with >=3 elements."""
    tid = _add_toponym(db, "Edwardingatun")
    e_head = _add_etymon(db, "Eadweard", "old-english", tags=("name",))
    e_inner = _add_etymon(db, "inga", "old-english", tags=("social",))
    e_tail = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e_head, e_inner, e_tail], attested_year=950)

    native, _loan, _summary = extract_priors(db)

    assert native[_NativeKey("english", "pre", "name", 950, "old-english:Eadweard")] == 1
    assert native[_NativeKey("english", "inner", "social", 950, "old-english:inga")] == 1
    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 1


@pytest.mark.parametrize(
    "raw_language, form",
    [
        ("brittonic", "penn"),  # coarse family tag — NOT in the emit map
        ("welsh", "caer"),  # emit-map-tracked celtic_mix member
        ("old-irish", "dun"),  # emit-map-tracked celtic_mix member
    ],
)
def test_extract_priors_celtic_family_lemma_ref_normalized_to_celtic(db, raw_language, form):
    """wyrd-3x9e: EVERY celtic-family language — the coarse 'brittonic'/'goidelic'
    tags AND the emit-map-tracked members (welsh, old-irish, …) — emits its NATIVE
    prior keyed on the 'celtic' representative, matching the runtime
    _lemma_ref_for(celtic_mix)->'celtic' lookup, not the raw language (whose key
    would be permanently unreachable). The LOAN-key donor keeps the raw language
    (separate axis) but the raw-language-keyed lemma_ref must NOT leak on either
    key — that would be a double-emit."""
    tid = _add_toponym(db, "Placename")
    e = _add_etymon(db, form, raw_language, tags=("landscape",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    native, loan, _summary = extract_priors(db)

    assert _NativeKey("english", "bare", "landscape", 950, f"celtic:{form}") in native
    assert _NativeKey("english", "bare", "landscape", 950, f"{raw_language}:{form}") not in native
    # loan donor keeps the raw language; the shared lemma_ref is normalized, and the
    # raw-language-keyed loan cell must not co-exist (guards against a double-emit).
    assert _LoanKey(raw_language, "english", "bare", "landscape", 950, f"celtic:{form}") in loan
    assert (
        _LoanKey(raw_language, "english", "bare", "landscape", 950, f"{raw_language}:{form}")
        not in loan
    )


def test_extract_priors_celtic_lemma_ref_reachable_from_runtime_lemma_ref_for(db):
    """wyrd-3x9e end-to-end reachability: a welsh etymon (an emit-map-tracked
    celtic_mix member, so a real welsh etymon DOES route into a runtime celtic_mix
    Meaning) gets a native prior whose lemma_ref is byte-identical to what the
    runtime _lemma_ref_for synthesizes for that Meaning — proving the baseline
    lookup HITS the cell rather than missing on a raw-language key ('welsh:caer'
    would never match the runtime's 'celtic:caer')."""
    tid = _add_toponym(db, "Caerwent")
    e = _add_etymon(db, "caer", "welsh", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    native, _loan, _summary = extract_priors(db)

    (extracted_key,) = native  # single element, single tag -> exactly one native cell
    runtime_ref = _lemma_ref_for(Meaning("Caer", [], [], {"celtic_mix": ["caer"]}))
    assert runtime_ref == "celtic:caer"
    assert extracted_key.lemma_ref == runtime_ref


def test_celtic_lemma_normalization_pins_runtime_representative_and_coarse_tags():
    """wyrd-3x9e tripwire: the extractor's celtic representative MUST equal the
    runtime bundle-field representative (_BUNDLE_FIELD_TO_L3_LANG['celtic_mix']).
    If those two constants drift, every celtic native prior silently misses again —
    this is the real cross-module invariant the fix rests on. Also pins that the
    two coarse family tags the emit map doesn't carry ('brittonic'/'goidelic') are
    included in the normalization set."""
    assert _BUNDLE_FIELD_TO_L3_LANG["celtic_mix"] == _CELTIC_LEMMA_REPRESENTATIVE
    assert {"brittonic", "goidelic"} <= _CELTIC_LEMMA_LANGUAGES


def test_extract_priors_multi_tag_lemma_emits_one_cell_per_tag(db):
    tid = _add_toponym(db, "Oakville")
    e_oak = _add_etymon(db, "āc", "old-english", tags=("plant", "tree"))
    _add_etymology(db, toponym_id=tid, elements=[e_oak], attested_year=950)

    native, _loan, summary = extract_priors(db)

    # Single-element toponym → 'bare' (wyrd-nkiq).
    assert native[_NativeKey("english", "bare", "plant", 950, "old-english:āc")] == 1
    assert native[_NativeKey("english", "bare", "tree", 950, "old-english:āc")] == 1
    # LEFT JOIN: 2 rows (one per tag), both emitted.
    assert summary.rows_scanned == 2
    assert summary.rows_emitted == 2


def test_extract_priors_aggregates_across_toponyms(db):
    t1 = _add_toponym(db, "X")
    t2 = _add_toponym(db, "Y")
    e_eadwine = _add_etymon(db, "Eadwine", "old-english", tags=("name",))
    e_tun = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=t1, elements=[e_eadwine, e_tun], attested_year=950)
    _add_etymology(db, toponym_id=t2, elements=[e_eadwine, e_tun], attested_year=950)

    native, _loan, _summary = extract_priors(db)

    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 2


def test_extract_priors_different_era_cells_are_separate(db):
    """Two toponyms attested in different centuries must land in
    different era cells (not merged). Pins the era-bucket integration."""
    t1 = _add_toponym(db, "Earlyplace")
    t2 = _add_toponym(db, "Lateplace")
    e_tun = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=t1, elements=[e_tun], attested_year=950)  # oe-late
    _add_etymology(db, toponym_id=t2, elements=[e_tun], attested_year=1200)  # me

    native, _loan, _summary = extract_priors(db)

    # Single-element toponyms → 'bare' (wyrd-nkiq).
    assert native[_NativeKey("english", "bare", "architecture", 950, "old-english:tūn")] == 1
    assert native[_NativeKey("english", "bare", "architecture", 1300, "old-english:tūn")] == 1


def test_extract_priors_skips_country_unknown_when_null(db):
    """toponym.country IS NULL -> skipped_country_unknown."""
    tid = _add_toponym(db, "Mysteryville", country=None)
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.skipped_country_unknown == 1
    assert summary.skipped_country_unmapped == 0


def test_extract_priors_skips_country_unmapped_when_not_in_map(db):
    """toponym.country='Ireland' -> deliberate gap, counts as
    skipped_country_unmapped (NOT _unknown). Pins the policy that
    Ireland/Wales/Scotland are intentionally deferred."""
    tid = _add_toponym(db, "Dublin", country="Ireland")
    e = _add_etymon(db, "dubh", "old-irish", tags=("color",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=900)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.skipped_country_unknown == 0
    assert summary.skipped_country_unmapped == 1


def test_extract_priors_skips_year_unknown_when_attested_year_is_none(db):
    """attested_year IS NULL -> skipped_year_unknown (data-quality:
    operator should backfill from source citations)."""
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=None)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.skipped_year_unknown == 1
    assert summary.skipped_year_out_of_range == 0


def test_extract_priors_skips_year_out_of_range_when_year_past_cells(db):
    """attested_year present but no matching era cell -> the
    sibling counter (skipped_year_out_of_range). Production English
    cells span (None, ...) → (..., None) so every English year
    resolves; this test injects a temporary culture mapped to a
    closed-bound family so the loop-exhaust path is exercised."""
    # latin's last cell ends at 1800; year=2000 has no match.
    _CULTURE_TO_FAMILY["__test_latin"] = "latin"
    _COUNTRY_TO_CULTURE["__TestCountry"] = "__test_latin"
    try:
        tid = _add_toponym(db, "X", country="__TestCountry")
        e = _add_etymon(db, "x", "latin", tags=("plant",))
        _add_etymology(db, toponym_id=tid, elements=[e], attested_year=2000)

        native, _loan, summary = extract_priors(db)

        assert native == {}
        assert summary.skipped_year_unknown == 0
        assert summary.skipped_year_out_of_range == 1
    finally:
        del _CULTURE_TO_FAMILY["__test_latin"]
        del _COUNTRY_TO_CULTURE["__TestCountry"]


def test_extract_priors_skips_no_tag_when_etymon_has_no_tags(db):
    """LEFT JOIN brings untagged rows through with tag=NULL; the
    extractor counts them into skipped_no_tag rather than silently
    dropping at the SQL layer."""
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=())
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    native, _loan, summary = extract_priors(db)

    assert native == {}
    assert summary.rows_scanned == 1
    assert summary.skipped_no_tag == 1


def test_extract_priors_loan_uses_recipient_family_for_era(db):
    """An Old Norse element in an English toponym at year=850 must
    use the ENGLISH family's era cell, not the NORSE family's. With
    year=850: english/oe-late = 950; norse/on-classical = 1000. The
    extractor must report midpoint=950 (recipient) for BOTH the
    native and loan rows — choosing the donor family would silently
    bucket Norse elements into era=1000."""
    tid = _add_toponym(db, "Carlton")
    e_karl = _add_etymon(db, "karl", "old-norse", tags=("social",))
    e_tun = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e_karl, e_tun], attested_year=850)

    native, loan, _summary = extract_priors(db)

    # Both elements at year=850 land in english/oe-late (midpoint 950).
    assert native[_NativeKey("english", "pre", "social", 950, "old-norse:karl")] == 1
    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 1
    # Loan: per-donor partition, but RECIPIENT family resolves the era.
    assert loan[_LoanKey("old-norse", "english", "pre", "social", 950, "old-norse:karl")] == 1
    assert (
        loan[_LoanKey("old-english", "english", "post", "architecture", 950, "old-english:tūn")]
        == 1
    )
    # The wrong (donor-family) resolution would have produced
    # midpoint=1000 for the Norse element — confirm it's NOT there.
    assert (
        loan.get(_LoanKey("old-norse", "english", "pre", "social", 1000, "old-norse:karl")) is None
    )


def test_extract_priors_partition_invariant_with_mixed_skips(db):
    """A DB with several skip-reason mixes should preserve the
    partition invariant: rows_scanned == emitted + all skips."""
    # One row that succeeds.
    tid_ok = _add_toponym(db, "Goodplace")
    e_ok = _add_etymon(db, "ok", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid_ok, elements=[e_ok], attested_year=950)
    # One row with null country.
    tid_null = _add_toponym(db, "NullPlace", country=None)
    e_null = _add_etymon(db, "n", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid_null, elements=[e_null], attested_year=950)
    # One row with unmapped country.
    tid_unmap = _add_toponym(db, "Dublin", country="Ireland")
    e_unmap = _add_etymon(db, "d", "old-irish", tags=("color",))
    _add_etymology(db, toponym_id=tid_unmap, elements=[e_unmap], attested_year=900)
    # One row with null attested_year.
    tid_noera = _add_toponym(db, "NoEra")
    e_noera = _add_etymon(db, "ne", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid_noera, elements=[e_noera], attested_year=None)
    # One row with untagged etymon.
    tid_notag = _add_toponym(db, "NoTag")
    e_notag = _add_etymon(db, "nt", "old-english", tags=())
    _add_etymology(db, toponym_id=tid_notag, elements=[e_notag], attested_year=950)

    _native, _loan, summary = extract_priors(db)

    # Each toponym produces 1 SQL row (1 element x 1 tag-or-NULL).
    assert summary.rows_scanned == 5
    assert summary.rows_emitted == 1
    assert summary.skipped_country_unknown == 1
    assert summary.skipped_country_unmapped == 1
    assert summary.skipped_year_unknown == 1
    assert summary.skipped_year_out_of_range == 0
    assert summary.skipped_no_tag == 1
    # The dataclass __post_init__ already enforced the invariant —
    # asserting we got the ExtractionResult at all means it held.


def test_extract_priors_progress_emits_to_stderr(db, capsys):
    """progress_every emits a periodic stderr line — CLAUDE.md rule."""
    # Add 5 successful rows.
    for i in range(5):
        tid = _add_toponym(db, f"P{i}")
        e = _add_etymon(db, f"e{i}", "old-english", tags=("plant",))
        _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    extract_priors(db, progress_every=2)

    captured = capsys.readouterr()
    # Expect 2 in-loop progress lines + 1 final line. CLAUDE.md:
    # "Always echo a final line at completion so the last partial
    # chunk shows up."
    assert "[2] scanned" in captured.err
    assert "[4] scanned" in captured.err
    # Final line — the loop saw 5 rows total; the final emission
    # fires because 5 % 2 != 0.
    assert "[5] scanned" in captured.err


def test_extract_priors_progress_emits_final_line_on_exact_multiple(db, capsys):
    """When total rows is an exact multiple of progress_every, the
    final periodic emission already covered the last chunk — no
    duplicate final line."""
    # 4 rows + progress_every=2 -> emissions at 2 and 4. No extra final.
    for i in range(4):
        tid = _add_toponym(db, f"P{i}")
        e = _add_etymon(db, f"e{i}", "old-english", tags=("plant",))
        _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    extract_priors(db, progress_every=2)

    captured = capsys.readouterr()
    # Exactly two progress lines, no duplicate at 4.
    assert captured.err.count("scanned") == 2


def test_extract_priors_mixed_tagged_and_untagged_compound(db):
    """A single toponym with one untagged element FIRST + one tagged
    element LAST produces (a) one skipped_no_tag from the untagged
    half (ordinal=0) and (b) one emitted cell from the tagged half at
    position='post' (ordinal=1 in 2-compound).

    Element order is critical to make this test catch the claimed
    regression. Under the regression — WHERE tag IS NOT NULL pushed
    into the cnt subquery — the cnt for this etymology would be 1
    (only the tagged element counted), so the tagged element at
    ordinal=1 would NOT exist (ordinal would be out of range). But
    if SQLite kept the row anyway with ordinal=1 against
    element_count=1, _position_label(1, 1) hits the single-element
    branch and returns 'bare'. With the LEFT JOIN preserved, cnt=2
    and _position_label(1, 2) returns 'post'. The diverging position
    keys catch the regression — a tagged-at-ordinal-0 setup would
    return 'pre' under both arms and silently pass.
    """
    tid = _add_toponym(db, "Mixedton")
    e_untagged = _add_etymon(db, "Eadwine", "old-english", tags=())
    e_tagged = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e_untagged, e_tagged], attested_year=950)

    native, _loan, summary = extract_priors(db)

    # Tagged element at ordinal=1 in 2-compound → position='post'.
    assert native[_NativeKey("english", "post", "architecture", 950, "old-english:tūn")] == 1
    # Untagged element at ordinal=0 → skipped_no_tag (its row reached
    # the extractor via LEFT JOIN with tag=NULL).
    assert summary.rows_scanned == 2
    assert summary.rows_emitted == 1
    assert summary.skipped_no_tag == 1


# ---------- mine_empirical_baselines ------------------------------------


def test_mine_empirical_baselines_dry_run_does_not_write(db):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    result = mine_empirical_baselines(db, apply=False)

    assert result["native_cells_written"] == 1
    assert _table_count(db, "empirical_priors_native") == 0
    assert _table_count(db, "empirical_priors_loan") == 0


def test_mine_empirical_baselines_apply_writes(db):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    mine_empirical_baselines(db, apply=True)

    assert _table_count(db, "empirical_priors_native") == 1
    assert _table_count(db, "empirical_priors_loan") == 1


def test_mine_empirical_baselines_apply_is_replace_not_merge_after_data_removed(db):
    """Second apply after deleting source rows leaves the tables empty."""
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    assert _table_count(db, "empirical_priors_native") == 1

    db.conn.execute("DELETE FROM toponym_etymology_element")
    db.conn.execute("DELETE FROM toponym_etymology")
    db.commit()
    mine_empirical_baselines(db, apply=True)
    assert _table_count(db, "empirical_priors_native") == 0
    assert _table_count(db, "empirical_priors_loan") == 0


def test_mine_empirical_baselines_double_apply_count_stays_one(db):
    """Pins replace-not-merge against a PK-collision-masking bug:
    applying twice on the SAME data should leave count=1, not 2.
    A hypothetical loss of the DELETE in _replace_priors would still
    pass test_..._is_idempotent (set comparison) but the count would
    be wrong if INSERT mode changed to OR REPLACE / OR IGNORE etc."""
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "tūn", "old-english", tags=("architecture",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    mine_empirical_baselines(db, apply=True)

    cur = db.conn.execute(
        "SELECT count FROM empirical_priors_native WHERE lemma_ref = 'old-english:tūn'"
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["count"] == 1


def test_mine_empirical_baselines_apply_is_idempotent(db):
    """Two consecutive applies on unchanged data produce identical
    table state."""
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)

    mine_empirical_baselines(db, apply=True)
    first_rows = sorted(db.conn.execute("SELECT * FROM empirical_priors_native").fetchall())
    mine_empirical_baselines(db, apply=True)
    second_rows = sorted(db.conn.execute("SELECT * FROM empirical_priors_native").fetchall())
    assert first_rows == second_rows


# ---------- dump_empirical_priors_to_json --------------------------------


def test_dump_json_structure(db, tmp_path):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    out = tmp_path / "priors.json"

    dump_empirical_priors_to_json(db, out, version="test-v1")

    artifact = json.loads(out.read_text())
    assert artifact["version"] == "test-v1"
    assert len(artifact["native"]) == 1
    cell = artifact["native"][0]
    assert cell["culture"] == "english"
    # Single-element toponym → 'bare' (wyrd-nkiq; matches runtime lookup).
    assert cell["position"] == "bare"
    assert cell["tag"] == "plant"
    assert cell["era_midpoint"] == 950
    assert cell["lemmas"] == {"old-english:x": 1}
    assert len(artifact["loan"]) == 1


def test_dump_json_empty_tables_produces_valid_artifact(db, tmp_path):
    """Dump before mine-apply still writes a well-formed file with
    empty native + loan lists."""
    out = tmp_path / "priors.json"
    result = dump_empirical_priors_to_json(db, out, version="empty")

    assert result == {"native_cells": 0, "loan_cells": 0}
    artifact = json.loads(out.read_text())
    assert artifact["version"] == "empty"
    assert artifact["native"] == []
    assert artifact["loan"] == []


def test_dump_json_byte_stable_across_runs(db, tmp_path):
    tid1 = _add_toponym(db, "X")
    tid2 = _add_toponym(db, "Y")
    e_a = _add_etymon(db, "a", "old-english", tags=("plant", "tree"))
    e_b = _add_etymon(db, "b", "old-norse", tags=("water",))
    _add_etymology(db, toponym_id=tid1, elements=[e_a, e_b], attested_year=950)
    _add_etymology(db, toponym_id=tid2, elements=[e_a], attested_year=950)
    mine_empirical_baselines(db, apply=True)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    dump_empirical_priors_to_json(db, out_a, version="v1")
    dump_empirical_priors_to_json(db, out_b, version="v1")

    assert out_a.read_bytes() == out_b.read_bytes()


def test_dump_json_byte_stable_across_db_reopen(db, tmp_path):
    """Stronger than intra-process: write priors, close + reopen the
    DB, dump again. Same bytes — exercises any state that might have
    been carried in the open connection."""
    tid = _add_toponym(db, "X")
    e_a = _add_etymon(db, "a", "old-english", tags=("plant", "tree"))
    e_b = _add_etymon(db, "b", "old-norse", tags=("water",))
    _add_etymology(db, toponym_id=tid, elements=[e_a, e_b], attested_year=950)
    mine_empirical_baselines(db, apply=True)

    first = tmp_path / "first.json"
    dump_empirical_priors_to_json(db, first, version="v1")
    db_path = db.path
    db.close()

    db2 = LexiconDB(db_path)
    second = tmp_path / "second.json"
    dump_empirical_priors_to_json(db2, second, version="v1")
    db2.close()

    assert first.read_bytes() == second.read_bytes()


def test_dump_json_byte_stable_across_independent_dbs(tmp_path):
    """Two DBs built with DIFFERENT insert orders produce identical
    JSON dumps. This pins the END-TO-END D36.9 contract that priors
    are content-addressable, not insert-time-rowid-dependent — the
    user-visible byte-stability that downstream caches key on.

    The byte-stability is the joint product of: (a) commutative
    dict-accumulation in extract_priors (order of element rows
    doesn't change the per-cell counts); (b) content-PRIMARY-KEY on
    both priors tables (no rowid-based ordering survives into
    storage); (c) the dump SELECTs' own ORDER BY clauses sorting on
    content fields; (d) _cell_records re-sorting on the cell key
    tuple. The extract iteration ORDER BY (_EXTRACT_SQL_ORDERED) is
    NOT what guarantees byte-stable JSON output — it's about
    debuggable iteration. A regression to ORDER BY te.id in
    _EXTRACT_SQL_ORDERED would still produce identical dumps and
    pass this test."""

    def _build_db(
        db_path: Path, insert_order: list[tuple[str, list[tuple[str, str, str]]]]
    ) -> Path:
        """Build a fresh DB and insert the given (toponym, [(canonical, language, tag), ...])
        rows in the order provided so the AUTOINCREMENT ids end up in
        different sequences across the two DBs."""
        init_schema(db_path)
        d = LexiconDB(db_path)
        d.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'S')")
        d.commit()
        for top_name, elements in insert_order:
            top_id = _add_toponym(d, top_name)
            etymon_ids = []
            for canon, lang, tag in elements:
                etymon_ids.append(_add_etymon(d, canon, lang, tags=(tag,)))
            _add_etymology(d, toponym_id=top_id, elements=etymon_ids, attested_year=950)
        mine_empirical_baselines(d, apply=True)
        out = db_path.parent / f"{db_path.stem}.json"
        dump_empirical_priors_to_json(d, out, version="v1")
        d.close()
        return out

    db_a_path = tmp_path / "a.db"
    db_b_path = tmp_path / "b.db"
    # Same content, opposite insert orders. Both DBs end up with
    # identical L2 facts but different AUTOINCREMENT rowids on
    # toponym / etymon / toponym_etymology.
    forward = [
        ("Alphaton", [("alpha", "old-english", "plant")]),
        ("Betaton", [("beta", "old-english", "tree")]),
        ("Gammaton", [("gamma", "old-norse", "water")]),
    ]
    reverse = list(reversed(forward))

    out_a = _build_db(db_a_path, forward)
    out_b = _build_db(db_b_path, reverse)

    assert out_a.read_bytes() == out_b.read_bytes()


def test_dump_json_sorts_lemmas_in_cell(db, tmp_path):
    tid1 = _add_toponym(db, "X")
    tid2 = _add_toponym(db, "Y")
    e_zeta = _add_etymon(db, "zeta", "old-english", tags=("plant",))
    e_alpha = _add_etymon(db, "alpha", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid1, elements=[e_zeta], attested_year=950)
    _add_etymology(db, toponym_id=tid2, elements=[e_alpha], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    out = tmp_path / "priors.json"
    dump_empirical_priors_to_json(db, out, version="v1")

    artifact = json.loads(out.read_text())
    cell = artifact["native"][0]
    lemma_keys = list(cell["lemmas"].keys())
    assert lemma_keys == sorted(lemma_keys)


# ---------- schema migration --------------------------------------------


def test_migration_helper_creates_tables_on_legacy_db(tmp_path):
    """Build a fresh DB but DROP the priors tables to simulate a
    legacy snapshot, then run migrate_schema and verify the tables
    + indexes get created and mine_empirical_baselines succeeds."""
    db_path = tmp_path / "legacy.db"
    init_schema(db_path)
    legacy = LexiconDB(db_path)
    legacy.conn.execute("DROP TABLE empirical_priors_native")
    legacy.conn.execute("DROP TABLE empirical_priors_loan")
    legacy.commit()
    legacy.close()

    db = LexiconDB(db_path)
    applied = migrate_schema(db)
    assert applied["empirical_priors_native_table"] is True
    assert applied["empirical_priors_loan_table"] is True

    # Verify tables + indexes exist and the pipeline runs.
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'S')")
    db.commit()
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    mine_empirical_baselines(db, apply=True)
    assert _table_count(db, "empirical_priors_native") == 1


# ---------- CLI ----------------------------------------------------------


def test_cli_mine_empirical_baselines_dry_run(db, tmp_path):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
    runner = CliRunner()

    result = runner.invoke(
        cli_root,
        ["lexicon", "mine-empirical-baselines", "--db", str(db.path)],
    )

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert _table_count(db, "empirical_priors_native") == 0


def test_cli_mine_empirical_baselines_apply(db, tmp_path):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
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
    # The apply-branch summary line should mention emitted rows.
    assert "emitted=" in result.output
    assert _table_count(db, "empirical_priors_native") == 1


def test_cli_dump_empirical_priors(db, tmp_path):
    tid = _add_toponym(db, "X")
    e = _add_etymon(db, "x", "old-english", tags=("plant",))
    _add_etymology(db, toponym_id=tid, elements=[e], attested_year=950)
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


def test_collect_empirical_priors_tolerates_missing_tables(tmp_path):
    """``collect_empirical_priors`` is called by the L4 emit; it MUST
    NOT crash when the L3 was snapshotted before the priors tables
    were added (alembic migration hadn't run). Returns None — the
    emit then writes no priors row, and the runtime loader degrades
    to empty ``EmpiricalPriors``."""
    from wyrd.generators.kenning.lexicon.empirical_priors import collect_empirical_priors

    db_path = tmp_path / "pre_priors_l3.db"
    # Hand-built minimal L3 WITHOUT the priors tables (mimics a
    # pre-empirical-priors-migration L3 snapshot).
    conn = sqlite3.connect(str(db_path))
    conn.executescript("CREATE TABLE etymon (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    with LexiconDB(db_path) as db:
        assert collect_empirical_priors(db) is None


def test_collect_empirical_priors_tolerates_partial_migration(tmp_path):
    """Partial-migration corner: ``empirical_priors_native`` exists
    but ``empirical_priors_loan`` doesn't (or vice versa). A failed /
    interrupted alembic upgrade could leave the DB in this shape.
    The try/except wraps BOTH SELECTs so either failure returns None
    rather than crashing the L4 emit. Pin so a future refactor that
    narrows the try-block (e.g. wraps the SELECTs individually) still
    handles the partial-migration case."""
    from wyrd.generators.kenning.lexicon.empirical_priors import collect_empirical_priors

    db_path = tmp_path / "partial.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        "CREATE TABLE etymon (id INTEGER PRIMARY KEY);"
        # Native table exists, loan table is missing.
        "CREATE TABLE empirical_priors_native ("
        "  culture TEXT, position TEXT, tag TEXT, era_midpoint INTEGER, "
        "  lemma_ref TEXT, count INTEGER);"
    )
    conn.commit()
    conn.close()

    with LexiconDB(db_path) as db:
        # Native is empty + loan is absent — function returns None
        # without crashing on the loan SELECT.
        assert collect_empirical_priors(db) is None


# ---------- helpers ------------------------------------------------------


def _table_count(db: LexiconDB, table: str) -> int:
    cur = db.conn.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]
