"""Tests for the region → country derivation helper + the ingest-path
+ backfill consumers of it."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.backfill_country import backfill_toponym_country
from wyrd.generators.kenning.lexicon.ingest import _upsert_toponym
from wyrd.generators.kenning.lexicon.regions import country_for_region, known_regions

# --- region → country map ----------------------------------------------------


def test_country_for_region_english_counties():
    """Every English county the Skeat book registry references must
    map to ``England``."""
    for region in (
        "Bedfordshire",
        "Cambridgeshire",
        "Lancashire",
        "Northumberland and Durham",
        "Suffolk",
        "Yorkshire",
        "England",
        "England (Danelaw)",
    ):
        assert country_for_region(region) == "England", region


def test_country_for_region_scottish():
    for region in ("Scotland", "Argyll", "Stirlingshire", "Scottish Highlands and Islands"):
        assert country_for_region(region) == "Scotland", region


def test_country_for_region_welsh():
    for region in ("Wales", "Wales and Monmouthshire"):
        assert country_for_region(region) == "Wales", region


def test_country_for_region_irish():
    assert country_for_region("Ireland") == "Ireland"


def test_country_for_region_isle_of_man():
    assert country_for_region("Isle of Man") == "Isle of Man"


def test_country_for_region_french():
    for region in ("France", "Brittany", "Loire-Inférieure (Brittany)"):
        assert country_for_region(region) == "France", region


def test_country_for_region_multi_country_returns_none():
    """Catch-all regions that span multiple countries deliberately
    return None — the operator backfills these by hand if/when they
    decide which country to attribute them to."""
    for region in ("British Isles", "Europe", "England and Wales"):
        assert country_for_region(region) is None, region


def test_country_for_region_unknown_returns_none():
    """An unrecognized region returns None — never raises."""
    assert country_for_region("Some Future Province") is None


def test_country_for_region_none_input():
    """``None`` input is safe — returns ``None`` rather than crashing."""
    assert country_for_region(None) is None


def test_known_regions_covers_all_skeat_book_regions():
    """Pin: every region referenced by ``_KNOWN_SKEAT_BOOKS`` either
    maps to a country OR is in the deliberately-omitted multi-country
    set. Catches the case where a new book is added to the registry
    with a region the map doesn't cover yet."""
    from wyrd.generators.kenning.cli.lexicon.build import _KNOWN_SKEAT_BOOKS

    multi_country = {"British Isles", "Europe", "England and Wales"}
    covered = known_regions()
    book_regions = {meta["region"] for meta in _KNOWN_SKEAT_BOOKS.values() if meta.get("region")}
    uncovered = book_regions - covered - multi_country
    assert not uncovered, (
        f"_KNOWN_SKEAT_BOOKS references regions not in country_for_region's map "
        f"(and not in the deliberately-omitted multi-country set): {sorted(uncovered)}"
    )


# --- ingest auto-derives country from region --------------------------------


def test_upsert_toponym_derives_country_from_region(tmp_path: Path):
    """``_upsert_toponym`` with only ``region`` set populates ``country``
    automatically via the region→country map. Fixes the historical
    ``country=NULL`` gap that zeroed out the empirical-priors miner."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        toponym_id = _upsert_toponym(db, "Yarmouth", region="Suffolk")
        db.commit()
        row = db.conn.execute(
            "SELECT country, region FROM toponym WHERE id = ?", (toponym_id,)
        ).fetchone()
    assert row["country"] == "England"
    assert row["region"] == "Suffolk"


def test_upsert_toponym_explicit_country_overrides_region_derivation(tmp_path: Path):
    """Caller-supplied ``country`` wins over the region-derived value
    — e.g. operator manually classifying a multi-country source."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        toponym_id = _upsert_toponym(db, "Cardiff", region="British Isles", country="Wales")
        db.commit()
        row = db.conn.execute("SELECT country FROM toponym WHERE id = ?", (toponym_id,)).fetchone()
    assert row["country"] == "Wales"


def test_upsert_toponym_unmappable_region_leaves_country_null(tmp_path: Path):
    """When the region is in the multi-country / unknown bucket and no
    explicit country is supplied, ``toponym.country`` stays NULL — the
    operator-review path picks these up via the backfill helper or by
    hand."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        toponym_id = _upsert_toponym(db, "Mystery", region="Europe")
        db.commit()
        row = db.conn.execute("SELECT country FROM toponym WHERE id = ?", (toponym_id,)).fetchone()
    assert row["country"] is None


def test_upsert_toponym_finds_existing_row_when_country_matches(tmp_path: Path):
    """Re-running the same ``(modern_name, region)`` on a populated DB
    returns the existing row's id rather than inserting a duplicate.
    Critical: the SELECT clause must match on country too, otherwise
    a second ingest pass would create twin rows (the historical bug
    that split the gazetteer + parser populations apart)."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        first = _upsert_toponym(db, "Bath", region="Somerset")
        db.commit()
        second = _upsert_toponym(db, "Bath", region="Somerset")
        db.commit()
    assert first == second


# --- backfill_toponym_country -----------------------------------------------


def test_backfill_toponym_country_updates_null_rows(tmp_path: Path):
    """A row with ``country IS NULL`` and a known region gets country
    populated in place. Idempotent — re-running is a no-op."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Acton", "Cambridgeshire"),
        )
        db.commit()
        result = backfill_toponym_country(db)
        row = db.conn.execute("SELECT country FROM toponym").fetchone()
    assert result.inspected == 1
    assert result.updated == 1
    assert result.merged == 0
    assert result.region_unknown == 0
    assert row["country"] == "England"


def test_backfill_toponym_country_leaves_unknown_regions(tmp_path: Path):
    """Multi-country / unknown regions stay NULL after backfill —
    operator-review territory."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("X", "British Isles"),
        )
        db.commit()
        result = backfill_toponym_country(db)
        row = db.conn.execute("SELECT country FROM toponym").fetchone()
    assert result.region_unknown == 1
    assert result.updated == 0
    assert row["country"] is None


def test_backfill_toponym_country_merges_when_collision(tmp_path: Path):
    """When the country-backfilled row would collide with an existing
    ``(name, country, region)`` row, the dependent ``toponym_etymology``
    rows move to the existing row and the orphan is dropped. This is
    the historical-split repair: gazetteer-side row has country set,
    parser-side has the etymology — backfill merges them."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Gazetteer row (country set, no etymology).
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Yarmouth", "England", "Suffolk"),
        )
        gazetteer_id = db.conn.execute(
            "SELECT id FROM toponym WHERE country IS NOT NULL"
        ).fetchone()["id"]
        # Parser row (country NULL, etymology attached).
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Yarmouth", "Suffolk"),
        )
        parser_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        # Pre-create a source row so the etymology FK satisfies.
        db.upsert_source(id="test-source", title="test")
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, ?, ?)",
            (parser_id, "test-source", "high"),
        )
        db.commit()

        result = backfill_toponym_country(db)

        remaining = db.conn.execute("SELECT id FROM toponym").fetchall()
        ety = db.conn.execute("SELECT toponym_id FROM toponym_etymology").fetchall()

    assert result.merged == 1
    assert result.updated == 0
    # Only the gazetteer row survives.
    assert [row["id"] for row in remaining] == [gazetteer_id]
    # Etymology FK was moved to the gazetteer row.
    assert [row["toponym_id"] for row in ety] == [gazetteer_id]


def test_backfill_merge_preserves_attestations_and_decompositions(tmp_path: Path):
    """Critical: ``_merge_toponym_into`` MUST re-parent every FK
    dependent before the source row is dropped. ``toponym_attestation``
    and ``toponym_decomposition`` both FK to ``toponym(id)`` with
    ``ON DELETE CASCADE`` — if the merge only handled
    ``toponym_etymology``, the from-side attestations + decompositions
    would silently disappear when the from-side toponym row is
    deleted. Pins that they DON'T."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Gazetteer row with country set.
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Bath", "England", "Somerset"),
        )
        gazetteer_id = db.conn.execute(
            "SELECT id FROM toponym WHERE country IS NOT NULL"
        ).fetchone()["id"]
        # Parser row (country NULL) with a full set of FK dependents.
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Bath", "Somerset"),
        )
        parser_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        db.upsert_source(id="test-src", title="test")
        # One attestation + one decomposition on the from-side.
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (parser_id, "Bathe", 1086, "test-src"),
        )
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments) VALUES (?, ?, ?, ?)",
            (parser_id, "sig-from-side", "[]", "[]"),
        )
        db.commit()

        backfill_toponym_country(db)

        attestations = db.conn.execute(
            "SELECT toponym_id, form FROM toponym_attestation"
        ).fetchall()
        decompositions = db.conn.execute(
            "SELECT toponym_id, decomposition_signature FROM toponym_decomposition"
        ).fetchall()

    # Attestation re-parented (not destroyed).
    assert len(attestations) == 1
    assert attestations[0]["toponym_id"] == gazetteer_id
    assert attestations[0]["form"] == "Bathe"
    # Decomposition re-parented (not destroyed).
    assert len(decompositions) == 1
    assert decompositions[0]["toponym_id"] == gazetteer_id
    assert decompositions[0]["decomposition_signature"] == "sig-from-side"


def test_backfill_merge_promotes_canonical_flag_on_collision(tmp_path: Path):
    """``toponym_decomposition`` carries ``is_canonical`` +
    ``canonical_source`` flags. When both rows share the same
    signature on collision, the from-side's canonical=1 must promote
    onto the surviving target row before the from-side is dropped.
    Otherwise the scholar pick silently disappears in the merge."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Bath", "England", "Somerset"),
        )
        target_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NOT NULL").fetchone()[
            "id"
        ]
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Bath", "Somerset"),
        )
        from_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        # Target side: same signature, NOT canonical.
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 0, NULL)",
            (target_id, "shared-sig", "[]", "[]"),
        )
        # From-side: same signature, canonical=1, scholar-sourced.
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 1, 'scholar')",
            (from_id, "shared-sig", "[]", "[]"),
        )
        db.commit()

        backfill_toponym_country(db)

        rows = db.conn.execute(
            "SELECT toponym_id, is_canonical, canonical_source FROM toponym_decomposition"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["toponym_id"] == target_id
    # The from-side's canonical flag + source were promoted onto the
    # surviving target row.
    assert rows[0]["is_canonical"] == 1
    assert rows[0]["canonical_source"] == "scholar"


def test_backfill_merge_honors_canonical_source_priority(tmp_path: Path):
    """When BOTH sides carry ``is_canonical=1`` for the same signature,
    the merge picks the higher-priority ``canonical_source`` per the
    schema's hierarchy (scholar > scholar-disagreement >
    unique-zero-unaccounted > tiebreaker, per
    ``data/lexicon.sql:418-429``). The surviving row is the target by
    id (FKs stable), but its flags reflect whichever side had the
    higher-priority source. Pin both directions so a reversed
    condition wouldn't pass."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Bath", "England", "Somerset"),
        )
        target_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NOT NULL").fetchone()[
            "id"
        ]
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Bath", "Somerset"),
        )
        from_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        # Target: lower-priority unique-zero-unaccounted. From-side:
        # higher-priority scholar. From-side WINS on priority — its
        # source gets promoted onto the target row.
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 1, 'unique-zero-unaccounted')",
            (target_id, "shared-sig", "[]", "[]"),
        )
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 1, 'scholar')",
            (from_id, "shared-sig", "[]", "[]"),
        )
        db.commit()

        backfill_toponym_country(db)

        rows = db.conn.execute(
            "SELECT toponym_id, is_canonical, canonical_source FROM toponym_decomposition"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["toponym_id"] == target_id
    assert rows[0]["is_canonical"] == 1
    # From-side's higher-priority 'scholar' source promoted; target's
    # 'unique-zero-unaccounted' demoted.
    assert rows[0]["canonical_source"] == "scholar"


def test_backfill_merge_target_wins_when_priority_equal_or_higher(tmp_path: Path):
    """Inverse case: target has the higher-priority source (or both
    are equal). The target's flags stay; from-side is silently
    dropped. Pin the no-promotion path so a regression wouldn't lower
    the target's provenance."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Bath", "England", "Somerset"),
        )
        target_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NOT NULL").fetchone()[
            "id"
        ]
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Bath", "Somerset"),
        )
        from_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        # Target: scholar (highest). From: unique-zero-unaccounted (lower).
        # Target wins.
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 1, 'scholar')",
            (target_id, "shared-sig", "[]", "[]"),
        )
        db.conn.execute(
            "INSERT INTO toponym_decomposition "
            "(toponym_id, decomposition_signature, morpheme_ids, "
            "unaccounted_fragments, is_canonical, canonical_source) "
            "VALUES (?, ?, ?, ?, 1, 'unique-zero-unaccounted')",
            (from_id, "shared-sig", "[]", "[]"),
        )
        db.commit()

        backfill_toponym_country(db)

        rows = db.conn.execute(
            "SELECT toponym_id, is_canonical, canonical_source FROM toponym_decomposition"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["toponym_id"] == target_id
    assert rows[0]["canonical_source"] == "scholar"


def test_backfill_merge_collapses_duplicate_decomposition_signatures(tmp_path: Path):
    """``toponym_decomposition`` has UNIQUE(toponym_id,
    decomposition_signature). When both the from-side and the target
    side have a row with the SAME signature, the merge must collapse
    the from-side row (rather than crashing on UNIQUE constraint).
    Pin the dedup behavior."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            ("Bath", "England", "Somerset"),
        )
        target_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NOT NULL").fetchone()[
            "id"
        ]
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Bath", "Somerset"),
        )
        from_id = db.conn.execute("SELECT id FROM toponym WHERE country IS NULL").fetchone()["id"]
        # Same signature on both sides.
        for tid in (target_id, from_id):
            db.conn.execute(
                "INSERT INTO toponym_decomposition "
                "(toponym_id, decomposition_signature, morpheme_ids, "
                "unaccounted_fragments) VALUES (?, ?, ?, ?)",
                (tid, "shared-sig", "[]", "[]"),
            )
        db.commit()

        backfill_toponym_country(db)

        rows = db.conn.execute(
            "SELECT toponym_id, decomposition_signature FROM toponym_decomposition"
        ).fetchall()

    # Both sides had 'shared-sig'; the from-side row was dropped, the
    # target keeps its row. Net: 1 row left, attached to target.
    assert len(rows) == 1
    assert rows[0]["toponym_id"] == target_id
    assert rows[0]["decomposition_signature"] == "shared-sig"


def test_upsert_toponym_repairs_legacy_null_country_row(tmp_path: Path):
    """Migration-window self-repair: when an existing legacy row has
    ``country=NULL`` and the new caller derives a non-null country,
    ``_upsert_toponym`` updates the legacy row in place rather than
    creating a twin. Without this, every re-ingest pass against a
    not-yet-backfilled DB silently doubles up rows."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Pre-populate a legacy NULL-country row (the historical shape).
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Yarmouth", "Suffolk"),
        )
        legacy_id = db.conn.execute("SELECT id FROM toponym").fetchone()["id"]
        db.commit()
        # New ingest derives 'England' from 'Suffolk'. Must NOT create
        # a duplicate; instead, must UPDATE the legacy row in place.
        returned_id = _upsert_toponym(db, "Yarmouth", region="Suffolk")
        db.commit()
        rows = db.conn.execute("SELECT id, country FROM toponym").fetchall()

    assert returned_id == legacy_id
    assert len(rows) == 1
    assert rows[0]["country"] == "England"


def test_upsert_toponym_legacy_repair_respects_region(tmp_path: Path):
    """The legacy-repair fallback in ``_upsert_toponym`` MUST gate on
    region — same modern_name + different region is a different place,
    not the same row needing country backfilled. A regression that
    drops the region check would silently merge unrelated places
    (Yarmouth/Norfolk vs Yarmouth/Suffolk) under one row.

    Pin: the legacy NULL-country row in region=Norfolk must NOT
    satisfy an ingest call with region=Suffolk."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Yarmouth", "Norfolk"),
        )
        norfolk_id = db.conn.execute("SELECT id FROM toponym").fetchone()["id"]
        db.commit()
        # Cross-region ingest — must create a NEW row, not repair the
        # Norfolk one.
        suffolk_id = _upsert_toponym(db, "Yarmouth", region="Suffolk")
        db.commit()
        rows = sorted(
            db.conn.execute("SELECT id, region, country FROM toponym").fetchall(),
            key=lambda r: r["id"],
        )

    assert suffolk_id != norfolk_id
    assert len(rows) == 2
    # Norfolk row untouched; Suffolk row newly created with derived country.
    assert (rows[0]["region"], rows[0]["country"]) == ("Norfolk", None)
    assert (rows[1]["region"], rows[1]["country"]) == ("Suffolk", "England")


def test_backfill_toponym_country_cli_smoke(tmp_path: Path):
    """End-to-end CLI smoke: ``wyrd kenning lexicon backfill-toponym-country
    --db <path>`` opens the L3, runs the in-place backfill, prints the
    operator-readable summary, and exits 0. Pins the CLI shim's
    LexiconDB context-manager wiring + the stderr summary format."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Acton", "Cambridgeshire"),
        )
        db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, NULL, ?)",
            ("Mystery", "British Isles"),
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "backfill-toponym-country", "--db", str(db_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Operator summary lands on stderr in production; CliRunner merges
    # stderr into ``output`` by default which is fine for shape pinning.
    assert "backfill-toponym-country:" in result.output
    assert "inspected=2" in result.output
    assert "updated=1" in result.output
    assert "region_unknown=1" in result.output
    # Side-effect: the Acton row's country is now populated; the
    # Mystery row stayed NULL.
    with LexiconDB(db_path) as db:
        rows = sorted(
            db.conn.execute("SELECT modern_name, country FROM toponym").fetchall(),
            key=lambda r: r["modern_name"],
        )
    assert (rows[0]["modern_name"], rows[0]["country"]) == ("Acton", "England")
    assert (rows[1]["modern_name"], rows[1]["country"]) == ("Mystery", None)
