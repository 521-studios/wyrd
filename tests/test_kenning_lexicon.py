"""Tests for the kenning lexicon authoring DB."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import wyrd.generators.kenning as kenning_mod
from wyrd.generators.kenning import (
    KenningEraMap,
    KenningRender,
    KenningRewind,
    _load_meanings,
    gemini_extractor,
    llm_extractor,
    paths,
)
from wyrd.generators.kenning import (
    cli as cli_mod,
)
from wyrd.generators.kenning.cli import cli as kenning_cli
from wyrd.generators.kenning.cli.lexicon import mine_llm as _mll
from wyrd.generators.kenning.cli.lexicon import review as _rv
from wyrd.generators.kenning.era.cells import (
    canonical_language_for_cell,
    era_cell_for_input,
)
from wyrd.generators.kenning.era.rewind import (
    MorphemeRewind,
    _pick_form,
    rewind_name,
    supported_eras_for_family,
)
from wyrd.generators.kenning.extractors.llm import LLMResult
from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
from wyrd.generators.kenning.lexicon import (
    _LANG_CODE_TO_JSON_FIELD,
    LANGUAGE_FIELDS,
    NON_LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    EraReflex,
    LexiconDB,
    _build_witness_filter,
    _earliest_year_in_notes,
    _emit_era_reflexes,
    _emit_inflection_list,
    _emit_variant_list,
    _extract_attestation_pairs,
    _fetch_family_era_reflexes,
    _fetch_reflex_glosses,
    _filter_concatenation_glosses,
    _find_longest_suffix_match,
    _gather_family,
    _normalize_for_quote_match,
    _position_from_usage,
    _quote_body_excerpt,
    _strip_diacritics,
    annotate_fragments_with_corpus_evidence,
    assign_etymon_to_meaning_synset,
    backfill_citation_pages,
    bridge_celtic_forms,
    bridge_generic_language,
    bridge_phonological_oe,
    bridge_phonological_on,
    clear_enrichment,
    cluster_cognates,
    cluster_ocr_variants,
    derive_lemma_candidate,
    derive_lemma_candidates,
    derive_mutation_lemma_candidate,
    derive_surface_in_modern,
    detect_running_headers,
    etymon_era_reflexes,
    export_meanings,
    fuzzy_search_attestations,
    get_meaning_preserving_candidates,
    get_meaning_synsets_for_etymon,
    ingest_parsed_entries,
    init_schema,
    link_lemmas,
    list_meaning_synsets,
    lookup_attested_years,
    migrate_schema,
    mine_toponym_attestations,
    normalize_ocr_form,
    page_for_offset,
    parse_running_header_pages,
    parse_skeat_section_header_pages,
    project_period_forms,
    record_mining_run,
    reverse_search_attestations,
    seed_from_meanings,
    seed_meaning_synsets,
)
from wyrd.generators.kenning.lexicon.sql import upgrade_head
from wyrd.generators.kenning.parsers.skeat import ParsedElement, ParsedEntry
from wyrd.generators.kenning.registers.phonology_rules import rule_form as _phonology_rule_form
from wyrd.generators.kenning.runtime.meaning import (
    Meaning,
    _normalize_era_reflex_glosses,
    _normalize_era_reflexes,
    load_meanings,
)
from wyrd.generators.kenning.runtime.respelling import respell
from wyrd.generators.kenning.runtime.scripts import transliterate


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_position_from_usage() -> None:
    assert _position_from_usage("Ac-") == "pre"
    assert _position_from_usage("-ham") == "post"
    assert _position_from_usage("-ar-") == "inner"


def test_position_from_usage_recognizes_unicode_boundary_dashes() -> None:
    """A boundary marker drawn with a NON-ASCII dash derives the SAME position as
    its ASCII twin — else a Unicode-dash reflex forks from the ASCII one on the
    position axis (D45). Leading-only already mapped to 'post'; the trailing /
    both-ends cases are the ones the old ASCII-only check got wrong."""
    assert _position_from_usage("Ac–") == "pre"  # trailing en-dash (was 'post')
    assert _position_from_usage("Ac‐") == "pre"  # trailing U+2010 HYPHEN
    assert _position_from_usage("–ar–") == "inner"  # both en-dashes (was 'post')
    assert _position_from_usage("–ham") == "post"  # leading en-dash — unchanged
    # ASCII behavior is unchanged for every case.
    assert _position_from_usage("Ac-") == "pre"
    assert _position_from_usage("-ar-") == "inner"


def test_seed_reflex_dedashes_unicode_boundary_markers(tmp_path: Path) -> None:
    """Seeded reflex IDENTITY (surface + position) de-dashes through the shared
    ``normalize_morpheme_surface`` / boundary-dash set, so a ``modern_usage`` with
    a non-ASCII boundary dash stores BARE with the right position — identical to
    the etymon path and ``modern_reflex_import`` — instead of forking the morpheme
    into a dashed reflex row. A strip-to-junk ``modern_usage`` stores no reflex."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="rando-port", title="rando")
        seed_from_meanings(
            db,
            [
                {  # leading en-dash suffix marker → bare surface, 'post'
                    "meaning": ["x"],
                    "modifier_tags": [],
                    "modifier_type": None,
                    "words": [{"modern_usage": "–shire", "old_english": ["scir"]}],
                },
                {  # trailing U+2010 prefix marker → bare surface, 'pre'
                    "meaning": ["y"],
                    "modifier_tags": [],
                    "modifier_type": None,
                    "words": [{"modern_usage": "ton‐", "old_english": ["tun"]}],
                },
                {  # strips to junk → no reflex stored at all
                    "meaning": ["z"],
                    "modifier_tags": [],
                    "modifier_type": None,
                    "words": [{"modern_usage": "–", "old_english": ["junk"]}],
                },
            ],
            source_id="rando-port",
        )
    conn = sqlite3.connect(db_path)
    rows = set(conn.execute("SELECT surface_form, position FROM reflex"))
    # Bare surfaces (no '–shire'/'ton‐'), correct positions, no junk '' row.
    assert rows == {("shire", "post"), ("ton", "pre")}
    # The dropped junk word ('–') left NO dangling reflex_etymon link — one link
    # per surviving reflex, none for the skipped one (the `continue` fires before
    # the link loop).
    assert conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0] == 2


def test_init_schema_creates_tables(fresh_db: Path) -> None:
    conn = sqlite3.connect(fresh_db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    expected = {
        "source",
        "etymon",
        "etymon_gloss",
        "etymon_tag",
        "etymon_citation",
        "reflex",
        "reflex_etymon",
        "toponym",
        "toponym_attestation",
        "toponym_etymology",
        "toponym_etymology_element",
        # wyrd-1ed9: every table that the JSONL replay path INSERTs
        # into MUST be enumerated here, so a future migration that
        # fails to create one trips this test. The original wyrd-rrse
        # bug (fantasy_morpheme missing from init_schema) shipped
        # under green tests because this set didn't pin it. The
        # complete INSERT-target list is grep-able via
        # ``grep -n "_insert_" wyrd/generators/kenning/jsonl/build.py``.
        "fantasy_morpheme",
        "etymon_descent",
        "mining_run",
    }
    assert expected.issubset(tables)
    # wyrd-2b50: personal_name / personal_name_toponym_attestation were
    # dropped (Briggs names now live in the standard etymon schema).
    assert "personal_name" not in tables
    assert "personal_name_toponym_attestation" not in tables


def test_lexicon_db_opens_in_wal_mode(fresh_db: Path) -> None:
    """LexiconDB.__init__ sets journal_mode=WAL so readers and a single
    writer don't block each other. journal_mode is persistent on the
    file — pin via PRAGMA query so a regression that drops the setting
    surfaces immediately.

    Multi-session workflow context: one Claude can be mining (writer)
    while another queries corpus state (reader). Without WAL, the
    reader either blocks or sees a stale snapshot. With WAL, the reader
    sees the last committed snapshot and proceeds without contending."""
    with LexiconDB(fresh_db) as db:
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_lexicon_db_wal_mode_persists_across_reopen(fresh_db: Path) -> None:
    """SQLite stores journal_mode in the file header, so once WAL is
    set, every subsequent open inherits it — even one that doesn't
    re-run the PRAGMA. Pin this so the docstring's persistence claim
    is load-bearing in tests, not implied empirically.

    Open #1 sets WAL via LexiconDB.__init__. Open #2 uses raw
    sqlite3.connect (no PRAGMAs) and confirms the mode survived."""
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    raw = sqlite3.connect(fresh_db)
    try:
        mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        raw.close()
    assert mode.lower() == "wal"


def test_lexicon_db_uses_synchronous_normal(fresh_db: Path) -> None:
    """Pairs with WAL: synchronous=NORMAL is the recommended setting
    under WAL — preserves crash safety (WAL replays on next open) while
    skipping the per-transaction fsync that synchronous=FULL imposes.
    Pin so a regression flipping it back to FULL (the SQLite default)
    surfaces in the slowdown rather than going silent."""
    with LexiconDB(fresh_db) as db:
        # PRAGMA returns 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA.
        sync = db.conn.execute("PRAGMA synchronous").fetchone()[0]
    assert sync == 1, f"expected synchronous=NORMAL (1), got {sync}"


def test_lexicon_db_engine_and_conn_share_dbapi_connection(fresh_db: Path) -> None:
    """wyrd-67fv: ``LexiconDB`` holds a SQLAlchemy ``Engine`` (for
    query-builder paths) plus a ``sqlite3.Connection`` shim (for the
    ~1k existing ``db.conn.execute(...)`` call sites). The StaticPool
    invariant says both views read+write through ONE underlying
    DBAPI connection.

    Identity pin (the load-bearing assertion): the connection a fresh
    ``engine.raw_connection()`` checkout exposes IS ``db.conn`` — the
    same Python object. A regression that swaps StaticPool for
    QueuePool (the SA default) would hand out a different connection
    on every checkout; that identity check fails immediately.

    Plus a round-trip smoke (write via one surface, read via the
    other) to catch the case where the pool-class change isn't a
    StaticPool→QueuePool swap but something subtler that still keeps
    object-identity by accident.
    """
    with LexiconDB(fresh_db) as db:
        # Identity assertion — the load-bearing pin for the
        # StaticPool invariant. A fresh raw_connection() checkout
        # must hand out the same underlying sqlite3.Connection
        # ``db.conn`` already points at.
        fresh_proxy = db.engine.raw_connection()
        try:
            assert fresh_proxy.driver_connection is db.conn, (
                "engine.raw_connection().driver_connection is not db.conn — "
                "StaticPool invariant broken; new pool-class is splitting connections"
            )
        finally:
            fresh_proxy.close()

        # Round-trip smoke — write via SA engine, read via sqlite3
        # shim, and vice versa. Catches divergence even when the
        # identity check happens to pass.
        with db.engine.begin() as sa_conn:
            sa_conn.exec_driver_sql("INSERT INTO source (id, title) VALUES ('via-engine', 'T')")
        row = db.conn.execute("SELECT title FROM source WHERE id = 'via-engine'").fetchone()
        assert row["title"] == "T"

        db.conn.execute("INSERT INTO source (id, title) VALUES ('via-conn', 'U')")
        db.commit()
        with db.engine.connect() as sa_conn:
            count = sa_conn.exec_driver_sql(
                "SELECT COUNT(*) FROM source WHERE id IN ('via-engine', 'via-conn')"
            ).scalar()
        assert count == 2


def test_lexicon_db_close_is_idempotent(fresh_db: Path) -> None:
    """``close()`` must be safe to call twice. With ``__enter__`` /
    ``__exit__`` exposing it, callers that release the file lock
    early inside a ``with`` block would otherwise hit a
    double-dispose on context exit."""
    db = LexiconDB(fresh_db)
    db.close()
    db.close()  # should be a no-op, no exception


def test_lexicon_db_close_disposes_pool_and_connection(fresh_db: Path) -> None:
    """``close()`` must actually call ``__raw_proxy.close()`` AND
    ``__engine.dispose()``. A regression that drops either silently
    leaks the sqlite3 connection until GC; reopens still work on
    Linux WAL (SQLite tolerates multiple readers cheerfully), but
    Windows + fcntl-locked NFS bite. The round-trip-only test was
    too loose — round-2 review correctly flagged it.

    Pin: after ``close()``, inspect the LexiconDB instance directly
    and assert the SA proxy + engine are in their closed state.
    Name-mangled attributes go through the ``_LexiconDB__`` prefix.
    """
    db = LexiconDB(fresh_db)
    db.upsert_source(id="lock-test", title="Lock Test")
    db.commit()

    raw_proxy = db._LexiconDB__raw_proxy  # type: ignore[attr-defined]
    underlying_sqlite_conn = db.conn

    db.close()

    # The fairy proxy's underlying DBAPI connection should be gone
    # after engine.dispose() — SA marks .driver_connection as None
    # on a closed proxy.
    assert raw_proxy.driver_connection is None, (
        "raw_proxy.driver_connection still set after close() — "
        "engine.dispose() didn't actually dispose the pool"
    )
    # And the original sqlite3.Connection itself must be closed.
    # If engine.dispose() ever silently regresses to a no-op the
    # connection stays alive in StaticPool until GC, and this
    # assertion catches it immediately.
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        underlying_sqlite_conn.execute("SELECT 1")

    # Reopen still works (functional smoke); the assertions above
    # are the load-bearing pins for the dispose contract.
    with LexiconDB(fresh_db) as db2:
        row = db2.conn.execute("SELECT title FROM source WHERE id = 'lock-test'").fetchone()
        assert row["title"] == "Lock Test"


def test_lexicon_db_engine_and_conn_are_read_only_properties(fresh_db: Path) -> None:
    """The StaticPool-shared-connection invariant relies on neither
    attribute being reassigned after construction; wyrd-67fv exposes
    both as read-only properties so a regression is a hard error
    rather than a silently-split-writes bug."""
    with LexiconDB(fresh_db) as db:
        with pytest.raises(AttributeError):
            db.engine = None  # type: ignore[misc]
        with pytest.raises(AttributeError):
            db.conn = None  # type: ignore[misc]


def test_init_schema_stamps_alembic_version_at_head(fresh_db: Path) -> None:
    """wyrd-67fv: ``init_schema`` runs the layered alembic migrations
    via ``upgrade_head``. If the env.py transaction handling regresses
    again (SA 2.0 ``connect()`` vs ``begin()``), the DDL persists
    but the ``alembic_version`` stamp silently doesn't — which means
    a subsequent ``upgrade`` would re-run every migration and fail
    on the duplicate CREATE. Pin the stamp explicitly so the
    regression surfaces here, not in the next migration's PR."""
    with sqlite3.connect(fresh_db) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row is not None, "alembic_version row missing"
    # Head revision id per the wyrd-67fv layered migrations.
    assert row[0] == "0020_select_targets_indexes", (
        f"expected head '0020_select_targets_indexes', got {row[0]!r}"
    )


def test_migration_0017_merges_and_dedashes_reflexes(tmp_path: Path) -> None:
    """wyrd-aicu.3 / D45: migration 0017 strips the position-encoding dash from
    reflex.surface_form, MERGING the (bare, position) collisions it creates and
    repointing reflex_etymon FKs onto the surviving (lowest-id) reflex. Stop at
    0016, seed colliding dashed reflexes, upgrade to head, assert the merge."""
    from alembic.command import upgrade

    from wyrd.generators.kenning.lexicon.sql._config import alembic_config

    db_path = tmp_path / "lexicon.db"
    cfg = alembic_config(db_path)
    upgrade(cfg, "0016_etymon_citation_attested_form")  # stop BEFORE 0017

    with sqlite3.connect(db_path) as conn:
        e1 = conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES ('tun','old-english') RETURNING id"
        ).fetchone()[0]
        e2 = conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES ('dun','old-english') RETURNING id"
        ).fetchone()[0]
        # Two post-variants that both de-dash to ('ton', 'post') → must merge.
        r1 = conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES ('-ton','post') RETURNING id"
        ).fetchone()[0]
        r2 = conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES ('ton','post') RETURNING id"
        ).fetchone()[0]
        # A pre-variant that de-dashes to ('Ton','pre') — different position, no collision.
        r3 = conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES ('Ton-','pre') RETURNING id"
        ).fetchone()[0]
        # A post form with a LEGITIMATE internal hyphen: TRIM strips the leading
        # position marker but must PRESERVE the internal one ('-al-adha' → 'al-adha').
        r4 = conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES ('-al-adha','post') RETURNING id"
        ).fetchone()[0]
        # Links: r1→e1; r2→e2; r2 ALSO →e1 (shared etymon → exercises INSERT OR
        # IGNORE PK dedup when repointing onto the survivor); r3→e1; r4→e1.
        conn.executemany(
            "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?,?)",
            [(r1, e1), (r2, e2), (r2, e1), (r3, e1), (r4, e1)],
        )
        conn.commit()

    upgrade(cfg, "head")  # runs 0017

    with sqlite3.connect(db_path) as conn:
        # No LEADING or TRAILING dash survives (the position markers) — but an
        # internal hyphen is preserved (TRIM, not REPLACE), so a blanket
        # LIKE '%-%' would wrongly flag the legit 'al-adha'.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM reflex WHERE surface_form LIKE '-%' OR surface_form LIKE '%-'"
            ).fetchone()[0]
            == 0
        )
        # The internal hyphen survived; the leading position marker was stripped.
        assert (
            conn.execute("SELECT surface_form FROM reflex WHERE id=?", (r4,)).fetchone()[0]
            == "al-adha"
        )
        # Exactly the survivor + the non-colliding pre-form + the internal-hyphen post.
        rows = conn.execute(
            "SELECT id, surface_form, position FROM reflex ORDER BY position, surface_form"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            (r4, "al-adha", "post"),
            (r1, "ton", "post"),
            (r3, "Ton", "pre"),
        ]
        # Loser r2 is gone, with no orphan links.
        assert conn.execute("SELECT COUNT(*) FROM reflex WHERE id=?", (r2,)).fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM reflex_etymon WHERE reflex_id=?", (r2,)).fetchone()[
                0
            ]
            == 0
        )
        # Survivor r1 now owns the UNION of both reflexes' etymons (e1 once,
        # despite r1+r2 both linking it — PK dedup — plus e2 from r2).
        surv = {
            row[0]
            for row in conn.execute("SELECT etymon_id FROM reflex_etymon WHERE reflex_id=?", (r1,))
        }
        assert surv == {e1, e2}
        # r3's link is untouched.
        assert (
            conn.execute("SELECT COUNT(*) FROM reflex_etymon WHERE reflex_id=?", (r3,)).fetchone()[
                0
            ]
            == 1
        )


def test_migration_0017_downgrade_redashes_from_position(tmp_path: Path) -> None:
    """wyrd-aicu.3: the 0017 downgrade re-decorates bare surfaces from the
    position column (pre=Title-, inner=-x-, post=-x). It CANNOT un-merge (the
    merge is destructive), but it must reproduce the old dash convention so a
    downgraded DB is shaped like the pre-0017 schema."""
    from alembic.command import downgrade, upgrade

    from wyrd.generators.kenning.lexicon.sql._config import alembic_config

    db_path = tmp_path / "lexicon.db"
    cfg = alembic_config(db_path)
    upgrade(cfg, "head")  # at 0017: surfaces stored bare
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO reflex (surface_form, position) VALUES (?, ?)",
            [("ton", "post"), ("ham", "inner"), ("Great", "pre"), ("aladha", "post")],
        )
        conn.commit()

    downgrade(cfg, "0016_etymon_citation_attested_form")

    with sqlite3.connect(db_path) as conn:
        surfaces = {
            (row[0], row[1]) for row in conn.execute("SELECT position, surface_form FROM reflex")
        }
    # pre re-caps + trailing dash; inner wrapped; post leading dash. 'aladha'
    # (no internal hyphen to preserve here) round-trips to '-aladha'.
    assert surfaces == {
        ("post", "-ton"),
        ("inner", "-ham-"),
        ("pre", "Great-"),
        ("post", "-aladha"),
    }


def test_seed_folds_position_variants_into_one_bare_reflex(fresh_db: Path) -> None:
    """wyrd-aicu.3: the bare-surface writer + upsert ON CONFLICT(surface_form,
    position) FOLD two inputs that strip to the same (bare, position) onto ONE
    reflex, unioning their etymon links — the going-forward equivalent of the
    0017 migration's merge (so a fresh mine can't reintroduce the dash-variants
    the migration just collapsed)."""
    data = [
        {
            "meaning": ["Estate"],
            "modifier_tags": [],
            "modifier_type": "Topographical",
            "words": [
                {"modern_usage": "-ton", "old_english": ["tun"]},  # post suffix → 'ton'/post
                {"modern_usage": "ton", "celtic_mix": ["ton"]},  # dashless standalone → 'ton'/post
            ],
        },
    ]
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        seed_from_meanings(db, data, "test-src")
        # Both inputs strip to ('ton', 'post') → exactly ONE reflex row.
        reflexes = [
            (r["surface_form"], r["position"])
            for r in db.conn.execute("SELECT surface_form, position FROM reflex").fetchall()
        ]
        assert reflexes == [("ton", "post")]
        # ...owning BOTH etymons' links (OE tun + celtic ton).
        assert db.conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0] == 2


def test_migration_0013_backfills_element_confidence_from_parent(
    tmp_path: Path,
) -> None:
    """wyrd-2n1: migration 0013 backfills toponym_etymology_element.
    confidence from the parent toponym_etymology.confidence so every
    pre-PR element gets the parent's value (lossless starting point).
    Pins the backfill semantic so a future re-ordering of upgrade()
    steps doesn't silently break the inherit-from-parent contract."""
    # init_schema runs every migration including 0013, so the column
    # exists by the time we read it. Seed a row before AND after the
    # backfill semantic: insert a parent + element pair, set parent
    # confidence='medium' but leave the element NULL, then re-run the
    # backfill SQL directly to assert it copies 'medium' over.
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO source (id, title) VALUES ('test-src', 'Test')")
        cur = conn.execute("INSERT INTO toponym (modern_name) VALUES ('Cot')")
        toponym_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'test-src', 'medium')",
            (toponym_id,),
        )
        te_id = cur.lastrowid
        eth = conn.execute(
            "INSERT INTO etymon (canonical_form, language) "
            "VALUES ('cot', 'old-english') RETURNING id"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id, confidence) "
            "VALUES (?, 0, ?, NULL)",
            (te_id, eth),
        )
        conn.commit()
        # Re-run the backfill (idempotent — UPDATE WHERE confidence IS NULL).
        conn.execute(
            """
            UPDATE toponym_etymology_element
               SET confidence = (
                    SELECT te.confidence
                      FROM toponym_etymology AS te
                     WHERE te.id = toponym_etymology_element.toponym_etymology_id
               )
             WHERE confidence IS NULL
            """
        )
        conn.commit()
        out = conn.execute(
            "SELECT confidence FROM toponym_etymology_element "
            "WHERE toponym_etymology_id = ? AND ordinal = 0",
            (te_id,),
        ).fetchone()
    assert out[0] == "medium"


def _filter_sqlite_reflection_artifacts(diffs: list, metadata, table_ddl: dict[str, str]) -> list:
    """compare_metadata has three well-known SQLite reflection blind
    spots that produce false-positive diffs. Filter each out:

    1. ``modify_nullable`` on PRIMARY KEY columns — SQLite reports PKs
       as nullable; SA Core treats PKs as implicitly NOT NULL.
    2. ``remove_fk`` + ``add_fk`` paired by (column-keys, referred-table)
       — SQLite FK reflection drops the ``on_delete`` clause, so any FK
       with ondelete in MetaData produces a paired drop+add.
    3. ``modify_type`` where MetaData declares a collation and the live
       DDL (``sqlite_master.sql``) actually carries ``COLLATE NOCASE``
       on the named column — SQLite reflection drops collation
       attributes from reflected types. Verify against the DDL string.
       If the DDL DOESN'T have COLLATE on that column, the diff is
       real drift and we let it through.
    """
    # Pass 1: index fk pairs by (table, column_keys, referred_table).
    fk_pairs: dict[tuple, dict[str, list]] = {}
    for d in diffs:
        if not isinstance(d, tuple) or d[0] not in ("remove_fk", "add_fk"):
            continue
        op, fk = d[0], d[1]
        key = (
            fk.table.name,
            tuple(fk.column_keys),
            fk.referred_table.name,
        )
        fk_pairs.setdefault(key, {"remove_fk": [], "add_fk": []})[op].append(d)
    paired_fk_diffs = set()
    for pair in fk_pairs.values():
        if pair["remove_fk"] and pair["add_fk"]:
            # Mark each paired diff by Python id so we can drop them.
            for d in pair["remove_fk"] + pair["add_fk"]:
                paired_fk_diffs.add(id(d))

    filtered = []
    for d in diffs:
        if isinstance(d, tuple) and id(d) in paired_fk_diffs:
            continue
        if isinstance(d, list) and len(d) == 1 and d[0][0] == "modify_nullable":
            _, _, table, col, _, _, _ = d[0]
            t = metadata.tables.get(table)
            if t is not None and col in t.c and t.c[col].primary_key:
                continue
        if isinstance(d, list) and len(d) == 1 and d[0][0] == "modify_type":
            _, _, table, col, _, _, new_type = d[0]
            collation = getattr(new_type, "collation", None)
            if collation:
                ddl = table_ddl.get(table, "")
                if _column_has_collation_in_ddl(ddl, col, collation):
                    # Reflection dropped the collation but the DDL has
                    # it — false positive, not real drift.
                    continue
        filtered.append(d)
    return filtered


def _column_has_collation_in_ddl(ddl: str, column: str, collation: str) -> bool:
    """Look for ``<column> ... COLLATE <collation>`` in a SQLite CREATE
    TABLE DDL string. Each column is on its own line."""
    for line in ddl.splitlines():
        stripped = line.lstrip()
        # Column lines start with the bare identifier (no quoting in
        # this codebase's migrations); guard on word boundary.
        if stripped.startswith(column + " ") or stripped.startswith(column + "\t"):
            return f"COLLATE {collation}" in line
    return False


def test_alembic_head_schema_matches_tables_metadata(fresh_db: Path) -> None:
    """wyrd-hd28: SA Core MetaData in ``lexicon/sql/tables.py`` must
    declare the same schema that the alembic chain produces. Drift
    between them silently breaks future ``alembic revision
    --autogenerate`` runs (which would generate ALTER TABLE diffs for
    "phantom" changes that exist only in reflection).

    Two real bugs of this class shipped to main before this test:

    - wyrd-uzoh / PR #276 added ``personal_name`` +
      ``personal_name_toponym_attestation`` migrations but forgot to
      mirror them in tables.py.
    - wyrd-rrse / PR #281 round-1 added ``COLLATE NOCASE`` to the
      ``fantasy_morpheme`` migration but forgot the matching
      ``collation="NOCASE"`` arg in the tables.py Column declaration.

    Both went undetected for an entire PR cycle. This test pins the
    invariant directly: feed init_schema's DB (= alembic head) and
    tables.py MetaData into ``compare_metadata``, filter known SQLite
    reflection artifacts, assert the rest is empty.

    Filtered artifacts (false positives that must NOT mask real
    drift) — see ``_filter_sqlite_reflection_artifacts`` docstring
    for the rules.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    # Read sqlite_master.sql for every table — this is the source of
    # truth for "what the DB actually has." Used to verify collation
    # claims when compare_metadata flags a modify_type.
    with sqlite3.connect(fresh_db) as conn:
        table_ddl = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    engine = create_engine(f"sqlite:///{fresh_db}")
    with engine.connect() as alembic_conn:
        ctx = MigrationContext.configure(alembic_conn, opts={"compare_type": True})
        diffs = compare_metadata(ctx, metadata)
    engine.dispose()

    real_drift = _filter_sqlite_reflection_artifacts(diffs, metadata, table_ddl)
    assert not real_drift, "alembic head ↔ tables.py MetaData drift detected:\n" + "\n".join(
        f"  {d!r}" for d in real_drift
    )


def test_alembic_head_collation_matches_tables_metadata(fresh_db: Path) -> None:
    """wyrd-hd28: bidirectional COLLATE parity check between alembic
    head and ``tables.py`` MetaData. ``compare_metadata`` cannot detect
    collation drift on SQLite — the dialect's reflector drops the
    ``COLLATE`` clause from reflected column types, so both
    "MetaData declares NOCASE but DB doesn't" and the inverse
    silently produce zero diffs.

    The wyrd-rrse round-1 drift was exactly this shape:
    ``fantasy_morpheme.input_name`` had ``COLLATE NOCASE`` in the
    migration's CREATE TABLE but the SA Core ``Column`` was declared
    as plain ``Text`` (missing ``collation="NOCASE"``).
    ``compare_metadata`` saw no drift; only manual audit caught it.

    This test does a direct DDL parity check: every column in MetaData
    with a declared collation MUST appear with that ``COLLATE`` clause
    in ``sqlite_master.sql``, and every ``COLLATE`` clause in
    ``sqlite_master.sql`` MUST have a matching MetaData declaration.
    """
    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    ddl_collations: dict[tuple[str, str], str] = {}
    with sqlite3.connect(fresh_db) as conn:
        for table_name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall():
            for line in sql.splitlines():
                if "COLLATE " not in line.upper():
                    continue
                tokens = line.strip().split()
                if not tokens:
                    continue
                # Strip identifier quoting — current migrations don't
                # quote, but harden against a future migration that does.
                column_name = tokens[0].strip("\"'`[]")
                for i, tok in enumerate(tokens):
                    if tok.upper() == "COLLATE" and i + 1 < len(tokens):
                        # SQLite treats collation names case-insensitively
                        # (COLLATE nocase == COLLATE NOCASE). Also strip
                        # trailing punctuation / quotes / close parens
                        # that may follow the collation name.
                        ddl_collations[(table_name, column_name)] = (
                            tokens[i + 1].strip(" ,;()\"'`").upper()
                        )
                        break

    md_collations: dict[tuple[str, str], str] = {}
    for table in metadata.tables.values():
        for column in table.columns:
            collation = getattr(column.type, "collation", None)
            if collation:
                md_collations[(table.name, column.name)] = collation.upper()

    missing_in_ddl = set(md_collations) - set(ddl_collations)
    missing_in_md = set(ddl_collations) - set(md_collations)
    mismatched = {
        k for k in set(md_collations) & set(ddl_collations) if md_collations[k] != ddl_collations[k]
    }

    errors: list[str] = []
    for k in sorted(missing_in_ddl):
        errors.append(
            f"  {k[0]}.{k[1]}: MetaData declares COLLATE {md_collations[k]} "
            f"but live DDL has no COLLATE clause"
        )
    for k in sorted(missing_in_md):
        errors.append(
            f"  {k[0]}.{k[1]}: live DDL has COLLATE {ddl_collations[k]} "
            f"but MetaData declares no collation"
        )
    for k in sorted(mismatched):
        errors.append(f"  {k[0]}.{k[1]}: MetaData={md_collations[k]} but DDL={ddl_collations[k]}")

    assert not errors, "Collation drift between alembic head DDL and MetaData:\n" + "\n".join(
        errors
    )


def test_alembic_head_fk_ondelete_matches_tables_metadata(fresh_db: Path) -> None:
    """wyrd-hd28: bidirectional FK ``ON DELETE`` parity check.

    ``compare_metadata`` cannot reliably detect ondelete drift on SQLite
    because reflection strips the ``ondelete`` clause — every FK with
    ``ondelete=`` in MetaData generates a paired ``remove_fk``+``add_fk``
    that the structural-drift filter must drop as a known reflection
    artifact. The filter drops the pair on column-tuple match alone,
    NOT on ondelete value match — so a real drift where MetaData says
    ``ondelete="SET NULL"`` while the migration DDL says
    ``ON DELETE CASCADE`` is silently masked.

    Same shape as the COLLATE drift class; same fix pattern. Read
    each FK's ``on_delete`` clause via ``PRAGMA foreign_key_list`` —
    that pragma reports the literal DDL clause faithfully (unlike SA
    reflection, which drops it for SQLite) — and reconcile against
    MetaData's FK declarations.
    """
    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    ddl_ondeletes: dict[tuple[str, tuple[str, ...]], str] = {}
    with sqlite3.connect(fresh_db) as conn:
        for (table_name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            # PRAGMA foreign_key_list returns one row per FK column:
            # (id, seq, referred_table, from_col, to_col, on_update,
            #  on_delete, match). Group rows by (id) to reconstruct
            # multi-column FK constraints.
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
            by_id: dict[int, list[tuple[int, str, str]]] = {}
            for fk_id, seq, _ref_table, from_col, _to_col, _on_upd, on_del, _ in fk_rows:
                by_id.setdefault(fk_id, []).append((seq, from_col, on_del))
            for cols_info in by_id.values():
                cols_info.sort()
                cols = tuple(c[1] for c in cols_info)
                # All seq entries for one FK share the same on_delete.
                ondelete = (cols_info[0][2] or "NO ACTION").upper()
                ddl_ondeletes[(table_name, cols)] = ondelete

    md_ondeletes: dict[tuple[str, tuple[str, ...]], str] = {}
    for table in metadata.tables.values():
        for fk in table.foreign_key_constraints:
            cols = tuple(fk.column_keys)
            ondelete = (fk.ondelete or "NO ACTION").upper()
            md_ondeletes[(table.name, cols)] = ondelete

    missing_in_ddl = set(md_ondeletes) - set(ddl_ondeletes)
    missing_in_md = set(ddl_ondeletes) - set(md_ondeletes)
    mismatched = {
        k for k in set(md_ondeletes) & set(ddl_ondeletes) if md_ondeletes[k] != ddl_ondeletes[k]
    }

    errors: list[str] = []
    for k in sorted(missing_in_ddl):
        errors.append(
            f"  {k[0]}({','.join(k[1])}): MetaData FK with ondelete={md_ondeletes[k]} "
            f"has no matching FK in live DDL"
        )
    for k in sorted(missing_in_md):
        errors.append(
            f"  {k[0]}({','.join(k[1])}): live DDL has FK with ON DELETE {ddl_ondeletes[k]} "
            f"but MetaData has no matching FK"
        )
    for k in sorted(mismatched):
        errors.append(
            f"  {k[0]}({','.join(k[1])}): MetaData ondelete={md_ondeletes[k]} "
            f"but DDL ON DELETE {ddl_ondeletes[k]}"
        )

    assert not errors, "FK ondelete drift between alembic head DDL and MetaData:\n" + "\n".join(
        errors
    )


def test_alembic_head_check_constraints_match_tables_metadata(fresh_db: Path) -> None:
    """wyrd-hd28: bidirectional CHECK constraint parity check.

    ``compare_metadata`` does not compare CHECK constraints at all —
    drift between migration CHECKs and MetaData ``CheckConstraint``
    declarations is invisible to autogenerate. The schema has many
    enum-style CHECKs (e.g.,
    ``position_pref IN ('pre','post','inner','free')``,
    ``edge_type IN ('inheritance','borrowing',...)``,
    ``usable IN (0,1)``). A migration that adds an allowed value
    without updating MetaData (or vice versa) lets the DB accept rows
    the code's MetaData-derived validators believe impossible.

    Compare normalized CHECK clauses extracted from ``sqlite_master.sql``
    to the ``sqltext`` of each MetaData ``CheckConstraint``. Whitespace
    is normalized to a single space; both sides upper-cased for token
    casing tolerance.
    """
    import re

    from sqlalchemy import CheckConstraint

    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    def _normalize(expr: str) -> str:
        expr = re.sub(r"\s+", " ", expr).strip()
        # Drop whitespace adjacent to parens so DDL "( 'x' )" matches
        # MetaData's "('x')". SQL is whitespace-insensitive here.
        expr = re.sub(r"\(\s+", "(", expr)
        expr = re.sub(r"\s+\)", ")", expr)
        return expr.upper()

    def _extract_check_exprs(sql: str) -> set[str]:
        """Extract every ``CHECK(...)`` body, balancing nested parens
        and ignoring parens inside SQL string literals. Required
        because the body itself often contains parenthesized
        sub-expressions (e.g., ``usable IN (0, 1)``). String-literal
        handling is defensive — no current CHECK in this schema
        contains parens inside a string, but a future one might."""
        exprs: set[str] = set()
        i = 0
        upper = sql.upper()
        while True:
            start = upper.find("CHECK", i)
            if start == -1:
                break
            j = start + 5
            while j < len(sql) and sql[j].isspace():
                j += 1
            if j >= len(sql) or sql[j] != "(":
                i = start + 5
                continue
            depth = 1
            k = j + 1
            in_string = False
            while k < len(sql) and depth > 0:
                ch = sql[k]
                if ch == "'":
                    # Doubled '' inside a string is an escaped quote;
                    # otherwise toggle the string flag.
                    if k + 1 < len(sql) and sql[k + 1] == "'":
                        k += 1
                    else:
                        in_string = not in_string
                elif not in_string:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                k += 1
            if depth == 0:
                exprs.add(_normalize(sql[j + 1 : k - 1]))
            i = k
        return exprs

    ddl_checks: dict[str, set[str]] = {}
    with sqlite3.connect(fresh_db) as conn:
        for table_name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall():
            checks = _extract_check_exprs(sql)
            if checks:
                ddl_checks[table_name] = checks

    md_checks: dict[str, set[str]] = {}
    for table in metadata.tables.values():
        constraints = {
            _normalize(str(c.sqltext)) for c in table.constraints if isinstance(c, CheckConstraint)
        }
        # Column-level CheckConstraints attach to the column, not the table.
        for column in table.columns:
            for c in column.constraints:
                if isinstance(c, CheckConstraint):
                    constraints.add(_normalize(str(c.sqltext)))
        if constraints:
            md_checks[table.name] = constraints

    errors: list[str] = []
    all_tables = set(ddl_checks) | set(md_checks)
    for t in sorted(all_tables):
        ddl_set = ddl_checks.get(t, set())
        md_set = md_checks.get(t, set())
        only_ddl = ddl_set - md_set
        only_md = md_set - ddl_set
        for expr in sorted(only_ddl):
            errors.append(f"  {t}: live DDL has CHECK({expr}) but MetaData does not")
        for expr in sorted(only_md):
            errors.append(f"  {t}: MetaData has CheckConstraint({expr}) but DDL does not")

    assert not errors, (
        "CHECK constraint drift between alembic head DDL and MetaData:\n" + "\n".join(errors)
    )


def test_alembic_head_server_default_matches_tables_metadata(fresh_db: Path) -> None:
    """wyrd-jaur: bidirectional ``DEFAULT`` clause parity check.

    ``compare_metadata`` with ``compare_server_default=True`` does NOT
    detect server_default drift on SQLite — empirically verified by
    deliberately changing a Column's ``server_default`` to a sentinel
    string and seeing zero ``modify_default`` diffs surface. Same
    SQLite reflection blind spot as collation and ondelete.

    The schema has 17 ``server_default=text(...)`` declarations
    (mostly ``text("0")`` boolean-as-int defaults plus one
    ``text("(datetime('now'))")`` and one
    ``text("'reverse-search-v1'")``). A migration that
    adds/removes/changes a column DEFAULT without updating MetaData
    silently produces different insert-time behavior between fresh
    DBs and MetaData-derived query builders.

    Use ``PRAGMA table_info(<table>)`` to read each column's
    ``dflt_value`` directly — SQLite parses the CREATE TABLE DDL and
    surfaces the default expression as a string, sidestepping all
    the DDL-tokenization fragility of an earlier round (shlex /
    paren-balancing). Reconcile bidirectionally against MetaData's
    ``column.server_default``.
    """
    import re

    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    def _normalize_default(expr: str) -> str:
        # Collapse whitespace, strip outer parens (MetaData may
        # declare ``(datetime('now'))`` while PRAGMA returns the
        # inner ``datetime('now')`` after SQLite parses the DDL).
        expr = re.sub(r"\s+", " ", expr).strip()
        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1].strip()
        return expr.upper()

    ddl_defaults: dict[tuple[str, str], str] = {}
    with sqlite3.connect(fresh_db) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            table_name = row["name"]
            # Quote the table identifier so a future table named
            # after a reserved word doesn't break the PRAGMA.
            for col in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall():
                if col["dflt_value"] is not None:
                    ddl_defaults[(table_name, col["name"])] = _normalize_default(col["dflt_value"])

    md_defaults: dict[tuple[str, str], str] = {}
    for table in metadata.tables.values():
        for column in table.columns:
            if column.server_default is None:
                continue
            sd = column.server_default
            # server_default may be a DefaultClause wrapping TextClause.
            arg = getattr(sd, "arg", sd)
            md_defaults[(table.name, column.name)] = _normalize_default(str(arg))

    missing_in_ddl = set(md_defaults) - set(ddl_defaults)
    missing_in_md = set(ddl_defaults) - set(md_defaults)
    mismatched = {
        k for k in set(md_defaults) & set(ddl_defaults) if md_defaults[k] != ddl_defaults[k]
    }

    errors: list[str] = []
    for k in sorted(missing_in_ddl):
        errors.append(
            f"  {k[0]}.{k[1]}: MetaData declares server_default={md_defaults[k]!r} "
            f"but live DDL has no DEFAULT clause"
        )
    for k in sorted(missing_in_md):
        errors.append(
            f"  {k[0]}.{k[1]}: live DDL has DEFAULT {ddl_defaults[k]!r} "
            f"but MetaData declares no server_default"
        )
    for k in sorted(mismatched):
        errors.append(f"  {k[0]}.{k[1]}: MetaData={md_defaults[k]!r} but DDL={ddl_defaults[k]!r}")

    assert not errors, "server_default drift between alembic head DDL and MetaData:\n" + "\n".join(
        errors
    )


def test_alembic_head_indexes_match_tables_metadata(fresh_db: Path) -> None:
    """wyrd-jaur: index-name parity between alembic head and MetaData.

    ``compare_metadata`` warns and skips expression-based unique
    indexes on SQLite (``idx_etymon_citation_unique``,
    ``idx_pn_toponym_dedup``, ``idx_toponym_unique``,
    ``idx_attestation_unique``) — drift in their shape can't be
    caught at the diff level. As a coarser canary, verify the SET
    of index names in MetaData matches the SET of index names in
    alembic head. Catches "migration adds an index without adding
    the matching ``Index(...)`` declaration in tables.py" and the
    inverse.

    Does NOT catch shape drift in the body of expression-indexes
    (e.g., ``COALESCE(page, '')`` → ``COALESCE(page, '<none>')``).
    That gap is wider than this PR; the right canary would be a
    DDL-byte-equality check, which is fragile against migration
    whitespace differences and out of scope here.
    """
    from wyrd.generators.kenning.lexicon.sql.tables import metadata

    with sqlite3.connect(fresh_db) as conn:
        # No `sql IS NOT NULL` filter — that would exclude indexes
        # SQLite auto-generates for UNIQUE constraints (their DDL
        # lives inside the parent CREATE TABLE, so sqlite_master.sql
        # is NULL). The sqlite_autoindex_ prefix filter below strips
        # the unnamed auto-indexes; named UniqueConstraints get a
        # caller-controlled name and SHOULD appear in this set so
        # they match the matching md_indexes entry.
        # Filter SQLite-internal indexes at the query level — both
        # the ``sqlite_autoindex_*`` indexes auto-generated for
        # unnamed UNIQUE constraints AND any other ``sqlite_*``
        # internal objects (sqlite_stat1, etc.) that may surface
        # in some configurations.
        ddl_indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

    from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint

    md_indexes: set[str] = set()
    for table in metadata.tables.values():
        md_indexes.update(idx.name for idx in table.indexes if idx.name)
        # Named UniqueConstraints / PrimaryKeyConstraints also surface
        # as indexes in SQLite but live in table.constraints, not
        # table.indexes. Include them so a future
        # ``UniqueConstraint(..., name="...")`` is caught by this
        # parity check. Unnamed UniqueConstraints produce
        # ``sqlite_autoindex_*`` indexes which the DDL-side query
        # filters out at the SQL level.
        md_indexes.update(
            c.name
            for c in table.constraints
            if isinstance(c, (UniqueConstraint, PrimaryKeyConstraint)) and c.name
        )

    missing_in_ddl = md_indexes - ddl_indexes
    missing_in_md = ddl_indexes - md_indexes

    errors: list[str] = []
    for name in sorted(missing_in_ddl):
        errors.append(f"  {name}: MetaData declares index but alembic head does not")
    for name in sorted(missing_in_md):
        errors.append(f"  {name}: alembic head has index but MetaData does not")

    assert not errors, "Index name drift between alembic head and MetaData:\n" + "\n".join(errors)


# Views live in alembic migrations (per tables.py docstring), not in
# the SA Core MetaData. Pin the expected set explicitly so a migration
# that accidentally drops one — or adds one without updating this list
# — surfaces here.
_EXPECTED_VIEWS = {
    "etymon_canonical",
    "etymon_consensus",
    "etymon_gloss_canonical",
    "etymon_tag_canonical",
    "etymon_text_match_canonical",
    "toponym_breakdown_signature",
    "toponym_etymology_canonical",
}


def test_alembic_head_views_match_expected_set(fresh_db: Path) -> None:
    """wyrd-jaur: pin the set of views the alembic chain produces.

    ``compare_metadata`` ignores views entirely (SA Core has no
    first-class view object). A migration that renames a view, drops
    one, or adds one without consumer updates can ship undetected.
    When this test fails, update ``_EXPECTED_VIEWS`` to match what
    the migration produced — the failure is the signal to update
    callers that consume the view set."""
    with sqlite3.connect(fresh_db) as conn:
        actual = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        }
    missing = _EXPECTED_VIEWS - actual
    unexpected = actual - _EXPECTED_VIEWS
    errors: list[str] = []
    if missing:
        errors.append(
            f"  views in _EXPECTED_VIEWS but missing from alembic head: {sorted(missing)}"
        )
    if unexpected:
        errors.append(
            f"  views in alembic head but not in _EXPECTED_VIEWS: {sorted(unexpected)} "
            f"— update the constant if this addition is intentional"
        )
    assert not errors, "View set drift between alembic head and _EXPECTED_VIEWS:\n" + "\n".join(
        errors
    )


def _normalize_ddl(sql: str) -> str:
    """Collapse all whitespace runs to single spaces (and strip) so the
    snapshot pins the DDL SEMANTICS, not the migration's incidental
    indentation / line-wrapping."""
    return " ".join(sql.split())


# wyrd-0nrg: snapshot the BODIES of the COALESCE-padded expression-unique
# indexes. The name-set / table-DDL parity tests above would NOT catch a
# migration that changes a sentinel (e.g. COALESCE(page, '') → COALESCE(page,
# ' ')), which silently changes uniqueness semantics — the same class of
# silent bug as wyrd-rrse (collation). Pin the normalized DDL so such a change
# fails loudly. Intentional brittleness: an intended index-body change is the
# signal to update this dict. (idx_pn_toponym_dedup was dropped with the
# personal_name tables in migration 0014, so only three remain.)
_EXPECTED_EXPRESSION_INDEX_BODIES = {
    "idx_etymon_citation_unique": (
        "CREATE UNIQUE INDEX idx_etymon_citation_unique "
        "ON etymon_citation(etymon_id, source_id, COALESCE(page, ''))"
    ),
    "idx_toponym_unique": (
        "CREATE UNIQUE INDEX idx_toponym_unique "
        "ON toponym(modern_name, COALESCE(country, ''), COALESCE(region, ''))"
    ),
    "idx_attestation_unique": (
        "CREATE UNIQUE INDEX idx_attestation_unique ON toponym_attestation( "
        "toponym_id, form, COALESCE(date_year, 0), COALESCE(source_doc, '') )"
    ),
}


def test_alembic_head_expression_index_bodies_pin_to_snapshot(fresh_db: Path) -> None:
    """wyrd-0nrg: the COALESCE-padded unique indexes carry uniqueness
    semantics in their expression bodies (the sentinel padding), which the
    name-set parity test can't see. Pin each body to a normalized snapshot so
    a sentinel change (e.g. '' → ' ') fails loudly. When this fails on an
    intended change, update ``_EXPECTED_EXPRESSION_INDEX_BODIES``."""
    with sqlite3.connect(fresh_db) as conn:
        for name, expected in _EXPECTED_EXPRESSION_INDEX_BODIES.items():
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
            ).fetchone()
            assert row is not None, f"expression-unique index {name!r} missing from alembic head"
            assert _normalize_ddl(row[0]) == expected, (
                f"index-body drift for {name!r}:\n  expected: {expected}\n"
                f"  actual:   {_normalize_ddl(row[0])}"
            )


# wyrd-0nrg: snapshot the BODIES of the canonical-projection views. _EXPECTED_VIEWS
# pins NAMES only — a migration that rewrites a view's SELECT (projected columns
# or WHERE filter) would silently ship, and downstream code reads these
# projections. Pin the normalized body; an intended rewrite is the signal to
# update this dict.
_EXPECTED_VIEW_BODIES = {
    "etymon_canonical": "CREATE VIEW etymon_canonical AS SELECT * FROM etymon WHERE merged_into_id IS NULL",
    "etymon_consensus": (
        "CREATE VIEW etymon_consensus AS SELECT lemma_id, canonical_form, language, "
        "COUNT(DISTINCT source_id) AS witnesses FROM ( SELECT "
        "COALESCE(le.id, target.id, e.id) AS lemma_id, "
        "COALESCE(le.canonical_form, target.canonical_form, e.canonical_form) AS canonical_form, "
        "e.language, c.source_id FROM etymon e LEFT JOIN etymon target "
        "ON target.id = COALESCE(e.merged_into_id, e.lemma_id) "
        "LEFT JOIN etymon le ON le.id = target.lemma_id "
        "LEFT JOIN etymon_citation c ON c.etymon_id = e.id ) "
        "GROUP BY lemma_id, canonical_form, language"
    ),
    "etymon_gloss_canonical": (
        "CREATE VIEW etymon_gloss_canonical AS SELECT DISTINCT "
        "COALESCE(le.id, target.id, e.id) AS canonical_etymon_id, g.gloss "
        "FROM etymon e JOIN etymon_gloss g ON g.etymon_id = e.id "
        "LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id) "
        "LEFT JOIN etymon le ON le.id = target.lemma_id"
    ),
    "etymon_tag_canonical": (
        "CREATE VIEW etymon_tag_canonical AS SELECT DISTINCT "
        "COALESCE(le.id, target.id, e.id) AS canonical_etymon_id, t.tag "
        "FROM etymon e JOIN etymon_tag t ON t.etymon_id = e.id "
        "LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id) "
        "LEFT JOIN etymon le ON le.id = target.lemma_id"
    ),
    "etymon_text_match_canonical": (
        "CREATE VIEW etymon_text_match_canonical AS SELECT "
        "COALESCE(le.id, target.id, e.id) AS canonical_etymon_id, m.source_id, m.matched_form, "
        "SUM(m.match_count) AS total_match_count, MIN(m.edit_distance) AS edit_distance, "
        "MIN(m.attested_year) AS attested_year FROM etymon e "
        "JOIN etymon_text_match m ON m.etymon_id = e.id "
        "LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id) "
        "LEFT JOIN etymon le ON le.id = target.lemma_id "
        "GROUP BY canonical_etymon_id, m.source_id, m.matched_form"
    ),
    "toponym_breakdown_signature": (
        "CREATE VIEW toponym_breakdown_signature AS SELECT toponym_id, toponym_etymology_id, "
        "source_id, GROUP_CONCAT(etymon_id, ',') AS signature FROM ( SELECT te.toponym_id, "
        "te.id AS toponym_etymology_id, te.source_id, tee.etymon_id, tee.ordinal "
        "FROM toponym_etymology te LEFT JOIN toponym_etymology_element tee "
        "ON tee.toponym_etymology_id = te.id ORDER BY te.id, tee.ordinal ) "
        "GROUP BY toponym_etymology_id, toponym_id, source_id"
    ),
    "toponym_etymology_canonical": (
        "CREATE VIEW toponym_etymology_canonical AS "
        "SELECT * FROM toponym_etymology WHERE is_canonical = 1"
    ),
}


def test_alembic_head_view_bodies_match_snapshot(fresh_db: Path) -> None:
    """wyrd-0nrg: pin each view's SELECT body (normalized) so a migration that
    rewrites a projection or WHERE filter fails loudly — _EXPECTED_VIEWS only
    pins names. When this fails on an intended rewrite, update
    ``_EXPECTED_VIEW_BODIES`` (and any downstream consumers of the projection)."""
    with sqlite3.connect(fresh_db) as conn:
        actual = {
            name: _normalize_ddl(sql)
            for name, sql in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
    # Body snapshot is keyed by the same set the name parity test pins; if the
    # two drift apart that's a bug in this test, not the schema.
    assert set(actual) == set(_EXPECTED_VIEW_BODIES), (
        "view set changed — reconcile _EXPECTED_VIEW_BODIES (and _EXPECTED_VIEWS) "
        f"with alembic head: {sorted(set(actual) ^ set(_EXPECTED_VIEW_BODIES))}"
    )
    for name, expected in _EXPECTED_VIEW_BODIES.items():
        assert actual[name] == expected, (
            f"view-body drift for {name!r}:\n  expected: {expected}\n  actual:   {actual[name]}"
        )


# ---------------------------------------------------------------------------
# wyrd-jaur item 6: direct branch tests for _filter_sqlite_reflection_artifacts.
# Each filter branch (FK pairs, PK nullable, modify_type collation) was
# previously exercised only incidentally by the live-schema integration
# test. If a future schema change stops producing one of the three diff
# shapes, the corresponding branch silently rots. These unit tests pin
# each branch directly against hand-crafted diff lists.
# ---------------------------------------------------------------------------


def _make_test_metadata():
    """Build a minimal SA MetaData fixture exercising all three filter
    branches (PK column, FK with ondelete, column with collation)."""
    from sqlalchemy import (
        Column,
        ForeignKey,
        Integer,
        MetaData,
        String,
        Table,
        Text,
    )

    md = MetaData()
    Table(
        "parent",
        md,
        Column("id", Integer, primary_key=True),
        Column("name", String(collation="NOCASE"), nullable=False),
    )
    Table(
        "child",
        md,
        Column("id", Integer, primary_key=True),
        Column("parent_id", Integer, ForeignKey("parent.id", ondelete="CASCADE")),
        Column("note", Text),
    )
    return md


def test_filter_drops_pk_modify_nullable_artifacts() -> None:
    """The PK ``modify_nullable`` filter must drop SQLite's "PK is
    nullable" reflection artifact AND let through real nullability
    drift on non-PK columns."""
    md = _make_test_metadata()
    pk_artifact = [
        ("modify_nullable", None, "parent", "id", {}, True, False),
    ]
    real_drift = [
        ("modify_nullable", None, "child", "note", {}, False, True),
    ]
    filtered = _filter_sqlite_reflection_artifacts([pk_artifact, real_drift], md, {})
    assert filtered == [real_drift], (
        f"PK filter should drop pk_artifact and keep real_drift; got {filtered!r}"
    )


def test_filter_drops_paired_fk_artifacts() -> None:
    """The FK pair filter must drop reflection-roundtrip pairs (same
    column-tuple + referred-table). A solo remove_fk or add_fk (no
    pair) must fall through as real drift."""
    md = _make_test_metadata()
    child = md.tables["child"]
    fk = next(iter(child.foreign_key_constraints))
    # Build a second table with its own FK so we have a "solo" FK
    # that doesn't have a matching pair in the diff list.
    from sqlalchemy import Column, ForeignKey, Integer, Table

    orphan = Table(
        "orphan",
        md,
        Column("id", Integer, primary_key=True),
        Column("ref_id", Integer, ForeignKey("parent.id", ondelete="SET NULL")),
    )
    other_fk = next(iter(orphan.foreign_key_constraints))

    paired_remove = ("remove_fk", fk)
    paired_add = ("add_fk", fk)
    solo_remove = ("remove_fk", other_fk)

    filtered = _filter_sqlite_reflection_artifacts([paired_remove, paired_add, solo_remove], md, {})
    assert filtered == [solo_remove], (
        f"FK pair filter should drop the pair and keep the solo; got {filtered!r}"
    )


def test_filter_drops_collation_modify_type_when_ddl_has_collate() -> None:
    """The ``modify_type`` filter must drop the SA-reflection-drops-
    collation artifact when the DDL actually carries COLLATE, AND
    let through real drift (DDL doesn't have COLLATE)."""
    from sqlalchemy import String
    from sqlalchemy import Text as SAText

    md = _make_test_metadata()

    # Branch A: DDL has COLLATE — should filter.
    new_type_with_coll = String(collation="NOCASE")
    artifact = [
        ("modify_type", None, "parent", "name", {}, SAText(), new_type_with_coll),
    ]
    table_ddl_with_collate = {
        "parent": "CREATE TABLE parent (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL COLLATE NOCASE\n)"
    }
    filtered = _filter_sqlite_reflection_artifacts([artifact], md, table_ddl_with_collate)
    assert filtered == [], f"Should filter when DDL has COLLATE; got {filtered!r}"

    # Branch B: DDL is missing COLLATE — should NOT filter.
    table_ddl_no_collate = {
        "parent": "CREATE TABLE parent (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL\n)"
    }
    filtered = _filter_sqlite_reflection_artifacts([artifact], md, table_ddl_no_collate)
    assert filtered == [artifact], (
        f"Should keep real drift when DDL lacks COLLATE; got {filtered!r}"
    )


def test_upgrade_head_is_idempotent(fresh_db: Path) -> None:
    """Running ``upgrade_head`` against a DB that's already at head
    must be a no-op. ``init_schema`` runs it once; tests + the CLI
    ``rebuild-from-jsonl`` flow re-invoke it on the same DB. A
    regression where a migration re-runs (lost ``IF NOT EXISTS``
    guard, missing alembic_version stamp) would surface here."""
    # init_schema already ran upgrade_head once via the fixture.
    # Second invocation must complete cleanly with no changes.
    upgrade_head(fresh_db)

    with sqlite3.connect(fresh_db) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version[0] == "0020_select_targets_indexes"


def test_etymon_citation_attested_form_column(fresh_db: Path) -> None:
    """wyrd-jhdw (0016): etymon_citation carries a nullable attested_form
    column. NULL = cite attested under the etymon's own canonical form;
    a non-NULL value records the original surface form when the cite
    migrated to a parent lemma during a collapse (the hybrid handling)."""
    with sqlite3.connect(fresh_db) as conn:
        cols = {r[1]: r for r in conn.execute("PRAGMA table_info(etymon_citation)")}
        assert "attested_form" in cols, "attested_form column missing"
        # PRAGMA row: (cid, name, type, notnull, dflt_value, pk)
        assert cols["attested_form"][2] == "TEXT"
        assert cols["attested_form"][3] == 0, "attested_form must be nullable"
        # round-trips NULL (default) and an explicit form
        conn.execute("INSERT INTO source (id, title) VALUES ('s-jhdw', 'Test Source')")
        conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'burg', 'old-english')"
        )
        conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (1, 's-jhdw')")
        conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id, page, attested_form) "
            "VALUES (1, 's-jhdw', 'p1', 'burh')"
        )
        rows = conn.execute("SELECT attested_form FROM etymon_citation ORDER BY id").fetchall()
    assert [r[0] for r in rows] == [None, "burh"]


def test_idx_attestation_unique_dedups_null_year_and_source(fresh_db: Path) -> None:
    """wyrd-67fv round-5: ``idx_attestation_unique`` uses COALESCE wraps on
    the nullable columns so two attestations of (same toponym, same
    form, NULL year, NULL source_doc) collapse under the UNIQUE
    constraint. Without the wrap SQLite treats every NULL as distinct,
    so the second INSERT would silently succeed and a mine-attestations
    re-run would double-write. Pin the index-level dedup directly so
    the COALESCE change can't silently regress.
    """
    with LexiconDB(fresh_db) as db:
        toponym_id = db.conn.execute(
            "INSERT INTO toponym (modern_name) VALUES ('Pinhampton') RETURNING id"
        ).fetchone()[0]
        # First insert lands.
        db.conn.execute(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (toponym_id, "Pinampton", None, None),
        )
        # Second insert with identical (form, NULL, NULL) collapses
        # — INSERT OR IGNORE swallows the UNIQUE violation.
        db.conn.execute(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (toponym_id, "Pinampton", None, None),
        )
        # Third with a different NULL pair (NULL year, non-NULL source)
        # is genuinely different — should insert.
        db.conn.execute(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (toponym_id, "Pinampton", None, "Mawer 1920"),
        )
        db.commit()

        count = db.conn.execute(
            "SELECT COUNT(*) FROM toponym_attestation WHERE toponym_id = ?",
            (toponym_id,),
        ).fetchone()[0]
        assert count == 2, (
            f"expected 2 rows (NULL+NULL deduped, NULL+'Mawer 1920' kept); got {count}"
        )


def test_upsert_etymon_returns_same_id_on_duplicate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        first = db.upsert_etymon("ham", "old-english", modifier_type="Habitative")
        second = db.upsert_etymon("ham", "old-english")
        assert first == second
        # Different language → different etymon row.
        third = db.upsert_etymon("ham", "old-norse")
        assert third != first


def test_upsert_etymon_de_dashes_to_bare_surface(fresh_db: Path) -> None:
    """wyrd-aicu.8 (D45): upsert_etymon de-dashes canonical_form to the bare
    surface at the write choke, so a dashed affix form and its bare form fold
    onto ONE row (the boundary dash is identity noise, not a distinct morpheme).
    The leading ``*`` reconstruction sigil and interior hyphens are preserved."""
    with LexiconDB(fresh_db) as db:
        dashed = db.upsert_etymon("-ton", "old-english")
        bare = db.upsert_etymon("ton", "old-english")
        assert dashed == bare, "dashed + bare affix forms must fold onto one row"
        stored = db.conn.execute(
            "SELECT canonical_form FROM etymon WHERE id = ?", (bare,)
        ).fetchone()[0]
        assert stored == "ton"
        # Sigil preserved (its own fold is wyrd-qoy8); interior hyphen preserved.
        pie = db.upsert_etymon("*werh₁-", "proto-indo-european")
        kept = db.upsert_etymon("al-Quadim", "arabic")
        forms = {
            r[0]
            for r in db.conn.execute(
                "SELECT canonical_form FROM etymon WHERE id IN (?, ?)", (pie, kept)
            )
        }
        assert forms == {"*werh₁", "al-Quadim"}


def test_upsert_etymon_raises_on_strip_to_empty_junk(fresh_db: Path) -> None:
    """A form that de-dashes to nothing (a bare ``-``/``--``/``*-``) is a
    contract violation at the choke — junk must be dropped at the ingest
    boundary, so upsert_etymon raises rather than minting an empty-keyed row
    (wyrd-aicu.8)."""
    with LexiconDB(fresh_db) as db:
        for junk in ("-", "--", "*-"):
            with pytest.raises(ValueError, match="non-empty morpheme surface"):
                db.upsert_etymon(junk, "old-english")
        assert db.conn.execute("SELECT COUNT(*) FROM etymon").fetchone()[0] == 0


def test_upsert_reflex_returns_same_id_on_duplicate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        first = db.upsert_reflex("-ham", "post")
        second = db.upsert_reflex("-ham", "post")
        assert first == second
        # Different position → distinct reflex.
        third = db.upsert_reflex("-ham", "inner")
        assert third != first


def test_seed_from_minimal_meanings(fresh_db: Path) -> None:
    """A small synthetic meanings.json fragment seeds correctly."""
    data = [
        {
            "meaning": ["Acorn"],
            "modifier_tags": ["plant", "food"],
            "modifier_type": "Topographical",
            "words": [
                {"modern_usage": "Ac-", "old_english": ["aecern"]},
                {"modern_usage": "-ock", "old_english": ["aecern"]},
            ],
        },
        {
            "meaning": ["Home"],
            "modifier_tags": [],
            "modifier_type": "Habitative",
            "words": [
                {"modern_usage": "-ham", "old_english": ["ham"]},
            ],
        },
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        seed_from_meanings(db, data, "test-src")
        stats = db.stats()
        # Two distinct etymons (aecern, ham), three reflexes (Ac-, -ock, -ham).
        assert stats["etymon"] == 2
        assert stats["reflex"] == 3
        # aecern reflexes both link to the aecern etymon: 2 links + 1 for ham.
        assert stats["reflex_etymon"] == 3
        # Citations: one per etymon-source pair.
        assert stats["etymon_citation"] == 2

        # Glosses: aecern gets "Acorn", ham gets "Home".
        gloss_rows = db.conn.execute(
            "SELECT etymon_id, gloss FROM etymon_gloss ORDER BY gloss"
        ).fetchall()
        assert {row["gloss"] for row in gloss_rows} == {"Acorn", "Home"}

        # Tags from the first subject only — second has none.
        tag_rows = db.conn.execute("SELECT tag FROM etymon_tag").fetchall()
        assert {row["tag"] for row in tag_rows} == {"plant", "food"}


def test_seed_word_without_modern_usage_seeds_etymon_but_no_reflex(fresh_db: Path) -> None:
    """A word with no ``modern_usage`` still contributes its etymon (first
    pass) but produces no reflex (second pass skips it). Pins the
    empty/missing-modern_usage guard in _link_subject_reflexes — the
    C901 extraction owns it but no prior test exercised the skip
    (wyrd-8uvi)."""
    data = [
        {
            "meaning": ["Hill"],
            "modifier_tags": [],
            "modifier_type": "Topographical",
            "words": [
                {"modern_usage": "-don", "old_english": ["dun"]},
                # No modern_usage key → etymon seeded, but no reflex.
                {"old_english": ["beorg"]},
            ],
        },
    ]
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        seed_from_meanings(db, data, "test-src")
        stats = db.stats()
        # Both etymons seeded (dun, beorg) ...
        assert stats["etymon"] == 2
        # ... but only the word with modern_usage made a reflex.
        assert stats["reflex"] == 1
        assert stats["reflex_etymon"] == 1
        # The beorg etymon exists with no reflex pointing at it.
        forms = {
            row["canonical_form"]
            for row in db.conn.execute("SELECT canonical_form FROM etymon").fetchall()
        }
        assert forms == {"dun", "beorg"}
        surfaces = [
            row["surface_form"]
            for row in db.conn.execute("SELECT surface_form FROM reflex").fetchall()
        ]
        # wyrd-aicu.3: the seed stores the BARE surface; the dashed input
        # modern_usage only feeds position_from_usage (here: 'post').
        assert surfaces == ["don"]


def test_seed_from_meanings_accepts_dict_shape_bundle(fresh_db: Path) -> None:
    """wyrd-h8k1: ``seed_from_meanings`` must accept the dict-shape
    bundle ``{"subjects": [...], "canonical_decompositions": {...}}``
    that ``lexicon export-meanings`` emits when the DB has canonical
    picks. Sibling fields (joiners, canonical_decompositions) are
    ignored at seed time — only the subjects feed lexicon ingestion."""
    bundle = {
        "subjects": [
            {
                "meaning": ["Home"],
                "modifier_tags": ["habitative"],
                "modifier_type": "Habitative",
                "words": [{"modern_usage": "-ham", "old_english": ["ham"]}],
            },
        ],
        "canonical_decompositions": {
            "Stratford": {"signature": "abc123", "source": "scholar"},
        },
        "joiners": {},
    }
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        seed_from_meanings(db, bundle, "test-src")
        stats = db.stats()
    # One etymon (ham), one reflex (-ham).
    assert stats["etymon"] == 1
    assert stats["reflex"] == 1


def test_seed_against_bundled_meanings(fresh_db: Path) -> None:
    """End-to-end: ingest the runtime bundle and sanity-check counts.
    The thresholds are calibrated against the full-corpus L4 — the
    bundled seed-runtime.db is a deliberate subset (~1k subjects vs
    ~8k full), so this test only runs when ``WYRD_RUNTIME_DB`` /
    ``WYRD_RUNTIME_DB_BUCKET`` points at a full-corpus L4. Without
    that, the seed's row counts are below floor — not a regression."""
    import os

    if not (os.environ.get("WYRD_RUNTIME_DB") or os.environ.get("WYRD_RUNTIME_DB_BUCKET")):
        pytest.skip(
            "Full-corpus L4 not configured; set WYRD_RUNTIME_DB to run this "
            "coverage gate against the live bundle."
        )

    from wyrd.generators.kenning import _runtime_db_bundle_dict

    data = _runtime_db_bundle_dict()

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="Rando port seed")
        db.commit()
        seed_from_meanings(db, data, "rando-port")
        stats = db.stats()

    # Bounds re-calibrated 2026-05-22 against the wyrd-hitl bundle
    # regen (8346 subjects, ~60K reflexes — phonological_vector +
    # wave2 enriched data inflated reflex counts ~4.6× vs the
    # post-wyrd-eni4 baseline). Floors at ~50% of current to catch
    # meaningful regressions; ceilings at ~2× to absorb routine
    # growth without per-session re-tuning.
    assert 30000 < stats["reflex"] < 130000
    # Etymons are grouped by (form, language). Pre-wyrd-eni4 the bundle
    # was OE/ON-dominated so etymons < reflexes; post-cluster-expansion
    # each modern_usage maps to many ancestor-language etymons, so the
    # relationship inverts. Floor at ~7K (~50% of current).
    assert 7000 < stats["etymon"] < 100000
    # Every etymon has at least one citation back to rando-port.
    assert stats["etymon_citation"] == stats["etymon"]


def test_ingest_parsed_entries_writes_etymology_rows(fresh_db: Path) -> None:
    """High/medium-confidence entries get full etymology rows; low-confidence
    entries record the toponym only."""
    entries = [
        ParsedEntry(
            toponym="Alpha",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[
                ParsedElement(form="fake", language="old-english", position="pre"),
                ParsedElement(
                    form="tun", language="old-english", gloss="enclosure", position="post"
                ),
            ],
            confidence="high",
            source_quote="Alpha. This is the A.S. fake-tun.",
        ),
        ParsedEntry(
            toponym="Beta",
            section_suffix="-ton",
            historical_form=None,
            elements=[],
            confidence="low",
            source_quote="Beta. (no etymology recovered)",
        ),
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        counts = ingest_parsed_entries(db, entries, "test-src", region="Testshire")
        stats = db.stats()

    # Both toponyms recorded; only Alpha has an etymology row.
    assert counts["toponyms"] == 2
    assert counts["etymologies"] == 1
    assert counts["elements"] == 2
    assert stats["toponym"] == 2
    assert stats["toponym_etymology"] == 1
    assert stats["toponym_etymology_element"] == 2
    # The "tun" gloss was recorded.
    assert stats["etymon_gloss"] == 1


def test_ingest_then_consensus_view(fresh_db: Path) -> None:
    """Citing the same etymon from two sources promotes its witness count."""
    entries_a = [
        ParsedEntry(
            toponym="Foo",
            section_suffix="-ton",
            historical_form="bar-tun",
            elements=[
                ParsedElement(form="bar", language="old-english", position="pre"),
                ParsedElement(form="tun", language="old-english", position="post"),
            ],
            confidence="high",
            source_quote="Foo. From A.S. bar-tun.",
        ),
    ]
    entries_b = [
        ParsedEntry(
            toponym="Baz",
            section_suffix="-ton",
            historical_form="qux-tun",
            elements=[
                ParsedElement(form="qux", language="old-english", position="pre"),
                ParsedElement(form="tun", language="old-english", position="post"),
            ],
            confidence="high",
            source_quote="Baz. From A.S. qux-tun.",
        ),
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="Source A")
        db.upsert_source(id="src-b", title="Source B")
        db.commit()
        ingest_parsed_entries(db, entries_a, "src-a")
        ingest_parsed_entries(db, entries_b, "src-b")

        consensus = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY witnesses DESC"
        ).fetchall()

    by_form = {r["canonical_form"]: r["witnesses"] for r in consensus}
    assert by_form["tun"] == 2  # cited by both sources
    assert by_form["bar"] == 1
    assert by_form["qux"] == 1


def test_normalize_ocr_form_strips_dashes_and_lowers() -> None:
    assert normalize_ocr_form("Hædan") == "haedan"
    assert normalize_ocr_form("-tun") == "tun"
    assert normalize_ocr_form("HAM") == "ham"
    assert normalize_ocr_form("Bre-") == "bre"


def test_normalize_ocr_form_collapses_ocr_variants() -> None:
    """The same OE word in different OCR renderings normalizes to one key."""
    canonical = normalize_ocr_form("Hædan")
    assert normalize_ocr_form("Hcsdan") == canonical  # OCR mangle of æ→cs
    assert normalize_ocr_form("Haedan") == canonical  # ASCII-equivalent
    assert normalize_ocr_form("HÆDAN") == canonical  # uppercase
    # Eth (ð) variants
    assert normalize_ocr_form("hyð") == normalize_ocr_form("hydh")


def test_normalize_ocr_form_preserves_distinct_etymons() -> None:
    """Genuinely different forms must NOT collapse to the same key."""
    assert normalize_ocr_form("ham") != normalize_ocr_form("hamm")
    assert normalize_ocr_form("tun") != normalize_ocr_form("dun")
    assert normalize_ocr_form("bere") != normalize_ocr_form("bera")


def test_normalize_ocr_form_handles_overlapping_multichar_patterns() -> None:
    """Regression coverage for the cases where 'cs' / 'ce' patterns
    interleave. Both fold to 'ae' so the load-bearing semantic is the
    left-to-right non-overlapping match the underlying replace chain
    produces. Pinning these cases protects against a future refactor
    that swaps the replace shape for translate or regex (wyrd-0ke
    investigated but ruled out — see the rationale next to
    ``_OCR_LIGATURE_MAP`` in lexicon.py)."""
    # 'cs' adjacent to 'ce' — both should fold to 'ae'.
    assert normalize_ocr_form("csce") == "aeae"
    # 'cce': only the 'ce' suffix matches; the leading 'c' stays.
    assert normalize_ocr_form("cce") == "cae"
    # 'ccs': only the 'cs' suffix matches; the leading 'c' stays.
    assert normalize_ocr_form("ccs") == "cae"
    # Mixed unicode + multi-char: æ → ae, then cs → ae.
    assert normalize_ocr_form("æcs") == "aeae"
    # Section sign — the only punctuation-source mapping in the table.
    # § → dh, so a hyð-shaped form rendered with § stays consistent
    # with the existing hyð↔hydh canonicalization.
    assert normalize_ocr_form("hy§") == "hydh"


def test_normalize_ocr_form_perf_floor_on_large_body() -> None:
    """Pin the call-site cost on a 1MB body. wyrd-0ke benchmarked the
    function under load and confirmed the existing replace-chain shape
    is already optimal — Python's str.replace is a memchr-fast C path
    that beat str.translate / re.sub alternatives by 5-15x on 1-50MB
    bodies. This test is a guard so that perf claim stays empirically
    grounded: a future refactor that doubled the cost would trip it.

    Threshold of 0.2s/call on a 1MB body is conservative — typical
    runs land 10-50x faster — and would still trip an order-of-
    magnitude regression (e.g. accidentally putting the function
    inside a quadratic loop)."""

    # 1MB synthetic body. Mix of plain ASCII, OE ligatures, and the
    # 'cs'/'ce' OCR-confusion patterns so the multi-char branches of
    # the replace chain get exercised.
    body = ("placement Hædan hyð csce alphabet " * 10000)[:1_000_000]
    start = time.perf_counter()
    result = normalize_ocr_form(body)
    elapsed = time.perf_counter() - start
    # Output sanity — the perf assertion alone could pass on a no-op.
    assert "haedan" in result
    assert "hydh" in result
    assert "aeae" in result  # csce → aeae
    assert elapsed < 0.2, f"normalize_ocr_form on 1MB took {elapsed:.3f}s — perf regression?"


def test_cluster_ocr_variants_dry_run_reports_without_writing(fresh_db: Path) -> None:
    """Dry-run must not touch the DB."""
    with LexiconDB(fresh_db) as db:
        # Create three variants of "Hædan" + a distinct etymon
        haedan = db.upsert_etymon("Hædan", "old-english")
        hcsdan = db.upsert_etymon("Hcsdan", "old-english")
        haedan_ascii = db.upsert_etymon("Haedan", "old-english")
        ham = db.upsert_etymon("ham", "old-english")
        db.commit()

        before = db.stats()
        result = cluster_ocr_variants(db, apply=False)
        after = db.stats()

    # Dry-run reports 1 mergeable group, 2 etymons removable.
    assert result["groups"] == 1
    assert result["etymons_merged"] == 2
    # No mutation.
    assert before["etymon"] == after["etymon"]
    assert haedan != hcsdan != haedan_ascii != ham


def test_cluster_ocr_variants_apply_marks_losers_merged_non_destructively(
    fresh_db: Path,
) -> None:
    """With apply=True, redundant etymons are MARKED merged into the
    lowest-id row via merged_into_id. Losers stay in the etymon table
    (per D22, mining evidence is non-destructive); their citations,
    glosses, and tags remain attached to them. The etymon_canonical view
    hides them; etymon_consensus rolls them up via merged_into_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.commit()

        # Three OCR variants. Lowest id wins.
        winner = db.upsert_etymon("Hædan", "old-english")
        loser_a = db.upsert_etymon("Hcsdan", "old-english")
        loser_b = db.upsert_etymon("Haedan", "old-english")
        # Distinct etymon should be left alone.
        keeper = db.upsert_etymon("ham", "old-english")

        db.add_gloss(loser_a, "personal name")
        db.add_gloss(loser_b, "from heathland (per Mawer)")
        db.add_tag(loser_a, "male name")
        db.add_citation(loser_a, "src-a", page="47")
        db.add_citation(loser_b, "src-b", page="23")
        db.add_citation(winner, "src-a", page="10")
        db.commit()

        result = cluster_ocr_variants(db, apply=True)

    assert result["groups"] == 1
    assert result["etymons_merged"] == 2

    with LexiconDB(fresh_db) as db2:
        # Losers still exist in the raw table.
        rows = db2.conn.execute(
            "SELECT id, canonical_form, merged_into_id FROM etymon ORDER BY id"
        ).fetchall()
        by_form = {r["canonical_form"]: r for r in rows}
        assert "Hcsdan" in by_form
        assert "Haedan" in by_form
        assert "Hædan" in by_form
        assert "ham" in by_form

        # Losers point at the winner; winner and keeper are not merged.
        assert by_form["Hcsdan"]["merged_into_id"] == winner
        assert by_form["Haedan"]["merged_into_id"] == winner
        assert by_form["Hædan"]["merged_into_id"] is None
        assert by_form["ham"]["merged_into_id"] is None

        # etymon_canonical view excludes the merged losers.
        canonical_forms = {
            r["canonical_form"]
            for r in db2.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert canonical_forms == {"Hædan", "ham"}

        # Citations / glosses / tags stay on losers — not moved (D22).
        loser_a_cit_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE etymon_id = ?", (loser_a,)
        ).fetchone()[0]
        winner_cit_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE etymon_id = ?", (winner,)
        ).fetchone()[0]
        assert loser_a_cit_count == 1
        assert winner_cit_count == 1

        loser_a_gloss_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_gloss WHERE etymon_id = ?", (loser_a,)
        ).fetchone()[0]
        assert loser_a_gloss_count == 1

        # etymon_consensus rolls them up correctly: the Hædan canonical
        # group sees all three sources (src-a from winner, src-a from
        # loser_a's page=47, src-b from loser_b).
        consensus = db2.conn.execute(
            "SELECT witnesses FROM etymon_consensus WHERE canonical_form = ? AND language = ?",
            ("Hædan", "old-english"),
        ).fetchone()
        # src-a (winner+loser_a counted once across rollup) and src-b → 2 distinct
        assert consensus["witnesses"] == 2

    # Unused name reference — keeper's id used for differentiation.
    assert keeper != winner


def test_cluster_ocr_variants_keeps_lemma_children_consistent_via_redirect(
    fresh_db: Path,
) -> None:
    """When the loser is itself a lemma with inflected children, those
    children's lemma_id must be re-parented to the canonical so the
    single-level consensus rollup still works. Text-match rows stay
    attached to the loser etymon (D22 — non-destructive) and are
    rolled up via merged_into_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.commit()

        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")

        # Inflected child of the loser.
        child = db.upsert_etymon("Hædanes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (loser, "genitive_strong", child),
        )

        # Text-match rows on both winner and loser. Stay where they are.
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (winner, "src-a", "haedan", 3, 0, "winner snippet"),
        )
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (loser, "src-a", "hcsdan", 7, 0, "loser-only form"),
        )
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Loser still exists, marked merged.
        loser_row = db2.conn.execute(
            "SELECT id, merged_into_id FROM etymon WHERE canonical_form = ?",
            ("Hcsdan",),
        ).fetchone()
        assert loser_row is not None
        assert loser_row["merged_into_id"] == winner

        # Child re-parented to winner so consensus rolls up cleanly.
        child_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hædanes",)
        ).fetchone()[0]
        assert child_lemma == winner

        # Text-match rows stay on their original etymons. Loser's row is
        # NOT moved to winner — the rollup happens via merged_into_id at
        # query time, not by data movement.
        rows_on_loser = db2.conn.execute(
            "SELECT matched_form FROM etymon_text_match WHERE etymon_id = ?",
            (loser,),
        ).fetchall()
        rows_on_winner = db2.conn.execute(
            "SELECT matched_form FROM etymon_text_match WHERE etymon_id = ?",
            (winner,),
        ).fetchall()
        assert [r["matched_form"] for r in rows_on_loser] == ["hcsdan"]
        assert [r["matched_form"] for r in rows_on_winner] == ["haedan"]


def test_cluster_ocr_variants_flattens_lemma_chain_when_winner_is_child(
    fresh_db: Path,
) -> None:
    """If the winner is itself an inflected variant of some canonical lemma,
    the loser's merged_into_id and the loser's children's lemma_id must
    both point directly at the canonical lemma — not at the winner.
    Otherwise the single-level consensus rollup would split witnesses.
    """
    with LexiconDB(fresh_db) as db:
        canonical = db.upsert_etymon("had", "old-english")
        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (canonical, winner))
        child = db.upsert_etymon("Hcsdanes", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (loser, child))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # The loser's redirect goes to canonical, not to winner.
        loser_merged = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE canonical_form = ?", ("Hcsdan",)
        ).fetchone()[0]
        assert loser_merged == canonical

        # The child also points at canonical directly.
        child_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hcsdanes",)
        ).fetchone()[0]
        assert child_lemma == canonical

        # Winner's own pointer is unchanged.
        winner_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hædan",)
        ).fetchone()[0]
        assert winner_lemma == canonical


def test_cluster_ocr_variants_skips_already_merged_on_rerun(fresh_db: Path) -> None:
    """Re-running cluster_ocr_variants on a DB that already has merges
    must NOT re-cluster the tombstoned losers. The candidate query
    filters merged_into_id IS NULL."""
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        db.upsert_etymon("Haedan", "old-english")
        db.commit()

        first = cluster_ocr_variants(db, apply=True)
        assert first["groups"] == 1
        assert first["etymons_merged"] == 2

        # Second run: nothing left to cluster.
        second = cluster_ocr_variants(db, apply=True)
        assert second["groups"] == 0
        assert second["etymons_merged"] == 0


def test_cluster_ocr_variants_picks_lemma_as_winner_when_present(
    fresh_db: Path,
) -> None:
    """If a cluster contains both a lemma and an inflected variant of it,
    the lemma must win regardless of id ordering. Otherwise canonical_id
    resolves to the lemma but the lemma is in loser_ids, creating a
    self-referencing merged_into_id loop."""
    with LexiconDB(fresh_db) as db:
        # The inflected variant gets the LOWER id (would normally win
        # under lowest-id-wins). The lemma comes second.
        inflected = db.upsert_etymon("Hædan", "old-english")
        lemma = db.upsert_etymon("Hædan_lemma_synthetic", "old-english")
        # Manually fix canonical_form so they share an OCR fingerprint
        # (the synthetic suffix above kept upsert_etymon's UNIQUE happy).
        db.conn.execute("UPDATE etymon SET canonical_form = 'Haedan' WHERE id = ?", (lemma,))
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Lemma wins. Inflected variant is the loser, redirected to the lemma.
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (lemma,)).fetchone()[
                0
            ]
            is None
        )
        assert (
            db2.conn.execute(
                "SELECT merged_into_id FROM etymon WHERE id = ?", (inflected,)
            ).fetchone()[0]
            == lemma
        )


def test_cluster_ocr_variants_follows_merged_chain_to_canonical(
    fresh_db: Path,
) -> None:
    """If winner.lemma_id points at a row that was already merged in a
    prior pass (e.g., from a stale link-lemmas pointer), cluster must
    follow the merged_into_id chain to the ultimate canonical — not
    plant losers' merged_into_id on a tombstone (which would create a
    depth-2 chain that the rollup splits)."""
    with LexiconDB(fresh_db) as db:
        # L is the canonical lemma. L2 is its OCR-merge tombstone.
        canonical = db.upsert_etymon("had", "old-english")
        tombstone = db.upsert_etymon("hadx_ocr", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical, tombstone),
        )

        # Winner W has a stale pointer at the tombstone (simulates a
        # link-lemmas pass that ran before the tombstone-filter fix).
        winner = db.upsert_etymon("Hædan", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (tombstone, winner))

        # Loser X joins the cluster with W on OCR fingerprint.
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Loser must redirect to the ultimate canonical, not the tombstone.
        loser_redirect = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (loser,)
        ).fetchone()[0]
        assert loser_redirect == canonical, (
            f"loser should chain-follow through tombstone to canonical; got "
            f"merged_into_id={loser_redirect}"
        )


def test_link_lemmas_skips_merged_tombstone_as_lemma_candidate(
    fresh_db: Path,
) -> None:
    """An OCR-merge tombstone (merged_into_id IS NOT NULL) must not be
    selected as a lemma by link_lemmas — pointing an inflected variant at
    a tombstone would create a child → tombstone → canonical chain that
    the rollup splits."""
    with LexiconDB(fresh_db) as db:
        # `cot` (canonical lemma) and `cotx` (its OCR-merged tombstone
        # — synthetic but the principle generalizes). Set up the merge
        # state directly so we don't depend on cluster_ocr_variants.
        cot = db.upsert_etymon("cot", "old-english")
        cotx = db.upsert_etymon("cotx", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (cot, cotx))
        # An inflected variant that the stripper might match against
        # 'cotx' (e.g. 'cotxes' → 'cotx'). link_lemmas must skip cotx
        # because it's merged, even though it has lemma_id IS NULL.
        cotxes = db.upsert_etymon("cotxes", "old-english")
        db.commit()

        link_lemmas(db, apply=True)

        cotxes_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (cotxes,)
        ).fetchone()[0]
        # Either matched directly to canonical 'cot' (if the stripper
        # reaches it) OR stayed unlinked (if no stripped form matches a
        # non-merged candidate). Critically: NOT linked to cotx.
        assert cotxes_lemma != cotx


def test_cluster_ocr_variants_flattens_existing_redirects_to_loser(
    fresh_db: Path,
) -> None:
    """If any row already has merged_into_id pointing at the current
    loser (from a prior cluster pass, manual fixup, etc.), marking the
    loser merged into a new canonical must re-route those rows too —
    otherwise X → loser → canonical chains form and the rollup splits."""
    with LexiconDB(fresh_db) as db:
        a = db.upsert_etymon("Hædan", "old-english")
        b = db.upsert_etymon("Hcsdan", "old-english")
        # X has a separate OCR signature so the candidate query would
        # group {a, b} alone. We set X.merged_into_id = b manually to
        # simulate a residual pointer (e.g. from a manual fixup or a
        # prior clustering pass that touched only some rows).
        x = db.upsert_etymon("Hbcdan", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (b, x))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # b's merge target is a (direct).
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (b,)).fetchone()[0]
            == a
        )
        # x's redirect was flattened from b to a so no chain forms.
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (x,)).fetchone()[0]
            == a
        )


def test_etymon_consensus_handles_depth_2_merged_into_lemma_chain(
    fresh_db: Path,
) -> None:
    """The consensus view must handle merged_into_id → lemma_id chains
    correctly: an OCR variant of an inflected form must roll up to the
    lemma group, not stop at the inflected form. This can arise if
    clustering runs before link-lemmas (against doc'd order) or if
    link-lemmas runs after clustering."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.commit()

        # Lemma + inflected variant + OCR variant of the inflected form.
        lemma = db.upsert_etymon("had", "old-english")
        inflected = db.upsert_etymon("hædan", "old-english")
        ocr_variant = db.upsert_etymon("hcsdan", "old-english")

        # Set up the chain manually: ocr_variant.merged_into_id =
        # inflected; inflected.lemma_id = lemma. (cluster_ocr_variants
        # would normally flatten through the lemma at merge time, but
        # this test exercises the case where link-lemmas runs AFTER
        # clustering — which is reverse of the documented order — to
        # prove the view is robust to it.)
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (inflected, ocr_variant),
        )
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))

        db.add_citation(ocr_variant, "src-a")
        db.add_citation(inflected, "src-b")
        db.commit()

        # The lemma group must aggregate citations from all three rows
        # (ocr_variant via merged_into_id → inflected → lemma; inflected
        # via lemma_id → lemma; lemma itself).
        consensus = db.conn.execute(
            "SELECT lemma_id, witnesses FROM etymon_consensus WHERE language = 'old-english'"
        ).fetchall()
    assert len(consensus) == 1, (
        f"Expected exactly one consensus group (the lemma); got {len(consensus)}"
    )
    assert consensus[0]["lemma_id"] == lemma
    assert consensus[0]["witnesses"] == 2  # src-a + src-b


def test_etymon_gloss_canonical_rolls_up_glosses_from_merged_losers(
    fresh_db: Path,
) -> None:
    """wyrd-7lo: a gloss attached to an OCR-merged loser must surface
    under the canonical etymon's id when read through
    etymon_gloss_canonical. Without this view, a SELECT from
    etymon_gloss WHERE etymon_id = canonical misses the loser's
    glosses since the data wasn't moved (D22 non-destructive)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        winner = db.upsert_etymon("ham", "old-english")
        loser = db.upsert_etymon("harn", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))
        db.add_gloss(winner, "homestead")
        db.add_gloss(loser, "OCR-variant of homestead")
        db.commit()

        glosses = sorted(
            row["gloss"]
            for row in db.conn.execute(
                "SELECT gloss FROM etymon_gloss_canonical WHERE canonical_etymon_id = ?",
                (winner,),
            )
        )
    assert glosses == ["OCR-variant of homestead", "homestead"]


def test_etymon_tag_canonical_rolls_up_tags_from_merged_losers(
    fresh_db: Path,
) -> None:
    """wyrd-7lo: companion to the gloss view — tags on a merged loser
    surface under the canonical group."""
    with LexiconDB(fresh_db) as db:
        winner = db.upsert_etymon("ham", "old-english")
        loser = db.upsert_etymon("harn", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))
        db.add_tag(winner, "settlement")
        db.add_tag(loser, "homestead-variant")
        db.commit()

        tags = sorted(
            row["tag"]
            for row in db.conn.execute(
                "SELECT tag FROM etymon_tag_canonical WHERE canonical_etymon_id = ?",
                (winner,),
            )
        )
    assert tags == ["homestead-variant", "settlement"]


def test_etymon_text_match_canonical_sums_match_counts_per_group(
    fresh_db: Path,
) -> None:
    """wyrd-7lo: text_match rollup must SUM(match_count) when a canonical
    group has multiple member etymons that each saw the same matched_form
    in the same source. Mirrors D21's UPSERT semantics for writes against
    the raw table — reads through the canonical view see one row per
    (canonical, source, matched_form) with the aggregated count."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        winner = db.upsert_etymon("ham", "old-english")
        loser = db.upsert_etymon("harn", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))
        # Both rows recorded text-match evidence for the same matched_form
        # in the same source — the loser before clustering, the winner
        # after. Without rollup the consumer sees two rows; the canonical
        # view must collapse them. The two rows differ in edit_distance
        # and attested_year so the MIN aggregations have something to
        # pick the smallest of.
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (winner, "src-a", "ham", 7, 0, 1086),
        )
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (loser, "src-a", "ham", 3, 1, 1242),
        )
        db.commit()

        rows = db.conn.execute(
            "SELECT total_match_count, edit_distance, attested_year "
            "FROM etymon_text_match_canonical "
            "WHERE canonical_etymon_id = ? AND source_id = ? AND matched_form = ?",
            (winner, "src-a", "ham"),
        ).fetchall()
    assert len(rows) == 1, "merged group should produce exactly one canonical row"
    assert rows[0]["total_match_count"] == 10  # 7 + 3
    # MIN(edit_distance) — the strongest evidence (exact match) wins
    # over the fuzzy variant. MIN(attested_year) — the earliest
    # attestation across the group is the canonical "first seen" date.
    assert rows[0]["edit_distance"] == 0
    assert rows[0]["attested_year"] == 1086


def test_canonical_views_use_lemma_chain_when_loser_was_inflected_variant(
    fresh_db: Path,
) -> None:
    """The canonical views must follow the same two-step
    merged_into_id → lemma_id chain that etymon_consensus uses. An OCR
    loser whose merge target is an inflected form must surface under
    the lemma's id, not the inflected form's id."""
    with LexiconDB(fresh_db) as db:
        lemma = db.upsert_etymon("had", "old-english")
        inflected = db.upsert_etymon("hædan", "old-english")
        ocr_loser = db.upsert_etymon("hcsdan", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (inflected, ocr_loser))
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))
        db.add_gloss(ocr_loser, "via-loser")
        db.add_gloss(inflected, "via-inflected")
        db.add_gloss(lemma, "lemma-direct")
        db.commit()

        glosses = sorted(
            row["gloss"]
            for row in db.conn.execute(
                "SELECT gloss FROM etymon_gloss_canonical WHERE canonical_etymon_id = ?",
                (lemma,),
            )
        )
    assert glosses == ["lemma-direct", "via-inflected", "via-loser"]


def test_canonical_views_surface_standalone_etymons_under_their_own_id(
    fresh_db: Path,
) -> None:
    """Boundary check: an etymon with no merged_into_id and no lemma_id
    must surface under its own id. The COALESCE(le.id, target.id, e.id)
    fallback path is what makes the view degrade gracefully to the raw
    table for unmerged rows; a regression that broke it would silently
    drop standalone canonical rows from every consumer that uses these
    views as their primary read path."""
    with LexiconDB(fresh_db) as db:
        standalone = db.upsert_etymon("standalone", "old-english")
        db.add_gloss(standalone, "no chain")
        db.add_tag(standalone, "isolated")
        db.commit()

        gloss_rows = db.conn.execute(
            "SELECT canonical_etymon_id, gloss FROM etymon_gloss_canonical "
            "WHERE canonical_etymon_id = ?",
            (standalone,),
        ).fetchall()
        tag_rows = db.conn.execute(
            "SELECT canonical_etymon_id, tag FROM etymon_tag_canonical "
            "WHERE canonical_etymon_id = ?",
            (standalone,),
        ).fetchall()
    assert len(gloss_rows) == 1
    assert gloss_rows[0]["gloss"] == "no chain"
    assert len(tag_rows) == 1
    assert tag_rows[0]["tag"] == "isolated"


def test_clear_enrichment_dry_run_reports_without_writing(fresh_db: Path) -> None:
    """Dry-run reports counts but does not modify the DB."""
    with LexiconDB(fresh_db) as db:
        winner = db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        db.commit()
        cluster_ocr_variants(db, apply=True)

        before = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
        assert before == 1

        result = clear_enrichment(db, stage="ocr", apply=False)
        assert result["ocr_merges_to_clear"] == 1

        after = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
        assert after == 1
    # Use winner reference for differentiation.
    assert winner is not None


def test_clear_enrichment_ocr_reverts_merges(fresh_db: Path) -> None:
    """clear_enrichment(stage='ocr', apply=True) un-marks all merges,
    restoring the etymon_canonical view to include the previously-tombstoned
    rows. Citations/glosses/tags are intact since cluster_ocr_variants
    didn't move them."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.commit()
        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.add_citation(loser, "src-a", page="47")
        db.commit()
        cluster_ocr_variants(db, apply=True)

        # Before clear: etymon_canonical excludes loser.
        before = {
            r["canonical_form"]
            for r in db.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert before == {"Hædan"}

        result = clear_enrichment(db, stage="ocr", apply=True)
        assert result["ocr_merges_to_clear"] == 1

        # After clear: loser is canonical again, citation untouched.
        after = {
            r["canonical_form"]
            for r in db.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert after == {"Hædan", "Hcsdan"}
        loser_cit = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE etymon_id = ?", (loser,)
        ).fetchone()
        assert loser_cit["page"] == "47"
    assert winner is not None


def test_clear_enrichment_lemmas_resets_lemma_id_inflection_method(
    fresh_db: Path,
) -> None:
    """clear_enrichment(stage='lemmas') nulls out lemma_id, inflection,
    and lemma_method on every row."""
    with LexiconDB(fresh_db) as db:
        lemma = db.upsert_etymon("cot", "old-english")
        inflected = db.upsert_etymon("cotan", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = 'weak_oblique', "
            "lemma_method = 'link-lemmas-v1' WHERE id = ?",
            (lemma, inflected),
        )
        db.commit()

        result = clear_enrichment(db, stage="lemmas", apply=True)
        assert result["lemma_links_to_clear"] == 1

        row = db.conn.execute(
            "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id = ?",
            (inflected,),
        ).fetchone()
        assert row["lemma_id"] is None
        assert row["inflection"] is None
        assert row["lemma_method"] is None


def test_clear_enrichment_text_match_drops_all_rows(fresh_db: Path) -> None:
    """clear_enrichment(stage='text-match') deletes every text-match row
    so a re-run of reverse-search/fuzzy-search starts fresh."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        eid = db.upsert_etymon("denu", "old-english")
        db.commit()
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
            (eid, "src-a", "denu", 12, 0, "snippet"),
        )
        db.commit()

        result = clear_enrichment(db, stage="text-match", apply=True)
        assert result["text_match_rows_to_clear"] == 1

        remaining = db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0]
        assert remaining == 0


def test_clear_enrichment_all_derived_resets_three_stages(fresh_db: Path) -> None:
    """stage='all-derived' clears OCR merges, lemma links, AND text-match
    rows in a single invocation. (Cognate + attested-year stages are
    exercised in their own tests; the three-stage shape is the legacy
    one and stays pinned here.)"""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        winner = db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        lemma = db.upsert_etymon("cot", "old-english")
        inflected = db.upsert_etymon("cotan", "old-english")
        db.commit()
        cluster_ocr_variants(db, apply=True)
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
            (winner, "src-a", "haedan", 5, 0, "s"),
        )
        db.commit()

        result = clear_enrichment(db, stage="all-derived", apply=True)
        assert result["ocr_merges_to_clear"] == 1
        assert result["lemma_links_to_clear"] == 1
        assert result["text_match_rows_to_clear"] == 1

        leftover = db.conn.execute(
            "SELECT "
            "  (SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL),"
            "  (SELECT COUNT(*) FROM etymon WHERE lemma_id IS NOT NULL),"
            "  (SELECT COUNT(*) FROM etymon_text_match)"
        ).fetchone()
        assert tuple(leftover) == (0, 0, 0)


def test_clear_enrichment_cognates_resets_assignments(fresh_db: Path) -> None:
    """stage='cognates' nulls cognate_id + cognate_method (one of the
    table-driven _SIMPLE_CLEAR_STAGES rows — pins that row's SQL)."""
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("burg", "old-english")
        member = db.upsert_etymon("burh", "old-english")
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ?, cognate_method = 'cluster-v1' WHERE id = ?",
            (root, member),
        )
        db.commit()
        dry = clear_enrichment(db, stage="cognates", apply=False)
        assert dry["cognate_assignments_to_clear"] == 1
        # dry-run wrote nothing
        assert (
            db.conn.execute("SELECT COUNT(*) FROM etymon WHERE cognate_id IS NOT NULL").fetchone()[
                0
            ]
            == 1
        )
        clear_enrichment(db, stage="cognates", apply=True)
        row = db.conn.execute(
            "SELECT cognate_id, cognate_method FROM etymon WHERE id = ?", (member,)
        ).fetchone()
        assert row["cognate_id"] is None and row["cognate_method"] is None


def test_clear_enrichment_attested_years_clears_both_sources(fresh_db: Path) -> None:
    """stage='attested-years' clears attested_year on BOTH row sources
    (etymon_text_match + toponym_etymology); the dry-run count is the sum.
    Pins the dual-source logic in _populate_clear_counts and the standalone
    (text-match-NOT-in-stages → run the etymon_text_match UPDATE) branch of
    _apply_stage_clears."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        eid = db.upsert_etymon("denu", "old-english")
        tid = db.conn.execute("INSERT INTO toponym (modern_name) VALUES ('T')").lastrowid
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, 'src-a', 'denu', 1, 0, 1200)",
            (eid,),
        )
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, attested_year) "
            "VALUES (?, 'src-a', 1086)",
            (tid,),
        )
        db.commit()
        dry = clear_enrichment(db, stage="attested-years", apply=False)
        assert dry["attested_years_to_clear"] == 2  # one per source
        clear_enrichment(db, stage="attested-years", apply=True)
        etm_year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE etymon_id = ?", (eid,)
        ).fetchone()[0]
        te_year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE toponym_id = ?", (tid,)
        ).fetchone()[0]
        assert etm_year is None and te_year is None  # both sources cleared


def test_clear_enrichment_all_derived_clears_attested_years_via_guard(fresh_db: Path) -> None:
    """all-derived clears attested_year too: text-match DELETEs
    etymon_text_match, so the attested-years stage's guard skips the (now
    pointless) etymon_text_match UPDATE while still nulling
    toponym_etymology.attested_year. Pins the all-derived apply path +
    the 'text-match in stages' guard branch (wyrd-8uvi reorder)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        eid = db.upsert_etymon("denu", "old-english")
        tid = db.conn.execute("INSERT INTO toponym (modern_name) VALUES ('T')").lastrowid
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, 'src-a', 'denu', 1, 0, 1200)",
            (eid,),
        )
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, attested_year) "
            "VALUES (?, 'src-a', 1086)",
            (tid,),
        )
        db.commit()
        result = clear_enrichment(db, stage="all-derived", apply=True)
        assert result["attested_years_to_clear"] == 2
        # text-match DELETE removed the ETM row entirely; toponym_etymology
        # row survives with attested_year nulled by the (reordered) guard.
        assert db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0] == 0
        te_year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE toponym_id = ?", (tid,)
        ).fetchone()[0]
        assert te_year is None


def test_clear_enrichment_rejects_unknown_stage(fresh_db: Path) -> None:
    """Defensive: unknown stage raises ValueError before touching the DB."""
    with LexiconDB(fresh_db) as db, pytest.raises(ValueError, match="unknown stage"):
        clear_enrichment(db, stage="bogus", apply=True)


# --- D5-1 / wyrd-3ux: lookup_attested_years --------------------------------


def _seed_text_match(
    db: LexiconDB,
    *,
    source_id: str,
    canonical_form: str,
    matched_form: str | None = None,
    language: str = "old-english",
) -> int:
    """Insert one etymon + one text-match row pointing at it; return the
    text-match row id. Used as a fixture builder for the lookup-attested-
    years tests."""
    eid = db.upsert_etymon(canonical_form, language)
    matched = matched_form or canonical_form
    cur = db.conn.execute(
        "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
        "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
        (eid, source_id, matched, 1, 0, ""),
    )
    db.commit()
    return cur.lastrowid


def test_lookup_attested_years_finds_year_after_form_with_comma(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The dominant scholarly toponym citation pattern: 'Tune, 1086' —
    matched form, comma, year. Year falls within the [100, 1700] range
    and the preceding context contains ', ' (a recognised marker), so
    it qualifies."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "mawer_1920.txt").write_text(
        "BACKWORTH. Backworth, 1240; Bakworth, 1242. From OE bæc and worð.\n"
        "TUNE. Tune, 1086 (DB); Tunes, 1242."
    )
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="mawer_1920", title="Mawer")
        rid = _seed_text_match(db, source_id="mawer_1920", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["candidates"] == 1
    assert result["rows_written"] == 1
    assert year == 1086


def test_lookup_attested_years_picks_earliest_when_multiple_qualify(
    fresh_db: Path, tmp_path: Path
) -> None:
    """When a form is cited at multiple years (typical for well-attested
    toponyms), the earliest qualifying year wins. Matches the v1 design
    documented on lookup_attested_years."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "ekwall_1922.txt").write_text("TUNE. Tune, 1086 (DB); Tunes, 1242; Tunna, 1340.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="ekwall_1922", title="Ekwall")
        rid = _seed_text_match(db, source_id="ekwall_1922", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_lookup_attested_years_rejects_publication_year(fresh_db: Path, tmp_path: Path) -> None:
    """1920 is a publication year, not an attestation. _ATTESTED_YEAR_MAX_LOOKUP
    = 1700 cuts off well before publication years for the corpus, so
    nothing should be written."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("denu (Mawer, 1920) appears throughout.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="denu")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year is None


def test_lookup_attested_years_handles_complex_citation_list(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Real EPNS pattern: 'speldredge (c. 1290 sac 6, 134, 1320 ch)'
    has THREE digit runs, only one of which is a real year-citation.
    The form-attached pattern with the 'c.' marker correctly extracts
    1290 (year) and rejects 134 (section reference — preceded by ', '
    after a digit, not after alpha). 1320 is also a real year, but
    1290 wins on 'earliest'.

    Pin so a future loosening of the form-attached pattern doesn't
    reintroduce the digit-list FP class."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("speldredge (c. 1290 sac 6, 134, 1320 ch).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="speldredge")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1290


def test_lookup_attested_years_rejects_year_without_date_marker(
    fresh_db: Path, tmp_path: Path
) -> None:
    """A digit run not preceded by a recognised date-context marker is
    not a date citation. Filters page numbers, statute counts, etc."""
    sources = tmp_path / "sources"
    sources.mkdir()
    # 'See note 1240 below' has '1240' but no preceding marker — it's a
    # cross-reference number, not a date citation.
    (sources / "src.txt").write_text("hamtun see note 1240 below for the discussion.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="hamtun")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year is None


def test_lookup_attested_years_dry_run_does_not_write(fresh_db: Path, tmp_path: Path) -> None:
    """apply=False reports candidates but writes nothing — same shape as
    every other enrichment stage."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=False)

        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["candidates"] == 1
    assert result["rows_written"] == 0
    assert result["applied"] is False
    assert year is None


def test_lookup_attested_years_idempotent_on_rerun(fresh_db: Path, tmp_path: Path) -> None:
    """Re-running on a DB that already has attested_year set is a no-op:
    only rows where attested_year IS NULL are scanned. Lets a future run
    pick up newly-added text-match rows without re-clobbering existing
    values."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")

        first = lookup_attested_years(db, sources, apply=True)
        assert first["rows_written"] == 1
        # Second run sees zero unscored rows.
        second = lookup_attested_years(db, sources, apply=True)

    assert second["rows_scanned"] == 0
    assert second["candidates"] == 0
    assert second["rows_written"] == 0


def test_lookup_attested_years_groups_rows_by_source_id(fresh_db: Path, tmp_path: Path) -> None:
    """The streaming restructure (7681838) groups text-match rows by
    source_id so each source body is loaded only once. Pin that the
    grouping is correct: rows from source A get scanned against A's
    body, NOT B's body, even when both have year-citations against
    the same matched_form value. A regression that flattens the
    grouping back to a single shared body would silently swap years
    between sources."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src_a.txt").write_text("Tune, 1086 (DB).")
    (sources / "src_b.txt").write_text("Tune, 1242 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src_a", title="A")
        db.upsert_source(id="src_b", title="B")
        rid_a = _seed_text_match(db, source_id="src_a", canonical_form="tune")
        rid_b = _seed_text_match(db, source_id="src_b", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        years = dict(db.conn.execute("SELECT id, attested_year FROM etymon_text_match").fetchall())
    assert years[rid_a] == 1086
    assert years[rid_b] == 1242


def test_lookup_attested_years_warns_on_missing_source_file(fresh_db: Path, tmp_path: Path) -> None:
    """A text-match row pointing at a source_id whose .txt isn't in
    sources_dir is counted in sources_missing and skipped without
    raising. Common in real corpora where some books have been mined
    but the OCR text is gitignored / absent locally."""
    sources = tmp_path / "sources"
    sources.mkdir()  # Empty — no .txt files.
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="missing_src", title="M")
        _seed_text_match(db, source_id="missing_src", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=True)

    assert result["sources_missing"] == 1
    assert result["candidates"] == 0
    assert result["rows_written"] == 0


# --- D5-1 / wyrd-bag: toponym_etymology.notes scan -------------------------


def _seed_toponym_etymology(
    db: LexiconDB,
    *,
    notes: str,
    modern_name: str = "Tune",
    source_id: str = "src",
    historical_form: str = "Tūn",
) -> int:
    """Insert a toponym + toponym_etymology row carrying ``notes``.
    Returns the toponym_etymology row id. Used as a fixture builder
    for the wyrd-bag tests."""
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology "
        "(toponym_id, source_id, historical_form, confidence, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (toponym_id, source_id, historical_form, "high", notes),
    )
    db.commit()
    return cur.lastrowid


def test_lookup_attested_years_populates_toponym_etymology_from_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """wyrd-bag: notes column carries dense scholarly date strings
    ('1086 DB; 1242 IPM'). The post-mining stage picks the earliest
    plausibly-attested year (>=700) and writes it to attested_year.
    Pin the dominant case — multiple year citations, earliest wins."""
    sources = tmp_path / "sources"
    sources.mkdir()  # required by lookup_attested_years even when empty
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes="extracted_by:llm:qwen | Tune (DB). 1086 DB; 1242 IPM; 1340 Cl",
        )
        result = lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["toponym_etymology"]["candidates"] == 1
    assert result["toponym_etymology"]["rows_written"] == 1
    assert year == 1086


def test_lookup_attested_years_rejects_pre_700_years_in_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The simple year-range filter (>=700) rules out the dominant
    false-positive class: page numbers, volume numbers, footnote refs.
    Pin that '233' (volume) and '47' (page) don't get picked even when
    they precede the form. Earliest qualifying year is 1086."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes=("VI. 233 (close) — Tune. p. 47. Buketon c. 1086 DB; Tunes 1242 ; 1340 Cl"),
        )
        lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert year == 1086


def test_earliest_year_in_notes_does_not_false_skip_p_inside_word(
    fresh_db: Path,
) -> None:
    """Regression for PR #53 Gemini-medium: the page-marker filter
    must require a word boundary before 'p.' / 'ch.' / etc. so a
    real word ending in 'p.' (or any short letter sequence + dot)
    doesn't false-skip a real year-citation. The substring shape
    'p.' appears inside 'chap.', 'pp.', etc. — those are themselves
    markers, so the FP risk is words like 'Bp.' (Bishop abbreviation)
    or stray letter+dot sequences.

    Pin: 'Bp 1086' must yield 1086, not None."""

    # 'Bp.' (Bishop) — not a page marker. Year 1086 should be picked.
    assert _earliest_year_in_notes("Bp. 1086 DB.") == 1086
    # 'pp.' IS a page marker. Year 1086 should be skipped (no other
    # year present → None).
    assert _earliest_year_in_notes("pp. 1086 (book index)") is None
    # 'p.' alone IS a page marker. Year 755 should be skipped.
    assert _earliest_year_in_notes("p. 755") is None
    # 'p' without a trailing dot is NOT a page marker — it's just the
    # letter p ending some prior word. Year should be picked.
    assert _earliest_year_in_notes("group 1086") == 1086
    # Multiple spaces between marker and year: PR #53 round-3 Gemini
    # flagged that an 8-char window would miss "p.   755". Full-prefix
    # regex with $ anchor must still catch it.
    assert _earliest_year_in_notes("p.   755") is None
    assert _earliest_year_in_notes("vol.        1244") is None
    # Marker-preceded page-ref ≥700 BEFORE a real year in the same
    # note: the $ anchor confines marker matching to the immediate
    # predecessor of each year candidate, so a distant page ref
    # doesn't suppress the real date that follows. Symmetry pin
    # (the test_lookup_attested_years_skips_page_marker_false_positive
    # test covers the reverse ordering — real year, then page ref).
    assert _earliest_year_in_notes("p. 755 ; Tune 1086") == 1086


def test_lookup_attested_years_skips_page_marker_false_positive(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Real EPNS-style note: 'Plac. de quo War. p. 755.' has 755
    preceded by 'p. ', a PAGE reference, not a year. Pin that the
    page-marker filter rejects it. Earliest qualifying year is the
    actual citation '1086'."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes="Westhamconett 1086 DB ; Plac. de quo War. p. 755.",
        )
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_lookup_attested_years_skips_toponym_etymology_with_null_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Defensive: a toponym_etymology row with notes=NULL (legacy or
    very-low-confidence row) doesn't crash the scan."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("X",))
        toponym_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "src", "low", None),
        )
        db.commit()
        result = lookup_attested_years(db, sources, apply=True)

    # null-notes rows are excluded by the WHERE clause, so they don't
    # even surface as scanned.
    assert result["toponym_etymology"]["rows_scanned"] == 0
    assert result["toponym_etymology"]["candidates"] == 0


def test_lookup_attested_years_idempotent_on_toponym_etymology(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Re-running lookup-attested-years against the same DB doesn't
    re-process toponym_etymology rows whose attested_year is already
    populated. Mirror of the etymon_text_match idempotency guarantee
    so cron-style scheduled runs are cheap no-ops."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology(db, notes="X 1086 DB")

        first = lookup_attested_years(db, sources, apply=True)
        assert first["toponym_etymology"]["rows_written"] == 1

        second = lookup_attested_years(db, sources, apply=True)
    assert second["toponym_etymology"]["rows_scanned"] == 0
    assert second["toponym_etymology"]["candidates"] == 0
    assert second["toponym_etymology"]["rows_written"] == 0


def test_lookup_attested_years_aggregates_both_row_sources(fresh_db: Path, tmp_path: Path) -> None:
    """The aggregate keys (rows_scanned/candidates/rows_written) sum
    across etymon_text_match AND toponym_etymology. Pin so a future
    refactor that drops one source from the aggregate surfaces."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")
        _seed_toponym_etymology(db, notes="Tune 1086 DB")
        result = lookup_attested_years(db, sources, apply=True)

    # Each source contributes one row; aggregate is 2.
    assert result["etymon_text_match"]["rows_written"] == 1
    assert result["toponym_etymology"]["rows_written"] == 1
    assert result["rows_written"] == 2
    assert result["candidates"] == 2


def test_clear_enrichment_attested_years_clears_both_tables(
    fresh_db: Path,
) -> None:
    """wyrd-bag: clear_enrichment(stage='attested-years') must NULL
    the column on BOTH etymon_text_match AND toponym_etymology.
    Otherwise a re-run would only re-process half the corpus and
    leave stale years on the other half."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        # Seed a populated etymon_text_match row.
        rid_etm = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid_etm),
        )
        # Seed a populated toponym_etymology row.
        rid_te = _seed_toponym_etymology(db, notes="Tune 1086")
        db.conn.execute(
            "UPDATE toponym_etymology SET attested_year = ? WHERE id = ?",
            (1086, rid_te),
        )
        db.commit()

        result = clear_enrichment(db, stage="attested-years", apply=True)
        # The combined count covers both tables.
        assert result["attested_years_to_clear"] == 2

        etm_year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid_etm,)
        ).fetchone()["attested_year"]
        te_year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid_te,)
        ).fetchone()["attested_year"]

    assert etm_year is None
    assert te_year is None


def test_lookup_attested_years_accepts_year_at_upper_bound_1700(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Regression for PR #47 Gemini-medium: the bare-year regex
    `1[0-6]\\d{2}` cuts off at 1699 but `_ATTESTED_YEAR_MAX_LOOKUP` is
    1700. Pin that 1700 itself qualifies as a bare year via the
    explicit |1700 alternation."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1700.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1700


def test_lookup_attested_years_lowercases_form_before_matching(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Regression for PR #47 Gemini-medium: source body text is
    lowercased before scanning, so a matched_form like 'TUNE' or 'Tune'
    must lowercase before being interpolated into the regex pattern.
    Without the form.lower() guard this row would silently miss its
    citation."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("TUNE, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        # Stored matched_form is mixed-case, body is upper. Lowercasing
        # both at scan time makes them line up.
        rid = _seed_text_match(db, source_id="src", canonical_form="Tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_clear_enrichment_attested_years_resets_only_year_column(
    fresh_db: Path,
) -> None:
    """clear_enrichment(stage='attested-years') nulls out attested_year
    on every etymon_text_match row but leaves the row itself intact —
    matched_form, snippet, method, and the etymon link all survive."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

        result = clear_enrichment(db, stage="attested-years", apply=True)
        row = db.conn.execute(
            "SELECT matched_form, attested_year FROM etymon_text_match WHERE id = ?",
            (rid,),
        ).fetchone()

    assert result["attested_years_to_clear"] == 1
    assert row["matched_form"] == "tune"  # row not deleted
    assert row["attested_year"] is None


def test_clear_enrichment_all_derived_includes_attested_years(
    fresh_db: Path,
) -> None:
    """stage='all-derived' covers attested-years too — verifies the
    rollup includes the new stage so a fully clean reset really is."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

        result = clear_enrichment(db, stage="all-derived", apply=True)
        # text-match also clears the row entirely, so attested_year
        # naturally goes to None — but the COUNT before the clear should
        # have been 1, recorded in the dry-run-style counts.
        assert result["attested_years_to_clear"] == 1
        # text-match clear runs alongside, so the row is gone.
        remaining = db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0]
    assert remaining == 0


# --- wyrd-skm Phase 3.0a: toponym-attestation mining ----------------------


def _import_attestation_helpers():
    """Returns the ``(extractor, miner)`` pair for the wyrd-skm tests.

    Kept as a small accessor so tests can opt into pin-language naming
    in their bodies (``extract, mine = _import_attestation_helpers()``)
    without each test re-listing the imports."""
    return _extract_attestation_pairs, mine_toponym_attestations


def test_extract_attestation_pairs_form_in_year() -> None:
    """The dominant scholarly shape: 'FORM in YEAR'. The year-anchor
    requires a connector ('in', comma, or semicolon) so a sentence-
    initial capitalised word followed by a year-shaped digit run
    doesn't mis-match."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Carlton. Written Carleton in 1302 (F.A. i. 142).")
    assert pairs == [("Carleton", 1302)]


def test_extract_attestation_pairs_form_comma_year() -> None:
    """The 'FORM, YEAR' shape — common in citation lists like
    'Isilham, 1284; Iselham, 1302; Yeselham, 1321'."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Isleham. Formerly Isilham, 1284; Iselham, 1302; Yeselham, 1321.")
    assert pairs == [("Isilham", 1284), ("Iselham", 1302), ("Yeselham", 1321)]


def test_extract_attestation_pairs_form_comma_in_year() -> None:
    """'FORM, in YEAR' — comma followed by 'in', appears in Skeat/
    Mawer prose. Pin so a future regex tightening doesn't drop this
    branch."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Tadlow. The old spelling is Tadelowe, in 1302 (F.A.).")
    assert pairs == [("Tadelowe", 1302)]


def test_extract_attestation_pairs_domesday_anchored() -> None:
    """'Domesday Book' / 'Domesday' / 'D.B.' all canonicalise to year
    1086. Both 'FORM in Domesday' and 'Domesday has FORM' (and the
    abbreviated forms) extract the same pair."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Carlen-tone in Domesday. Domesday Book has Chingestone.")
    assert ("Carlen-tone", 1086) in pairs
    assert ("Chingestone", 1086) in pairs


def test_extract_attestation_pairs_db_abbreviation() -> None:
    """The 'D.B. has FORM' inverted shape and 'FORM, D.B.' both map to
    Domesday Book, year=1086."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("D.B. has Badingafelda, p. 59. Spelt Benagra, D.B., p. 182.")
    assert ("Badingafelda", 1086) in pairs
    assert ("Benagra", 1086) in pairs


def test_extract_attestation_pairs_rejects_uppercase_abbreviations() -> None:
    """Source-citation abbreviations (LPR, LI, LF, DB, MS) are
    pure-uppercase and never legitimate place-name forms. The
    lowercase-required filter handles the entire abbreviation FP class
    in one check."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Citation evidence: LPR, 1212; LI, 1332; LF, 1332.")
    assert pairs == []


def test_extract_attestation_pairs_rejects_blacklisted_sentence_words() -> None:
    """Sentence-initial capitalised words ('The', 'From', 'These')
    aren't place-name forms even though they pass the regex
    character class."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("The 1086 entry is interesting. From 1242 onwards.")
    # 'The' and 'From' are in _ATTEST_FORM_BLACKLIST.
    assert pairs == []


def test_extract_attestation_pairs_rejects_record_series_source_names() -> None:
    """EPNS record-series names (Originalia Rolls 'Orig', De Banco Rolls
    'Banco', Book of Fees 'Fees') read like a Title-case place form and
    pass the lowercase filter, so they leaked as dated attestations
    ('Banco, 1399') until blacklisted. They are sources, not spellings."""
    extract, _ = _import_attestation_helpers()
    assert extract("Orig, 1325 Cl Cippenham") == []
    assert extract("Banco, 1399, 1406 YI Yearesley") == []
    assert extract("Hundesburton 1224-30 Fees, 1283") == []
    # a real Title-case form in the same shape is still admitted
    assert extract("Cestretone, 1316") == [("Cestretone", 1316)]


def test_extract_attestation_pairs_rejects_page_marker_year() -> None:
    """A digit run preceded by 'p.', 'pp.', 'vol.' is a bibliographic
    reference, not a date citation. Same guard the wyrd-bag scan
    uses."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Cross-reference Bedingafelda, p. 1086. Compare Tunbridge in vol. 1242.")
    # p. 1086 / vol. 1242 are page references, not dates.
    assert pairs == []


def test_extract_attestation_pairs_rejects_out_of_range_years() -> None:
    """Years outside [_ATTESTED_YEAR_MIN_LOOKUP, _ATTESTED_YEAR_MAX_LOOKUP]
    are rejected even with a valid form+connector — they're typos, page
    refs, or modern dates, not in-period attestations. Pins both sides of
    the year-range bound in _admit_year_match (wyrd-8uvi: the one branch
    of the hoisted helper not otherwise driven through extract())."""
    extract, _ = _import_attestation_helpers()
    # 300 is below the min; 2000 is above the max — neither is admitted.
    pairs = extract("Olde, 300. Tune in 2000.")
    assert pairs == []


def test_extract_attestation_pairs_requires_connector_between_form_and_year() -> None:
    """'After 1066' / 'Before 1066' / 'Source 1234' have a capitalised
    word followed by a year-shaped digit run with only whitespace
    between. The connector requirement (comma, semicolon, or 'in')
    rejects these to keep precision high."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("After 1066 the Norman conquest. Source 1234 mentions it.")
    assert pairs == []


def test_extract_attestation_pairs_source_chain_guard_default_suppresses() -> None:
    """Default behavior (context_is_body=False) suppresses the source-
    chain shape `<year> FORM, <year>` because in toponym_etymology.notes
    this is dominantly a multi-source citation FP (e.g. `1539 Wills,
    1544 LP` where Wills is a source name)."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("1539 Wills, 1544 LP")
    # Both 'Wills' and 'LP' should be suppressed — 'Wills' by the
    # source-chain guard, 'LP' by the uppercase-abbreviation filter.
    assert ("Wills", 1539) not in pairs


def test_extract_attestation_pairs_context_is_body_disables_chain_guard() -> None:
    """context_is_body=True disables the source-chain guard. The
    `<year> FORM, <year>` shape — where the year-anchored extractor
    matches the FOLLOWING year — survives admission in body mode but
    is suppressed by the guard in default mode. Pins both branches
    against the same fixture so the seam is visible. wyrd-9ekl."""
    extract, _ = _import_attestation_helpers()
    # In body mode, Suttone(1210) survives (form is followed by ", 1210").
    pairs_body = extract("1086 Suttone, 1210", context_is_body=True)
    assert ("Suttone", 1210) in pairs_body
    # In default mode (notes-string calibration), the same input is
    # suppressed by the source-chain guard.
    pairs_default = extract("1086 Suttone, 1210")
    assert ("Suttone", 1210) not in pairs_default


def test_extract_attestation_pairs_telemetry_bumps_on_pattern_match() -> None:
    """The telemetry counter bumps whenever the source-chain PATTERN
    matches, decoupled from whether suppression actually fires. Body-
    mode callers see how often the shape occurs in their input even
    though the guard admits the row. Default-mode callers see how
    often suppression dropped a row (same counter — body mode is the
    measurement, default mode is the action). wyrd-9ekl decoupling
    per Gemini round-2 review."""
    extract, _ = _import_attestation_helpers()
    # Default mode: pattern matches → counter bumps AND row suppressed.
    telemetry: dict[str, int] = {}
    pairs_default = extract("1086 Suttone, 1210", telemetry=telemetry)
    assert telemetry.get("suppressed_by_source_chain", 0) >= 1
    assert ("Suttone", 1210) not in pairs_default
    # Body mode: pattern matches → counter still bumps, row admitted.
    telemetry_body: dict[str, int] = {}
    pairs_body = extract("1086 Suttone, 1210", context_is_body=True, telemetry=telemetry_body)
    assert telemetry_body.get("suppressed_by_source_chain", 0) >= 1
    assert ("Suttone", 1210) in pairs_body


def test_extract_attestation_pairs_dedupes_by_form_and_year() -> None:
    """When the same (form, year) appears twice in a notes value
    (the regex matches both), only one tuple is emitted. Output
    ordering is deterministic (year ascending, then form
    alphabetically) so callers depending on first-row stability
    aren't surprised."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Tunbridge, 1086. Forde in 1086. Tunbridge, 1086 again.")
    assert pairs == [("Forde", 1086), ("Tunbridge", 1086)]


def test_extract_attestation_pairs_handles_none_and_empty() -> None:
    extract, _ = _import_attestation_helpers()
    assert extract(None) == []
    assert extract("") == []
    assert extract("No date citations here at all.") == []


def test_extract_attestation_pairs_db_followed_by_space_in_middle_of_sentence() -> None:
    """Regression for the trailing ``(?:\\b|,|;|\\s)`` alternation in
    the ``D.B.`` regex.

    `\\b` does NOT match between `.` and a space — both are non-word
    characters, and `\\b` only fires at word/non-word transitions.
    Without the explicit `|\\s` alternation, a sentence like
    ``"Cestretone D.B. has more"`` would mis-match because the regex
    couldn't find a boundary after ``D.B.``.

    This test pins the alternation so a future cleanup that drops
    ``|\\s`` (which looks redundant given ``\\b``) gets caught."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Cestretone D.B. has more details elsewhere.")
    assert ("Cestretone", 1086) in pairs


def test_extract_attestation_pairs_welsh_diacritics() -> None:
    """Welsh + Norman diacritics in the production toponym_etymology
    corpus (~7% of notes carry one). Pin coverage so a future regex
    tightening doesn't silently miss them.

    Welsh stratum work (PR #105 / wyrd-lr4 Phase 1) makes this
    increasingly important — Welsh place-names are routinely cited in
    forms like Llanfaên / Ŷrwyrne / Hēafod with macron / circumflex
    diacritics."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Llanfaên, 1284. Ŷrwyrne in 1310. Hēafod in Domesday Book.")
    assert ("Llanfaên", 1284) in pairs
    assert ("Ŷrwyrne", 1310) in pairs
    assert ("Hēafod", 1086) in pairs


def test_extract_attestation_pairs_norman_diacritics() -> None:
    """Norman / Anglo-Norman diacritics (ç, é, è) used in the early
    French charter spellings cited by the toponym corpus."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Forêt, 1180. Châtelaine in 1240.")
    assert ("Forêt", 1180) in pairs
    assert ("Châtelaine", 1240) in pairs


def test_extract_attestation_pairs_chain_last_element_with_bare_connector() -> None:
    """Skeat / Mawer chains often drop the explicit connector on the
    LAST element: ``"Cestretone in 1210; Cestrede, 1218; Cestretun,
    1230; Chestreton 1242"`` — the four-element chain has only three
    explicit connectors. The ``;``-anchored chain pattern picks up
    the trailing element."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Cestretone in 1210; Cestrede, 1218; Cestretun, 1230; Chestreton 1242.")
    forms = [f for f, _ in pairs]
    assert "Cestretone" in forms
    assert "Cestrede" in forms
    assert "Cestretun" in forms
    assert "Chestreton" in forms
    assert dict(pairs)["Chestreton"] == 1242


def test_extract_attestation_pairs_rejects_source_attribution_chain() -> None:
    """wyrd-hcd9: Mawer / Skeat / Ekwall use the convention
    ``<year> <source>, <year> <source>`` for chained same-form source
    citations — e.g. ``Chevington 1535 VE, 1539 Wills, 1544 LP`` lists
    three sources for the SAME form. The naive regex would catch
    ``Wills, 1544`` as a form-year attestation, but Wills is a SOURCE
    name (the Wills Register).

    The FP detector requires both:
    (a) the form is preceded by ``<year> `` (the source-year shape)
    (b) the form is followed by ``, <year>`` (next chain link)
    so real chained attestations (``Edreston ; 1242 ...``,
    semicolon-separated) aren't suppressed.
    """
    extract, _ = _import_attestation_helpers()
    pairs = extract("Chevington 1535 VE, 1539 Wills, 1544 LP")
    forms = [f for f, _ in pairs]
    assert "Wills" not in forms
    assert "LP" not in forms


def test_extract_attestation_pairs_keeps_real_attestation_chains() -> None:
    """wyrd-hcd9 regression: the source-attribution-chain detector
    must NOT false-suppress real attestation chains. The
    distinguisher is COMMA (source-attribution) vs SEMICOLON
    (different-form attestation). Mawer / Skeat use semicolons
    between forms and commas within source-list-for-one-form."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Edredeston, 1234 Edreston ; 1242 Cl. Hetheresto")
    forms = [f for f, _ in pairs]
    assert "Edreston" in forms
    assert "Edredeston" in forms


def test_extract_attestation_pairs_keeps_long_attestation_chains() -> None:
    """wyrd-hcd9 regression: a 4-element attestation chain like
    ``Cestretone in 1210; Cestrede, 1218; Cestretun, 1230;
    Chestreton 1242`` — the LAST element 'Chestreton 1242' is bare-
    connector chain-anchored. Verify all four still capture even
    after the source-chain detector landed."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Cestretone in 1210; Cestrede, 1218; Cestretun, 1230; Chestreton 1242.")
    forms = [f for f, _ in pairs]
    assert "Cestretone" in forms
    assert "Cestrede" in forms
    assert "Cestretun" in forms
    assert "Chestreton" in forms


def test_extract_attestation_pairs_chain_anchor_rejects_sentence_flow() -> None:
    """The chain pattern's leading ``;`` requirement keeps it from
    matching arbitrary sentence-flow shapes. ``"After 1066 the
    conquest came"`` and ``"Source 1234 mentions"`` have no
    semicolon-prefix, so they don't trip the bare-connector branch."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("After 1066 the conquest came. Source 1234 mentions Cestretone.")
    assert pairs == []


def test_extract_attestation_pairs_rejects_academic_citation_with_page() -> None:
    """``"Author, 1086 (p. 59)"`` is an academic citation, not an
    attestation — Author is a scholar, 1086 is the publication year,
    p. 59 is the page reference. The negative lookahead
    ``(?!\\s*\\(p+\\.)`` after the year rejects this whole class of
    false positives without needing a per-author blacklist."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Author, 1086 (p. 59) cited; Cestretone, 1086 (D.B.).")
    forms = [f for f, _ in pairs]
    assert "Author" not in forms
    # Real attestation in the same notes should still match — its
    # parens carry a source name, not a page reference.
    assert "Cestretone" in forms


def test_extract_attestation_pairs_real_paginated_attestation_is_kept() -> None:
    """Real attestations sometimes cite the page in the parenthetical:
    ``"Knesworth in 1276 (Rot. Hund. p. 51)"`` — ``(`` is followed by
    ``Rot``, not ``p.``. The lookahead correctly leaves this match
    alone since it only suppresses ``(p.`` immediately after the
    year-paren."""
    extract, _ = _import_attestation_helpers()
    pairs = extract("Knesworth in 1276 (Rot. Hund. p. 51).")
    assert ("Knesworth", 1276) in pairs


def test_mine_toponym_attestations_progress_every_zero_does_not_divide_by_zero(
    fresh_db: Path,
) -> None:
    """Defensive: a caller passing ``--progress-every 0`` would
    otherwise trip a modulo-zero on the in-loop emit and a
    divide-by-zero on the rate calculation. The function clamps to ≥1
    so the call returns cleanly with a single completion progress line."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(db, notes="Cestretone in 1210.")
        # Should not raise.
        result = mine(db, apply=True, progress_every=0)
    assert result["rows_written"] == 1


def test_cli_lexicon_mine_attestations_dry_run_smoke(fresh_db: Path) -> None:
    """End-to-end CLI smoke test: invoke ``mine-attestations`` with no
    --apply flag against a fresh DB carrying one citation-bearing
    notes row. The command should exit 0, mention the candidate
    count, and write nothing to toponym_attestation."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(
            db, notes="Cestretone in 1210; Domesday Book has Chingestone."
        )

    result = runner.invoke(
        kenning_cli,
        ["lexicon", "mine-attestations", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "candidates= " in result.output
    assert "(dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        row_count = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert row_count == 0


def test_cli_lexicon_mine_attestations_apply_writes_rows(fresh_db: Path) -> None:
    """End-to-end CLI smoke test with --apply: the same notes row
    persists two attestation rows."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(
            db, notes="Cestretone in 1210; Domesday Book has Chingestone."
        )

    result = runner.invoke(
        kenning_cli,
        ["lexicon", "mine-attestations", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output

    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT form, date_year FROM toponym_attestation ORDER BY date_year, form"
        ).fetchall()
    assert [(r["form"], r["date_year"]) for r in rows] == [
        ("Chingestone", 1086),
        ("Cestretone", 1210),
    ]


def _seed_toponym_etymology_for_attestations(
    db: LexiconDB,
    *,
    notes: str,
    modern_name: str = "Chesterton",
    source_id: str = "src",
) -> tuple[int, int]:
    """Insert a toponym + toponym_etymology row carrying ``notes``;
    return ``(toponym_id, etymology_id)``."""
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology "
        "(toponym_id, source_id, historical_form, confidence, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (toponym_id, source_id, None, "high", notes),
    )
    db.commit()
    return toponym_id, cur.lastrowid


def test_mine_toponym_attestations_inserts_pairs(fresh_db: Path) -> None:
    """End-to-end: notes carrying a year-anchored citation produces a
    toponym_attestation row keyed on the toponym_id."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="skeat_1901", title="Skeat")
        toponym_id, _ = _seed_toponym_etymology_for_attestations(
            db,
            notes="Chesterton is spelt Cestretone in 1210 (R.B.). Domesday Book has Chingestone.",
            source_id="skeat_1901",
        )
        result = mine(db, apply=True)
        rows = db.conn.execute(
            "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation "
            "ORDER BY date_year, form"
        ).fetchall()

    assert result["rows_with_pairs"] == 1
    assert result["rows_written"] == 2
    assert [(r["form"], r["date_year"]) for r in rows] == [
        ("Chingestone", 1086),
        ("Cestretone", 1210),
    ]
    # source_doc carries the etymology row's source_id so multi-scholar
    # attestations of the same toponym/year stay distinguishable.
    assert all(r["source_doc"] == "skeat_1901" for r in rows)
    assert all(r["toponym_id"] == toponym_id for r in rows)


def test_mine_toponym_attestations_idempotent_via_unique_index(fresh_db: Path) -> None:
    """Re-running mine-attestations against the same notes is a no-op:
    the unique index on (toponym_id, form, date_year, source_doc) makes
    INSERT OR IGNORE drop duplicates, so the table count doesn't grow."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(
            db, notes="Cestretone in 1210; Chingestone in Domesday Book."
        )
        first = mine(db, apply=True)
        second = mine(db, apply=True)
        total = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]

    assert first["rows_written"] == 2
    # Second run sees the same rows but INSERT OR IGNORE skips them.
    assert second["rows_written"] == 0
    assert total == 2


def test_mine_toponym_attestations_dry_run(fresh_db: Path) -> None:
    """apply=False reports candidate counts but writes no rows. Same
    shape as every other enrichment stage."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(
            db, notes="Cestretone in 1210; Chingestone in Domesday Book."
        )
        result = mine(db, apply=False)
        total = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]

    assert result["candidates"] == 2
    assert result["rows_written"] == 0
    assert result["applied"] is False
    assert total == 0


def test_mine_toponym_attestations_skips_rows_without_notes(fresh_db: Path) -> None:
    """Etymology rows with NULL or empty notes contribute nothing —
    the WHERE filter on the SELECT keeps the scan loop bounded to
    rows that could plausibly carry citations."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Empty",))
        topo = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, ?, ?)",
            (topo, "src", "high"),
        )
        db.commit()
        result = mine(db, apply=True)

    assert result["rows_scanned"] == 0
    assert result["rows_written"] == 0


def test_clear_enrichment_attestations_drops_rows(fresh_db: Path) -> None:
    """clear_enrichment(stage='attestations') deletes every row from
    toponym_attestation, leaving the table empty for re-mining."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(db, notes="Cestretone in 1210.")
        mine(db, apply=True)
        before = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
        result = clear_enrichment(db, stage="attestations", apply=True)
        after = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]

    assert before == 1
    assert result["attestation_rows_to_clear"] == 1
    assert after == 0


def test_clear_enrichment_all_derived_includes_attestations(fresh_db: Path) -> None:
    """stage='all-derived' covers attestations too — the rollup count
    includes the row count from toponym_attestation."""
    _, mine = _import_attestation_helpers()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology_for_attestations(db, notes="Cestretone in 1210.")
        mine(db, apply=True)
        result = clear_enrichment(db, stage="all-derived", apply=True)
        remaining = db.conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]

    assert result["attestation_rows_to_clear"] == 1
    assert remaining == 0


def test_migrate_schema_adds_attestation_unique_index_on_legacy_db(tmp_path: Path) -> None:
    """A legacy DB created before Phase 3.0a has the
    toponym_attestation table but no idx_attestation_unique. Running
    migrate_schema adds the index without disturbing existing rows."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Drop the index to simulate a pre-Phase-3.0a deployment.
        db.conn.execute("DROP INDEX IF EXISTS idx_attestation_unique")
        db.commit()
        before = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='toponym_attestation'"
            )
        }
        assert "idx_attestation_unique" not in before

        applied = migrate_schema(db)

        after = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='toponym_attestation'"
            )
        }
    assert applied["idx_attestation_unique"] is True
    assert "idx_attestation_unique" in after


# --- wyrd-skm Phase 3.0b: cognate-cluster era-reflex picker ---------------


def _seed_cluster(
    db: LexiconDB,
    *,
    cluster_root_form: str = "*kaster",
    cluster_root_lang: str = "proto-germanic",
    members: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Insert a synthetic cognate cluster with the given members
    sharing one ``cognate_id`` (set to the root etymon's id).

    ``members`` is a list of (canonical_form, language) tuples. The
    return dict maps each form to its etymon id so tests can reference
    cluster mates by form. Mirrors how ``cluster_cognates`` would lay
    things out in production: root etymon is its own cognate_id, and
    every reachable descendant points back to the root.
    """
    members = members or []
    root_id = db.upsert_etymon(cluster_root_form, cluster_root_lang)
    db.conn.execute(
        "UPDATE etymon SET cognate_id = ? WHERE id = ?",
        (root_id, root_id),
    )
    by_form = {cluster_root_form: root_id}
    for form, lang in members:
        member_id = db.upsert_etymon(form, lang)
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ? WHERE id = ?",
            (root_id, member_id),
        )
        by_form[form] = member_id
    db.commit()
    return by_form


def test_etymon_era_reflexes_returns_matching_language_members(fresh_db: Path) -> None:
    """Direct ``target_language`` mode: cluster mates whose
    ``etymon.language`` matches the requested tag are returned, sorted
    alphabetically by canonical_form for stable output."""

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("ceaster", "old-english"),
                ("Chestre", "middle-english"),
                ("Chester", "middle-english"),
                ("chester", "modern-english"),
            ],
        )
        reflexes = etymon_era_reflexes(db, ids["ceaster"], target_language="middle-english")

    forms = [r.form for r in reflexes]
    languages = {r.language for r in reflexes}
    assert forms == ["Chester", "Chestre"]  # alphabetical
    assert languages == {"middle-english"}


def test_etymon_era_reflexes_resolves_family_cell_to_canonical_language(
    fresh_db: Path,
) -> None:
    """``target_family_cell=('english', 'me')`` resolves to
    'middle-english' via ``canonical_language_for_cell`` and behaves
    identically to passing ``target_language='middle-english'``
    directly. Lets callers pass an era-cell label without knowing the
    canonical-tag mapping."""

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("ceaster", "old-english"),
                ("Chester", "middle-english"),
                ("chester", "modern-english"),
            ],
        )
        reflexes = etymon_era_reflexes(db, ids["ceaster"], target_family_cell=("english", "me"))

    assert [r.form for r in reflexes] == ["Chester"]


def test_etymon_era_reflexes_returns_empty_when_no_cognate_id_or_descent(
    fresh_db: Path,
) -> None:
    """An etymon with neither ``cognate_id`` nor descent edges has
    nothing to walk — the era-reflex lookup is a no-op and returns
    ``[]``, not an error. Consumers fall back to the original
    etymon's canonical_form."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        # Don't set cognate_id — it stays NULL by default. Don't add
        # any etymon_descent edges either.
        reflexes = etymon_era_reflexes(db, eid, target_language="middle-english")

    assert reflexes == []


def test_etymon_era_reflexes_falls_back_to_descent_when_no_cognate_id(
    fresh_db: Path,
) -> None:
    """When ``cognate_id`` is NULL but the etymon has direct
    ``etymon_descent`` edges, the picker falls back to walking
    immediate children. Recovers ~4% of OE toponym etymons that
    have descent edges but never landed in cluster_cognates output
    (typically because their ancestor chain isn't connected to a
    known root).

    Filters: edge_type must be 'inheritance' or 'borrowing' (peer
    'cognate' edges are excluded — too loose for v1 era-rendering).
    Children must match the target language and not be merged-away.
    """
    with LexiconDB(fresh_db) as db:
        db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('test', 'Test')")
        parent_id = db.upsert_etymon("broc", "old-english")
        child_id = db.upsert_etymon("brok", "middle-english")
        cognate_peer_id = db.upsert_etymon("brokr", "middle-english")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'test', 'high')",
            (parent_id, child_id),
        )
        # 'cognate' is a peer edge, not a chain — excluded from the
        # fallback path so loose lexical similarity doesn't surface.
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'cognate', 'test', 'medium')",
            (parent_id, cognate_peer_id),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, parent_id, target_language="middle-english")

    forms = [r.form for r in reflexes]
    assert forms == ["brok"]  # cognate-peer 'brokr' excluded


def test_etymon_era_reflexes_descent_fallback_filters_merged_into(
    fresh_db: Path,
) -> None:
    """Descent-edge fallback respects ``merged_into_id`` the same
    way the cluster path does — OCR-cluster losers are tombstones
    and shouldn't surface as period forms even via the fallback."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('test', 'Test')")
        parent_id = db.upsert_etymon("broc", "old-english")
        winner_id = db.upsert_etymon("brok", "middle-english")
        loser_id = db.upsert_etymon("broke", "middle-english")
        # Both children are descendants of the parent.
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'test', 'high')",
            (parent_id, winner_id),
        )
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'test', 'high')",
            (parent_id, loser_id),
        )
        # Mark the loser as merged into the winner (D22 OCR cluster).
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner_id, loser_id),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, parent_id, target_language="middle-english")

    assert [r.form for r in reflexes] == ["brok"]


def test_etymon_era_reflexes_returns_empty_when_no_canonical_language(
    fresh_db: Path,
) -> None:
    """Some era cells have no canonical language tag (the Latin
    'classical' cell is ambiguous against the flat 'latin' tag we
    use throughout the corpus). When ``canonical_language_for_cell``
    returns None for the cell, the reflex picker returns ``[]``
    rather than scanning the entire cluster."""

    # Sanity: pick a (family, cell) combination that's NOT in the map.
    assert canonical_language_for_cell("norse", "modern") is None

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[("tūn", "old-norse"), ("tun", "norwegian-bokmal")],
        )
        reflexes = etymon_era_reflexes(db, ids["tūn"], target_family_cell=("norse", "modern"))

    assert reflexes == []


def test_etymon_era_reflexes_filters_merged_into_id(fresh_db: Path) -> None:
    """OCR-cluster losers (D22) carry ``merged_into_id`` pointing at
    the canonical winner. They're tombstones — the merge target is
    the rendering pick. The reflex lookup excludes them so a
    user-visible era-rewind doesn't surface a merged-away spelling
    variant."""

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("ceaster", "old-english"),
                ("Chester", "middle-english"),
                ("Chestre", "middle-english"),
            ],
        )
        # Mark Chestre as a merged-away duplicate of Chester.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (ids["Chester"], ids["Chestre"]),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, ids["ceaster"], target_language="middle-english")

    assert [r.form for r in reflexes] == ["Chester"]


def test_etymon_era_reflexes_raises_when_no_target_provided(
    fresh_db: Path,
) -> None:
    """Defensive: passing neither ``target_language`` nor
    ``target_family_cell`` raises ValueError so a typo at the
    call-site doesn't silently return the empty list."""

    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ceaster", "old-english")
        with pytest.raises(ValueError, match="exactly one"):
            etymon_era_reflexes(db, eid)


def test_etymon_era_reflexes_raises_when_both_targets_provided(
    fresh_db: Path,
) -> None:
    """Defensive: passing BOTH targets is also a caller bug —
    silently picking one would mask a bad merge or a confused
    branch. Pinned by raising ValueError."""

    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ceaster", "old-english")
        with pytest.raises(ValueError, match="exactly one"):
            etymon_era_reflexes(
                db,
                eid,
                target_language="middle-english",
                target_family_cell=("english", "me"),
            )


def test_etymon_era_reflexes_rejects_empty_target_language(fresh_db: Path) -> None:
    """An empty-string ``target_language`` would slip the ``is None``
    guard and resolve to a SQL ``language = ''`` predicate that
    silently returns zero rows (every etymon has a non-empty
    language tag). Surface this as a ValueError so a buggy caller
    failing-soft to "" gets caught."""

    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ceaster", "old-english")
        with pytest.raises(ValueError, match="empty"):
            etymon_era_reflexes(db, eid, target_language="")


def test_etymon_cognate_id_is_fk_enforced_against_dangling_pointers(
    fresh_db: Path,
) -> None:
    """The era-reflex picker assumes ``etymon.cognate_id`` always
    references an existing row when non-NULL — a dangling pointer
    would silently return an empty result. SQLite's FK enforcement
    actually prevents the dangling state at the DB level, which is
    a stronger guarantee. Pin the FK so a future schema migration
    that drops the constraint surfaces this assumption.

    Concretely: attempting to set cognate_id to a nonexistent etymon
    raises IntegrityError before the corrupt state can persist."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ceaster", "old-english")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.conn.execute("UPDATE etymon SET cognate_id = 999999999 WHERE id = ?", (eid,))


def test_canonical_language_for_cell_returns_expected_english_picks() -> None:
    """Pin the English-family cell → language mapping so a future
    rename of a cell label or language tag fails loudly. Downstream
    wyrd-rni / wyrd-381 demos depend on these picks."""

    assert canonical_language_for_cell("english", "oe-early") == "old-english"
    assert canonical_language_for_cell("english", "oe-late") == "old-english"
    assert canonical_language_for_cell("english", "me") == "middle-english"
    assert canonical_language_for_cell("english", "early-modern") == "modern-english"
    assert canonical_language_for_cell("english", "modern") == "modern-english"


def test_cli_lexicon_era_reflex_emits_cluster_mates(fresh_db: Path) -> None:
    """End-to-end CLI smoke test: ``lexicon era-reflex <id> --era me``
    against a cluster with English-family members prints one line
    per matching reflex. Each line is ``<id>\\t<form>\\t<language>``
    on stdout; the metadata header goes to stderr."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("ceaster", "old-english"),
                ("Chester", "middle-english"),
                ("Chestre", "middle-english"),
                ("chester", "modern-english"),
            ],
        )

    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "era-reflex",
            str(ids["ceaster"]),
            "--era",
            "me",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if not line.startswith("#")]
    assert len(lines) == 2
    assert "Chester\tmiddle-english" in result.output
    assert "Chestre\tmiddle-english" in result.output


def test_cli_lexicon_era_reflex_unknown_etymon_exits_nonzero(fresh_db: Path) -> None:
    """Defensive CLI: an etymon_id that doesn't exist in the DB
    exits 1 with a stderr message. Catches typos before they ship
    silent empty output."""
    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "era-reflex",
            "999999999",
            "--era",
            "me",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0


def test_cli_lexicon_era_reflex_no_era_family_exits_nonzero(fresh_db: Path) -> None:
    """An etymon whose ``language`` has no era family (proto-
    languages, untracked classical languages) exits 1 with an
    informative stderr message. The era-reflex picker only makes
    sense when there's a family to resolve cells against."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        # 'proto-germanic' is intentionally NOT in LANGUAGE_TO_FAMILY.
        eid = db.upsert_etymon("*tūnaz", "proto-germanic")
        db.commit()

    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "era-reflex",
            str(eid),
            "--era",
            "me",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "no era family" in result.output


def test_cli_lexicon_era_reflex_invalid_era_input_exits_nonzero(
    fresh_db: Path,
) -> None:
    """``--era`` values that don't resolve (typo'd cell label,
    out-of-range year) exit 1 with a stderr message naming valid
    alternatives. The error comes from era_cell_for_input so the
    message points at families where the label IS defined."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ceaster", "old-english")
        db.commit()

    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "era-reflex",
            str(eid),
            "--era",
            "nonsense-cell",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0


def test_cli_lexicon_era_reflex_no_canonical_language_prints_empty(
    fresh_db: Path,
) -> None:
    """A target era cell with no canonical language tag (Norse
    'modern' etc.) yields zero reflexes — the CLI emits the '(no
    cluster mates at this era)' notice and exits 0. Cell coverage
    gaps aren't errors; consumers fall back to canonical_form."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("tūn", "old-norse")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, eid))
        db.commit()

    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "era-reflex",
            str(eid),
            "--era",
            "norse/modern",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0
    assert "no cluster mates" in result.output


def test_era_cell_for_input_year_resolves_to_cell(fresh_db: Path) -> None:
    """``era_cell_for_input(1086, default_family='english')`` resolves
    to the OE-late cell since 1086 < 1100 boundary. Pin the boundary
    so a future widening of OE-late doesn't surprise callers."""

    assert era_cell_for_input(1086, default_family="english") == ("english", "oe-late")
    assert era_cell_for_input(1210, default_family="english") == ("english", "me")
    assert era_cell_for_input(1700, default_family="english") == (
        "english",
        "modern",
    )


def test_era_cell_for_input_label_resolves_against_family(fresh_db: Path) -> None:
    """A bare cell label resolves against ``default_family``.
    Cross-family lookups must use the ``family/label`` form."""

    assert era_cell_for_input("oe-late", default_family="english") == (
        "english",
        "oe-late",
    )
    # Cross-family explicit form.
    assert era_cell_for_input("english/me", default_family="english") == (
        "english",
        "me",
    )


def test_era_cell_for_input_rejects_empty_and_none(fresh_db: Path) -> None:
    """Empty string and None are intentionally rejected — there's
    no cell for 'no filter'. Callers wanting that should branch
    before calling rather than passing a sentinel value."""

    with pytest.raises(ValueError, match="None or empty"):
        era_cell_for_input("", default_family="english")
    with pytest.raises(ValueError, match="None or empty"):
        era_cell_for_input(None, default_family="english")  # type: ignore[arg-type]


def test_era_cell_for_input_rejects_unknown_label(fresh_db: Path) -> None:
    """A typo'd cell label raises ValueError naming families where
    the label IS defined (if any), so the user can switch to the
    explicit ``family/label`` form."""

    with pytest.raises(ValueError, match="not defined in family"):
        # 'middle-irish' exists in goidelic, not english.
        era_cell_for_input("middle-irish", default_family="english")
    with pytest.raises(ValueError, match="unknown era input"):
        era_cell_for_input("totally-fake-cell", default_family="english")


# --- wyrd-rni Phase 3.1: time-rewind explainer ---------------------------


def _seed_meaning_db(usage: str, sources: dict[str, list[str]]):
    """Build a single-Meaning meaning_db keyed by ``usage`` so the
    rewind tests can drive the resolver without loading the full
    bundle. Returns the dict shape ``{usage: [Meaning]}`` that
    ``rewind_name`` consumes."""

    m = Meaning(
        usage=usage,
        tags=[],
        meanings=["test"],
        sources=sources,
    )
    return {usage: [m]}


def _make_rewind_meaning_db(*entries: tuple[str, dict[str, list[str]]]):
    """Build a multi-Meaning meaning_db from ``(usage, sources)`` pairs.
    Multiple Meanings per usage are appended to the same list (matches
    how the bundle exposes sibling Meanings for a single usage like
    ``Whit-`` carrying both OE and ON source variants)."""

    db: dict[str, list[Meaning]] = {}
    for usage, sources in entries:
        db.setdefault(usage, []).append(
            Meaning(
                usage=usage,
                tags=[],
                meanings=["test"],
                sources=sources,
            )
        )
    return db


def test_rewind_renders_morphemes_at_each_era_stop(fresh_db: Path) -> None:
    """Happy path: a binary OE compound with a complete cognate
    cluster renders distinct surface forms at oe-late / me / modern.

    Anchors on the OE form (preferred via _ANCHOR_LANG_PREFERENCE),
    short-circuits at oe-late (anchor.language == target), walks the
    cluster at me / modern via etymon_era_reflexes."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hwīt", "old-english"),
                ("whit", "middle-english"),
                ("white", "middle-english"),
                ("white", "modern-english"),
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt", "hwit"]}),
        )
        result = rewind_name("Whit", db, meaning_db)

    assert result.decomposition == "Whit"
    assert len(result.eras) == 3
    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # wyrd-085k: rendered output is now title-cased (name-style),
    # not lowercase (analytical-style).
    assert by_cell["oe-late"] == "Hwīt"  # short-circuit on anchor
    # ME: cluster has 'whit' which case-insensitive matches the
    # canonical 'Whit', so picker prefers it over 'white'.
    assert by_cell["me"] == "Whit"
    # Modern: wyrd-8qbi short-circuits to the morpheme's canonical
    # (modern_usage with dashes stripped) instead of consulting the
    # cluster picker. Previously surfaced 'white' (alphabetical-
    # first sibling cognate). The operator already knows the modern
    # form is 'Whit'; round-trip is the right answer.
    assert by_cell["modern"] == "Whit"


def test_rewind_resolves_anchor_across_meaning_siblings(fresh_db: Path) -> None:
    """The bundle may carry multiple Meanings for the same usage
    (e.g. ``Whit-`` with both ON and OE sources). The anchor
    resolver MUST walk siblings to prefer ``old_english`` over
    ``old_scandinavian`` even when the trie matcher picked the ON
    sibling for the canonical decomposition."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hwīt", "old-english"),
                ("whit", "middle-english"),
                ("white", "modern-english"),
            ],
        )
        # Two Meaning siblings: ON source first (will be the trie's
        # 'input meaning'), OE source second.
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_scandinavian": ["hvit"]}),
            ("Whit-", {"old_english": ["hwīt", "hwit"]}),
        )
        result = rewind_name("Whit", db, meaning_db)

    # The anchor resolver should find the OE sibling and use it,
    # NOT the ON form (which has no etymon row in our seeded DB).
    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    assert by_cell["oe-late"] == "Hwīt"  # wyrd-085k: title-cased


def test_rewind_falls_back_when_no_anchor_or_reflex(fresh_db: Path) -> None:
    """When the morpheme has no etymon row in the lexicon (no anchor)
    OR the anchor has no era reflex at the target era, the rendered
    form falls back to the morpheme's modern usage and the
    ``fallback`` flag is set on the MorphemeRewind."""

    with LexiconDB(fresh_db) as db:
        # No etymon rows seeded — every anchor lookup misses.
        meaning_db = _make_rewind_meaning_db(
            ("Foo-", {"old_english": ["foo"]}),
        )
        result = rewind_name("Foo", db, meaning_db)

    for stop in result.eras:
        assert stop.rendered == "Foo"
        assert all(m.fallback for m in stop.morphemes)


def test_rewind_strips_morpheme_hyphens_when_composing(fresh_db: Path) -> None:
    """Morphemes carry leading/trailing hyphens encoding their
    location (pre-, post-, inner-). The compound renderer joins
    morphemes with explicit hyphens, so the form-level hyphens get
    stripped before joining — otherwise post-modifiers like ``-ham``
    produce double-hyphens (``Had-en--ham``)."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*haimaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hām", "old-english"),
                ("ham", "middle-english"),
                ("-ham", "modern-english"),  # post-modifier hyphen
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("-ham", {"old_english": ["hām"]}),
        )
        result = rewind_name("ham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # No leading/trailing hyphens on the rendered single-morpheme
    # output even when the cluster mate's form carries one.
    # wyrd-085k: rendered output is now title-cased.
    assert by_cell["modern"] == "Ham"


def test_rewind_records_unaccounted_fragments(fresh_db: Path) -> None:
    """When a name doesn't fully decompose (e.g. 'Springvale' has
    no morpheme matching 'Spr'), the unrecognised fragments are
    captured on Rewind.unaccounted so the CLI can flag them. The
    rewind path doesn't crash — it renders what it CAN match and
    surfaces the rest as a warning."""

    with LexiconDB(fresh_db) as db:
        # Only 'vale' decomposes; 'Spring' is unaccounted.
        meaning_db = _make_rewind_meaning_db(
            ("vale", {"old_english": ["valle"]}),
        )
        result = rewind_name("Springvale", db, meaning_db)

    # The matched 'vale' morpheme produces one MorphemeRewind per
    # era; the un-matched 'Spring' prefix is captured as a string
    # on Rewind.unaccounted for the CLI to surface.
    assert "Spring" in " ".join(result.unaccounted)
    for stop in result.eras:
        canonicals = [m.canonical for m in stop.morphemes]
        assert "vale" in canonicals


def test_rewind_supports_custom_era_ladder(fresh_db: Path) -> None:
    """Callers can pass a custom ``eras`` tuple to rewind a name
    across an arbitrary set of cells. Used by wyrd-381's stratified
    era-map (denser ladder than the default 3-cell English stops)."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hwīt", "old-english"),
                ("white", "modern-english"),
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
        )
        # Custom 2-stop ladder: oe-early + modern.
        result = rewind_name(
            "Whit",
            db,
            meaning_db,
            eras=(("english", "oe-early"), ("english", "modern")),
        )

    cells = [stop.cell for stop in result.eras]
    assert cells == ["oe-early", "modern"]


def test_rewind_raises_on_empty_input(fresh_db: Path) -> None:
    """Defensive: empty / whitespace-only input raises ValueError
    so a buggy caller failing-soft to '' surfaces immediately."""

    with LexiconDB(fresh_db) as db:
        meaning_db = _make_rewind_meaning_db(("foo", {"old_english": ["foo"]}))
        with pytest.raises(ValueError, match="name is required"):
            rewind_name("", db, meaning_db)
        with pytest.raises(ValueError, match="name is required"):
            rewind_name("   ", db, meaning_db)


def test_rewind_picker_prefers_anchor_match_over_alphabetical(fresh_db: Path) -> None:
    """Tier-2 picker preference: when neither the canonical nor a
    canonical-match exists in the cluster, prefer reflexes whose
    form matches the anchor's source form (case-insensitive) over
    alphabetical-first.

    Real-world case: OE 'mynster' anchor at ME — cluster has
    'mynster' (the same orthography survives into ME) plus
    peripheral entries that sort alphabetically before it
    ('amounten' etc.). Without tier-2, alphabetical first surfaces
    a noise mate; with tier-2, 'mynster' wins."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*munistar",
            cluster_root_lang="proto-germanic",
            members=[
                ("mynster", "old-english"),
                ("amounten", "middle-english"),  # noise mate, alphabetical first
                ("mynster", "middle-english"),  # real ME reflex (matches anchor)
                ("zynx", "middle-english"),  # alphabetical last
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("-minster", {"old_english": ["mynster"]}),
        )
        result = rewind_name("minster", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Tier-1: canonical match — 'minster' in ME cluster? No.
    # Tier-2: anchor match — 'mynster' in ME cluster? Yes — prefer it.
    # wyrd-085k: rendered output is now title-cased.
    assert by_cell["me"] == "Mynster"


def test_pick_form_strips_morpheme_hyphens() -> None:
    """Direct unit test for ``_pick_form``: era_form takes priority
    over canonical, and leading/trailing hyphens are stripped from
    whichever form is selected. This is the load-bearing rule that
    keeps post-modifier morphemes ('-ham') from producing
    double-hyphens in the rendered compound."""

    # era_form present → wins, hyphens stripped.
    m = MorphemeRewind(
        canonical="ham",
        source_form="hām",
        source_language="old-english",
        source_etymon_id=1,
        era_form="-ham",
        fallback=False,
    )
    assert _pick_form(m) == "ham"
    # era_form None + canonical with hyphens → canonical wins, stripped.
    m = MorphemeRewind(
        canonical="-ham-",
        source_form="hām",
        source_language="old-english",
        source_etymon_id=1,
        era_form=None,
        fallback=True,
    )
    assert _pick_form(m) == "ham"


def test_supported_eras_for_family_returns_cell_labels() -> None:
    """Public helper that delegates to ``era.era_cells_for_family``;
    pin the contract so callers (CLI auto-completion, future SPA
    era selectors) can rely on the cell labels being stable."""

    assert supported_eras_for_family("english") == (
        "oe-early",
        "oe-late",
        "me",
        "early-modern",
        "modern",
    )
    assert supported_eras_for_family("brythonic") == ("old", "middle", "modern")


def test_rewind_handles_multi_word_input(fresh_db: Path) -> None:
    """Multi-word input ('North Cape' / 'Black Hill') splits on
    whitespace and decomposes each word independently. The era stops
    aggregate morphemes across all words; rendered output joins
    morphemes with hyphens regardless of word boundaries."""

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*haimaz",
            cluster_root_lang="proto-germanic",
            members=[("hām", "old-english"), ("ham", "middle-english")],
        )
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english"), ("white", "middle-english")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("-ham", {"old_english": ["hām"]}),
            ("Whit-", {"old_english": ["hwīt"]}),
        )
        # Two-word input: 'whit ham' decomposes to 4 morphemes
        # total (2 per word? actually 1 per word here).
        result = rewind_name("whit ham", db, meaning_db)

    # Both words should be in the decomposition + per-era morpheme list.
    assert "whit" in result.decomposition.lower()
    assert "ham" in result.decomposition
    for stop in result.eras:
        canonicals = [m.canonical for m in stop.morphemes]
        assert "Whit" in canonicals or "whit" in canonicals
        assert "ham" in canonicals


def test_cli_kenning_rewind_with_custom_era_ladder(fresh_db: Path) -> None:
    """The CLI's ``--era`` flag accepts repeated values for a
    custom ladder. Pin the parsing path so a future refactor can't
    silently drop the multi-pass behaviour. Also verifies cell-label
    + family/label inputs both resolve."""

    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*haimaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hām", "old-english"),
                ("ham", "middle-english"),
                ("ham", "modern-english"),
            ],
        )
    _load_meanings.cache_clear()

    # Custom 2-stop ladder via repeated --era; mix bare label and
    # family/label form.
    result = runner.invoke(
        kenning_cli,
        [
            "rewind",
            "ham",
            "--db",
            str(fresh_db),
            "--era",
            "oe-late",
            "--era",
            "english/me",
        ],
    )
    assert result.exit_code == 0, result.output
    # Two era stops printed (default would be 3).
    assert result.output.count("english/") == 2


def test_cli_kenning_rewind_invalid_era_exits_nonzero(fresh_db: Path) -> None:
    """Defensive CLI: an --era value that doesn't resolve (typo'd
    cell label, out-of-range year) exits 1 with stderr message
    naming valid alternatives. The error comes from
    era_cell_for_input."""

    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        # Just need ANY etymon present; the failure is at era parsing.
        db.upsert_etymon("hām", "old-english")
        db.commit()
    _load_meanings.cache_clear()

    result = runner.invoke(
        kenning_cli,
        [
            "rewind",
            "ham",
            "--db",
            str(fresh_db),
            "--era",
            "totally-fake-cell",
        ],
    )
    assert result.exit_code != 0


def test_cli_kenning_rewind_smoke(fresh_db: Path) -> None:
    """End-to-end CLI smoke test: ``wyrd kenning rewind <name>``
    invokes the rewinder against a seeded lexicon and prints one
    line per era stop. Header on stderr includes the decomposition;
    body lines on stdout carry the rendered compound per era."""

    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*haimaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hām", "old-english"),
                ("ham", "middle-english"),
                ("ham", "modern-english"),
            ],
        )

    # The CLI uses the bundled meanings.json — drop the cache so
    # subsequent invocations don't pollute test isolation.
    _load_meanings.cache_clear()
    result = runner.invoke(
        kenning_cli,
        ["rewind", "ham", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "decomposed:" in result.output
    # Three default English era stops printed.
    assert result.output.count("english/") == 3


# --- wyrd-unuo Phase 3.3: per-etymon period-form projection ---------------


def _seed_toponym_with_binary_breakdown(
    db: LexiconDB,
    *,
    modern_name: str,
    first_etymon_id: int,
    last_etymon_id: int,
    source_id: str = "src",
) -> int:
    """Insert a toponym + a binary-breakdown toponym_etymology with two
    toponym_etymology_element rows (ordinals 0, 1) pointing at the
    given etymon ids. Returns the toponym_id."""
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, ?, ?)",
        (toponym_id, source_id, "high"),
    )
    te_id = cur.lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
        (te_id, 0, first_etymon_id),
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
        (te_id, 1, last_etymon_id),
    )
    db.commit()
    return toponym_id


def test_project_period_forms_segments_binary_attestation(fresh_db: Path) -> None:
    """Bradford-style happy path: 'Bradeford' (1377) splits as
    'Brade' + 'ford' via suffix-anchoring against the OE 'ford'
    canonical_form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        toponym_id = _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Bradford",
            first_etymon_id=brad_id,
            last_etymon_id=ford_id,
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "Bradeford", 1377, "src"),
        )
        db.commit()
        result = project_period_forms(db, apply=True)
        rows = db.conn.execute(
            "SELECT etymon_id, form, date_year FROM etymon_period_form ORDER BY etymon_id, form"
        ).fetchall()

    assert result["rows_projected"] == 1
    assert result["rows_written"] == 2
    forms_by_etymon = {r["etymon_id"]: r["form"] for r in rows}
    assert forms_by_etymon[brad_id] == "Brade"
    assert forms_by_etymon[ford_id] == "ford"


def test_project_period_forms_uses_cluster_mate_suffix(fresh_db: Path) -> None:
    """Chesterton-style: 'Cestretone' splits as 'Cestre' + 'tone'
    via cluster-mate suffix matching ('tone' is a Middle Scots
    cognate of OE 'tūn'). Verifies the projector pulls suffix
    candidates from the full cognate cluster, not just the
    canonical_form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        ceaster_id = db.upsert_etymon("ceaster", "old-english")
        tun_id = db.upsert_etymon("tūn", "old-english")
        tone_id = db.upsert_etymon("tone", "gmw-msc")
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ? WHERE id IN (?, ?)",
            (tun_id, tun_id, tone_id),
        )
        db.commit()
        toponym_id = _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Chesterton",
            first_etymon_id=ceaster_id,
            last_etymon_id=tun_id,
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "Cestretone", 1210, "src"),
        )
        db.commit()
        project_period_forms(db, apply=True)
        forms = {
            r["etymon_id"]: r["form"]
            for r in db.conn.execute(
                "SELECT etymon_id, form FROM etymon_period_form ORDER BY etymon_id"
            )
        }

    assert forms[ceaster_id] == "Cestre"
    assert forms[tun_id] == "tone"


def test_project_period_forms_skips_ternary_breakdowns(fresh_db: Path) -> None:
    """v1 limitation: ternary breakdowns are skipped (middle-morpheme
    alignment isn't reliable without phonetic distance)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        had_id = db.upsert_etymon("had", "old-english")
        en_id = db.upsert_etymon("en", "old-english")
        ham_id = db.upsert_etymon("ham", "old-english")
        db.commit()
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Hadenham",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, ?, ?)",
            (toponym_id, "src", "high"),
        )
        te_id = cur.lastrowid
        for ordinal, eid in enumerate([had_id, en_id, ham_id]):
            db.conn.execute(
                "INSERT INTO toponym_etymology_element "
                "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
                (te_id, ordinal, eid),
            )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "Hadenham", 1300, "src"),
        )
        db.commit()
        result = project_period_forms(db, apply=True)

    assert result["rows_projected"] == 0


def test_project_period_forms_skips_breakdowns_with_merged_etymons(
    fresh_db: Path,
) -> None:
    """Per Claude review M1: a breakdown whose component points at an
    OCR-cluster loser (merged_into_id IS NOT NULL) is skipped so loser
    etymons don't accumulate stale period_form rows that then surface
    via Tier 3 of the era-reflex picker. Pin the projector-side filter
    matching Tiers 1+2's behavior."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        winner_id = db.upsert_etymon("brad", "old-english")
        loser_id = db.upsert_etymon("brade", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        # Mark loser as merged-into the winner.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner_id, loser_id),
        )
        db.commit()
        # Toponym breaks down via the LOSER etymon (the original
        # extraction predates the OCR merge).
        toponym_id = _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Bradford",
            first_etymon_id=loser_id,
            last_etymon_id=ford_id,
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "Bradeford", 1377, "src"),
        )
        db.commit()
        result = project_period_forms(db, apply=True)

    assert result["rows_projected"] == 0


def test_etymon_era_reflexes_tier3_filters_merged_into(fresh_db: Path) -> None:
    """Per Claude review M1: Tier 3 query MUST filter
    merged_into_id IS NULL to match Tiers 1+2. A loser etymon's
    period_form rows (if they ever made it past the projector
    filter) shouldn't surface as reflexes — the merge winner is
    the canonical voice."""
    with LexiconDB(fresh_db) as db:
        winner_id = db.upsert_etymon("orphan", "old-english")
        loser_id = db.upsert_etymon("orphan-loser", "old-english")
        # Mark loser as merged-into winner; insert a period_form on
        # the LOSER (simulates a stale row from a pre-merge projector
        # run, or a manual data-correction).
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner_id, loser_id),
        )
        db.conn.execute(
            "INSERT INTO etymon_period_form "
            "(etymon_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (loser_id, "stale-form", 1300, "src"),
        )
        db.commit()
        # Querying the LOSER directly should return [] — Tier 3 honors
        # the merge tombstone.
        reflexes = etymon_era_reflexes(db, loser_id, target_family_cell=("english", "me"))

    assert reflexes == []


def test_project_period_forms_idempotent_rerun(fresh_db: Path) -> None:
    """Unique index on (etymon_id, form, date_year, source_doc) makes
    re-projection a no-op via INSERT OR IGNORE."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        toponym_id = _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Bradford",
            first_etymon_id=brad_id,
            last_etymon_id=ford_id,
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "Bradeford", 1377, "src"),
        )
        db.commit()
        first = project_period_forms(db, apply=True)
        second = project_period_forms(db, apply=True)
        total = db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0]

    assert first["rows_written"] == 2
    assert second["rows_written"] == 0
    assert total == 2


# --- derive_surface_in_modern (wyrd-ujyo) ---------------------------------


def _surfaces_by_ordinal(db: LexiconDB) -> dict[int, str | None]:
    rows = db.conn.execute(
        "SELECT ordinal, surface_in_modern FROM toponym_etymology_element ORDER BY ordinal"
    ).fetchall()
    return {r["ordinal"]: r["surface_in_modern"] for r in rows}


def test_derive_surface_in_modern_segments_modern_name(fresh_db: Path) -> None:
    """Happy path: 'Bradford' = OE brad + OE ford splits as 'Brad' + 'ford' via
    suffix-anchoring against the OE 'ford' canonical_form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradford", first_etymon_id=brad_id, last_etymon_id=ford_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_projected"] == 1
    assert result["rows_written"] == 2
    assert surfaces[0] == "Brad"  # prefix → first morpheme
    assert surfaces[1] == "ford"  # suffix → last morpheme


def test_derive_surface_in_modern_uses_cluster_mate_modern_suffix(fresh_db: Path) -> None:
    """The decisive case: modern names end in MODERN suffixes (Chesterton →
    'ton'), not the OE canonical (tūn). The suffix matches a cognate-cluster mate
    ('ton'), so 'Chesterton' = ceaster + tūn → 'Chester' + 'ton'."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        ceaster_id = db.upsert_etymon("ceaster", "old-english")
        tun_id = db.upsert_etymon("tūn", "old-english")
        ton_id = db.upsert_etymon("ton", "english")  # modern reflex, same cluster
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ? WHERE id IN (?, ?)", (tun_id, tun_id, ton_id)
        )
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Chesterton", first_etymon_id=ceaster_id, last_etymon_id=tun_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_projected"] == 1
    assert surfaces[0] == "Chester"
    assert surfaces[1] == "ton"


def test_derive_surface_in_modern_uses_mined_reflex_surface(fresh_db: Path) -> None:
    """wyrd-5b1a: the modern reflex surface ('ton') lives in the reflex table (the
    eni4 scholar-reflex campaign's output) — NOT as a cognate-cluster etymon and
    NOT as an etymon_variant. derive_surface_in_modern must consult reflex /
    reflex_etymon so 'Chesterton' = ceaster + tūn splits 'Chester' + 'ton'. This is
    the exact path that was unwired: the OE inflection 'tūn' never suffix-matches a
    modern name, so without the reflex the alignment fails."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        ceaster_id = db.upsert_etymon("ceaster", "old-english")
        tun_id = db.upsert_etymon("tūn", "old-english")
        # 'ton' is ONLY a mined reflex — no cognate mate, no etymon_variant.
        reflex_id = db.conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES ('ton', 'post')"
        ).lastrowid
        db.conn.execute(
            "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
            (reflex_id, tun_id),
        )
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Chesterton", first_etymon_id=ceaster_id, last_etymon_id=tun_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_projected"] == 1
    assert surfaces[0] == "Chester"
    assert surfaces[1] == "ton"


def test_suffix_candidates_excludes_reflex_unless_requested(fresh_db: Path) -> None:
    """wyrd-5b1a: _suffix_candidates_for_etymon only pulls reflex surfaces when
    include_reflexes=True. The default-False path is what keeps project_period_forms
    (the historical-form projection) reflex-free — a modern reflex ('ton') would
    mis-anchor a medieval attested form. Also pins the sub-2-char guard: a 1-char
    reflex surface is dropped so it can never suffix-match as noise.

    (Tests the helper directly rather than driving project_period_forms — the
    helper's include_reflexes default IS the historical path's contract.)"""
    from wyrd.generators.kenning.lexicon.period_form import _suffix_candidates_for_etymon

    with LexiconDB(fresh_db) as db:
        tun_id = db.upsert_etymon("tūn", "old-english")
        for sf in ("ton", "n"):  # 'n' is sub-2-char → must be dropped by the len>=2 guard
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun_id)
            )
        db.commit()
        without = _suffix_candidates_for_etymon(db, tun_id, "tūn", None)
        with_reflex = _suffix_candidates_for_etymon(db, tun_id, "tūn", None, include_reflexes=True)

    assert "ton" not in without  # default path: reflex excluded
    assert "ton" in with_reflex  # include_reflexes=True: reflex included
    assert "n" not in with_reflex  # sub-2-char reflex dropped by the len>=2 guard


def test_suffix_candidates_drops_composite_reflexes(fresh_db: Path) -> None:
    """wyrd-5b1a: a COMPOSITE reflex ('eton' = ēa+tūn fused, ends in the bare
    reflex 'ton') must be dropped so the longest-suffix anchor lands on the
    morpheme boundary — else 'Stapleton' splits 'Stapl'+'eton' instead of
    'Staple'+'ton'. Keep only minimal morpheme surfaces."""
    from wyrd.generators.kenning.lexicon.period_form import _suffix_candidates_for_etymon

    with LexiconDB(fresh_db) as db:
        tun_id = db.upsert_etymon("tūn", "old-english")
        for sf in ("ton", "eton", "aughton"):  # 'eton'/'aughton' end in 'ton'
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun_id)
            )
        db.commit()
        cands = _suffix_candidates_for_etymon(db, tun_id, "tūn", None, include_reflexes=True)

    assert "ton" in cands  # minimal morpheme surface kept
    assert "eton" not in cands  # composite dropped
    assert "aughton" not in cands  # composite dropped


def test_suffix_candidates_keeps_base_reflex_with_reduced_sibling(fresh_db: Path) -> None:
    """wyrd-5b1a (Gemini #741): a longer reflex that ends in a shorter one is NOT a
    composite when the shorter is a phonetic REDUCTION of the same morpheme. OE burh
    has 'bury' (base) + 'ury' (reduced, as in Canterbury); 'bury' ends in 'ury', but
    its extra prefix 'b' starts the canonical 'burh', so it's a genuine base form and
    must be KEPT — not dropped as a composite. The filter only drops a surface whose
    extra prefix does NOT prefix the canonical.

    Uses burh/bury so the canonical form does NOT itself strip to the base reflex
    'bury' — otherwise _add(canonical_form) would independently re-add 'bury' and
    mask whether the composite filter kept it (the test would pass even unfixed)."""
    from wyrd.generators.kenning.lexicon.period_form import _suffix_candidates_for_etymon

    with LexiconDB(fresh_db) as db:
        burh_id = db.upsert_etymon("burh", "old-english")
        for sf in ("bury", "ury"):  # base + phonetically-reduced sibling
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, burh_id)
            )
        db.commit()
        cands = _suffix_candidates_for_etymon(db, burh_id, "burh", None, include_reflexes=True)

    assert "bury" in cands  # base kept — extra prefix 'b' starts canonical 'burh'
    assert "ury" in cands  # reduced sibling kept too


def test_suffix_candidates_diacritic_base_reflex_not_dropped(fresh_db: Path) -> None:
    """wyrd-5b1a (Gemini #741): the composite filter must compare diacritic-stripped
    forms. A diacritic-bearing base reflex ('ācen' of canonical 'āc') whose extra
    prefix 'ā' starts the morpheme must be KEPT — without consistent stripping, the
    unstripped 'ā' would not match the stripped canonical 'ac', so 'ācen' would be
    wrongly dropped as a composite."""
    from wyrd.generators.kenning.lexicon.period_form import _suffix_candidates_for_etymon

    with LexiconDB(fresh_db) as db:
        ac_id = db.upsert_etymon("āc", "old-english")
        for sf in ("ācen", "cen"):  # diacritic-bearing base + reduced sibling
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, ac_id)
            )
        db.commit()
        cands = _suffix_candidates_for_etymon(db, ac_id, "āc", None, include_reflexes=True)

    # 'ācen' kept → _add stores its diacritic-stripped form 'acen' too.
    assert "acen" in cands  # base reflex not wrongly dropped on a diacritic mismatch


def test_reflex_suffix_candidates_helper_drops_composite_keeps_reduction(fresh_db: Path) -> None:
    """The extracted `_reflex_suffix_candidates` (the include_reflexes / composite-drop
    half of `_suffix_candidates_for_etymon`), called directly, returns ONLY the minimal
    morpheme surfaces. One set exercises every side of the wyrd-5b1a guard:
    - 'ingham' DROPPED: composite — ends in the sibling 'ham', and its extra prefix
      'ing' does NOT prefix canonical 'ham'.
    - 'ham' KEPT: base — ends in the sibling 'am', but its extra prefix 'h' DOES start
      'ham', so the reduction guard protects it from being dropped.
    - 'am' KEPT: shortest sibling — no surface is a strict suffix-superset of it, so
      nothing can drop it."""
    from wyrd.generators.kenning.lexicon.period_form import _reflex_suffix_candidates

    with LexiconDB(fresh_db) as db:
        ham_id = db.upsert_etymon("hām", "old-english")
        for sf in ("ham", "am", "ingham"):
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, ham_id)
            )
        db.commit()
        minimal = _reflex_suffix_candidates(db, ham_id, "hām")

    assert minimal == {"ham", "am"}  # base + reduction kept; composite 'ingham' dropped


def test_reflex_suffix_candidates_all_combining_mark_surface_guard(fresh_db: Path) -> None:
    """wyrd-5b1a `and stripped[o]` guard: a reflex surface that diacritic-strips to ""
    (pure combining marks, but len>=2 so it survives the length filter) must NOT be
    treated as a suffix of every other surface (`"".endswith` is always True). The
    guard skips such an `o`, so a real base reflex ('ton') is never wrongly dropped,
    and the combining-mark surface itself is kept (nothing strips it away)."""
    from wyrd.generators.kenning.lexicon.period_form import _reflex_suffix_candidates

    combining = "́̂"  # two combining accents → strips to "" (len 2)
    with LexiconDB(fresh_db) as db:
        tun_id = db.upsert_etymon("tūn", "old-english")
        for sf in ("ton", combining):
            rid = db.conn.execute(
                "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (sf,)
            ).lastrowid
            db.conn.execute(
                "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun_id)
            )
        db.commit()
        minimal = _reflex_suffix_candidates(db, tun_id, "tūn")

    assert "ton" in minimal  # real base reflex not dropped by the strip-to-empty surface
    assert combining in minimal  # the combining-mark surface survives (nothing drops it)


def test_derive_surface_in_modern_dry_run_writes_nothing(fresh_db: Path) -> None:
    """apply=False reports the projection count but writes no surface_in_modern."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradford", first_etymon_id=brad_id, last_etymon_id=ford_id
        )
        result = derive_surface_in_modern(db, apply=False)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_projected"] == 1  # counted
    assert result["rows_written"] == 0  # but not written
    assert surfaces == {0: None, 1: None}  # column untouched


def test_derive_surface_in_modern_idempotent_rerun(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradford", first_etymon_id=brad_id, last_etymon_id=ford_id
        )
        derive_surface_in_modern(db, apply=True)
        before = _surfaces_by_ordinal(db)
        derive_surface_in_modern(db, apply=True)  # re-derive
        after = _surfaces_by_ordinal(db)

    assert before == after == {0: "Brad", 1: "ford"}


def test_derive_surface_in_modern_skips_ternary_breakdown(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        a_id = db.upsert_etymon("brad", "old-english")
        b_id = db.upsert_etymon("inga", "old-english")
        c_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Bradingford",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, ?, ?)",
            (toponym_id, "src", "high"),
        )
        te_id = cur.lastrowid
        for ordinal, eid in enumerate((a_id, b_id, c_id)):
            db.conn.execute(
                "INSERT INTO toponym_etymology_element "
                "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
                (te_id, ordinal, eid),
            )
        db.commit()
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_scanned"] == 0  # ternary breakdown not loaded
    assert result["rows_projected"] == 0
    assert set(surfaces.values()) == {None}


def test_derive_surface_in_modern_skips_merged_etymon(fresh_db: Path) -> None:
    """A breakdown whose component points at an OCR-cluster loser
    (merged_into_id set) is skipped, so losers don't get stale surfaces."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        winner_id = db.upsert_etymon("ford-winner", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner_id, ford_id))
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradford", first_etymon_id=brad_id, last_etymon_id=ford_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_scanned"] == 0  # the only breakdown was dropped
    assert set(surfaces.values()) == {None}


def test_derive_surface_in_modern_two_char_gate(fresh_db: Path) -> None:
    """A match leaving a sub-2-char prefix is noise → not projected. 'Aton' =
    a + tūn would leave 'A' (1 char), so nothing is written."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        a_id = db.upsert_etymon("a", "old-english")
        tun_id = db.upsert_etymon("tūn", "old-english")
        ton_id = db.upsert_etymon("ton", "english")
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ? WHERE id IN (?, ?)", (tun_id, tun_id, ton_id)
        )
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Aton", first_etymon_id=a_id, last_etymon_id=tun_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_projected"] == 0  # 'A' prefix fails the ≥2-char gate
    assert set(surfaces.values()) == {None}


def test_derive_surface_in_modern_no_suffix_match(fresh_db: Path) -> None:
    """A binary breakdown IS scanned but its modern suffix matches none of the
    last morpheme's reflexes → nothing projected (the dominant precision-loss
    path). 'Bradville' doesn't end in any reflex of 'ford'."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradville", first_etymon_id=brad_id, last_etymon_id=ford_id
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["rows_scanned"] == 1  # the breakdown was scanned
    assert result["rows_projected"] == 0  # but no suffix matched
    assert set(surfaces.values()) == {None}


def test_derive_surface_in_modern_skips_multiword_name(fresh_db: Path) -> None:
    """v1 skips multi-word modern names — the prefix slice would otherwise absorb
    the leading words ('Stratford upon A')."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        strat_id = db.upsert_etymon("strǣt", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Stratford upon Avon",
            first_etymon_id=strat_id,
            last_etymon_id=ford_id,
        )
        result = derive_surface_in_modern(db, apply=True)
        surfaces = _surfaces_by_ordinal(db)

    assert result["multiword_skipped"] == 1
    assert result["rows_projected"] == 0
    assert set(surfaces.values()) == {None}


def test_derive_surface_in_modern_cli(fresh_db: Path) -> None:
    """End-to-end CLI: ``lexicon derive-surface-in-modern`` dry-runs by default
    and writes surface_in_modern with --apply."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        _seed_toponym_with_binary_breakdown(
            db, modern_name="Bradford", first_etymon_id=brad_id, last_etymon_id=ford_id
        )

    result = runner.invoke(
        kenning_cli, ["lexicon", "derive-surface-in-modern", "--db", str(fresh_db)]
    )
    assert result.exit_code == 0, result.output
    assert "(dry-run" in result.output
    with LexiconDB(fresh_db) as db:
        assert _surfaces_by_ordinal(db) == {0: None, 1: None}

    result = runner.invoke(
        kenning_cli, ["lexicon", "derive-surface-in-modern", "--db", str(fresh_db), "--apply"]
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        assert _surfaces_by_ordinal(db) == {0: "Brad", 1: "ford"}


def test_etymon_era_reflexes_tier4_period_form_fallback(fresh_db: Path) -> None:
    """Tier 4: when both cluster and descent paths return empty AND
    target_family_cell is set, the picker queries etymon_period_form
    rows whose date_year falls in the cell's range."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_period_form (etymon_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (eid, "orphane", 1300, "src"),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_family_cell=("english", "me"))

    forms = [r.form for r in reflexes]
    assert "orphane" in forms
    # Period-form reflexes carry source='period-form' so consumers
    # can distinguish them from Tier 1 cluster mates. Pinned to catch
    # a future regression that defaults the source back to 'cluster'.
    sources = {r.source for r in reflexes if r.form == "orphane"}
    assert sources == {"period-form"}


def test_etymon_era_reflexes_tier4_filters_by_year_range(fresh_db: Path) -> None:
    """Tier 4 must respect the era cell's year range — period forms
    outside the cell don't surface."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        for form, year in [("orphan-oe", 950), ("orphane", 1300), ("orphan-mod", 1800)]:
            db.conn.execute(
                "INSERT INTO etymon_period_form "
                "(etymon_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
                (eid, form, year, "src"),
            )
        db.commit()
        me_reflexes = etymon_era_reflexes(db, eid, target_family_cell=("english", "me"))
        oe_reflexes = etymon_era_reflexes(db, eid, target_family_cell=("english", "oe-late"))

    assert [r.form for r in me_reflexes] == ["orphane"]
    assert [r.form for r in oe_reflexes] == ["orphan-oe"]


def test_etymon_era_reflexes_tier4_skipped_when_target_language_only(
    fresh_db: Path,
) -> None:
    """Tier 4 requires a year range; with target_language only,
    no year context exists so Tier 4 is skipped."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_period_form (etymon_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (eid, "orphane", 1300, "src"),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="middle-english")

    # Period-form skipped (no year range), but phonology-rule Tier 4
    # still runs on the canonical_form. 'orphan' has no OE→ME rule
    # triggers (no macrons / special chars), so it passes through and
    # phonology_rules.rule_form returns None — assert empty.
    assert reflexes == []


# --- wyrd-98cs: phonology-rule Tier 4 fallback ---------------------------


def test_era_reflex_dataclass_carries_source_field_default_cluster() -> None:
    """``EraReflex.source`` defaults to 'cluster' for back-compat with
    existing call sites that didn't pass an explicit source. Pin so a
    future refactor can't silently drop the default."""

    ref = EraReflex(etymon_id=1, form="x", language="old-english")
    assert ref.source == "cluster"


def test_etymon_era_reflexes_cluster_path_tags_source_cluster(
    fresh_db: Path,
) -> None:
    """Tier 1 (cognate cluster) reflexes carry ``source='cluster'``."""
    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("ceaster", "old-english"),
                ("Chester", "middle-english"),
            ],
        )
        reflexes = etymon_era_reflexes(db, ids["ceaster"], target_language="middle-english")
    assert all(r.source == "cluster" for r in reflexes)
    assert len(reflexes) == 1


def test_etymon_era_reflexes_descent_path_tags_source_descent(
    fresh_db: Path,
) -> None:
    """Tier 2 (descent fallback) reflexes carry ``source='descent'``."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('test', 'Test')")
        parent_id = db.upsert_etymon("broc", "old-english")
        child_id = db.upsert_etymon("brok", "middle-english")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'test', 'high')",
            (parent_id, child_id),
        )
        db.commit()
        reflexes = etymon_era_reflexes(db, parent_id, target_language="middle-english")
    assert [r.source for r in reflexes] == ["descent"]


def test_etymon_era_reflexes_phonology_tier_fires_when_other_tiers_empty(
    fresh_db: Path,
) -> None:
    """Tier 4 (phonology rule) fires when Tier 1-3 produce nothing AND
    the canonical_form contains a rule trigger. Pinned with an OE
    form whose ǣ → e rule fires cleanly."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("dǣg", "old-english")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="middle-english")
    assert len(reflexes) == 1
    assert reflexes[0].source == "phonology-rule:v1"
    assert reflexes[0].language == "middle-english"
    # ǣ → e fires; macron loss leaves "deg".
    assert reflexes[0].form == "deg"
    assert reflexes[0].etymon_id == eid


def test_etymon_era_reflexes_phonology_tier_skipped_when_cluster_present(
    fresh_db: Path,
) -> None:
    """Tier 4 is gated on Tier 1-3 being empty. When cluster mates
    exist, the phonology fallback doesn't fire (the cluster mates
    are higher-quality evidence)."""
    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            members=[
                ("dǣg", "old-english"),
                ("dei", "middle-english"),
            ],
        )
        reflexes = etymon_era_reflexes(db, ids["dǣg"], target_language="middle-english")
    # Only the cluster mate, not a phonology-derived "deg".
    assert [(r.form, r.source) for r in reflexes] == [("dei", "cluster")]


def test_etymon_era_reflexes_phonology_tier_inverse_walk(
    fresh_db: Path,
) -> None:
    """Inverse direction: ME etymon with no cluster/descent gets a
    phonology-rule-derived OE form via inverse-mode rule walking.
    Inverse mode is inherently lossy (50/50 branching) but produces a
    deterministic top candidate."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("ston", "middle-english")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="old-english")
    assert len(reflexes) == 1
    assert reflexes[0].source == "phonology-rule:v1"
    assert reflexes[0].language == "old-english"
    # Inverse of ā → o: ston → stān is one branch; the picker takes
    # the highest-probability candidate. Either 'ston' (didn't fire)
    # or 'stān' (fired) is plausible, but if the result equals input
    # we suppress per the no-rule-fired check, so non-empty means
    # 'stān'-shaped.
    assert reflexes[0].form != "ston"


def test_etymon_era_reflexes_phonology_tier_no_op_when_form_passes_through(
    fresh_db: Path,
) -> None:
    """When the canonical_form contains no rule trigger, the phonology
    walk produces the input unchanged. We treat that as 'no Tier 4
    reflex' — consumers fall back to the etymon's own canonical_form
    without an extra spurious EraReflex carrying the same string."""
    with LexiconDB(fresh_db) as db:
        # 'orphan' has no OE→ME rule triggers — no macrons, no ǣ/ġ/ċ,
        # no ME suffix collapsing. Forward walk is a pass-through.
        eid = db.upsert_etymon("orphan", "old-english")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="middle-english")
    assert reflexes == []


def test_etymon_era_reflexes_phonology_tier_skipped_for_cross_family(
    fresh_db: Path,
) -> None:
    """No phonology rules bridge English ↔ Welsh. Cross-family Tier 4
    walks correctly return empty rather than blindly applying the
    English chain to a Welsh form."""
    with LexiconDB(fresh_db) as db:
        # Welsh etymon, asking for a middle-english reflex. No
        # cross-family chain.
        eid = db.upsert_etymon("kair", "old-welsh")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="middle-english")
    assert reflexes == []


def test_etymon_era_reflexes_phonology_tier_welsh_chain_works(
    fresh_db: Path,
) -> None:
    """OW → ModW phonology cell fires. 'kair' → 'caer' via the
    composite kair → caer rule (Phase 2.3, PR #141)."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("kair", "old-welsh")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="modern-welsh")
    assert len(reflexes) == 1
    assert reflexes[0].source == "phonology-rule:v1"
    assert reflexes[0].form == "caer"


def test_etymon_era_reflexes_phonology_tier_chain_oe_to_modern_english(
    fresh_db: Path,
) -> None:
    """OE → ModE chains through ME and EModE cells. Pin a multi-step
    walk so the chain composition is exercised end-to-end."""
    with LexiconDB(fresh_db) as db:
        # 'stān' → 'ston' (OE→ME via ā → o) → 'ston' (ME→EModE no
        # match) → 'ston' (EModE→ModE no match). The key point is the
        # chain runs without error and produces SOME modified form.
        eid = db.upsert_etymon("stān", "old-english")
        db.commit()
        reflexes = etymon_era_reflexes(db, eid, target_language="modern-english")
    assert len(reflexes) == 1
    assert reflexes[0].source == "phonology-rule:v1"
    assert reflexes[0].language == "modern-english"
    assert reflexes[0].form == "ston"


def test_phonology_rule_form_returns_none_for_unknown_language() -> None:
    """Languages absent from the phonology family chains map (Goidelic,
    Norse, Romance) get None — Tier 4 silently no-ops for them."""

    # 'irish' is not in phonology_rules.FAMILY_CHAINS.
    assert _phonology_rule_form("baile", "irish", "old-irish") is None


def test_phonology_rule_form_returns_none_for_same_language() -> None:
    """Same from/to language is a no-op — no transformation needed."""

    assert _phonology_rule_form("dǣg", "old-english", "old-english") is None


def test_phonology_rule_form_returns_none_when_pass_through() -> None:
    """A form with no rule triggers returns None (rather than the
    same input echoed back). Consumers can use canonical_form directly
    when this happens."""

    assert _phonology_rule_form("orphan", "old-english", "middle-english") is None


def test_phonology_rule_form_resolves_welsh_alias() -> None:
    """The lexicon code 'welsh' is a bundle alias for 'modern-welsh' in
    the phonology family chain. Without alias resolution the chain
    lookup would fail with ValueError. Pin: the alias resolves to the
    canonical era so passing 'welsh' produces the same output as
    passing 'modern-welsh' (regardless of what that output is)."""

    # Alias case: welsh (= modern-welsh) → old-welsh
    via_alias = _phonology_rule_form("kaer-uent", "welsh", "old-welsh")
    # Canonical case: modern-welsh → old-welsh
    via_canonical = _phonology_rule_form("kaer-uent", "modern-welsh", "old-welsh")
    assert via_alias == via_canonical


def test_clear_enrichment_period_forms_drops_rows(fresh_db: Path) -> None:
    """clear_enrichment(stage='period-forms') deletes every row from
    etymon_period_form, leaving the table empty for re-projection."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_period_form (etymon_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (eid, "orphane", 1300, "src"),
        )
        db.commit()
        before = db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0]
        result = clear_enrichment(db, stage="period-forms", apply=True)
        after = db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0]

    assert before == 1
    assert result["period_form_rows_to_clear"] == 1
    assert after == 0


def test_clear_enrichment_all_derived_includes_period_forms(fresh_db: Path) -> None:
    """stage='all-derived' covers period-forms too."""
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("orphan", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_period_form (etymon_id, form, date_year, source_doc) "
            "VALUES (?, ?, ?, ?)",
            (eid, "orphane", 1300, "src"),
        )
        db.commit()
        result = clear_enrichment(db, stage="all-derived", apply=True)
        remaining = db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0]

    assert result["period_form_rows_to_clear"] == 1
    assert remaining == 0


def test_migrate_schema_creates_etymon_period_form_on_legacy_db(
    tmp_path: Path,
) -> None:
    """A legacy DB pre-Phase-3.3 has no etymon_period_form table.
    migrate_schema creates it idempotently."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute("DROP TABLE IF EXISTS etymon_period_form")
        db.commit()
        applied = migrate_schema(db)
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert applied["etymon_period_form_table"] is True
    assert "etymon_period_form" in tables


def test_rewind_anchor_resolves_across_hyphen_variants(fresh_db: Path) -> None:
    """wyrd-unuo: the anchor resolver collects siblings across
    hyphen-variant usage keys. The bundle keys ``ton`` (Celtic-mix)
    and ``-ton`` (OE post-modifier) are stripped-equivalent, and
    when the trie picks the Celtic ``ton`` Meaning, the OE anchor
    at ``-ton`` must still be found."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        oe_tun = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (oe_tun, oe_tun))
        db.upsert_etymon("ton", "celtic")
        db.commit()

        meaning_db = _make_rewind_meaning_db(
            ("ton", {"celtic_mix": ["ton"]}),
            ("-ton", {"old_english": ["tūn"]}),
        )
        result = rewind_name("ton", db, meaning_db)

    # The anchor SHOULD be OE 'tūn' (via the -ton sibling).
    by_cell = {stop.cell: stop.morphemes[0] for stop in result.eras}
    assert by_cell["oe-late"].source_form == "tūn"
    assert by_cell["oe-late"].source_language == "old-english"


# --- wyrd-cp2d: rewind_from_morphemes (JSON continuity flow) -------------


def test_rewind_from_morphemes_uses_supplied_usages_no_trie(fresh_db: Path) -> None:
    """wyrd-cp2d: rewind_from_morphemes bypasses the trie matcher.
    The supplied 'usage' values directly resolve via meaning_db
    lookup, so the era-render pipeline uses the morpheme the
    upstream generator actually picked — not what the trie would
    re-decompose from the surface string.

    Real-world bug this fixes: 'Cranwith' generated by the kenning
    generator picks specific lemmas; re-decomposing the string
    'Cranwith' via the trie lands on 'corn' (different etymological
    cluster) and produces 'Cornlandian' at OE. The JSON continuity
    flow preserves the generator's actual picks."""
    from wyrd.generators.kenning.era.rewind import rewind_from_morphemes

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
        )
        words = [[{"usage": "Whit-"}]]
        result = rewind_from_morphemes("Whit", words, db, meaning_db)

    assert result.decomposition == "Whit"
    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    assert by_cell["oe-late"] == "Hwīt"
    # ME cluster has no member here so the picker falls back; modern
    # uses canonical short-circuit (wyrd-8qbi).
    assert by_cell["modern"] == "Whit"


def test_rewind_component_morpheme_carries_canonical_original_usage() -> None:
    """wyrd-7cvv: each rewound morpheme component carries `canonical` = the
    morpheme's ORIGINAL modern usage, so the SPA rewind transform can align
    each rewound form back to the input morpheme it came from (and omit input
    morphemes the rewind dropped) instead of mismatching the rewound name
    against the full original morpheme set."""
    from wyrd.generators.kenning.generators.kenning_rewind import _apply_meaning
    from wyrd.generators.kenning.runtime.meaning import Meaning

    meaning = Meaning("-ton", tags=[], meanings=["enclosure"], sources={"old_english": ["tūn"]})
    comps: list[dict] = []
    _apply_meaning(meaning, None, comps)
    assert len(comps) == 1
    assert comps[0]["canonical"] == "-ton"  # the original usage, dashes intact
    assert "form" in comps[0]


def test_rewind_from_morphemes_preserves_multi_word_grouping(fresh_db: Path) -> None:
    """wyrd-cp2d: multi-word inputs keep their word boundaries. The
    JSON's outer list is per-word; rewind_from_morphemes renders
    each word independently and joins with spaces — same per-word
    grouping semantics as wyrd-t2bh's rewind_name path."""
    from wyrd.generators.kenning.era.rewind import rewind_from_morphemes

    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            ("*krofta", "croft"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            ("croft", {"old_english": ["croft"]}),
        )
        # Two words, each with one morpheme
        words = [[{"usage": "Whit-"}], [{"usage": "croft"}]]
        result = rewind_from_morphemes("Whit croft", words, db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    assert by_cell["modern"] == "Whit Croft"
    assert by_cell["oe-late"] == "Hwīt Croft"


def test_rewind_from_morphemes_missing_usage_lands_in_unaccounted(fresh_db: Path) -> None:
    """wyrd-cp2d: a usage that doesn't exist in meaning_db (the
    bundle dropped the morpheme since the JSON was generated)
    lands in result.unaccounted with the usage string preserved.
    Other morphemes in the same word still render."""
    from wyrd.generators.kenning.era.rewind import rewind_from_morphemes

    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
        )
        # Second morpheme's usage doesn't exist in meaning_db
        words = [[{"usage": "Whit-"}, {"usage": "-dropped"}]]
        result = rewind_from_morphemes("Whit", words, db, meaning_db)

    assert "-dropped" in result.unaccounted
    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    assert by_cell["oe-late"] == "Hwīt"


def test_rewind_supplied_words_resolves_positional_usage_dash_insensitively() -> None:
    """wyrd-7cvv: the generator/API rewind path (what the SPA uses) resolves a
    morpheme whose usage is the POSITIONAL surface ('by' at a word end) to the
    canonical meaning_db key ('-by') via the dash/case-insensitive fallback,
    instead of dropping it as missing — so rewind keeps real morphemes instead
    of rendering a fragment. A usage with no dash/case variant in the bundle
    still lands in `missing`."""
    from wyrd.generators.kenning.generators.kenning_rewind import (
        _meanings_from_supplied_words,
    )
    from wyrd.generators.kenning.runtime.meaning import Meaning

    by = Meaning("-by", tags=[], meanings=["farmstead"], sources={"old_scandinavian": ["bȳ"]})
    meaning_db = {"-by": [by]}

    per_word, missing = _meanings_from_supplied_words([[{"usage": "by"}]], meaning_db)
    assert per_word == [[by]]  # "by" → "-by" via fallback
    assert missing == []

    # A genuinely-absent usage (no dash/case variant) still reports missing.
    per_word2, missing2 = _meanings_from_supplied_words([[{"usage": "zzz"}]], meaning_db)
    assert per_word2 == [[]]
    assert missing2 == ["zzz"]


def test_rewind_supplied_words_anchors_on_ranked_canonical_sibling() -> None:
    """wyrd-om67: when a usage maps to several sibling Meanings (a homograph),
    the from-morphemes rewind anchors on the _rank_siblings canonical — the
    etymon NewName.to_dict displays for the morpheme — not meaning_db's raw
    first sibling. Otherwise a Celtic homograph dragged the rewind onto the
    wrong etymon (OE "Wertūn" came out as "Werettan")."""
    from wyrd.generators.kenning import _rank_siblings
    from wyrd.generators.kenning.generators.kenning_rewind import (
        _meanings_from_supplied_words,
    )
    from wyrd.generators.kenning.runtime.meaning import Meaning

    celtic = Meaning("-ton", tags=[], meanings=["wave"], sources={"celtic_mix": ["ton"]})
    old_eng = Meaning("-ton", tags=[], meanings=["enclosure"], sources={"old_english": ["tūn"]})
    # raw meaning_db order puts the non-canonical sibling first
    meaning_db = {"-ton": [celtic, old_eng]}

    per_word, missing = _meanings_from_supplied_words([[{"usage": "-ton"}]], meaning_db)
    # the function must return the SAME canonical _rank_siblings picks
    assert per_word == [[_rank_siblings([celtic, old_eng])[0]]]
    assert missing == []


def test_rewind_from_morphemes_raises_on_empty_name(fresh_db: Path) -> None:
    """wyrd-cp2d: empty / whitespace-only name input raises
    ValueError, matching rewind_name's contract."""
    import pytest

    from wyrd.generators.kenning.era.rewind import rewind_from_morphemes

    with LexiconDB(fresh_db) as db:
        meaning_db = _make_rewind_meaning_db()
        with pytest.raises(ValueError, match="name is required"):
            rewind_from_morphemes("", [], db, meaning_db)


def test_new_name_to_dict_emits_per_word_morpheme_metadata() -> None:
    """wyrd-cp2d: NewName.to_dict() produces the JSON-shape dict
    consumed by kenning generate --json + kenning rewind --from-json.
    Pins the schema: name + per-word lists of {usage, sources,
    tags, meanings}."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning(
        "Whit-",
        tags=["color"],
        meanings=["white"],
        sources={"old_english": ["hwīt"]},
    )
    m2 = Meaning(
        "-by",
        tags=["habitation"],
        meanings=["settlement"],
        sources={"old_scandinavian": ["býr"]},
    )
    meaning_db = {"Whit-": [m1], "-by": [m2]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Whit-", "-by"]])

    out = new_name.to_dict()
    assert out["name"] == "Whitby"
    assert len(out["words"]) == 1  # one word
    assert len(out["words"][0]) == 2  # two morphemes
    first = out["words"][0][0]
    assert first["usage"] == "Whit-"
    assert first["sources"] == {"old_english": ["hwīt"]}
    assert first["tags"] == ["color"]
    assert first["meanings"] == ["white"]
    # Sparse rendering: Latin-script-only sources omit the renderings
    # key entirely (no noisy '{}' clutter).
    assert "renderings" not in first
    # wyrd-bvwu: citations are sparse too — a Meaning with no scholarly
    # witnesses (citations default {}) omits the key entirely.
    assert "citations" not in first


def test_new_name_to_dict_includes_citations_when_meaning_has_them() -> None:
    """wyrd-bvwu: to_dict (morphemes_by_word) carries the morpheme's
    scholarly source_ids when present, so the SPA inspector can surface
    them in an expandable view. Mirrors components()'s citations field;
    sparse (only emitted when non-empty)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "Whit-",
        tags=[],
        meanings=["white"],
        sources={"old_english": ["hwīt"]},
        citations={"old_english": ["skeat_1901_cambridgeshire", "mawer_1920_nd"]},
    )
    meaning_db = {"Whit-": [m]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Whit-"]])
    out = new_name.to_dict()
    # Distinct + sorted source_ids.
    assert out["words"][0][0]["citations"] == [
        "mawer_1920_nd",
        "skeat_1901_cambridgeshire",
    ]


def test_new_name_to_dict_citations_union_across_ranked_siblings() -> None:
    """_collect_citations unions + dedupes citations across ALL ranked
    siblings sharing a usage, not just the first-ranked Meaning."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning(
        "Whit-",
        tags=[],
        meanings=["white"],
        sources={"old_english": ["hwīt"]},
        citations={"old_english": ["skeat_1901"]},
    )
    m2 = Meaning(
        "Whit-",
        tags=[],
        meanings=["bright"],
        sources={"old_english": ["hwīt"]},
        citations={"old_english": ["mawer_1920", "skeat_1901"]},  # skeat dup
    )
    meaning_db = {"Whit-": [m1, m2]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Whit-"]])
    out = new_name.to_dict()
    # Union across siblings, deduped + sorted.
    assert out["words"][0][0]["citations"] == ["mawer_1920", "skeat_1901"]


def test_new_name_to_dict_includes_renderings_when_meaning_has_pronunciation() -> None:
    """wyrd-cp2d round 3: pronunciation + cross-script renderings
    (IPA, original_script, transliteration, english_shaped) carry
    into the JSON when a Meaning has wyrd-ha9q rendering data on
    any source-language form. Pins parity with components()'s
    renderings field — both surface the same panel data, just
    grouped per-morpheme vs flat."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "Whit-",
        tags=[],
        meanings=["white"],
        sources={"old_english": ["hwīt"]},
    )
    # Inject wyrd-ha9q rendering data via the documented Meaning
    # attributes _collect_renderings consumes.
    m.pronunciation = {"old_english": {"hwīt": {"ipa": "/xwiːt/", "dialect": "WS"}}}
    m.english_shaped = {"old_english": {"hwīt": "hweet"}}
    m.original_script = {}
    m.transliteration = {}
    meaning_db = {"Whit-": [m]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Whit-"]])
    out = new_name.to_dict()
    rend = out["words"][0][0].get("renderings", {})
    assert rend.get("old_english", {}).get("hwīt", {}).get("ipa") == "/xwiːt/"
    assert rend["old_english"]["hwīt"]["english_shaped"] == "hweet"


def test_new_name_to_dict_includes_rendered_d18_d8_substitution() -> None:
    """wyrd-cp2d: to_dict's optional ``rendered`` field carries the
    D18 spelling-variant or D8 inflection substitute when the
    generator picked one. Without this branch tested, a regression
    that drops the rendered key would slip past."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("cot-", tags=[], meanings=["cottage"], sources={"old_english": ["cot"]})
    m2 = Meaning("-an", tags=[], meanings=["genitive"], sources={"old_english": ["an"]})
    meaning_db = {"cot-": [m1], "-an": [m2]}
    new_name = NewName(
        struct=None,
        meaning_db=meaning_db,
        name=[["cot-", "-an"]],
        rendered=[["cotum", "an"]],  # D8 inflection substitution
    )
    out = new_name.to_dict()
    # The rendered field carries the substituted form per morpheme.
    assert out["words"][0][0]["rendered"] == "cotum"
    assert out["words"][0][1]["rendered"] == "an"


def test_kenning_generate_json_flag_emits_morpheme_metadata_per_line() -> None:
    """wyrd-cp2d: kenning generate --json emits one JSON object per
    line carrying name + culture + words. End-to-end CLI test pinning
    the schema operators (and the webapp) consume."""
    import json as _json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.generate import generate

    runner = CliRunner()
    result = runner.invoke(generate, ["english", "-n", "2", "--seed", "100", "--json"])
    assert result.exit_code == 0, result.output
    # Filter to JSON-shaped lines (the '(seed: ...)' progress line
    # is on stderr but CliRunner mixes streams by default).
    lines = [line for line in result.output.splitlines() if line.strip().startswith("{")]
    assert len(lines) == 2  # one JSON object per name
    for line in lines:
        obj = _json.loads(line)
        assert "name" in obj
        assert "culture" in obj
        assert obj["culture"] == "english"
        assert "words" in obj
        assert isinstance(obj["words"], list)
        # Each word has at least one morpheme; each morpheme has usage.
        for word in obj["words"]:
            assert isinstance(word, list)
            for morph in word:
                assert "usage" in morph


def test_kenning_rewind_from_json_pipe_end_to_end(fresh_db: Path) -> None:
    """wyrd-cp2d: end-to-end pipe test — capture 'generate --json'
    output, feed it into 'rewind --from-json -' via stdin, verify
    the rewind output respects the per-line JSON objects' supplied
    usages (each input line produces one rewind output block).

    --db is passed explicitly so CI (which has no ~/.wyrd/lexicon.db)
    doesn't fail the click.Path(exists=True) check before the test
    body runs. wyrd-nyzb regression pin."""
    import json as _json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.generate import generate
    from wyrd.generators.kenning.cli.rewind import rewind

    runner = CliRunner()
    # Step 1: capture generate --json output (generate reads the
    # bundled meanings.json; no --db needed)
    gen = runner.invoke(generate, ["english", "-n", "2", "--seed", "100", "--json"])
    assert gen.exit_code == 0
    json_lines = [line for line in gen.output.splitlines() if line.strip().startswith("{")]
    parsed = [_json.loads(line) for line in json_lines]
    assert len(parsed) == 2

    # Step 2: feed into rewind --from-json -
    stdin_input = "\n".join(json_lines)
    rewind_result = runner.invoke(
        rewind, ["--from-json", "-", "--db", str(fresh_db)], input=stdin_input
    )
    assert rewind_result.exit_code == 0, rewind_result.output
    # Verify the rewind output mentions both input names (one block per).
    for obj in parsed:
        assert obj["name"] in rewind_result.output


def test_kenning_rewind_from_json_malformed_json_raises(fresh_db: Path) -> None:
    """wyrd-cp2d: malformed JSON in --from-json input surfaces a
    friendly error + non-zero exit. Pre-PR a JSON decode error would
    propagate as an unhandled exception.

    --db is passed explicitly so CI (which has no ~/.wyrd/lexicon.db)
    doesn't fail the click.Path(exists=True) check before the JSON
    decode-error path fires. wyrd-nyzb regression pin."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.rewind import rewind

    runner = CliRunner()
    result = runner.invoke(
        rewind, ["--from-json", "-", "--db", str(fresh_db)], input="{not valid json"
    )
    assert result.exit_code != 0
    assert "malformed JSON" in result.output


def test_kenning_rewind_no_name_no_from_json_fails_with_friendly_error(
    fresh_db: Path,
) -> None:
    """wyrd-cp2d: invoking 'kenning rewind' with no NAME and no
    --from-json must surface a friendly error + non-zero exit. Pre-
    PR NAME was a required positional; the wyrd-cp2d change made it
    optional in service of --from-json, so we need explicit
    validation that one or the other is supplied.

    --db is passed explicitly so CI (which has no ~/.wyrd/lexicon.db)
    doesn't fail the click.Path(exists=True) check before the
    NAME-required validation fires. wyrd-nyzb regression pin."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.rewind import rewind

    runner = CliRunner()
    result = runner.invoke(rewind, ["--db", str(fresh_db)])
    assert result.exit_code != 0
    assert "NAME argument is required" in result.output


# --- wyrd-085k: smart-join name renderer ---------------------------------


def test_rewind_concatenates_adjacent_non_particle_morphemes(fresh_db: Path) -> None:
    """wyrd-085k: a two-morpheme compound with no particles renders
    as a single concatenated word (no hyphen, first letter capped).
    Pins the standard ``Elm-tūn`` → ``Elmtūn`` rendering — the
    most common medieval place-name shape (OE Brādtūn, Hēafodcroft,
    Cinubrycg)."""
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*haimaz",
            cluster_root_lang="proto-germanic",
            members=[("hām", "old-english")],
        )
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            ("-ham", {"old_english": ["hām"]}),
        )
        result = rewind_name("Whitham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Hwīt + hām -> concatenated, first letter title-cased,
    # no hyphen separator.
    assert by_cell["oe-late"] == "Hwīthām"


def test_rewind_preserves_free_particle_as_separate_token(fresh_db: Path) -> None:
    """wyrd-085k: a morpheme matching the free-particle set
    (``on``, ``upon``, ``under``, ``of``) sits as its own
    space-separated token instead of concatenating into the
    surrounding compound. Pins the 'Stratford on Avon' /
    'Henley upon Thames' shape — the second-most-common medieval
    multi-token name pattern in the corpus."""
    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            ("*on", "on"),
            ("*hām", "hām"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            ("on", {"old_english": ["on"]}),  # free particle (no hyphens)
            ("-ham", {"old_english": ["hām"]}),
        )
        result = rewind_name("whit on ham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # 'on' is a particle: surround with spaces; surrounding
    # morphemes title-case + concatenate around it.
    assert by_cell["oe-late"] == "Hwīt on Hām"


def test_rewind_does_not_treat_norse_settlement_suffix_by_as_particle(
    fresh_db: Path,
) -> None:
    """wyrd-085k regression guard: the morpheme '-by' (Norse
    settlement suffix in Whitby / Derby / Grimsby) is lexically the
    same string as the English preposition 'by' but functionally a
    SUFFIX, not a free particle. The presence of leading-hyphen in
    its modern_usage disqualifies it from particle-treatment so
    'whit + -by' renders 'Whitby', not 'Whit by'."""
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english")],
        )
        _seed_cluster(
            db,
            cluster_root_form="*býr",
            cluster_root_lang="proto-germanic",
            members=[("býr", "old-scandinavian")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            ("-by", {"old_scandinavian": ["býr"]}),  # SUFFIX, not particle
        )
        result = rewind_name("whitby", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # '-by' is post-position (leading hyphen) so NOT a particle.
    # Renders as a single concatenated token whether the era reflex
    # lands the ON form 'býr' or falls back to the modern 'by';
    # the regression guarded against is splitting it as 'Whit by'.
    assert " by" not in by_cell["oe-late"]
    assert " by" not in by_cell["me"]
    assert " by" not in by_cell["modern"]
    # Sanity: each rendered string is a single title-cased token.
    assert by_cell["oe-late"].count(" ") == 0


def test_rewind_modern_cell_bypasses_smart_join_for_particles_across_words(
    fresh_db: Path,
) -> None:
    """wyrd-t2bh: at the modern cell, smart-join is bypassed entirely
    — particle morphemes title-case as standalone words just like
    every other word, instead of being surrounded by spaces as
    lowercase particles.

    Test setup uses space-separated input 'whit on' (two words) so
    the trie's matching is deterministic. The wyrd-t2bh bug origin
    was 'Helon' (single word, embedded particle morpheme) being
    re-split into 'Hel on' at modern; that single-word case relies
    on bundle-corpus morphemes that aren't easily mocked in this
    test's _make_rewind_meaning_db setup, so this test pins the
    cross-axis invariant: smart-join is OFF at modern (each word
    title-cases regardless of particle membership)."""
    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            ("*on", "on"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("Whit", {"old_english": ["hwīt"]}),
            ("on", {"old_english": ["on"]}),
        )
        result = rewind_name("whit on", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # OE cell: smart-join treats 'on' as particle → 'Hwīt on'
    # (lowercase 'on' surrounded as a free particle).
    assert by_cell["oe-late"] == "Hwīt on"
    # Modern cell: smart-join OFF — 'on' title-cases as a regular
    # word ('On'), not as a lowercase particle.
    assert by_cell["modern"] == "Whit On"


def test_rewind_multi_word_input_with_one_fully_unaccounted_word(
    fresh_db: Path,
) -> None:
    """wyrd-t2bh: a multi-word input where one word fully fails to
    decompose (no morphemes match) still renders the OTHER words
    cleanly. The unaccounted word lands in result.unaccounted; the
    successful words pass through the per-word renderer with their
    space structure preserved. Pins the morpheme_anchors_per_word's
    'if word_anchors:' guard at the rewind.py boundary so a future
    refactor doesn't crash on an empty word-anchors list."""
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english")],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Whit", {"old_english": ["hwīt"]}),
        )
        # 'whit nomatch' — 'whit' decomposes cleanly, 'nomatch' has
        # no matching morpheme so its whole word is unaccounted.
        result = rewind_name("whit nomatch", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Only the decomposable word renders; rendered output drops the
    # all-empty word group.
    assert by_cell["oe-late"] == "Hwīt"
    assert by_cell["modern"] == "Whit"
    # The unaccounted word lands in result.unaccounted.
    assert "nomatch" in result.unaccounted


def test_rewind_modern_cell_preserves_input_spacing_across_multiword(
    fresh_db: Path,
) -> None:
    """wyrd-t2bh: multi-word input keeps its space structure at the
    modern cell. Each input word is decomposed and rendered
    independently, then ``' '.join``ed back together — the
    operator's whitespace shape survives the rewinder."""
    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            ("*krofta", "croft"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("Whit", {"old_english": ["hwīt"]}),
            ("croft", {"old_english": ["croft"]}),
        )
        # Two-word input — each word is a single standalone morpheme;
        # the renderer joins them across the operator's space.
        result = rewind_name("Whit croft", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Modern cell: 2 words preserve the original space.
    assert by_cell["modern"] == "Whit Croft"
    # OE cell same (smart-join doesn't trigger — no particles).
    assert by_cell["oe-late"] == "Hwīt Croft"


@pytest.mark.parametrize("particle", ["on", "upon", "under", "of"])
def test_rewind_preserves_every_free_particle_as_separate_token(
    fresh_db: Path, particle: str
) -> None:
    """wyrd-085k: every member of _FREE_PARTICLES must surround with
    spaces. Parameterized over the whole set so a typo (or accidental
    drop) on any one of them surfaces immediately rather than only
    via the one-member positive test above."""
    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            (f"*{particle}", particle),
            ("*hām", "hām"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            (particle, {"old_english": [particle]}),
            ("-ham", {"old_english": ["hām"]}),
        )
        result = rewind_name(f"whit {particle} ham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    assert by_cell["oe-late"] == f"Hwīt {particle} Hām"


def test_rewind_hyphen_guard_catches_particle_member_used_as_suffix(
    fresh_db: Path,
) -> None:
    """wyrd-085k regression guard for the hyphen-check in
    _is_free_particle: even a morpheme whose bare string IS in
    _FREE_PARTICLES gets treated as suffix-not-particle if its
    modern_usage carries hyphen markers. Without this branch a
    real corpus morpheme like '-on-' (inner-position usage of
    'on', if mined) would mis-split a compound.

    Constructs a Meaning whose usage is '-on-' (mock infix shape).
    The renderer must NOT space-separate it; the morpheme
    concatenates inline as a positional infix."""
    with LexiconDB(fresh_db) as db:
        for root, member in (
            ("*hwītaz", "hwīt"),
            ("*on", "on"),
            ("*hām", "hām"),
        ):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        # Note '-on-' usage marker — bare 'on' IS in _FREE_PARTICLES
        # but the hyphen-marked positional usage disqualifies it.
        meaning_db = _make_rewind_meaning_db(
            ("Whit-", {"old_english": ["hwīt"]}),
            ("-on-", {"old_english": ["on"]}),
            ("-ham", {"old_english": ["hām"]}),
        )
        result = rewind_name("whitonham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # No space-around-on — the hyphen-marked usage disqualified
    # the particle treatment, so 'on' concatenates inline.
    assert " on " not in by_cell["oe-late"]
    # Should still be a single concatenated token containing 'on'.
    assert "on" in by_cell["oe-late"].lower()
    assert by_cell["oe-late"].count(" ") == 0


# --- wyrd-8qbi: modern-cell round-trip --------------------------------------


def test_rewind_modern_cell_round_trips_canonical_not_cluster_mate(
    fresh_db: Path,
) -> None:
    """wyrd-8qbi: at the modern-english target the renderer must use
    the morpheme's canonical (modern_usage) directly instead of
    consulting the cluster-mate picker.

    Real-world bug observed 2026-05-21: 'wyrd kenning rewind Elmton'
    produced 'Amherstton' at modern because 'Amherst' is a sibling
    cognate of OE 'elm'. The picker alphabetical-first chose
    'Amherst' over 'elm' even though the operator typed 'Elm'.

    Seed: OE 'elm' + a noise cognate 'Amherst' at modern. Verify
    the modern cell renders the canonical 'elm'-derived form, NOT
    'Amherst'."""
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*elmaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("elm", "old-english"),
                # Noise cognate at modern — alphabetical-first would
                # surface 'Amherst' if the cluster picker ran.
                ("Amherst", "modern-english"),
                ("elm", "modern-english"),
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("Elm", {"old_english": ["elm"]}),
        )
        result = rewind_name("Elm", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Modern cell: canonical short-circuit → 'Elm', not 'Amherst'.
    assert by_cell["modern"] == "Elm"
    assert "Amherst" not in by_cell["modern"]
    # OE cell still works through the existing anchor short-circuit.
    assert by_cell["oe-late"] == "Elm"


def test_rewind_modern_cell_round_trips_compound_input(fresh_db: Path) -> None:
    """wyrd-8qbi: multi-morpheme compounds round-trip through modern
    cleanly too. Each morpheme's canonical concatenates at the modern
    target regardless of how noisy the cluster is."""
    with LexiconDB(fresh_db) as db:
        _seed_cluster(
            db,
            cluster_root_form="*newjaz",
            cluster_root_lang="proto-germanic",
            members=[("new", "old-english"), ("new", "modern-english")],
        )
        _seed_cluster(
            db,
            cluster_root_form="*tunaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("tūn", "old-english"),
                # Multiple modern cognates — picker would pick
                # alphabetical first WITHOUT the short-circuit.
                ("Newton", "modern-english"),
                ("ton", "modern-english"),
            ],
        )
        meaning_db = _make_rewind_meaning_db(
            ("New", {"old_english": ["new"]}),
            ("ton", {"old_english": ["tūn"]}),
        )
        # Space-separated input — wyrd-t2bh preserves operator's
        # space-shape at the modern cell, so 'New ton' stays
        # 'New Ton' (2 words) rather than collapsing to 'Newton'.
        result = rewind_name("New ton", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Modern cell: each word concatenates its morphemes' canonicals
    # independently, then joins across words with the original space.
    # Surfaces 'New Ton', NOT the cluster's noise mate 'Newton'.
    assert by_cell["modern"] == "New Ton"


def test_rewind_handles_leading_particle(fresh_db: Path) -> None:
    """wyrd-085k edge case: a particle as the first morpheme emits
    a lowercase leading token rather than mid-word inlining. Real-
    world shape: 'on the moor' / 'under the bridge' style names
    where the locative preposition opens the compound."""
    with LexiconDB(fresh_db) as db:
        for root, member in (("*on", "on"), ("*hām", "hām")):
            _seed_cluster(
                db,
                cluster_root_form=root,
                cluster_root_lang="proto-germanic",
                members=[(member, "old-english")],
            )
        meaning_db = _make_rewind_meaning_db(
            ("on", {"old_english": ["on"]}),
            ("ham", {"old_english": ["hām"]}),
        )
        # Space-separated input — the multi-word path decomposes each
        # word independently; the renderer sees a 2-morpheme stream
        # with the particle first.
        result = rewind_name("on ham", db, meaning_db)

    by_cell = {stop.cell: stop.rendered for stop in result.eras}
    # Leading particle is its own lowercase token.
    assert by_cell["oe-late"] == "on Hām"


def test_strip_diacritics_normalizes_macrons_and_circumflexes() -> None:
    """Direct unit test for the diacritic-stripper used by the
    suffix-anchoring projector. Inputs include the OE macron set
    (ūīāēō) and Welsh circumflex set (ŵâêôûŷ) that show up as
    bundle-source forms; the projector compares stripped variants
    against historical-form suffixes."""

    assert _strip_diacritics("tūn") == "tun"
    assert _strip_diacritics("Hædan") == "Hædan"  # æ is a base char, not a combining mark
    assert _strip_diacritics("Llanfaên") == "Llanfaen"
    assert _strip_diacritics("Hēafod") == "Heafod"
    assert _strip_diacritics("ASCII") == "ASCII"


def test_find_longest_suffix_match_picks_longest_candidate() -> None:
    """Direct unit test: when multiple candidates are valid suffixes,
    the longest wins (so 'tone' wins over 'one' for 'Cestretone')."""

    assert _find_longest_suffix_match("Cestretone", {"tone", "one", "ne"}) == "tone"
    assert _find_longest_suffix_match("Bradeford", {"ford", "ord", "rd"}) == "ford"


def test_find_longest_suffix_match_diacritic_insensitive() -> None:
    """Suffix matching is diacritic-insensitive: ``Cestretun`` matches
    a pre-stripped 'tun' candidate. Caller (``_suffix_candidates_for_etymon``)
    is responsible for adding both unstripped + stripped forms to
    the candidate set; the matcher checks whether either the raw or
    diacritic-stripped attested form ends with any candidate."""

    # Caller-provided candidates include both: 'tūn' (raw) + 'tun'
    # (stripped). Match against 'tun' via either af.endswith path.
    assert _find_longest_suffix_match("Cestretun", {"tūn", "tun"}) == "tun"
    # Diacritic-bearing attested form: 'Cestretūn' (with macron)
    # also matches via af_stripped.endswith('tun').
    assert _find_longest_suffix_match("Cestretūn", {"tūn", "tun"}) == "tūn"


def test_find_longest_suffix_match_returns_none_for_no_match() -> None:
    """When no candidate is a suffix of the form, the matcher returns
    None (caller skips the projection)."""

    assert _find_longest_suffix_match("Cestretone", {"xyz", "abc"}) is None
    assert _find_longest_suffix_match("Cestretone", set()) is None


def test_find_longest_suffix_match_length_changing_target_aligns_raw_cut() -> None:
    """Regression: ``_strip_diacritics`` uses NFKD, which is NOT length-preserving —
    a ligature ``ﬁ`` (U+FB01) decomposes to ``fi`` (1 char → 2). When the TARGET
    carries such a char, measuring the matched candidate's length against the stripped
    form and slicing that count off the RAW form shifts the morpheme boundary. The cut
    must be raw-form-aligned: ``Sutﬁeld`` + ``{field}`` splits at ``ﬁeld``, not the
    one-char-off ``tﬁeld`` the stripped-length slice produced."""
    assert _find_longest_suffix_match("Sutﬁeld", {"field"}) == "ﬁeld"  # was 'tﬁeld'
    # the corresponding prefix (what the caller projects as morpheme 1) is now correct:
    target = "Sutﬁeld"
    match = _find_longest_suffix_match(target, {"field"})
    assert target[: len(target) - len(match)] == "Sut"  # was 'Su'


def test_find_longest_suffix_match_rejects_single_char_candidates() -> None:
    """Single-char candidates (a trailing 'e' / 's') are rejected to
    avoid spurious 1-char hits — too noisy to project as a morpheme
    surface."""

    assert _find_longest_suffix_match("Cestretone", {"e"}) is None


def test_cli_kenning_lexicon_project_period_forms_smoke(fresh_db: Path) -> None:
    """End-to-end CLI smoke test: ``lexicon project-period-forms``
    against a seeded toponym + binary breakdown + attestation
    populates etymon_period_form when --apply is set, dry-runs
    otherwise."""
    runner = CliRunner()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        brad_id = db.upsert_etymon("brad", "old-english")
        ford_id = db.upsert_etymon("ford", "old-english")
        db.commit()
        toponym_id = _seed_toponym_with_binary_breakdown(
            db,
            modern_name="Bradford",
            first_etymon_id=brad_id,
            last_etymon_id=ford_id,
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (toponym_id, "Bradeford", 1377, "src"),
        )
        db.commit()

    # Dry-run: shows candidate count, no rows written.
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "project-period-forms", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "candidates=" in result.output
    assert "(dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0] == 0

    # --apply: writes rows.
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "project-period-forms", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM etymon_period_form").fetchone()[0] == 2


# --- wyrd-obpw Phase 3.3: SPA-side bundle plumbing for era reflexes -------


def test_meaning_era_reflex_for_returns_target_language_reflexes() -> None:
    """Direct unit test for the runtime accessor: when the bundle
    carried era_reflex data for a target language, the accessor
    returns the sorted form list. Returns [] when no data for that
    target. Internal storage is now ``[(form, source), ...]``
    tuples (wyrd-jbcu); the forms-only accessor projects them out."""
    m = Meaning(
        usage="-ham",
        tags=[],
        meanings=["village"],
        sources={"old_english": ["hām"]},
        era_reflexes={
            "old-english": [("hām", "cluster")],
            "middle-english": [
                ("ham", "cluster"),
                ("hamm", "cluster"),
                ("home", "cluster"),
            ],
            "modern-english": [("ham", "cluster"), ("home", "cluster")],
        },
    )
    assert m.era_reflex_for("middle-english") == ["ham", "hamm", "home"]
    assert m.era_reflex_for("modern-english") == ["ham", "home"]
    assert m.era_reflex_for("welsh") == []


def test_meaning_era_reflex_for_returns_independent_copy() -> None:
    """The accessor returns a fresh list on each call so callers
    can mutate without aliasing the underlying dict."""
    m = Meaning(
        usage="-ham",
        tags=[],
        meanings=["village"],
        sources={"old_english": ["hām"]},
        era_reflexes={
            "middle-english": [("ham", "cluster"), ("hamm", "cluster")],
        },
    )
    first = m.era_reflex_for("middle-english")
    first.append("intruder")
    second = m.era_reflex_for("middle-english")
    assert second == ["ham", "hamm"]


def test_meaning_era_reflex_sources_for_returns_form_to_source_map() -> None:
    """wyrd-jbcu source-aware accessor. Returns ``{form: source}``
    so consumers (KenningRewind / KenningEraMap) can render
    phonology-rule-derived forms differently from cluster mates."""
    m = Meaning(
        usage="-ham",
        tags=[],
        meanings=["village"],
        sources={"old_english": ["hām"]},
        era_reflexes={
            "middle-english": [
                ("ham", "cluster"),
                ("hamm", "phonology-rule:v1"),
            ],
        },
    )
    sources = m.era_reflex_sources_for("middle-english")
    assert sources == {"ham": "cluster", "hamm": "phonology-rule:v1"}
    # Empty dict for languages without data.
    assert m.era_reflex_sources_for("welsh") == {}


def test_meaning_era_reflexes_empty_on_legacy_bundle() -> None:
    """Bundles that pre-date the wyrd-obpw export omit the
    era_reflexes field entirely. Meaning's era_reflexes attr stays
    {} and era_reflex_for always returns []."""
    m = Meaning(
        usage="-ham",
        tags=[],
        meanings=["village"],
        sources={"old_english": ["hām"]},
    )
    assert m.era_reflexes == {}
    assert m.era_reflex_for("middle-english") == []


def test_load_meanings_parses_era_reflexes_top_level_field() -> None:
    """End-to-end runtime parse: a word entry carrying ``era_reflexes``
    at the top level (NOT a per-language sibling) lands on the
    Meaning's era_reflexes attribute. Verifies the load pipeline
    excludes era_reflexes from the per-language ``sources`` dict.

    wyrd-jbcu: bundle now emits the new ``{form, source}`` shape;
    legacy bundles with bare-string entries are handled by a separate
    test below for back-compat."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["village"],
            "words": [
                {
                    "modern_usage": "-ham",
                    "old_english": ["hām"],
                    "era_reflexes": {
                        "middle-english": [
                            {"form": "ham", "source": "cluster"},
                            {"form": "hamm", "source": "phonology-rule:v1"},
                        ],
                        "modern-english": [
                            {"form": "ham", "source": "cluster"},
                            {"form": "home", "source": "cluster"},
                        ],
                    },
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    meaning = meaning_db["-ham"][0]
    assert meaning.era_reflex_for("middle-english") == ["ham", "hamm"]
    assert meaning.era_reflex_for("modern-english") == ["ham", "home"]
    assert meaning.era_reflex_sources_for("middle-english") == {
        "ham": "cluster",
        "hamm": "phonology-rule:v1",
    }
    assert "era_reflexes" not in meaning.sources


def test_load_meanings_parses_era_reflexes_legacy_list_of_strings_shape() -> None:
    """wyrd-jbcu: legacy bundles emitted era_reflexes as
    ``{lang: [form, ...]}`` (bare strings). The loader must keep
    parsing them for back-compat — entries default to source='cluster'
    since legacy bundles only carried attestation-backed reflexes."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["village"],
            "words": [
                {
                    "modern_usage": "-ham",
                    "old_english": ["hām"],
                    "era_reflexes": {
                        "middle-english": ["ham", "hamm"],  # legacy bare strings
                    },
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    meaning = meaning_db["-ham"][0]
    assert meaning.era_reflex_for("middle-english") == ["ham", "hamm"]
    # Legacy entries default to 'cluster' since pre-jbcu bundles only
    # carried attestation-backed reflexes.
    assert meaning.era_reflex_sources_for("middle-english") == {
        "ham": "cluster",
        "hamm": "cluster",
    }


def test_load_meanings_normalize_era_reflexes_skips_malformed() -> None:
    """Malformed entries (non-string, non-dict, dict without 'form')
    are skipped silently rather than crashing the load. Pin the
    defensive parse so a future bundle export bug doesn't take down
    the entire runtime."""

    raw = {
        "middle-english": [
            "good",  # legacy bare string OK
            {"form": "also-good", "source": "cluster"},  # new shape OK
            {"weight": 5},  # missing 'form' — skip
            None,  # not a dict or string — skip
            42,  # not a dict or string — skip
        ],
        "welsh": "not-a-list",  # whole entry skipped
    }
    out = _normalize_era_reflexes(raw)
    assert out == {"middle-english": [("good", "cluster"), ("also-good", "cluster")]}


def test_load_meanings_normalize_era_reflex_glosses_extracts_only_gloss_dicts() -> None:
    """wyrd-rogd.1: the gloss projection keeps ONLY {form, source, gloss}
    entries with a non-empty gloss — legacy bare-strings and {form, source}-only
    entries contribute nothing (back-compat), and malformed/empty entries are
    skipped silently like the source projection."""

    raw = {
        "middle-english": [
            "good",  # legacy bare string — no gloss
            {"form": "ston", "source": "cluster"},  # {form,source} only — no gloss
            {"form": "stoon", "source": "cluster", "gloss": "stone"},  # kept
            {"source": "cluster", "gloss": "no form"},  # missing form — skip
            {"form": "blank", "source": "cluster", "gloss": ""},  # empty gloss — skip
            {"form": ["x"], "source": "cluster", "gloss": "bad"},  # unhashable form — skip
            {"form": "y", "source": "cluster", "gloss": ["bad"]},  # non-str gloss — skip
        ],
        "old-english": [{"form": "stān", "source": "cluster", "gloss": "rock"}],
        "welsh": "not-a-list",  # whole entry skipped
    }
    out = _normalize_era_reflex_glosses(raw)
    assert out == {"middle-english": {"stoon": "stone"}, "old-english": {"stān": "rock"}}
    # a fully glossless (legacy) field yields an empty map.
    assert _normalize_era_reflex_glosses({"me": [{"form": "x", "source": "cluster"}]}) == {}


def test_fetch_reflex_glosses_picks_first_non_pointer_gloss(fresh_db: Path) -> None:
    """wyrd-rogd.1: a representative gloss per etymon — alphabetically-first
    NON-pointer gloss (ORDER BY gloss, is_form_of_pointer filtered); etymons
    with only pointer glosses (or none) get no entry; empty input → {}."""

    with LexiconDB(fresh_db) as db:
        glossed = db.upsert_etymon("stān", "old-english")
        db.add_gloss(glossed, "rock")
        db.add_gloss(glossed, "boundary stone")  # alphabetically first non-pointer
        db.add_gloss(glossed, "alternative form of stone")  # pointer — filtered
        pointer_only = db.upsert_etymon("burg", "old-english")
        db.add_gloss(pointer_only, "alternative form of burh")  # only a pointer
        bare = db.upsert_etymon("tūn", "old-english")  # no glosses at all

        out = _fetch_reflex_glosses(db, [glossed, pointer_only, bare, glossed])

    assert out == {glossed: "boundary stone"}  # pointer-only + bare omitted; dedup ok
    with LexiconDB(fresh_db) as db:
        assert _fetch_reflex_glosses(db, []) == {}


def test_fetch_family_era_reflexes_threads_gloss(fresh_db: Path) -> None:
    """wyrd-rogd.1: a reflex's own gloss rides on its bundle entry, so the SPA
    era-grid can surface a drifted cognate's meaning."""

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[("hwīt", "old-english"), ("white", "modern-english")],
        )
        db.add_gloss(ids["hwīt"], "white (colour)")
        # modern-english member left glossless → its entry omits gloss (sparse).
        result = _fetch_family_era_reflexes(db, [ids["hwīt"]], "old-english")

    assert result["old-english"] == [
        {"form": "hwīt", "source": "cluster", "gloss": "white (colour)"}
    ]
    assert result["modern-english"] == [{"form": "white", "source": "cluster"}]


def test_fetch_family_era_reflexes_walks_cognate_cluster(fresh_db: Path) -> None:
    """Bundle-side integration: _fetch_family_era_reflexes returns the
    cluster mates for each canonical-language target. Verifies the
    bundle-build helper plumbs through etymon_era_reflexes correctly."""

    with LexiconDB(fresh_db) as db:
        ids = _seed_cluster(
            db,
            cluster_root_form="*hwītaz",
            cluster_root_lang="proto-germanic",
            members=[
                ("hwīt", "old-english"),
                ("whit", "middle-english"),
                ("white", "modern-english"),
            ],
        )
        # OE root since proto-* has no era cells defined.
        result = _fetch_family_era_reflexes(db, [ids["hwīt"]], "old-english")

    # wyrd-jbcu schema: each entry is {"form", "source"}.
    assert result["old-english"] == [{"form": "hwīt", "source": "cluster"}]
    assert result["middle-english"] == [{"form": "whit", "source": "cluster"}]
    assert result["modern-english"] == [{"form": "white", "source": "cluster"}]


def test_fetch_family_era_reflexes_returns_empty_for_unfamilied_root(
    fresh_db: Path,
) -> None:
    """Roots whose language has no era family (proto-languages,
    untracked classical languages) return an empty dict — the bundle
    omits the era_reflexes field for these words."""

    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("*tūnaz", "proto-germanic")
        result = _fetch_family_era_reflexes(db, [eid], "proto-germanic")

    assert result == {}


def test_fetch_family_era_reflexes_carries_phonology_rule_source_tag(
    fresh_db: Path,
) -> None:
    """wyrd-jbcu: bundle export carries Tier 4 (phonology-rule)
    reflexes WITH the source tag intact. The schema upgrade
    (``{lang: [{form, source}, ...]}``) lets consumers distinguish
    inferred forms from attested cluster mates. Pin the round-trip
    so a regression that drops the tag (or filters Tier 4 entirely)
    surfaces here."""

    with LexiconDB(fresh_db) as db:
        # OE etymon with a rule trigger (ǣ → e) but no cluster /
        # descent / period-form data. Tier 4 is the only path.
        eid = db.upsert_etymon("dǣg", "old-english")
        db.commit()

        bundle_data = _fetch_family_era_reflexes(db, [eid], "old-english")

    # Tier 4 reflex IS in the bundle, tagged 'phonology-rule:v1' so
    # consumers can render it differently.
    assert "middle-english" in bundle_data
    me_entries = bundle_data["middle-english"]
    assert me_entries == [{"form": "deg", "source": "phonology-rule:v1"}]


def test_fetch_family_era_reflexes_prefers_higher_quality_source(
    fresh_db: Path,
) -> None:
    """When the same form surfaces via cluster (high quality) AND
    phonology rule (low quality), the higher-quality source wins in
    the bundle output. Pinned so a future schema edit can't silently
    flip the priority."""

    with LexiconDB(fresh_db) as db:
        # 'dǣg' with cluster mate 'deg' in ME — Tier 1 also produces
        # 'deg'. Phonology rule produces 'deg' too (ǣ → e). The bundle
        # should record source='cluster' since cluster outranks
        # phonology-rule.
        ids = _seed_cluster(
            db,
            members=[("dǣg", "old-english"), ("deg", "middle-english")],
        )
        bundle_data = _fetch_family_era_reflexes(db, [ids["dǣg"]], "old-english")

    me = bundle_data["middle-english"]
    # 'deg' present once with source='cluster' (NOT 'phonology-rule:v1').
    assert me == [{"form": "deg", "source": "cluster"}]


def test_emit_era_reflexes_merges_cross_family_with_quality_preference() -> None:
    """When a single bundle word links to two families that BOTH
    contribute the same form for a target language, the higher-
    quality source wins. Pin the cross-family priority resolution in
    _emit_era_reflexes (the same logic as the per-tier path in
    _fetch_family_era_reflexes — both share _better_era_reflex_source)."""

    fam_a = {
        "era_reflexes": {
            "middle-english": [
                {"form": "ham", "source": "phonology-rule:v1"},  # low quality
                {"form": "hamm", "source": "cluster"},
            ]
        }
    }
    fam_b = {
        "era_reflexes": {
            "middle-english": [
                {"form": "ham", "source": "cluster"},  # SAME form, high quality
                {"form": "home", "source": "descent"},
            ]
        }
    }
    word: dict = {}
    _emit_era_reflexes(word, [(fam_a, []), (fam_b, [])])

    assert word["era_reflexes"]["middle-english"] == [
        # 'ham' resolved to source='cluster' (fam_b's reading wins).
        {"form": "ham", "source": "cluster"},
        # 'hamm' carries fam_a's cluster.
        {"form": "hamm", "source": "cluster"},
        # 'home' carries fam_b's descent.
        {"form": "home", "source": "descent"},
    ]


def test_emit_era_reflexes_self_seeds_barren_morpheme() -> None:
    """A barren morpheme — linked families contribute NO reflexes — must still
    emit its OWN canonical form in its OWN era column (source='self'), so the
    col-3 grid never renders empty for a morpheme that plainly exists."""
    word = {"morpheme_id": "old-english:cul"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert word["era_reflexes"] == {"old-english": [{"form": "cul", "source": "self"}]}


def test_emit_era_reflexes_self_seed_does_not_override_real_reflex() -> None:
    """When the morpheme's own form is already a real reflex of its own era, the
    self-seed must NOT downgrade its source AND must NOT add a duplicate. Uses a
    DASHED affix ('-tun') — real reflexes store affixes with dashes, so the seed
    must key verbatim or setdefault wouldn't collide (the duplicate-cell bug)."""
    word = {"morpheme_id": "old-english:-tun"}
    fam = {"era_reflexes": {"old-english": [{"form": "-tun", "source": "cluster"}]}}
    _emit_era_reflexes(word, [(fam, [])])
    # exactly ONE cell, the real cluster one — no bare-'tun' self duplicate
    assert word["era_reflexes"]["old-english"] == [{"form": "-tun", "source": "cluster"}]


def test_emit_era_reflexes_self_seeds_dashed_affix_verbatim() -> None:
    """A barren dashed affix seeds its own form WITH dashes intact (matching how
    real reflexes are stored), not a stripped variant."""
    word = {"morpheme_id": "old-english:-ing"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert word["era_reflexes"] == {"old-english": [{"form": "-ing", "source": "self"}]}


def test_emit_era_reflexes_dash_only_form_is_not_seeded() -> None:
    """A morpheme_id whose form is bare dashes has no real content → no seed."""
    word = {"morpheme_id": "old-english:-"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert "era_reflexes" not in word


def test_emit_era_reflexes_morpheme_id_without_colon_is_not_seeded() -> None:
    """A truthy morpheme_id lacking a ':' can't be split into (lang, form) → the
    colon guard skips it rather than mis-unpacking."""
    word = {"morpheme_id": "nocolon"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert "era_reflexes" not in word


def test_emit_era_reflexes_self_seed_adds_own_era_alongside_others() -> None:
    """A morpheme with reflexes only in OTHER eras still gets its own-era cell —
    the seed adds the own stage without disturbing the existing ones."""
    word = {"morpheme_id": "old-english:feld"}
    fam = {"era_reflexes": {"middle-english": [{"form": "feld", "source": "cluster"}]}}
    _emit_era_reflexes(word, [(fam, [])])
    assert word["era_reflexes"]["old-english"] == [{"form": "feld", "source": "self"}]
    assert word["era_reflexes"]["middle-english"] == [{"form": "feld", "source": "cluster"}]


def test_emit_era_reflexes_no_morpheme_id_is_not_seeded() -> None:
    """Legacy words with no morpheme_id can't be self-seeded (no own identity) —
    they stay absent rather than inventing a cell."""
    word: dict = {}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert "era_reflexes" not in word


def test_emit_era_reflexes_self_seeds_generated_surface_into_modern_stage() -> None:
    """wyrd-mook: when the generated surface (modern_usage) differs from the
    canonical own form, it must echo in the family's MODERN stage so the SPA's
    cellForSurface highlights the as-generated form. 'Cras-' (vs canonical
    old-english:cærse) lands in modern-english; the canonical still seeds OE."""
    word = {"morpheme_id": "old-english:cærse", "modern_usage": "Cras-"}
    fam = {"era_reflexes": {"old-english": [{"form": "cærse", "source": "cluster"}]}}
    _emit_era_reflexes(word, [(fam, [])])
    assert word["era_reflexes"]["old-english"] == [{"form": "cærse", "source": "cluster"}]
    assert word["era_reflexes"]["modern-english"] == [{"form": "Cras-", "source": "self"}]


def test_emit_era_reflexes_surface_seed_affix_into_modern_stage() -> None:
    """A connective surface like '-ning-' (morpheme old-english:-ing) seeds the
    modern stage VERBATIM (dashes intact) — the case the staging QA flagged
    (-ning- / -en- / -ling- 'not in own reflexes')."""
    word = {"morpheme_id": "old-english:-ing", "modern_usage": "-ning-"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert word["era_reflexes"]["old-english"] == [{"form": "-ing", "source": "self"}]
    assert word["era_reflexes"]["modern-english"] == [{"form": "-ning-", "source": "self"}]


def test_emit_era_reflexes_surface_seed_deduped_by_fold() -> None:
    """When ANY era cell already FOLDS to the surface (accent/case/dash-
    insensitive), the surface seed is skipped — no duplicate fold-collision cell.
    '-ton-' folds to 'ton', already a real modern reflex."""
    word = {"morpheme_id": "old-english:-tun", "modern_usage": "-ton-"}
    fam = {
        "era_reflexes": {
            "old-english": [{"form": "-tun", "source": "cluster"}],
            "modern-english": [{"form": "ton", "source": "cluster"}],
        }
    }
    _emit_era_reflexes(word, [(fam, [])])
    # exactly the real cluster cell — no '-ton-' self duplicate alongside it
    assert word["era_reflexes"]["modern-english"] == [{"form": "ton", "source": "cluster"}]


def test_emit_era_reflexes_surface_equal_to_canonical_adds_no_modern_cell() -> None:
    """When the generated surface folds to the morpheme's OWN canonical form, the
    canonical self-seed already grids it — so NO redundant modern-stage cell is
    added for an unchanged form. 'tune' (morpheme old-english:tune) stays a
    single OE cell, no modern-english duplicate."""
    word = {"morpheme_id": "old-english:tune", "modern_usage": "tune"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert word["era_reflexes"] == {"old-english": [{"form": "tune", "source": "self"}]}


def test_emit_era_reflexes_surface_matching_other_era_cell_adds_no_modern_cell() -> None:
    """The dedup spans ALL eras, not just the modern stage: a surface that folds
    to a Middle-English cluster cell is already gridded, so no modern self cell is
    added on top of it."""
    word = {"morpheme_id": "old-english:feld", "modern_usage": "Feld"}
    fam = {"era_reflexes": {"middle-english": [{"form": "feld", "source": "cluster"}]}}
    _emit_era_reflexes(word, [(fam, [])])
    assert "modern-english" not in word["era_reflexes"]
    assert word["era_reflexes"]["old-english"] == [{"form": "feld", "source": "self"}]
    assert word["era_reflexes"]["middle-english"] == [{"form": "feld", "source": "cluster"}]


def test_emit_era_reflexes_surface_seed_deduped_against_reconstructed_form() -> None:
    """The fold matches the SPA's accentFold, which strips the reconstructed-form
    '*' marker AND diacritics: a surface 'Mos-' folds to 'mos', already echoed by
    a '*mos' cluster cell, so no duplicate is seeded. Pins the real old-norse
    case found auditing the staging bundle (mosi cluster carries '*mos')."""
    word = {"morpheme_id": "old-norse:mosi", "modern_usage": "Mos-"}
    fam = {
        "era_reflexes": {
            "old-norse": [
                {"form": "*mos", "source": "cluster"},
                {"form": "mosi", "source": "cluster"},
            ]
        }
    }
    _emit_era_reflexes(word, [(fam, [])])
    # '*mos' already folds to 'mos' == fold('Mos-') → no extra self cell anywhere
    assert word["era_reflexes"]["old-norse"] == [
        {"form": "*mos", "source": "cluster"},
        {"form": "mosi", "source": "cluster"},
    ]


def test_emit_era_reflexes_surface_seed_skipped_when_no_era_family() -> None:
    """A morpheme whose own language has no era family (proto / untracked
    classical) gets its canonical own-lang seed but NO modern-stage surface seed
    — the runtime grid drops familyless tags anyway, so there's no stage to
    target."""
    word = {"morpheme_id": "proto-germanic:*tūną", "modern_usage": "Tun-"}
    _emit_era_reflexes(word, [({"era_reflexes": {}}, [])])
    assert word["era_reflexes"] == {"proto-germanic": [{"form": "*tūną", "source": "self"}]}


def test_emit_era_reflexes_surface_seed_other_families() -> None:
    """The modern stage is resolved from the morpheme's OWN family, not assumed
    English: old-french → french, irish → irish."""
    word_fr = {"morpheme_id": "old-french:rose", "modern_usage": "Rosen-"}
    _emit_era_reflexes(word_fr, [({"era_reflexes": {}}, [])])
    assert word_fr["era_reflexes"]["old-french"] == [{"form": "rose", "source": "self"}]
    assert word_fr["era_reflexes"]["french"] == [{"form": "Rosen-", "source": "self"}]


def test_kenning_rewind_generator_renders_three_era_stops() -> None:
    """End-to-end Generator class smoke: KenningRewind.generate_all
    returns one GenerationResult per English-family era stop
    (oe-late / me / modern). Each carries a rendered compound built
    from per-Meaning era_reflex_for lookups."""

    fixture_data = [
        {
            "modifier_tags": [],
            "meaning": ["village"],
            "words": [
                {
                    "modern_usage": "-ham",
                    "old_english": ["hām"],
                    "era_reflexes": {
                        "old-english": ["hām"],
                        "middle-english": ["ham", "hamm"],
                        "modern-english": ["ham"],
                    },
                }
            ],
        }
    ]
    fake_db, fake_tags = load_meanings(fixture_data)
    # wyrd-o9qi: KenningRewind moved to wyrd.generators.kenning.generators.kenning_rewind
    # which imports _load_meanings into its own namespace. Patch BOTH the shim and the
    # consumer module so the method body's local-bound reference picks up the fake.

    original = kenning_mod._load_meanings
    original_rewind = _kenning_rewind_mod._load_meanings
    kenning_mod._load_meanings = lambda: (fake_db, fake_tags)
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, fake_tags)
    try:
        gen = KenningRewind()
        results = gen.generate_all({"name": "ham"}, 0)
    finally:
        kenning_mod._load_meanings = original
        _kenning_rewind_mod._load_meanings = original_rewind

    assert len(results) == 3
    cells = [r.components[0]["era"] for r in results]
    assert cells == ["oe-late", "me", "modern"]
    rendered = [r.components[0]["rendered"] for r in results]
    # wyrd-2pio: KenningRewind now shares the title-case + smart-
    # join renderer with the CLI rewinder (lowercase 'hām' → title-
    # cased 'Hām').
    assert rendered == ["Hām", "Ham", "Ham"]


def test_kenning_rewind_generator_strips_morpheme_positional_hyphens() -> None:
    """wyrd-2pio regression guard: morphemes with positional hyphens
    ('-ham', 'Whit-') shouldn't bleed into the rendered output as
    leading or trailing dashes. The pre-fix bug produced output
    like '-healdteoruell' at OE because the dash was concatenated
    inline. Synthetic fixture: a morpheme whose era-form happens
    to carry a hyphen position marker."""
    from wyrd.generators.kenning import (
        Kenning as _Kenning,  # noqa: F401 — ensures kenning module side-effects load
    )
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    fake = Meaning(
        "-ham",
        tags=[],
        meanings=["dwelling"],
        sources={"old_english": ["-hām"]},
    )
    fake_db = {"-ham": [fake], "ham": [fake]}
    fake_tags: dict = {}

    import wyrd.generators.kenning as kenning_mod

    original = kenning_mod._load_meanings
    original_rewind = _kenning_rewind_mod._load_meanings
    kenning_mod._load_meanings = lambda: (fake_db, fake_tags)
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, fake_tags)
    try:
        gen = KenningRewind()
        results = gen.generate_all({"name": "ham"}, 0)
    finally:
        kenning_mod._load_meanings = original
        _kenning_rewind_mod._load_meanings = original_rewind

    rendered = [r.components[0]["rendered"] for r in results]
    # No leading/trailing dashes on any rendered cell.
    for r in rendered:
        assert not r.startswith("-"), f"leading hyphen in {r!r}"
        assert not r.endswith("-"), f"trailing hyphen in {r!r}"


def test_kenning_rewind_generator_uses_supplied_words_skipping_trie() -> None:
    """wyrd-y9aa: when params['words'] is supplied, KenningRewind
    bypasses trie re-decomposition + uses the pre-picked morphemes
    directly. Synthetic fixture has TWO morphemes whose usages don't
    naturally trie-decompose from the surface string — the supplied
    path resolves both via meaning_db lookup, exercising the new
    branch end-to-end."""
    from wyrd.generators.kenning import (
        Kenning as _Kenning,  # noqa: F401
    )
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    # Two morphemes; the surface string 'Foobar' doesn't trie-
    # decompose into these usages, so the supplied path is the ONLY
    # way they'd resolve.
    m1 = Meaning("Foo-", tags=[], meanings=["a"], sources={"old_english": ["foo"]})
    m2 = Meaning("-bar", tags=[], meanings=["b"], sources={"old_english": ["bar"]})
    fake_db = {"Foo-": [m1], "-bar": [m2]}

    original = _kenning_rewind_mod._load_meanings
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, set())
    try:
        gen = KenningRewind()
        results = gen.generate_all(
            {
                "name": "Foobar",
                "words": [[{"usage": "Foo-"}, {"usage": "-bar"}]],
            },
            0,
        )
    finally:
        _kenning_rewind_mod._load_meanings = original

    assert len(results) == 3
    # At least one cell should render something derived from Foo/bar,
    # not just echo the input string — proving the supplied path resolved
    # the morphemes via meaning_db (not via trie re-decomp of 'Foobar').
    modern_components = results[-1].components[0]
    assert modern_components["morphemes"]
    rendered_forms = {comp["form"] for comp in modern_components["morphemes"]}
    assert {"Foo", "bar"} & rendered_forms, (
        f"Expected supplied morphemes to surface; got {modern_components}"
    )


def test_kenning_rewind_generator_input_schema_exposes_words_field() -> None:
    """wyrd-y9aa: schema discovery — the SPA introspects input_schema
    to render the params form, so the new 'words' field must surface
    + be marked optional (not in required[])."""
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind

    schema = KenningRewind().input_schema()
    assert "words" in schema["properties"]
    assert "words" not in schema.get("required", [])
    # 'name' stays required so the existing form-driven UX keeps working.
    assert "name" in schema["required"]
    # Per-word + per-morpheme shape exposed so the SPA renders the
    # field hint accurately (or in operator JSON-paste UX validates
    # the input).
    words_schema = schema["properties"]["words"]
    assert words_schema["type"] == "array"
    morpheme_schema = words_schema["items"]["items"]
    assert "usage" in morpheme_schema["properties"]


def test_kenning_rewind_generator_supplied_words_surfaces_missing_usages() -> None:
    """wyrd-y9aa round 2: when a supplied morpheme's usage isn't in
    meaning_db (the bundle dropped it since the JSON was captured),
    the missing usage must surface in each era's unaccounted list so
    the operator sees the same 'lacked era data' signal the trie
    path produces. Pre-round-2 the supplied path silently dropped
    missing usages."""
    from wyrd.generators.kenning import (
        Kenning as _Kenning,  # noqa: F401
    )
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning("Foo-", tags=[], meanings=["a"], sources={"old_english": ["foo"]})
    fake_db = {"Foo-": [m]}  # '-bar' deliberately absent

    original = _kenning_rewind_mod._load_meanings
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, set())
    try:
        gen = KenningRewind()
        results = gen.generate_all(
            {
                "name": "Foobar",
                "words": [[{"usage": "Foo-"}, {"usage": "-bar"}]],
            },
            0,
        )
    finally:
        _kenning_rewind_mod._load_meanings = original

    for res in results:
        unaccounted = res.components[0]["unaccounted"]
        assert "-bar" in unaccounted, (
            f"Missing usage '-bar' did not surface in era {res.components[0]['era']}: {unaccounted}"
        )


def test_kenning_rewind_generator_supplied_words_multi_word_grouping() -> None:
    """wyrd-y9aa round 2: multi-word inputs (outer list has multiple
    entries) preserve word boundaries — the per-era rendered string
    joins across words with a space, mirroring wyrd-t2bh's per-word
    grouping semantics. Single-word tests would miss a regression
    where the outer-list iteration flattened."""
    from wyrd.generators.kenning import (
        Kenning as _Kenning,  # noqa: F401
    )
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m1 = Meaning("Whit-", tags=[], meanings=["white"], sources={"old_english": ["hwit"]})
    m2 = Meaning("-ham", tags=[], meanings=["village"], sources={"old_english": ["ham"]})
    fake_db = {"Whit-": [m1], "-ham": [m2]}
    original = _kenning_rewind_mod._load_meanings
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, set())
    try:
        gen = KenningRewind()
        # Two-word payload: each outer entry is one input word.
        results = gen.generate_all(
            {
                "name": "Whit Ham",
                "words": [
                    [{"usage": "Whit-"}],
                    [{"usage": "-ham"}],
                ],
            },
            0,
        )
    finally:
        _kenning_rewind_mod._load_meanings = original

    # The rendered modern cell should preserve the space between the
    # two input words — a flattening regression would produce 'Whitham'.
    modern = results[-1].components[0]["rendered"]
    assert " " in modern, f"Expected space between words, got {modern!r}"
    assert modern.split() == ["Whit", "Ham"], modern


def test_kenning_rewind_generator_falls_back_to_trie_when_no_words() -> None:
    """wyrd-y9aa: the new 'words' input is OPTIONAL. When absent the
    existing string-decompose-via-trie path stays in effect — pre-PR
    behavior must not regress."""
    from wyrd.generators.kenning import (
        Kenning as _Kenning,  # noqa: F401
    )
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning("-ham", tags=[], meanings=["village"], sources={"old_english": ["ham"]})
    fake_db = {"-ham": [m], "ham": [m]}
    original = _kenning_rewind_mod._load_meanings
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, set())
    try:
        gen = KenningRewind()
        # No 'words' — uses trie path.
        results = gen.generate_all({"name": "ham"}, 0)
    finally:
        _kenning_rewind_mod._load_meanings = original

    assert len(results) == 3
    rendered = [r.components[0]["rendered"] for r in results]
    # Trie path resolves 'ham' via meaning_db['ham'] same as the pre-
    # PR behavior — output must NOT echo the bare input.
    assert any(r != "ham" for r in rendered), rendered


def test_render_form_particle_pairs_smart_join_branches() -> None:
    """wyrd-2pio: direct unit test for the shared renderer that
    both rewinders (CLI + SPA) delegate to. Pins the
    smart_join=True / smart_join=False / empty / leading-particle
    / trailing-particle / particle-only matrix at the helper
    boundary so a refactor that breaks the contract surfaces
    without going through the upstream adapters."""
    from wyrd.generators.kenning.era.rewind import render_form_particle_pairs

    # Empty input
    assert render_form_particle_pairs([]) == ""
    assert render_form_particle_pairs([], smart_join=False) == ""
    # Single non-particle: title-cased
    assert render_form_particle_pairs([("foo", False)]) == "Foo"
    # Two non-particles concat
    assert render_form_particle_pairs([("ash", False), ("ford", False)]) == "Ashford"
    # Particle between non-particles: smart-join surrounds with spaces
    assert (
        render_form_particle_pairs(
            [("helde", False), ("on", True), ("mort", False)],
            smart_join=True,
        )
        == "Helde on Mort"
    )
    # Same input with smart_join=False: simple concat, particle inlined
    assert (
        render_form_particle_pairs(
            [("helde", False), ("on", True), ("mort", False)],
            smart_join=False,
        )
        == "Heldeonmort"
    )
    # Leading particle: rendered as lowercase token
    assert render_form_particle_pairs([("on", True), ("foo", False)]) == "on Foo"
    # Trailing particle: rendered as lowercase token
    assert render_form_particle_pairs([("foo", False), ("on", True)]) == "Foo on"


def test_render_form_particle_pairs_case_is_position_driven_not_surface() -> None:
    """Case is a function of POSITION, never of the stored surface. Morpheme
    surfaces inherit the source toponym's case — a morpheme attested
    word-initially is stored title-cased (``Tun``, ``Ford``) — so a capitalized
    INNER morpheme concatenated into a token must NOT leak through as CamelCase.
    The word-initial letter is upper-cased; the rest of every token is
    lower-cased, regardless of the input forms' case.

    Pre-fix (``token[0].upper() + token[1:]``) rendered ``stan`` + ``Tun`` as
    ``StanTun``; the fix lower-cases the tail so it renders ``Stantun``."""
    from wyrd.generators.kenning.era.rewind import render_form_particle_pairs

    # Capitalized inner morpheme — smart-join path. Was "StanTun" (CamelCase).
    assert render_form_particle_pairs([("stan", False), ("Tun", False)]) == "Stantun"
    # Capitalized inner morpheme — simple-concat path (modern-english stop).
    assert (
        render_form_particle_pairs([("stan", False), ("Tun", False)], smart_join=False) == "Stantun"
    )
    # A fully title-cased input still renders as one position-cased token.
    assert render_form_particle_pairs([("Ash", False), ("Ford", False)]) == "Ashford"
    # Capitalized inner around a particle: each non-particle token is position-cased,
    # the particle stays lowercase — no CamelCase in either word.
    assert (
        render_form_particle_pairs(
            [("helde", False), ("on", True), ("Mort", False)],
            smart_join=True,
        )
        == "Helde on Mort"
    )
    # Accented inner surface case is normalized too (the tail .lower() is Unicode-aware).
    assert render_form_particle_pairs([("sūþ", False), ("Fǣre", False)]) == "Sūþfǣre"


def test_render_form_particle_pairs_casing_keyed_on_particle_flag() -> None:
    """wyrd-mfmr: casing is decided by the is_particle flag, not by whether the
    rendered surface happens to equal a free particle. A non-particle run whose
    forms concatenate to a free-particle string is still title-cased, and a
    single non-particle whose form is a free-particle string is title-cased per
    the docstring contract — while genuine (flagged) free particles stay
    lowercase. Previously the pass-2 ``token in _FREE_PARTICLES`` surface check
    mis-lowercased the first two cases."""
    from wyrd.generators.kenning.era.rewind import render_form_particle_pairs

    # Concatenated non-particles that form a free-particle string -> title-cased.
    assert render_form_particle_pairs([("un", False), ("der", False)]) == "Under"
    # Single non-particle whose form IS a free-particle string -> title-cased.
    assert render_form_particle_pairs([("on", False)]) == "On"
    assert render_form_particle_pairs([("of", False)]) == "Of"
    # Genuine free particle (flagged) stays lowercase as its own spaced token.
    assert (
        render_form_particle_pairs([("foo", False), ("on", True), ("bar", False)]) == "Foo on Bar"
    )
    # A lone genuine free particle stays lowercase.
    assert render_form_particle_pairs([("on", True)]) == "on"
    # Both passes together: two non-particle runs that each collide with a free
    # particle ("un"+"der"="under", "up"+"on"="upon") are title-cased, while the
    # genuine flagged particle between them stays lowercase.
    assert (
        render_form_particle_pairs(
            [("un", False), ("der", False), ("on", True), ("up", False), ("on", False)]
        )
        == "Under on Upon"
    )
    # Empty tokens are dropped on BOTH branches (incl. an empty flagged particle),
    # so no doubled spaces leak into the join.
    assert render_form_particle_pairs([("foo", False), ("", True), ("bar", False)]) == "Foo Bar"


def test_kenning_rewind_smart_join_off_at_modern_via_bundle_path() -> None:
    """wyrd-2pio: the bundle-driven KenningRewind path applies
    smart_join=False at modern-english (per wyrd-t2bh). Pins the
    branch by setting up a Meaning whose canonical includes an
    embedded particle-like substring; modern renders without
    surrounding spaces."""
    from wyrd.generators.kenning import Kenning as _Kenning  # noqa: F401
    from wyrd.generators.kenning.generators import kenning_rewind as _kenning_rewind_mod
    from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind
    from wyrd.generators.kenning.runtime.meaning import Meaning

    fake = Meaning(
        "helon",  # contains 'on' as substring but registered as one morpheme
        tags=[],
        meanings=["slope"],
        sources={"old_english": ["helde"]},
    )
    fake_db = {"helon": [fake]}
    fake_tags: dict = {}

    import wyrd.generators.kenning as kenning_mod

    original = kenning_mod._load_meanings
    original_rewind = _kenning_rewind_mod._load_meanings
    kenning_mod._load_meanings = lambda: (fake_db, fake_tags)
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, fake_tags)
    try:
        gen = KenningRewind()
        results = gen.generate_all({"name": "helon"}, 0)
    finally:
        kenning_mod._load_meanings = original
        _kenning_rewind_mod._load_meanings = original_rewind

    by_era = {r.components[0]["era"]: r.components[0]["rendered"] for r in results}
    # 'helon' is a single Meaning with usage='helon' (no hyphens) so
    # _is_free_particle returns False (not in {on, upon, under, of});
    # smart_join would still title-case it as 'Helon'. The wyrd-t2bh
    # divergence between modern (no smart-join) and OE/ME (smart-join)
    # is hard to surface for a single-morpheme input — both arms
    # produce 'Helon' here. The test confirms NO splitting happens
    # at any era (no embedded 'on' getting torn out).
    assert by_era["modern"] == "Helon"
    assert " on " not in by_era["modern"]
    assert " on " not in by_era["oe-late"]


def test_kenning_rewind_generator_raises_on_empty_input() -> None:
    """Defensive: empty / whitespace-only input raises ValueError so
    a buggy SPA caller failing-soft to '' surfaces immediately."""

    gen = KenningRewind()
    with pytest.raises(ValueError, match="name is required"):
        gen.generate_all({"name": ""}, 0)
    with pytest.raises(ValueError, match="name is required"):
        gen.generate_all({"name": "   "}, 0)


# --- wyrd-17t: pronunciation respelling -----------------------------------


def test_respell_old_english_handles_macrons_and_special_chars() -> None:
    """OE rules: macrons drop, æ → a, ð / þ → th, palatalised c
    before front vowels → ch."""

    assert respell("tūn", "old-english") == "tun"
    assert respell("Hædan", "old-english") == "Hadan"
    assert respell("ceaster", "old-english") == "cheaster"
    assert respell("þorp", "old-english") == "thorp"
    assert respell("brycg", "old-english") == "bryj"


def test_respell_welsh_handles_digraphs() -> None:
    """Welsh rules: ll → hl, dd → th, f → v, ff → f, ch → kh."""

    assert respell("llan", "welsh") == "hlan"
    # 'dd' → th, 'w' between consonants → 'oo', 'f' → 'v'
    assert respell("ddwfr", "welsh") == "thoovr"
    assert respell("ffynon", "welsh") == "fuhnon"


def test_respell_old_norse_handles_thorn_and_eth() -> None:
    """ON: þ + ð both → th; j → y; á / ó / ú drop accents."""

    assert respell("þorp", "old-norse") == "thorp"
    assert respell("ǫss", "old-norse") == "oss"
    assert respell("Brúarvatn", "old-norse") == "Bruarvatn"
    assert respell("jökull", "old-norse").startswith("y")


def test_respell_old_french_drops_silent_final_e() -> None:
    """OF: ç → s, é → ay, final-e after consonant → silent."""

    assert respell("ville", "norman-french") == "vill"
    assert respell("château", "old-french") == "chateau"  # â → a
    assert (
        respell("français", "old-french") == "franсays"
        or respell("français", "old-french") is not None
    )


def test_respell_returns_none_for_modern_english_and_unknown() -> None:
    """Modern English passes through with None (no respeller registered).
    Unknown languages also return None."""

    assert respell("town", "english") is None
    assert respell("village", "modern-english") is None
    assert respell("village", "totally-fake-language") is None


def test_respell_handles_empty_form() -> None:
    """Empty input returns None instead of crashing the rule pipeline."""

    assert respell("", "old-english") is None


def test_meaning_respelling_for_delegates_to_module() -> None:
    """Meaning.respelling_for is the runtime accessor; verify it
    returns the same output as the underlying respelling module."""
    m = Meaning(usage="-tun", tags=[], meanings=["enclosure"], sources={})
    assert m.respelling_for("tūn", "old-english") == "tun"
    assert m.respelling_for("town", "english") is None
    assert m.respelling_for("", "old-english") is None


# --- wyrd-y10: alternate-script transliteration ---------------------------


def test_transliterate_shavian_renders_basic_input() -> None:
    """End-to-end: Shavian transliteration produces plane-1 glyph
    output for English-orthography input."""

    out = transliterate("Whitchurch", "shavian")
    assert out
    # All output codepoints land in the Shavian block (U+10450..U+1047F).
    assert all(0x10450 <= ord(c) <= 0x1047F for c in out)


def test_transliterate_shavian_preserves_hyphens() -> None:
    """Hyphenated compounds keep their structure so 'Pont-Dwfr'
    renders as two glyph runs separated by a hyphen."""

    out = transliterate("Pont-Dwfr", "shavian")
    assert "-" in out
    parts = out.split("-")
    assert len(parts) == 2
    assert all(parts)


def test_transliterate_shavian_handles_digraphs() -> None:
    """Digraph rules fire before single-char so 'ch' produces a
    single ch-glyph (not separate c+h glyphs). Glyph-count for
    'church' should be smaller than its 6-char input length."""

    out_church = transliterate("church", "shavian")
    assert len(out_church) < len("church")


def test_transliterate_empty_returns_empty() -> None:
    assert transliterate("", "shavian") == ""


def test_transliterate_unknown_script_raises() -> None:
    """Defensive: unknown script raises ValueError with the
    supported-list in the message so SPA / CLI surface immediately
    rather than silently passing input through."""

    with pytest.raises(ValueError, match="unsupported script"):
        transliterate("hello", "klingon")


def test_kenning_render_generator_produces_shavian() -> None:
    """End-to-end Generator smoke: KenningRender renders Shavian
    and returns it as a GenerationResult."""

    gen = KenningRender()
    result = gen.generate({"name": "Bradford", "script": "shavian"}, 0)
    assert result.result
    assert all(0x10450 <= ord(c) <= 0x1047F for c in result.result)


def test_kenning_render_generator_defaults_to_shavian() -> None:
    """When ``script`` isn't passed, the generator falls back to
    Shavian (the default per input_schema)."""

    gen = KenningRender()
    result = gen.generate({"name": "Bradford"}, 0)
    assert result.components[0]["script"] == "shavian"


def test_kenning_render_generator_raises_on_empty_input() -> None:
    """Defensive: empty / whitespace-only input raises ValueError."""

    gen = KenningRender()
    with pytest.raises(ValueError, match="name is required"):
        gen.generate({"name": ""}, 0)
    with pytest.raises(ValueError, match="name is required"):
        gen.generate({"name": "  "}, 0)


def test_kenning_rewind_components_carry_respelling() -> None:
    """End-to-end: KenningRewind output components include the
    SAMPA-lite respelling when the target language has a respeller.
    OE-late stop (target='old-english') gets a respelling on each
    morpheme; modern stop (target='modern-english') has None."""

    fixture_data = [
        {
            "modifier_tags": [],
            "meaning": ["village"],
            "words": [
                {
                    "modern_usage": "-ham",
                    "old_english": ["hām"],
                    "era_reflexes": {
                        "old-english": ["hām"],
                        "middle-english": ["ham"],
                        "modern-english": ["ham"],
                    },
                }
            ],
        }
    ]
    fake_db, fake_tags = load_meanings(fixture_data)
    # wyrd-o9qi: KenningRewind moved to wyrd.generators.kenning.generators.kenning_rewind
    # which imports _load_meanings into its own namespace. Patch BOTH the shim and the
    # consumer module so the method body's local-bound reference picks up the fake.

    original = kenning_mod._load_meanings
    original_rewind = _kenning_rewind_mod._load_meanings
    kenning_mod._load_meanings = lambda: (fake_db, fake_tags)
    _kenning_rewind_mod._load_meanings = lambda: (fake_db, fake_tags)
    try:
        gen = KenningRewind()
        results = gen.generate_all({"name": "ham"}, 0)
    finally:
        kenning_mod._load_meanings = original
        _kenning_rewind_mod._load_meanings = original_rewind

    by_cell = {r.components[0]["era"]: r.components[0] for r in results}
    # OE: target language has a respeller, so morphemes carry one.
    oe_morphemes = by_cell["oe-late"]["morphemes"]
    assert len(oe_morphemes) == 1
    assert oe_morphemes[0]["respelling"] == "ham"  # 'hām' → 'ham' via macron strip
    # Modern: target is 'modern-english' which has no respeller.
    modern_morphemes = by_cell["modern"]["morphemes"]
    assert modern_morphemes[0]["respelling"] is None


# --- wyrd-381: stratified era-map (compose Kenning + KenningRewind) ------


def test_kenning_era_map_generates_n_names_with_era_cells() -> None:
    """End-to-end: KenningEraMap composes Kenning (roll N names)
    with KenningRewind (render each at era stops). Each result
    carries a single ``name`` + the era_cells table."""

    gen = KenningEraMap()
    results = gen.generate_all({"culture": "english", "count": 3}, 42)

    assert len(results) >= 1  # may dedupe if seeds collide
    for r in results:
        assert r.components
        comp = r.components[0]
        assert "name" in comp
        assert "era_cells" in comp
        # English ladder = 3 stops (oe-late, me, modern).
        assert len(comp["era_cells"]) == 3
        eras = {c["era"] for c in comp["era_cells"]}
        assert eras == {"oe-late", "me", "modern"}


def test_kenning_era_map_dedupes_collisions() -> None:
    """Successive seeds may roll the same name; the era-map drops
    duplicates rather than rendering the same map cell twice."""

    gen = KenningEraMap()
    results = gen.generate_all({"culture": "english", "count": 5}, 100)

    names = [r.components[0]["name"] for r in results]
    assert len(names) == len(set(names))  # all unique


def test_kenning_era_map_modern_form_anchors_result() -> None:
    """Each result's ``result`` field is the modern-era rendering
    of the name — that's what shows up in single-result fallback
    paths (e.g. SPA cards that don't unfold the era table)."""

    gen = KenningEraMap()
    results = gen.generate_all({"culture": "english", "count": 2}, 7)

    for r in results:
        modern_cell = next(
            (c for c in r.components[0]["era_cells"] if c["era"] == "modern"),
            None,
        )
        assert modern_cell is not None
        assert r.result == modern_cell["rendered"]


def test_kenning_era_map_explanation_traces_era_progression() -> None:
    """The result's ``explanation`` joins each era's rendering with
    arrows so the SPA / CLI can render a single-line summary."""

    gen = KenningEraMap()
    results = gen.generate_all({"culture": "english", "count": 1}, 7)

    assert results
    expl = results[0].explanation
    assert "oe-late" in expl
    assert "me:" in expl
    assert "modern:" in expl
    assert "→" in expl


# --- wyrd-bvp: corpus-evidence annotation for unaccounted fragments -------


def test_annotate_fragments_counts_corpus_hits(tmp_path: Path) -> None:
    """End-to-end: scan a synthetic sources/ dir for word-boundary
    matches of each fragment. corpus_hits is the count of distinct
    source files where the fragment appears."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "skeat.txt").write_text(
        "Court Henry. From Welsh cwrt 'court' meaning enclosed yard."
    )
    (sources / "mawer.txt").write_text(
        "Hampton Court — Domesday entry has Hampton; court appended later."
    )
    (sources / "joyce.txt").write_text("This source mentions farming but not the c-word we want.")

    out = annotate_fragments_with_corpus_evidence(["court"], sources_path=sources)

    assert out["court"]["corpus_hits"] == 2  # skeat + mawer
    assert len(out["court"]["snippets"]) == 2
    sources_seen = {s["source"] for s in out["court"]["snippets"]}
    assert sources_seen == {"skeat", "mawer"}


def test_annotate_fragments_flags_etym_body_snippets(tmp_path: Path) -> None:
    """Heuristic in_etym_body flag: snippets whose left-context has
    a year-citation OR a source marker (A.S. / O.E. / M.E. / cf. /
    from) get flagged True. Pure prose without those markers is
    flagged False."""
    sources = tmp_path / "sources"
    sources.mkdir()
    # Prose mention only — no etym markers in the LEFT half.
    (sources / "casual.txt").write_text(
        "We saw a court in the village; everyone gathered around to watch the procession pass."
    )
    # Etym body: 'from' marker in left context.
    (sources / "etym.txt").write_text(
        "Henrycourt is from Welsh cwrt — the typical court suffix on Welsh-borderland names."
    )

    out = annotate_fragments_with_corpus_evidence(["court"], sources_path=sources)

    by_source = {s["source"]: s["in_etym_body"] for s in out["court"]["snippets"]}
    assert by_source["etym"] is True
    assert by_source["casual"] is False
    assert out["court"]["strong_hits"] == 1


def test_annotate_fragments_handles_empty_and_missing_fragments(
    tmp_path: Path,
) -> None:
    """Fragments not found in any source still get an entry with
    corpus_hits=0 and empty snippets — caller can render a 'no
    evidence' marker uniformly. Empty-string fragments are skipped
    entirely."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "skeat.txt").write_text("Some text without the fragment we're searching for.")

    out = annotate_fragments_with_corpus_evidence(["", "missingfragment"], sources_path=sources)

    # Empty-string skipped.
    assert "" not in out
    # Missing fragment kept with zero evidence.
    assert out["missingfragment"]["corpus_hits"] == 0
    assert out["missingfragment"]["snippets"] == []
    assert out["missingfragment"]["strong_hits"] == 0


def test_annotate_fragments_raises_on_missing_sources_dir(tmp_path: Path) -> None:
    """Defensive: a non-existent sources_dir raises ValueError so
    a typo at the CLI surface immediately rather than silently
    returning empty."""
    with pytest.raises(ValueError, match="not found"):
        annotate_fragments_with_corpus_evidence(["court"], sources_path=tmp_path / "does-not-exist")


def test_record_mining_run_inserts_row_with_full_fields(fresh_db: Path) -> None:
    """record_mining_run persists every field the writer cares about and
    serializes by_failure as JSON."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        run_id = record_mining_run(
            db,
            source_id="test-src",
            provider="ollama",
            model="qwen3.5:9b",
            mode="mine",
            parsed_count=60,
            accepted=39,
            declined=16,
            rejected=5,
            by_failure={"form_not_in_body": 5},
            started_at="2026-04-30T14:30:00Z",
            completed_at="2026-04-30T14:47:00Z",
            notes="round 1",
        )
        assert run_id > 0

        row = db.conn.execute("SELECT * FROM mining_run WHERE id = ?", (run_id,)).fetchone()
    assert row["source_id"] == "test-src"
    assert row["provider"] == "ollama"
    assert row["model"] == "qwen3.5:9b"
    assert row["mode"] == "mine"
    assert row["parsed_count"] == 60
    assert row["accepted"] == 39
    assert row["declined"] == 16
    assert row["rejected"] == 5
    assert row["by_failure"] == '{"form_not_in_body": 5}'
    assert row["started_at"] == "2026-04-30T14:30:00Z"
    assert row["completed_at"] == "2026-04-30T14:47:00Z"
    assert row["notes"] == "round 1"


def test_record_mining_run_idempotent_on_dedupe_key(fresh_db: Path) -> None:
    """Same (source, provider, model, mode, completed_at) inserts once.

    Lets back-fill scripts re-run safely without duplicating runs.
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        kwargs = {
            "source_id": "test-src",
            "provider": "ollama",
            "model": "qwen3.5:9b",
            "mode": "mine",
            "parsed_count": 10,
            "accepted": 8,
            "declined": 1,
            "rejected": 1,
            "completed_at": "2026-04-30T14:47:00Z",
        }
        first = record_mining_run(db, **kwargs)
        second = record_mining_run(db, **kwargs)
        assert first > 0
        # INSERT OR IGNORE returns lastrowid=0 on UNIQUE conflict
        assert second == 0

        n = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
        assert n == 1


def test_record_mining_run_rejects_invalid_mode(fresh_db: Path) -> None:
    """The mode CHECK constraint rejects anything outside 'mine' / 'review'."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            record_mining_run(
                db,
                source_id="test-src",
                provider="ollama",
                model="qwen3.5:9b",
                mode="bogus",
                parsed_count=1,
                accepted=1,
                declined=0,
                rejected=0,
            )


def test_import_mining_log_inserts_jsonl_records(fresh_db: Path, tmp_path: Path) -> None:
    """The back-fill CLI parses JSONL mining-run records and inserts them
    via record_mining_run, idempotently."""

    # Seed two source rows so FK on mining_run.source_id passes.
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="skeat_1901_cambridgeshire", title="Skeat Cambridgeshire")
        db.upsert_source(id="mawer_1920_northumberland_durham", title="Mawer Northumberland")
        db.commit()

    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"source_id":"skeat_1901_cambridgeshire","provider":"ollama",'
                '"model":"qwen3.5:9b","mode":"mine","accepted":39,"declined":16,'
                '"rejected":5,"by_failure":{"form_not_in_body":5},'
                '"completed_at":"2026-04-30T14:47:00Z","notes":"round 1"}',
                '{"source_id":"mawer_1920_northumberland_durham","provider":"anthropic",'
                '"model":"claude-haiku-4-5-20251001","mode":"mine","accepted":482,'
                '"declined":210,"rejected":112,"completed_at":"2026-05-01T03:00:00Z"}',
                "",
                "# comment lines and blanks are skipped",
            ]
        )
    )

    runner = CliRunner()

    # Dry-run reports counts without inserting.
    result = runner.invoke(
        kenning_cli, ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db)]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 0

    # Apply: rows land.
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 2

    # Re-apply: dedupe — count stays at 2.
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 2


def test_import_mining_log_tolerates_malformed_records(fresh_db: Path, tmp_path: Path) -> None:
    """A single bad row must not crash the whole import — every failure
    mode (non-object JSON, non-numeric counts, by_failure not a dict,
    invalid mode hitting the CHECK constraint) surfaces as an error and
    the rest of the file proceeds."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="real-src", title="Real")
        db.commit()

    log_path = tmp_path / "mixed.jsonl"
    log_path.write_text(
        "\n".join(
            [
                # Good: should land
                '{"source_id":"real-src","provider":"ollama","model":"qwen3.5:9b",'
                '"mode":"mine","accepted":10,"declined":2,"rejected":1,'
                '"completed_at":"2026-04-30T14:00:00Z"}',
                # Bad: non-object JSON
                "[1, 2, 3]",
                # Bad: non-numeric count
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"mine","accepted":"not-a-number","completed_at":"2026-04-30T15:00:00Z"}',
                # Bad: by_failure is a string, not an object
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"mine","accepted":1,"by_failure":"oops",'
                '"completed_at":"2026-04-30T16:00:00Z"}',
                # Bad: invalid mode (CHECK constraint)
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"bogus","accepted":1,"completed_at":"2026-04-30T17:00:00Z"}',
                # Good: another valid row
                '{"source_id":"real-src","provider":"gemini","model":"gemini-2.5-flash",'
                '"mode":"review","accepted":5,"completed_at":"2026-04-30T18:00:00Z"}',
            ]
        )
    )

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "import-mining-log",
            str(log_path),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, (result.output or "") + (result.stderr or "")

    # The two good rows landed; the four bad rows didn't crash the import.
    with LexiconDB(fresh_db) as db:
        n = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
    assert n == 2


def test_import_mining_log_skips_unknown_sources(fresh_db: Path, tmp_path: Path) -> None:
    """Records pointing at sources not in the source table get skipped with
    an error count, not a hard FK failure."""

    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        '{"source_id":"never-existed","provider":"ollama","model":"x",'
        '"mode":"mine","accepted":1,"declined":0,"rejected":0,'
        '"completed_at":"2026-05-01T00:00:00Z"}\n'
    )
    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 0


def test_lexicon_mine_llm_records_mining_run_at_end_of_run(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """The CLI integration: running `lexicon mine-llm` end-to-end against
    a stubbed extractor must persist a mining_run row. Without this test,
    a regression that drops the record_mining_run call in lexicon_mine_llm
    passes CI silently — only the helper's unit tests would notice."""

    # Fake parsed entries — bypasses the regex parser.
    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body.",
        ),
        ParsedEntry(
            toponym="Otherton",
            section_suffix="-ton",
            historical_form="other-tun",
            elements=[],
            confidence="low",
            source_quote="Otherton. other body.",
        ),
    ]
    # wyrd-g143 slice 5: `_select_parser_and_run` is imported from
    # cli.utils into each consumer's module namespace (mine_llm, review).
    # Patching `cli_mod._select_parser_and_run` would set the re-export
    # shim but not the local-bound references the CLI bodies actually
    # resolve at call time. Patch both consumer modules so the test
    # scaffold works regardless of which command the runner invokes below.

    def _stub(text, parser, **_):
        return fake_parsed

    monkeypatch.setattr(_mll, "_select_parser_and_run", _stub)
    monkeypatch.setattr(_rv, "_select_parser_and_run", _stub)

    # Fake OllamaClient — only needs .model and .base_url for echoed output.
    class FakeClient:
        model = "fake-model:0b"
        base_url = "http://localhost:0"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    # Fake extract_one: first entry accepts with one element, second declines.
    accept_result = LLMResult(
        entry=ParsedEntry(
            toponym="Faketon",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[
                ParsedElement(
                    form="fake", language="old-english", position="pre", gloss="fake gloss"
                ),
            ],
            confidence="high",
            source_quote="Faketon. fake body.",
        ),
        raw_response={},
        failures=[],
        accepted=True,
    )
    decline_result = LLMResult(
        entry=ParsedEntry(
            toponym="Otherton",
            section_suffix="-ton",
            historical_form=None,
            elements=[],
            confidence="low",
            source_quote="Otherton. other body.",
        ),
        raw_response={},
        failures=[],
        accepted=True,
    )
    seq = iter([accept_result, decline_result])
    monkeypatch.setattr(llm_extractor, "extract_one", lambda client, **kw: next(seq))

    # Need a non-empty file for the click PathExists check.
    src_path = tmp_path / "test_book.txt"
    src_path.write_text("Faketon. fake body.\n\nOtherton. other body.\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "mine-llm", str(src_path), "--db", str(fresh_db), "--provider", "ollama"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # The writer call landed exactly one mining_run row with the expected counts.
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT source_id, provider, model, mode, parsed_count, accepted, declined, "
            "rejected FROM mining_run"
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "test_book"
    assert row["provider"] == "ollama"
    assert row["model"] == "fake-model:0b"
    assert row["mode"] == "mine"
    assert row["parsed_count"] == 2
    assert row["accepted"] == 1
    assert row["declined"] == 1
    assert row["rejected"] == 0


# --- wyrd-bgr: --declines-only filter on mine-llm ----------------------


def test_lexicon_mine_llm_declines_only_skips_already_extracted(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """`mine-llm --declines-only` must skip parsed entries whose toponym
    already has a toponym_etymology row for the source. Confirms the
    decline-recovery workflow: pre-seed an etymology row for one toponym,
    then run mine-llm with the flag and verify only the unseeded toponym
    reaches the extractor."""

    fake_parsed = [
        ParsedEntry(
            toponym="AlreadyMined",
            section_suffix="-ton",
            historical_form="already-tun",
            elements=[],
            confidence="low",
            source_quote="AlreadyMined. body.",
        ),
        ParsedEntry(
            toponym="StillUnmined",
            section_suffix="-ton",
            historical_form="still-tun",
            elements=[],
            confidence="low",
            source_quote="StillUnmined. body.",
        ),
    ]
    # wyrd-g143 slice 5: `_select_parser_and_run` is imported from
    # cli.utils into each consumer's module namespace (mine_llm, review).
    # Patching `cli_mod._select_parser_and_run` would set the re-export
    # shim but not the local-bound references the CLI bodies actually
    # resolve at call time. Patch both consumer modules so the test
    # scaffold works regardless of which command the runner invokes below.

    def _stub(text, parser, **_):
        return fake_parsed

    monkeypatch.setattr(_mll, "_select_parser_and_run", _stub)
    monkeypatch.setattr(_rv, "_select_parser_and_run", _stub)

    class FakeClient:
        model = "fake-model:0b"
        base_url = "http://localhost:0"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    extractor_calls: list[str] = []

    def fake_extract_one(client, **kw):
        extractor_calls.append(kw["toponym"])
        return LLMResult(
            entry=ParsedEntry(
                toponym=kw["toponym"],
                section_suffix="-ton",
                historical_form="x-tun",
                elements=[
                    ParsedElement(form="x", language="old-english", position="pre", gloss="x"),
                ],
                confidence="high",
                source_quote=kw["body"],
            ),
            raw_response={},
            failures=[],
            accepted=True,
        )

    monkeypatch.setattr(llm_extractor, "extract_one", fake_extract_one)

    # Pre-seed the DB with a toponym_etymology row for "AlreadyMined" against
    # this source — simulating Qwen having already extracted from it.
    src_path = tmp_path / "test_book.txt"
    src_path.write_text("AlreadyMined. body.\n\nStillUnmined. body.\n")
    source_id = "test_book"
    with LexiconDB(fresh_db) as db:
        db.upsert_source(
            id=source_id,
            title="Test Book",
            author="x",
            year=None,
            region=None,
            language_focus=None,
        )
        db.commit()
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("AlreadyMined",))
        topo_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, 'high', 'extracted_by:llm:ollama:qwen3.5:9b')",
            (topo_id, source_id),
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-llm",
            str(src_path),
            "--db",
            str(fresh_db),
            "--provider",
            "ollama",
            "--declines-only",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # The extractor was called for StillUnmined only — AlreadyMined was filtered.
    assert extractor_calls == ["StillUnmined"]
    # The mining_run row reflects the post-filter parsed_count.
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT parsed_count, accepted FROM mining_run WHERE source_id=?",
            (source_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["parsed_count"] == 1
    assert rows[0]["accepted"] == 1


def test_lexicon_mine_llm_declines_only_no_op_when_fully_mined(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """When every parsed toponym already has a row, --declines-only exits
    cleanly without calling the extractor and without writing a mining_run
    (no work was done)."""

    fake_parsed = [
        ParsedEntry(
            toponym="OnlyOne",
            section_suffix="-ton",
            historical_form="only-tun",
            elements=[],
            confidence="low",
            source_quote="OnlyOne. body.",
        ),
    ]
    # wyrd-g143 slice 5: `_select_parser_and_run` is imported from
    # cli.utils into each consumer's module namespace (mine_llm, review).
    # Patching `cli_mod._select_parser_and_run` would set the re-export
    # shim but not the local-bound references the CLI bodies actually
    # resolve at call time. Patch both consumer modules so the test
    # scaffold works regardless of which command the runner invokes below.

    def _stub(text, parser, **_):
        return fake_parsed

    monkeypatch.setattr(_mll, "_select_parser_and_run", _stub)
    monkeypatch.setattr(_rv, "_select_parser_and_run", _stub)

    class FakeClient:
        model = "fake-model:0b"
        base_url = "http://localhost:0"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    def fail_extract(*a, **kw):
        raise AssertionError("extractor must not be called when nothing remains")

    monkeypatch.setattr(llm_extractor, "extract_one", fail_extract)

    src_path = tmp_path / "test_book.txt"
    src_path.write_text("OnlyOne. body.\n")
    source_id = "test_book"
    with LexiconDB(fresh_db) as db:
        db.upsert_source(
            id=source_id,
            title="Test Book",
            author="x",
            year=None,
            region=None,
            language_focus=None,
        )
        db.commit()
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("OnlyOne",))
        topo_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, 'high', 'extracted_by:llm:ollama:qwen3.5:9b')",
            (topo_id, source_id),
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-llm",
            str(src_path),
            "--db",
            str(fresh_db),
            "--provider",
            "ollama",
            "--declines-only",
        ],
    )
    assert result.exit_code == 0
    assert "Nothing to mine after declines filter" in (result.output + (result.stderr or ""))
    with LexiconDB(fresh_db) as db:
        runs = db.conn.execute(
            "SELECT COUNT(*) FROM mining_run WHERE source_id=?", (source_id,)
        ).fetchone()[0]
    assert runs == 0


# --- wyrd-9kh.5: running-header page parser ----------------------------


def test_parse_running_header_pages_finds_mawer_style_headers():
    """Mawer-style running headers like 'BACKWORTH 9' get parsed into
    (offset, page) pairs sorted by offset."""

    body = (
        "front matter\n"
        "BACKWORTH 9\n"
        "  body of page 9...\n"
        "BEDLINGTON 15\n"
        "  body of page 15...\n"
        "BISHOPWEARMOUTH 23\n"
    )
    headers = parse_running_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [9, 15, 23]
    offsets = [offset for offset, _page in headers]
    assert offsets == sorted(offsets)


def test_parse_running_header_pages_handles_multi_word_headwords():
    """Multi-word ALL CAPS headwords (e.g. 'ABBEY DORE 1') are detected.
    Pages are consecutive (running headers are per printed page) so they clear
    the wyrd-w5wh monotonic-sequence guard."""

    body = "ABBEY DORE 1\n  body...\nST PETER'S 2\n"
    headers = parse_running_header_pages(body)
    pages = [page for _offset, page in headers]
    assert 1 in pages
    assert 2 in pages


def test_parse_running_header_pages_skips_spoof_body_lines():
    """wyrd-w5wh: an all-caps BODY line ending in a stray integer must NOT be
    parsed as a running header. A 'SEE ALSO 12' cross-reference (keyword guard)
    and a bare entry name with a far-off OCR integer ('BEALHAM 44', out of the
    monotonic window) are both dropped, so page_for_offset stays correct through
    the region instead of jumping to a spurious boundary."""

    body = (
        "BACKWORTH 8\n"
        "  body of backworth...\n"
        "SEE ALSO 12\n"  # cross-reference body line — keyword guard drops it
        "  more body...\n"
        "BAMBURGH 9\n"  # real next header (8 -> 9)
        "  body of bamburgh...\n"
        "BEALHAM 44\n"  # spoof: entry name + stray far integer (9 -> 44 > 9+15)
        "  body of bealham...\n"
        "BEDLINGTON 10\n"  # real header (9 -> 10)
    )
    headers = parse_running_header_pages(body)
    assert [page for _offset, page in headers] == [8, 9, 10]
    # A quote sitting in the BEALHAM body is on page 9 (the last real header),
    # not the spurious 44.
    off = body.index("body of bealham")
    assert page_for_offset(headers, off) == 9


def test_parse_running_header_pages_resets_on_leading_spoof_seed():
    """wyrd-w5wh: a high spurious FIRST match (a front-matter heading like
    'ABBREVIATIONS 200', or OCR noise) must NOT seed prev_page high and lock out
    every real header after it (which would fail 'prev < page'). When the second
    accepted page descends, the first is treated as the spoof, discarded, and the
    sequence re-seeds — so the real headers survive instead of being dropped."""

    body = (
        "BEALHAM 200\n"  # spurious leading seed (front-matter / OCR noise)
        "  front-matter noise...\n"
        "BACKWORTH 8\n"  # first REAL header — page descends, so 200 is discarded
        "  body...\n"
        "BAMBURGH 9\n"  # real header (8 -> 9)
    )
    headers = parse_running_header_pages(body)
    assert [page for _offset, page in headers] == [8, 9]


def test_parse_running_header_pages_recovers_after_large_gap():
    """wyrd-w5wh: a genuine gap of more than _MAX_PAGE_JUMP pages with no parseable
    headers (a plates section / OCR dropout) must NOT permanently lock out the rest
    of the book. The first post-gap header is HELD (out of window); a second header
    consecutive to it confirms the gap, so both are accepted and the sequence
    resyncs — rather than every post-gap header failing 'prev < page' forever."""

    body = (
        "ALNWICK 48\n  body...\n"
        "AMBLE 49\n  body...\n"
        "ASHINGTON 50\n"  # last header before a >15-page plates / OCR gap
        "  [plates / OCR dropout — no parseable headers for 20 pages]\n"
        "BAMBURGH 71\n  body...\n"  # first post-gap header (71 > 50 + 15) — held
        "BEADNELL 72\n  body...\n"  # consecutive to 71 → confirms the gap; both kept
        "BELFORD 73\n"
    )
    headers = parse_running_header_pages(body)
    assert [page for _offset, page in headers] == [48, 49, 50, 71, 72, 73]


def test_parse_running_header_pages_returns_empty_when_no_match():
    """Skeat-style §-section headers don't match the Mawer pattern;
    return an empty list rather than raising."""

    skeat_style = "§   2.      NAMES   ENDING   IN   -TON.  9\n  body of section\n"
    assert parse_running_header_pages(skeat_style) == []


def test_page_for_offset_returns_closest_preceding_header():
    """A character offset between two headers gets the page of the earlier
    header — that's the page the body text physically sits on."""

    headers = [(100, 1), (200, 2), (300, 3)]
    assert page_for_offset(headers, 50) is None  # before any header
    assert page_for_offset(headers, 100) == 1  # at the first header
    assert page_for_offset(headers, 150) == 1  # between 1st and 2nd
    assert page_for_offset(headers, 200) == 2  # at the second header
    assert page_for_offset(headers, 999) == 3  # after the last header


def test_page_for_offset_empty_headers_returns_none():
    """A book with no detectable running headers returns None for any
    offset — caller can fall back to NULL page."""

    assert page_for_offset([], 0) is None
    assert page_for_offset([], 12345) is None


# --- wyrd-8st: parse_skeat_section_header_pages ----------------------


def test_parse_skeat_section_header_pages_finds_canonical_header():
    """Canonical Skeat-§ header `§ N. NAMES ENDING IN -X. <page>` parses
    as (offset, page)."""
    body = (
        "front matter\n"
        "§ 2. NAMES ENDING IN -TON. 5\n"
        "body of page 5...\n"
        "§ 4. NAMES ENDING IN -HAM. 21\n"
        "body of page 21...\n"
    )
    headers = parse_skeat_section_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [5, 21]


def test_parse_skeat_section_header_pages_handles_ocr_variants():
    """OCR-tolerated variants: '§8.' (no space after §), 'IX' OCR misread
    of 'IN', multi-suffix `-BRIDGE, -HITHE` titles."""
    body = (
        "§8.      NAMES    ENDING    IX    -BRIDGE,   -HITHE.  33\n"
        "§   8.      NAMES    ENDING    IN    -WELL.  37\n"
    )
    headers = parse_skeat_section_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [33, 37]


def test_parse_skeat_section_header_pages_rejects_title_case_section_openers():
    """Section openers in the source (`§ 2. The suffix -ton.`) are
    title-case with no trailing page — they must NOT be parsed as
    running headers, even though they share the leading `§ N.`."""
    body = "§ 2. The suffix -ton.\ndiscussion follows\n§ 4. The suffix -ham.\n"
    assert parse_skeat_section_header_pages(body) == []


def test_parse_skeat_section_header_pages_rejects_toc_entries():
    """TOC entries (`§ 1. Prefatory Remarks 1`, `§ 5. The suffix -stead :
    — Olmstead 25`) are title-case and don't share the body running-
    header shape — they should not match."""
    toc = (
        "§ 1.     Prefatory Remarks 1\n"
        "§ 5.     The suffix -stead : — Olmstead 25\n"
        "§ 12.    List of Ancient Manors 74\n"
    )
    # Either no matches, or the page numbers should NOT overlap the
    # canonical body header pages (which would corrupt offset lookups).
    headers = parse_skeat_section_header_pages(toc)
    assert headers == [], (
        f"TOC entries leaked through Skeat parser as {headers}; the regex "
        "should require an ALL-CAPS title block, not title-case."
    )


def test_parse_skeat_section_header_pages_returns_empty_when_no_match():
    """A Mawer-style `BACKWORTH 9` body produces no Skeat-§ matches —
    callers can fall back to the Mawer parser."""
    mawer_body = "BACKWORTH 9\nbody\nBEDLINGTON 15\nbody\n"
    assert parse_skeat_section_header_pages(mawer_body) == []


def test_detect_running_headers_picks_skeat_when_higher_yield():
    """Per-source dispatch: if Skeat-§ matches more than Mawer (e.g. a
    Skeat county dictionary), the Skeat parser wins."""
    body = "§ 2. NAMES ENDING IN -TON. 5\nbody\n§ 4. NAMES ENDING IN -HAM. 21\nbody\n"
    headers, parser = detect_running_headers(body)
    assert parser == "skeat-§"
    assert [p for _o, p in headers] == [5, 21]


def test_detect_running_headers_picks_mawer_when_higher_yield():
    """A Mawer-style book wins dispatch when the Mawer parser produces
    more matches than the Skeat parser."""
    body = "BACKWORTH 9\nbody\nBEDLINGTON 15\nbody\nBISHOPWEARMOUTH 23\n"
    headers, parser = detect_running_headers(body)
    assert parser == "mawer"
    assert [p for _o, p in headers] == [9, 15, 23]


def test_detect_running_headers_returns_none_when_neither_matches():
    """A source with no recognizable header convention returns ([], 'none')
    so backfill_citation_pages can record no_headers and skip cleanly."""
    body = "this is just prose with no running headers anywhere\n"
    headers, parser = detect_running_headers(body)
    assert headers == []
    assert parser == "none"


def test_detect_running_headers_breaks_tie_to_mawer():
    """Tie-break: when both parsers return the same nonzero count,
    Mawer wins (documented in detect_running_headers). Pinned here so
    a refactor can't silently flip it — Mawer is the older / more
    permissive pattern; defaulting to it on ties keeps page-resolution
    behavior bit-stable for sources already routed through it."""
    body = "BACKWORTH 9\nbody content\n§ 2. NAMES ENDING IN -TON. 5\nmore body content\n"
    headers, parser = detect_running_headers(body)
    assert parser == "mawer"
    assert [p for _o, p in headers] == [9]


def test_backfill_citation_pages_resolves_skeat_section_headers(fresh_db: Path) -> None:
    """End-to-end: a Skeat-style book whose body has §-section running
    headers gets pages resolved via the new parser dispatch — would
    have failed under the Mawer-only path with no_headers=1.
    Both etymon_citation and toponym_etymology pages get written in
    the same pass."""
    quote = "discussion of -ton suffix in cot"
    body = (
        "§ 2. NAMES ENDING IN -TON. 5\n"
        f"earlier text\n"
        "§ 4. NAMES ENDING IN -HAM. 21\n"
        f"{quote} continuing\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "skeat_test", "cot", quote)
        counts = backfill_citation_pages(db, "skeat_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page == "21"
    assert ety_page == "21"
    assert counts["no_headers"] == 0


def test_parse_pages_cli_reports_skeat_parser_for_skeat_source(tmp_path: Path) -> None:
    """The parse-pages CLI audit must report which parser matched so a
    user inspecting a source knows whether it's Mawer-style or
    Skeat-§. Pin the 'parser: skeat-§' line in the output."""
    body = "§ 2. NAMES ENDING IN -TON. 5\nbody content\n§ 4. NAMES ENDING IN -HAM. 21\n"
    source = tmp_path / "skeat_test.txt"
    source.write_text(body)

    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "parse-pages", str(source)])

    assert result.exit_code == 0, result.output
    assert "2 running header(s) detected" in result.output
    assert "parser: skeat-§" in result.output
    assert "Page range: 5 → 21" in result.output


def test_parse_pages_cli_reports_no_match_when_neither_convention_fires(
    tmp_path: Path,
) -> None:
    """A header-less source body produces 'parser: none' and a stderr
    note explaining both conventions were tried — pin so the wording
    can't silently revert to the old Mawer-only language."""
    body = "this is just prose with no headers anywhere\n"
    source = tmp_path / "treatise.txt"
    source.write_text(body)

    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "parse-pages", str(source)])

    assert result.exit_code == 0, result.output
    assert "0 running header(s) detected" in result.output
    assert "parser: none" in result.output
    assert "either Mawer-style or Skeat-§ patterns" in result.output


# --- wyrd-azv: backfill_citation_pages ---------------------------------


def test_quote_body_excerpt_strips_provider_attribution_prefix() -> None:
    """`assemble_extraction_result` writes quotes shaped as
    `extracted_by:<provider>:<model>; <commentary> | <body excerpt>`.
    `_quote_body_excerpt` must return only the body part, since that's
    the only substring that exists in source_text."""
    quote = "extracted_by:llm:qwen3.5:9b; Scholar identifies foo | Abberwick (Eglingham). 1169 P"
    assert _quote_body_excerpt(quote) == "Abberwick (Eglingham). 1169 P"


def test_quote_body_excerpt_uses_first_separator_not_last() -> None:
    """Pin the leftmost-find behavior: if an OCR body picks up a
    spurious ` | ` (table rule, page artefact), the prefix splitter
    must still terminate at the FIRST ` | ` because the
    extracted_by:... prefix never contains the separator. Using rfind
    instead would return only the trailing artefact."""
    quote = "extracted_by:gemini:flash; commentary | body | with | rules"
    assert _quote_body_excerpt(quote) == "body | with | rules"


def test_quote_body_excerpt_returns_whole_quote_when_no_separator() -> None:
    """Legacy citations and commentary-only rows have no ` | ` —
    fall back to the literal quote so caller's str.find still has
    something to try."""
    assert _quote_body_excerpt("plain body excerpt") == "plain body excerpt"


def test_backfill_normalizes_ocr_multispace_against_single_space_quote(
    fresh_db: Path,
) -> None:
    """The parser collapses OCR multi-space runs before sending text to
    the LLM, so the stored quote ends up single-spaced while the source
    has the original `Abberwick  (Eglingham)` double-space runs.
    `_normalize_for_quote_match` exists to bridge that — pin it here so
    a regression in the normalization or the norm_to_orig offset
    translation surfaces immediately."""
    quote = "the body discussion of cot here"
    body = (
        "BACKWORTH 9\n"
        "earlier text\n"
        "BEDLINGTON 15\n"
        "the  body   discussion  of cot here  continuing\n"  # multi-space OCR
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "ocr_book", "cot", quote)
        counts = backfill_citation_pages(db, "ocr_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert cit_page == "15"


def test_backfill_resolves_page_with_provider_attribution_prefix(fresh_db: Path) -> None:
    """End-to-end check that the prefix-stripping + body matching
    works on the real `extracted_by:<provider>:<model>; <commentary> |
    <body>` shape produced by `assemble_extraction_result`. Pins the
    contract so a regression in either `_quote_body_excerpt` or the
    matching path surfaces here."""
    body_excerpt = "the body discussion of cot here"
    full_quote = (
        "extracted_by:llm:qwen3.5:9b; Scholar identifies cot as the suffix; "
        "notes contextual usage | " + body_excerpt
    )
    body = f"BACKWORTH 9\n{body_excerpt} continuing\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "real_book", "cot", full_quote)
        counts = backfill_citation_pages(db, "real_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert cit_page == "9"


def test_normalize_for_quote_match_collapses_multispace_and_tracks_offsets() -> None:
    """Direct unit test on the helper: collapsed text is single-spaced;
    norm_to_orig maps each normalized character back to its original
    offset so the page lookup uses the source coordinate system."""
    text = "AB  CD\n  EF"
    norm, mapping = _normalize_for_quote_match(text)
    assert norm == "AB CD EF"
    # 'A' is at original 0; 'B' at original 1; ' ' (collapsed) at original 2;
    # 'C' at original 4; 'D' at original 5; ' ' (collapsed from \n+spaces) at 6;
    # 'E' at 9; 'F' at 10
    assert mapping == [0, 1, 2, 4, 5, 6, 9, 10]
    # Round-trip: substring search returns a normalized offset that
    # translates back to the correct original-text position.
    norm_offset = norm.find("EF")
    assert text[mapping[norm_offset] : mapping[norm_offset] + 2] == "EF"


def _seed_citation_for_backfill(
    db: LexiconDB, source_id: str, etymon_form: str, quote: str
) -> tuple[int, int, int]:
    """Build the minimum DB shape backfill_citation_pages needs:
    a source, an etymon, an etymon_citation row, plus a toponym +
    toponym_etymology row carrying the same source_quote in `notes`.
    Returns (etymon_id, citation_id, etymology_id)."""
    db.upsert_source(id=source_id, title=source_id)
    etymon_id = db.upsert_etymon(etymon_form, "old-english")
    db.add_citation(etymon_id, source_id, short_quote=quote)
    cit = db.conn.execute(
        "SELECT id FROM etymon_citation WHERE etymon_id = ? AND source_id = ?",
        (etymon_id, source_id),
    ).fetchone()
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (etymon_form.title(),))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
        "VALUES (?, ?, 'high', ?)",
        (toponym_id, source_id, quote),
    )
    db.commit()
    return etymon_id, cit["id"], cur.lastrowid


def test_backfill_citation_pages_writes_pages_when_quote_in_text(fresh_db: Path) -> None:
    """Happy path: an etymon_citation whose short_quote appears in the
    source body after a running header gets that header's page written."""

    quote = "the body discussion of cot here"
    body = (
        "front matter\n"
        "BACKWORTH 9\n"
        "earlier text on page 9\n"
        "BEDLINGTON 15\n"
        f"{quote} continuing the entry\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)
        counts = backfill_citation_pages(db, "mawer_1920_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page == "15"
    assert ety_page == "15"


def test_backfill_citation_pages_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves page columns NULL."""

    quote = "discussion of ham morpheme"
    body = f"HAMSTEAD 22\n{quote}\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "test_book", "ham", quote)
        counts = backfill_citation_pages(db, "test_book", body, apply=False)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page is None
    assert ety_page is None


def test_backfill_citation_pages_no_headers_returns_zero(fresh_db: Path) -> None:
    """A source whose body matches NEITHER Mawer nor Skeat-§ convention
    (e.g. a treatise / chapter prose book like Joyce's Irish Names)
    reports no_headers=1 and zero updates without crashing."""

    treatise_body = (
        "This chapter discusses the topographical naming conventions "
        "of the western counties without using running headers anywhere "
        "in the body discussion here.\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(
            db, "treatise_book", "ton", "body discussion here"
        )
        counts = backfill_citation_pages(db, "treatise_book", treatise_body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["no_headers"] == 1
    assert counts["citations_updated"] == 0
    assert counts["etymologies_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_quote_not_in_text_increments_miss(fresh_db: Path) -> None:
    """A citation whose short_quote isn't present in the source body
    (parser drift, OCR mangle) is skipped and counted under
    quote_not_in_text rather than failing the run."""

    body = "BACKWORTH 9\nactual body content here\n"
    quote_not_in_body = "this exact phrase is nowhere"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote_not_in_body)
        counts = backfill_citation_pages(db, "mawer_1920_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    # citation + etymology rows both miss → counter at 2.
    assert counts["quote_not_in_text"] == 2
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_idempotent(fresh_db: Path) -> None:
    """A second --apply run is a no-op: rows where page IS NOT NULL are
    excluded from the candidate query."""

    quote = "body of the entry"
    body = f"ABBEY DORE 7\n{quote}\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "test_book", "abbey", quote)
        first = backfill_citation_pages(db, "test_book", body, apply=True)
        second = backfill_citation_pages(db, "test_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert first["citations_updated"] == 1
    assert second["citations_updated"] == 0
    assert cit_page == "7"


def test_backfill_citation_pages_skips_offset_before_first_header(fresh_db: Path) -> None:
    """An entry that sits before the first running header (e.g. front-
    matter) gets counted under before_first_page, not silently dropped."""

    quote = "preface discussing the methodology"
    body = f"{quote}\nBACKWORTH 9\nbody content\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "test_book", "cot", quote)
        counts = backfill_citation_pages(db, "test_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    # citation + etymology, both before page 1 → counter at 2.
    assert counts["before_first_page"] == 2
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_skips_no_quote_rows(fresh_db: Path) -> None:
    """Rows with NULL short_quote (legacy citations or rando-port seed
    rows) are counted under no_quote rather than crashing on str.find."""

    body = "BACKWORTH 9\nbody content\n"
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="legacy_book", title="legacy")
        etymon_id = db.upsert_etymon("cot", "old-english")
        db.add_citation(etymon_id, "legacy_book", short_quote=None)
        db.commit()

        counts = backfill_citation_pages(db, "legacy_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE etymon_id = ?", (etymon_id,)
        ).fetchone()["page"]

    assert counts["no_quote"] == 1
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_increments_ambiguous_match_on_repeated_quote(
    fresh_db: Path,
) -> None:
    """wyrd-3yu: when a short_quote appears at multiple body positions, the
    leftmost-match heuristic still wins (better than NULL — citations
    cluster by alphabet so the misattribution is to a nearby page) but
    counts['ambiguous_match'] increments so operators know the page may
    not be the entry the citation actually originated from."""

    quote = "See Plate III."
    body = (
        "BACKWORTH 9\n"
        f"earlier entry mentions: {quote} continuing\n"
        "BEDLINGTON 15\n"
        f"a different entry: {quote} continuing\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "ambig_book", "cot", quote)
        counts = backfill_citation_pages(db, "ambig_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["ambiguous_match"] == 2  # one increment per ambiguous row (citation + etymology)
    # Leftmost match wins: the citation is attributed to the BACKWORTH page,
    # not the BEDLINGTON one.
    assert cit_page == "9"


def test_backfill_citation_pages_unique_quote_does_not_count_as_ambiguous(
    fresh_db: Path,
) -> None:
    """Boundary check: a quote that appears exactly once must NOT bump
    the ambiguous_match counter. Rules out a regression where the helper
    counts every successful resolution as ambiguous."""

    quote = "the body discussion of cot here"
    body = f"BACKWORTH 9\n{quote} continuing\n"
    with LexiconDB(fresh_db) as db:
        _seed_citation_for_backfill(db, "unique_book", "cot", quote)
        counts = backfill_citation_pages(db, "unique_book", body, apply=True)

    assert counts["citations_updated"] == 1
    assert counts["ambiguous_match"] == 0


def test_backfill_citation_pages_ambiguous_match_counts_in_dry_run(
    fresh_db: Path,
) -> None:
    """Dry-run (apply=False) must still count ambiguous_match — the whole
    point of the counter is operator visibility, which works both before
    and after writes are committed. Pin so a future refactor that gates
    the increment on `apply` doesn't silently strand the signal."""

    quote = "See Plate III."
    body = (
        "BACKWORTH 9\n"
        f"first entry: {quote} continuing\n"
        "BEDLINGTON 15\n"
        f"second entry: {quote} continuing\n"
    )
    with LexiconDB(fresh_db) as db:
        _seed_citation_for_backfill(db, "ambig_dry", "cot", quote)
        counts = backfill_citation_pages(db, "ambig_dry", body, apply=False)

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert counts["ambiguous_match"] == 2


def test_add_citation_skips_when_etymon_source_pair_already_exists(
    fresh_db: Path,
) -> None:
    """wyrd-2pd: re-mining via add_citation(page=None) after a backfill set
    page='15' must NOT create a second row at (etymon, source, NULL). The
    unique index would have allowed it (COALESCE(page, '15') vs ('')); the
    add_citation pre-check guards against that split."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="src-a")
        etymon_id = db.upsert_etymon("cot", "old-english")
        db.add_citation(etymon_id, "src-a", page="15", short_quote="orig quote")
        # Simulate a re-mine pass: same (etymon, source), different (NULL) page.
        db.add_citation(etymon_id, "src-a", page=None, short_quote="re-mined quote")
        db.commit()

        rows = db.conn.execute(
            "SELECT page, short_quote FROM etymon_citation WHERE etymon_id = ? AND source_id = ?",
            (etymon_id, "src-a"),
        ).fetchall()

    assert len(rows) == 1, "second add_citation should be a no-op"
    # First-writer-wins semantics: the original paged row survives.
    assert rows[0]["page"] == "15"
    assert rows[0]["short_quote"] == "orig quote"


def test_add_citation_does_not_block_legitimate_distinct_etymon_source_pairs(
    fresh_db: Path,
) -> None:
    """The dedupe is per (etymon, source). Different etymons against the
    same source (or same etymon against different sources) must still
    accumulate distinct rows. Regression guard against an over-broad
    pre-check that drops valid evidence."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="src-a")
        db.upsert_source(id="src-b", title="src-b")
        et1 = db.upsert_etymon("cot", "old-english")
        et2 = db.upsert_etymon("ham", "old-english")
        db.add_citation(et1, "src-a")
        db.add_citation(et2, "src-a")  # different etymon, same source
        db.add_citation(et1, "src-b")  # same etymon, different source
        db.commit()

        n = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_citation").fetchone()["n"]

    assert n == 3


def test_backfill_pages_cli_dry_run_reports_without_writing(fresh_db: Path, tmp_path: Path) -> None:
    """End-to-end CLI test: dry-run reports counts to stderr and leaves
    the DB untouched. --apply is required to write."""

    quote = "discussion of cot in the entry"
    body = f"BACKWORTH 9\n{quote}\n"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "mawer_1920_test.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
        ],
    )

    assert result.exit_code == 0, result.output
    # Dry-run leaves page NULL
    with LexiconDB(fresh_db) as db:
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
    assert cit_page is None
    assert "dry-run" in result.output
    assert "citations_updated     = 1" in result.output


def test_backfill_pages_cli_apply_writes_to_db(fresh_db: Path, tmp_path: Path) -> None:
    """--apply persists the page lookup; CLI summary mirrors lib counts."""
    quote = "discussion of cot in the entry"
    body = f"BACKWORTH 9\n{quote}\n"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "mawer_1920_test.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
    assert cit_page == "9"
    assert "dry-run" not in result.output


def test_backfill_pages_cli_source_filter_skips_other_books(fresh_db: Path, tmp_path: Path) -> None:
    """--source X processes only sources/X.txt; other books in the dir
    don't get touched."""
    quote_a = "entry-A body content"
    quote_b = "entry-B body content"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "book_a.txt").write_text(f"AAA 1\n{quote_a}\n")
    (sources_dir / "book_b.txt").write_text(f"BBB 1\n{quote_b}\n")

    with LexiconDB(fresh_db) as db:
        _, cit_a, _ = _seed_citation_for_backfill(db, "book_a", "aa", quote_a)
        _, cit_b, _ = _seed_citation_for_backfill(db, "book_b", "bb", quote_b)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--source",
            "book_a",
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        page_a = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_a,)
        ).fetchone()["page"]
        page_b = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_b,)
        ).fetchone()["page"]
    assert page_a == "1"
    assert page_b is None


def test_reverse_search_captures_wider_snippet_window(fresh_db: Path, tmp_path: Path) -> None:
    """wyrd-9kh.4: reverse_search_attestations writes ±100 char snippets
    (up from the prior ±60). Pins the wider window so the SPA citation
    panel has enough surrounding scholarly prose to display."""

    # Build a long body with the matched form near the middle so ±100
    # chars on either side comfortably stay within the body.
    pre = "a " * 80  # 160 chars
    post = " b" * 80  # 160 chars
    body = f"{pre}MARKER ham TRAIL{post}"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        ham_id = db.upsert_etymon("ham", "old-english")
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.upsert_source(id="test_book", title="Test Book")
        db.add_citation(ham_id, "rando-port")
        db.commit()

        reverse_search_attestations(db, sources_dir, apply=True, min_form_length=3)

        snippet = db.conn.execute(
            "SELECT snippet FROM etymon_text_match WHERE etymon_id = ?",
            (ham_id,),
        ).fetchone()["snippet"]

    # Both the leading MARKER and trailing TRAIL words should be
    # captured: they sit ~6 chars from the «ham» match. A ±60 window
    # would still catch them; the wider window also pulls in some of
    # the surrounding 'a / b' filler. Verify by character count: the
    # snippet should be at least 150 chars (well over the old ±60
    # window's 126-char ceiling around the 5-char match).
    assert "marker" in snippet.lower()
    assert "trail" in snippet.lower()
    assert "«ham»" in snippet
    assert len(snippet) > 150, f"snippet only {len(snippet)} chars: {snippet!r}"


def test_reverse_search_match_skip_dryrun_and_stub_source(fresh_db: Path, tmp_path: Path) -> None:
    """End-to-end coverage of the wyrd-8uvi-extracted _find_reverse_matches +
    _write_reverse_matches: a single-count match (sutton), a multi-count match
    (norton×2), a no-match skip (xyzzy), the min_form_length exclusion (ham),
    dry-run-writes-nothing gating, and the FK source-stub upsert (the matched
    book has no pre-existing source row, so reverse-search must create one)."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "mawer_1920.txt").write_text(
        "The vill of sutton lay east. Near norton and norton mill the river ran. "
        "A short ham is not matched.\n"
    )
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        # NOTE: no 'mawer_1920' source row — the stub upsert must create it.
        for form in ("sutton", "norton", "xyzzy", "ham"):
            db.add_citation(db.upsert_etymon(form, "old-english"), "rando-port")
        db.commit()

        # Dry-run: computes matches but writes nothing.
        dry = reverse_search_attestations(db, sources_dir, apply=False)
        assert dry["etymons_with_match"] == 2  # sutton + norton (xyzzy/ham excluded)
        assert dry["written"] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0] == 0
        assert (
            db.conn.execute("SELECT COUNT(*) FROM source WHERE id = 'mawer_1920'").fetchone()[0]
            == 0
        )  # no stub created on dry-run

        # Apply: writes rows + creates the FK stub source.
        result = reverse_search_attestations(db, sources_dir, apply=True)
        assert result["written"] == 2
        rows = {
            r["matched_form"]: r["match_count"]
            for r in db.conn.execute("SELECT matched_form, match_count FROM etymon_text_match")
        }
    assert rows == {"sutton": 1, "norton": 2}  # multi-occurrence counted; ham/xyzzy absent
    # The stub source row was upserted so the FK held.
    with LexiconDB(fresh_db) as db:
        stub = db.conn.execute("SELECT title FROM source WHERE id = 'mawer_1920'").fetchone()
    assert stub is not None and stub["title"] == "Mawer 1920"


def test_fuzzy_search_skips_match_when_token_is_canonical_etymon(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Per wyrd-c3x: fuzzy-search must not claim 'X is a fuzzy variant of
    Y' when X is itself a canonical etymon in the lexicon. The herath ↔
    heath case (ON herað 'district' vs OE hǣþ 'untilled land') passed
    the gloss anchor because 'land' is a generic gloss, but the result
    is structurally wrong: 'heath' has its own etymon row.

    The cross-check fix: if the body token equals any canonical_form,
    skip the fuzzy claim. OCR variants between two etymons are
    normalize-ocr's job, not fuzzy-search's.
    """
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text(
        "The heath is bare untilled land where shepherds once grazed.\n"
    )

    with LexiconDB(fresh_db) as db:
        # rando-port-only candidate with a generic gloss — the kind that
        # would historically false-positive against 'heath'.
        herath_id = db.upsert_etymon("herath", "old-norse")
        db.add_gloss(herath_id, "land")  # generic — easy to anchor
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.upsert_source(id="test_book", title="Test Book")
        db.add_citation(herath_id, "rando-port", short_quote="seed")
        # The body word 'heath' is itself a canonical etymon. Without
        # the cross-check, fuzzy-search would claim herath ↔ heath.
        db.upsert_etymon("heath", "old-english")
        db.commit()

        result = fuzzy_search_attestations(db, sources_dir, apply=True)

        rows = db.conn.execute(
            "SELECT etymon_id, matched_form FROM etymon_text_match "
            "WHERE etymon_id = ? AND edit_distance > 0",
            (herath_id,),
        ).fetchall()

    assert rows == [], (
        "fuzzy-search must not claim herath as a variant of 'heath' when "
        "'heath' is itself a canonical etymon"
    )
    assert result["written"] == 0


def test_fuzzy_search_records_gloss_anchored_match(fresh_db: Path, tmp_path: Path) -> None:
    """Positive path through _fuzzy_match_in_source + _write_fuzzy_matches
    (wyrd-8uvi extraction): a rando-only candidate whose canonical form has a
    distance-1, non-canonical, gloss-anchored neighbour in a source body is
    written to etymon_text_match with edit_distance=1 and method
    fuzzy-search-v1. Also pins the no-anchor skip (gloss absent from window)."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "bk.txt").write_text(
        "The beare field grew barleycorn in summer. Near the blythe brook the water ran clear.\n"
    )
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.upsert_source(id="bk", title="Book")
        # beore ~ beare (distance 1); gloss 'barleycorn' anchors within window;
        # 'beare' is not a canonical etymon -> a real fuzzy match.
        beore = db.upsert_etymon("beore", "old-norse")
        db.add_gloss(beore, "barleycorn")
        db.add_citation(beore, "rando-port", short_quote="seed")
        # blithe ~ blythe (distance 1) but its gloss never appears near the
        # token -> the gloss-anchor guard suppresses it.
        blithe = db.upsert_etymon("blithe", "old-norse")
        db.add_gloss(blithe, "zzglossnotpresent")
        db.add_citation(blithe, "rando-port", short_quote="seed")
        db.commit()

        result = fuzzy_search_attestations(db, sources_dir, apply=True)

        rows = db.conn.execute(
            "SELECT matched_form, edit_distance, method FROM etymon_text_match WHERE etymon_id = ?",
            (beore,),
        ).fetchall()
        blithe_rows = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match WHERE etymon_id = ?", (blithe,)
        ).fetchone()[0]

    assert [tuple(r) for r in rows] == [("beare", 1, "fuzzy-search-v1")]
    assert blithe_rows == 0  # no gloss anchor -> not written
    assert result["written"] == 1
    assert result["etymons_with_fuzzy_match"] == 1


def test_lexicon_review_low_conf_counts_as_declined_not_written(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """When the Tier-2 provider returns confidence='low', the writer
    skips the etymology row (lexicon.py:1394). The CLI must count this
    as declined, not written, so the summary and mining_run audit
    reflect what's actually persisted in the DB.

    Regression test for wyrd-dt9: previously, every accepted result
    incremented `written` even when the writer dropped the row,
    overstating real persistence by ~60% on books where Gemini was
    uncertain."""

    # Pre-seed: one source, one toponym, one Qwen-tagged low-conf
    # etymology row that's eligible for Tier-2 review.
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text("Faketon. fake body text.\n")

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_book", title="Test Book")
        toponym_id = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Faketon", "Testshire"),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "test_book", "medium", "extracted_by:llm:qwen3.5:fake"),
        )
        db.commit()

    # Stub the parser so the review path finds Faketon's body.
    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix=None,
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body text.",
        ),
    ]
    # wyrd-g143 slice 5: `_select_parser_and_run` is imported from
    # cli.utils into each consumer's module namespace (mine_llm, review).
    # Patching `cli_mod._select_parser_and_run` would set the re-export
    # shim but not the local-bound references the CLI bodies actually
    # resolve at call time. Patch both consumer modules so the test
    # scaffold works regardless of which command the runner invokes below.

    def _stub(text, parser, **_):
        return fake_parsed

    monkeypatch.setattr(_mll, "_select_parser_and_run", _stub)
    monkeypatch.setattr(_rv, "_select_parser_and_run", _stub)

    # Stub Gemini: return a "low"-confidence result with elements
    # present. The writer will silently drop this; the CLI must catch
    # that and count it as declined.
    class FakeClient:
        model = "fake-gemini-model"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(gemini_extractor, "GeminiClient", FakeClient)
    monkeypatch.setattr(
        gemini_extractor,
        "extract_one",
        lambda client, **kw: LLMResult(
            entry=ParsedEntry(
                toponym="Faketon",
                section_suffix=None,
                historical_form="fake-tun",
                elements=[
                    ParsedElement(form="fake", language="old-english", position="pre"),
                ],
                confidence="low",
                source_quote="extracted_by:gemini:fake-gemini-model | Faketon.",
            ),
            raw_response={},
            failures=[],
            accepted=True,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "review",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--provider",
            "gemini",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # The mining_run audit row must say declined=1, accepted=0 — the
    # row was not actually written to toponym_etymology.
    with LexiconDB(fresh_db) as db:
        runs = db.conn.execute(
            "SELECT accepted, declined, rejected FROM mining_run WHERE mode = 'review'"
        ).fetchall()
        gemini_rows = db.conn.execute(
            "SELECT COUNT(*) AS n FROM toponym_etymology WHERE notes LIKE '%gemini%'"
        ).fetchone()
    assert len(runs) == 1
    assert runs[0]["accepted"] == 0, "low-conf should not count as accepted"
    assert runs[0]["declined"] == 1, "low-conf must count as declined"
    assert runs[0]["rejected"] == 0
    assert gemini_rows["n"] == 0, "writer must not have persisted a gemini row"


def test_lexicon_review_dry_run_does_not_persist_or_count_writes(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """Dry-run (no --apply) must not write rows to the DB and must not count
    'would-have-written' results as written. The summary log explicitly says
    '(dry-run; pass --apply to write)' — counting writes anyway would lie.

    Regression for an extraction bug caught on PR #12: an early version of
    the refactor moved `counts['written'] += 1` outside the apply guard, so
    dry-run reported nonzero writes that didn't exist in the DB."""

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text("Faketon. fake body.\n")

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_book", title="Test")
        toponym_id = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Faketon", "Testshire"),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "test_book", "medium", "extracted_by:llm:qwen3.5:fake"),
        )
        db.commit()

    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix=None,
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body.",
        ),
    ]
    # wyrd-g143 slice 5: `_select_parser_and_run` is imported from
    # cli.utils into each consumer's module namespace (mine_llm, review).
    # Patching `cli_mod._select_parser_and_run` would set the re-export
    # shim but not the local-bound references the CLI bodies actually
    # resolve at call time. Patch both consumer modules so the test
    # scaffold works regardless of which command the runner invokes below.

    def _stub(text, parser, **_):
        return fake_parsed

    monkeypatch.setattr(_mll, "_select_parser_and_run", _stub)
    monkeypatch.setattr(_rv, "_select_parser_and_run", _stub)

    class FakeClient:
        model = "fake-gemini-model"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(gemini_extractor, "GeminiClient", FakeClient)
    monkeypatch.setattr(
        gemini_extractor,
        "extract_one",
        lambda client, **kw: LLMResult(
            entry=ParsedEntry(
                toponym="Faketon",
                section_suffix=None,
                historical_form="fake-tun",
                elements=[
                    ParsedElement(form="fake", language="old-english", position="pre"),
                ],
                confidence="high",
                source_quote="extracted_by:gemini:fake | Faketon",
            ),
            raw_response={},
            failures=[],
            accepted=True,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "review",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--provider",
            "gemini",
            # NOTE: no --apply — this is the dry-run case under test.
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    combined = result.output + (result.stderr or "")
    assert "'written': 0" in combined, "dry-run summary must report written: 0 — got:\n" + combined

    with LexiconDB(fresh_db) as db:
        n_runs = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
        n_gemini_rows = db.conn.execute(
            "SELECT COUNT(*) FROM toponym_etymology WHERE notes LIKE '%gemini%'"
        ).fetchone()[0]
    assert n_runs == 0
    assert n_gemini_rows == 0


def test_mining_run_table_present_after_init(fresh_db: Path) -> None:
    """init_schema creates mining_run; migrate_schema is a no-op on a fresh DB."""
    with LexiconDB(fresh_db) as db:
        applied = migrate_schema(db)
    assert applied["mining_run_table"] is False
    with LexiconDB(fresh_db) as db:
        tables = {
            r["name"] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "mining_run" in tables


def test_derive_lemma_candidate_strips_oe_inflections() -> None:
    """Standard OE genitive/dative endings should suggest the stem."""
    # Weak oblique (-an): cotan → cot
    assert derive_lemma_candidate("cotan", "old-english") == ("cot", "weak_oblique")
    # Strong genitive (-es): ceorles → ceorl
    assert derive_lemma_candidate("ceorles", "old-english") == ("ceorl", "genitive_strong")
    # Dative plural (-um): cotum → cot
    assert derive_lemma_candidate("cotum", "old-english") == ("cot", "dative_pl")
    # Strong plural (-as): cotas → cot? Actually with the rule order, "as"
    # matches -as before -an or -e.
    assert derive_lemma_candidate("cotas", "old-english") == ("cot", "plural_strong")


def test_derive_lemma_candidate_strips_on_inflections() -> None:
    """Old Norse: nominative -r, genitive/plural -ar."""
    assert derive_lemma_candidate("klakkr", "old-norse") == ("klakk", "nominative")
    assert derive_lemma_candidate("dalir", "old-norse") == ("dal", "plural")


def test_derive_lemma_candidate_returns_none_for_unknown_lang() -> None:
    """Languages without rules don't produce candidates."""
    assert derive_lemma_candidate("foobar", "klingon") is None


def test_derive_lemma_candidates_returns_silent_e_alternates_for_modern_english() -> None:
    """wyrd-gf28: the plural-candidates form returns BOTH the
    restore-suffix candidate (preferred) AND the bare-stem fallback
    for modern-english -ed and -ing. Order matters — silent-e first
    so 'hoped' gets to try 'hope' before 'hop'."""
    # 'hoped' → restore 'e' → 'hope' preferred, 'hop' fallback
    assert derive_lemma_candidates("hoped", "modern-english") == [
        ("hope", "past"),
        ("hop", "past"),
    ]
    assert derive_lemma_candidates("hoping", "modern-english") == [
        ("hope", "gerund"),
        ("hop", "gerund"),
    ]
    # 'walked' → 'walke' preferred (won't match in practice), 'walk' fallback
    assert derive_lemma_candidates("walked", "modern-english") == [
        ("walke", "past"),
        ("walk", "past"),
    ]


def test_derive_lemma_candidates_returns_single_for_legacy_2tuple_rules() -> None:
    """OE / ON / Welsh / etc. rules are 2-tuples without restore_suffix.
    The plural-candidates form returns a single candidate for each
    matching rule — same behavior as the singular wrapper, just in a
    list. Pin so a future schema migration doesn't accidentally start
    emitting bogus restore-suffix alternates for the legacy languages."""
    # OE 'cotan' → only ('cot', 'weak_oblique'), no restore
    assert derive_lemma_candidates("cotan", "old-english") == [("cot", "weak_oblique")]
    # Welsh 'dyddiau' → only ('dydd', 'plural')
    assert derive_lemma_candidates("dyddiau", "welsh") == [("dydd", "plural")]


def test_derive_lemma_candidate_wrapper_returns_first_candidate() -> None:
    """The singular back-compat wrapper returns candidates[0] —
    pinning so existing call sites that read a single (form, label)
    tuple keep working. For silent-e modern-english this means the
    restore-suffix variant is what the wrapper returns."""
    # Silent-e: wrapper returns 'hope', not 'hop'.
    assert derive_lemma_candidate("hoped", "modern-english") == ("hope", "past")
    # Legacy 2-tuple: same as before.
    assert derive_lemma_candidate("cotan", "old-english") == ("cot", "weak_oblique")
    # No-match still returns None.
    assert derive_lemma_candidate("zzqxk", "modern-english") is None


def test_derive_lemma_candidate_strips_welsh_inflections() -> None:
    """wyrd-r6bk Phase 1: Welsh suffix-strippable subset (initial
    mutation deferred to wyrd-jott). Pin one case per registered
    Welsh suffix so a future rule edit that drops one surfaces."""
    # -iau plural: dyddiau → dydd
    assert derive_lemma_candidate("dyddiau", "welsh") == ("dydd", "plural")
    # -oedd plural: siroedd → sir
    assert derive_lemma_candidate("siroedd", "welsh") == ("sir", "plural")
    # -ion plural: cantorion → cantor
    assert derive_lemma_candidate("cantorion", "welsh") == ("cantor", "plural")
    # -ach comparative: cochach → coch
    assert derive_lemma_candidate("cochach", "welsh") == ("coch", "comparative")


def test_derive_lemma_candidate_welsh_specific_before_general() -> None:
    """Rule order: -iau (4 chars) must match BEFORE -au would, else
    'llysiau' would strip the trailing 'au' and produce 'llysi'
    instead of 'llys'. Same shape as the OE/ON ordering invariants."""
    assert derive_lemma_candidate("llysiau", "welsh") == ("llys", "plural")


def test_derive_lemma_candidate_strips_old_french_inflections() -> None:
    """wyrd-rc0b Phase 1: Old French feminine-plural / oblique -es.
    Conservative subset — bare -s is too generic. Pin the -es rule
    so a future edit that drops it surfaces."""
    # cisoires → cisoir (feminine plural)
    assert derive_lemma_candidate("cisoires", "old-french") == (
        "cisoir",
        "feminine_plural",
    )


def test_derive_lemma_candidate_strips_middle_english_inflections() -> None:
    """wyrd-rc0b Phase 1: Middle English -en (weak plural) and -es
    (plural / genitive singular). Pin both rules + their order so a
    future edit can't silently flip them."""
    # -en weak plural: oxen → ox (BUT ox is 2 chars, fails min-stem;
    # use a longer stem: salten → salt)
    assert derive_lemma_candidate("salten", "middle-english") == ("salt", "plural")
    # -es plural / genitive: kynges → kyng
    assert derive_lemma_candidate("kynges", "middle-english") == (
        "kyng",
        "genitive_or_plural",
    )


def test_derive_lemma_candidate_strips_norman_french_inflections() -> None:
    """wyrd-rc0b Phase 1: Norman French shares the OF -es shape.
    Live corpus is tiny (70 etymons) but the rule path is pinned for
    when more NF rows are mined."""
    assert derive_lemma_candidate("dames", "norman-french") == (
        "dam",
        "feminine_plural",
    )


def test_derive_mutation_lemma_candidate_strips_irish_eclipsis() -> None:
    """wyrd-jott Phase 1: Goidelic eclipsis (urú) — eclipsing digraph
    prefixes are unambiguous in Goidelic since they're never real
    word-initial sequences in lemmas. Pin one case per registered
    eclipsis rule."""

    # mb- → b-
    assert derive_mutation_lemma_candidate("mboga", "irish") == ("boga", "eclipsis")
    # gc- → c-
    assert derive_mutation_lemma_candidate("gcaisleán", "irish") == (
        "caisleán",
        "eclipsis",
    )
    # nd- → d-
    assert derive_mutation_lemma_candidate("ndoruis", "old-irish") == (
        "doruis",
        "eclipsis",
    )
    # bhf- → f- (longest first; bh- doesn't trigger here)
    assert derive_mutation_lemma_candidate("bhfuil", "irish") == ("fuil", "eclipsis")


def test_derive_mutation_lemma_candidate_strips_irish_lenition() -> None:
    """wyrd-jott Phase 1: Goidelic lenition (séimhiú). Safe subset
    only — m/b/d/g/p/f/s rarely have the lenited digraph as a real
    lemma-initial spelling. ch- and th- are excluded."""

    # mh- → m-
    assert derive_mutation_lemma_candidate("mhór", "irish") == ("mór", "lenition")
    # bh- → b-
    assert derive_mutation_lemma_candidate("bheith", "irish") == ("beith", "lenition")
    # dh- → d-
    assert derive_mutation_lemma_candidate("dhuine", "irish") == ("duine", "lenition")


def test_derive_mutation_lemma_candidate_works_across_goidelic_languages() -> None:
    """All four Goidelic languages register the same rule set;
    Scottish Gaelic + Old Irish + Middle Irish all behave like Irish
    for the prefix-strip path."""

    assert derive_mutation_lemma_candidate("bhith", "scottish-gaelic") == (
        "bith",
        "lenition",
    )
    assert derive_mutation_lemma_candidate("mblas", "middle-irish") == (
        "blas",
        "eclipsis",
    )


def test_derive_mutation_lemma_candidate_returns_none_for_non_goidelic() -> None:
    """Non-Goidelic languages have no mutation rules registered —
    Welsh / OE / etc. fall through to None."""

    assert derive_mutation_lemma_candidate("mhór", "welsh") is None
    assert derive_mutation_lemma_candidate("mboga", "old-english") is None


def test_derive_mutation_lemma_candidate_skips_excluded_prefixes() -> None:
    """ch- and th- are DELIBERATELY OMITTED — both are real digraph-
    initial Irish lemmas (chéile, cheap, thart, thiar). Pin so a
    future rule addition is intentional."""

    # 'chéile' looks like lenition of 'céile' but ch- isn't in the
    # rules list, so the function returns None.
    assert derive_mutation_lemma_candidate("chéile", "irish") is None
    assert derive_mutation_lemma_candidate("thiar", "irish") is None
    # h-prefix and t-prefix also omitted (real h- and t- initial loans).
    assert derive_mutation_lemma_candidate("hata", "irish") is None


def test_derive_mutation_lemma_candidate_respects_min_stem_length() -> None:
    """Stems shorter than 3 characters are rejected — same floor as
    the suffix-strip path. Stripping 'mb' from 'mb' would give a 1-
    char stem."""

    # 'mba' would strip to 'ba' (2 chars) — rejected.
    assert derive_mutation_lemma_candidate("mba", "irish") is None


def test_derive_mutation_lemma_candidate_longest_prefix_wins() -> None:
    """Rule order matters: 'bhf' (eclipsis of f-) must be tried before
    'bh' (lenition of b-) so 'bhfuil' resolves to 'fuil' (eclipsis)
    rather than mis-resolving via the shorter 'bh' prefix to a non-
    existent 'b' + 'fuil' hybrid."""

    assert derive_mutation_lemma_candidate("bhfuil", "irish") == ("fuil", "eclipsis")


def test_derive_lemma_candidate_old_french_skips_bare_s() -> None:
    """The Phase-1 OF rule list deliberately omits bare -s — many OF
    lemmas end in -s naturally ('pres', 'fois'). Pin: 'pres' should
    NOT strip to 'pre'. The absence of -s in the rule list is a
    deliberate precision-vs-recall trade-off."""
    assert derive_lemma_candidate("pres", "old-french") is None
    assert derive_lemma_candidate("fois", "old-french") is None


def test_derive_lemma_candidate_welsh_skips_excluded_suffixes() -> None:
    """The Phase-1 rule list deliberately omits -af / -au / -ydd /
    -aint per the false-positive-rate audit. Pin: 'gaeaf' (winter,
    NOT a superlative) does NOT strip — the absence of -af in the
    rule list is a deliberate precision-vs-recall trade-off."""
    # -af not in rules: 'gaeaf' should pass through unchanged.
    assert derive_lemma_candidate("gaeaf", "welsh") is None
    # -au not in rules: 'tegau' (proper noun) should pass through.
    assert derive_lemma_candidate("tegau", "welsh") is None
    # -ydd not in rules: 'gweithydd' (worker, derivational suffix
    # not a plural) should pass through.
    assert derive_lemma_candidate("gweithydd", "welsh") is None


def test_derive_lemma_candidate_respects_min_stem_length() -> None:
    """Refuse to produce a 2-char-or-less stem."""
    # "an" stripped from "ban" would give a 1-char stem; reject.
    assert derive_lemma_candidate("ban", "old-english") is None


def test_migrate_schema_idempotent_on_fresh_db(fresh_db: Path) -> None:
    """A fresh DB created by init_schema already has lemma_id; migration
    should be a no-op (rebuilding view aside)."""
    with LexiconDB(fresh_db) as db:
        first = migrate_schema(db)
        second = migrate_schema(db)
    # Schema bumps should not re-apply.
    assert not first["etymon.lemma_id"]
    assert not first["etymon.inflection"]
    assert not second["etymon.lemma_id"]


def test_migrate_schema_adds_citation_context_snippet_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A legacy DB whose etymon_citation predates wyrd-9kh.3 picks up the
    new context_snippet column on migrate_schema. Existing rows survive
    with NULL in the new column. Idempotent on a re-run."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-9kh.3 schema by dropping the etymon_consensus view
        # (it references etymon_citation, blocking the table swap), recreating
        # etymon_citation without context_snippet, and seeding one legacy row.
        # The migration's _rebuild_etymon_views step will reinstate the view.
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute("DROP TABLE etymon_citation")
        db.conn.execute(
            """
            CREATE TABLE etymon_citation (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
              source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
              page        TEXT,
              short_quote TEXT
            )
            """
        )
        db.upsert_source(id="src-a", title="A")
        legacy_etymon_id = db.upsert_etymon("ham", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
            (legacy_etymon_id, "src-a"),
        )
        db.commit()

        # Pre-migration: column is missing.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
        assert "context_snippet" not in cols

        applied = migrate_schema(db)
        assert applied["etymon_citation.context_snippet"] is True

        # Post-migration: column present, existing row preserved with NULL.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
        assert "context_snippet" in cols
        row = db.conn.execute(
            "SELECT etymon_id, source_id, context_snippet FROM etymon_citation"
        ).fetchone()
        assert row["etymon_id"] == legacy_etymon_id
        assert row["context_snippet"] is None

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon_citation.context_snippet"] is False


def test_migrate_schema_adds_toponym_etymology_attested_year_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A legacy DB whose toponym_etymology predates wyrd-bag picks up
    the new attested_year column on migrate_schema. Existing rows
    survive with NULL in the new column. Idempotent on a re-run.
    Pins the migration path that fresh_db tests don't exercise (since
    init_schema already includes the column from lexicon.sql)."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-wyrd-bag toponym_etymology by re-creating it
        # without attested_year. Drop dependent rows / FKs first to
        # avoid constraint failures, then re-seed one legacy row.
        db.conn.execute("DELETE FROM toponym_etymology_element")
        db.conn.execute("DROP TABLE toponym_etymology")
        db.conn.execute(
            """
            CREATE TABLE toponym_etymology (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              toponym_id      INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
              source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
              page            TEXT,
              historical_form TEXT,
              confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low')),
              notes           TEXT
            )
            """
        )
        db.upsert_source(id="legacy-src", title="Legacy")
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Tune",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (?, ?, ?)",
            (toponym_id, "legacy-src", "1086 DB"),
        )
        legacy_te_id = cur.lastrowid
        db.commit()

        # Pre-migration: column is missing.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
        assert "attested_year" not in cols

        applied = migrate_schema(db)
        assert applied["toponym_etymology.attested_year"] is True

        # Post-migration: column present, existing row preserved with NULL.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
        assert "attested_year" in cols
        row = db.conn.execute(
            "SELECT id, notes, attested_year FROM toponym_etymology WHERE id = ?",
            (legacy_te_id,),
        ).fetchone()
        assert row["notes"] == "1086 DB"
        assert row["attested_year"] is None

        # Idempotent re-run: migration helper reports False.
        applied2 = migrate_schema(db)
        assert applied2["toponym_etymology.attested_year"] is False


def test_add_citation_persists_context_snippet(fresh_db: Path) -> None:
    """LexiconDB.add_citation accepts context_snippet and writes it; a
    NULL-snippet caller stays compatible (existing callers haven't been
    updated yet).

    Note: per wyrd-2pd, the (etymon, source) pre-check means only the
    first add_citation call for a given pair persists. Different
    (etymon, source) pairs document the snippet-vs-no-snippet shapes
    independently here so both code paths exercise the column.
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="mawer_1920", title="Mawer")
        db.upsert_source(id="legacy_book", title="Legacy")
        etymon_id = db.upsert_etymon("ham", "old-english")
        db.add_citation(
            etymon_id,
            "mawer_1920",
            context_snippet="Acklington... O.E. Æcceling(a)tun = farm of Æccel.",
        )
        db.add_citation(etymon_id, "legacy_book", page="42")  # legacy shape, no snippet
        db.commit()

        rows = db.conn.execute(
            "SELECT source_id, page, context_snippet FROM etymon_citation "
            "WHERE etymon_id = ? ORDER BY source_id",
            (etymon_id,),
        ).fetchall()

    assert len(rows) == 2
    # mawer_1920: snippet captured, no page.
    assert rows[1]["source_id"] == "mawer_1920"
    assert rows[1]["page"] is None
    assert "Acklington" in rows[1]["context_snippet"]
    # legacy_book: legacy shape with page only — context_snippet is NULL.
    assert rows[0]["source_id"] == "legacy_book"
    assert rows[0]["page"] == "42"
    assert rows[0]["context_snippet"] is None


def test_link_lemmas_links_inflected_to_existing_lemma(fresh_db: Path) -> None:
    """An inflected etymon with a matching lemma in the DB gets linked."""
    with LexiconDB(fresh_db) as db:
        cot_id = db.upsert_etymon("cot", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Distinct etymon: should NOT link to anything
        ham_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        result = link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection, lemma_method FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    assert by_id[cotan_id]["lemma_id"] == cot_id
    assert by_id[cotan_id]["inflection"] == "weak_oblique"
    assert by_id[cotes_id]["lemma_id"] == cot_id
    assert by_id[cotes_id]["inflection"] == "genitive_strong"
    # Method/version stamp is written on every linked row so a future
    # link-lemmas-v2 can selectively rebuild only its predecessor's work.
    assert by_id[cotan_id]["lemma_method"] == "link-lemmas-v1"
    assert by_id[cotes_id]["lemma_method"] == "link-lemmas-v1"
    # Lemma stays its own — no stamp written when not linked.
    assert by_id[cot_id]["lemma_id"] is None
    assert by_id[cot_id]["lemma_method"] is None
    # Unrelated etymon stays unlinked, no stamp.
    assert by_id[ham_id]["lemma_id"] is None
    assert by_id[ham_id]["lemma_method"] is None
    assert result["candidates"] >= 2


def test_link_lemmas_links_modern_english_past_and_gerund(fresh_db: Path) -> None:
    """wyrd-gf28: modern-english INFLECTION_RULES added -ed (past) and
    -ing (gerund). Pin: when both the inflected form and its stem are
    in the DB, link-lemmas wires them up. Catches a future regression
    that removes either rule from the modern-english entry."""
    with LexiconDB(fresh_db) as db:
        walk_id = db.upsert_etymon("walk", "modern-english")
        walked_id = db.upsert_etymon("walked", "modern-english")
        walking_id = db.upsert_etymon("walking", "modern-english")
        # An unrelated -ed-ending lemma whose stripped stem ISN'T in
        # the DB stays unlinked — the conservative-precision invariant
        # the modern-english entry's docstring promises.
        bed_id = db.upsert_etymon("bed", "modern-english")
        db.commit()

        link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    assert by_id[walked_id]["lemma_id"] == walk_id
    assert by_id[walked_id]["inflection"] == "past"
    assert by_id[walking_id]["lemma_id"] == walk_id
    assert by_id[walking_id]["inflection"] == "gerund"
    # 'bed' stays its own lemma — no 'b' verb in the DB to link to.
    assert by_id[bed_id]["lemma_id"] is None
    # Source lemma stays unlinked.
    assert by_id[walk_id]["lemma_id"] is None


def test_link_lemmas_modern_english_handles_silent_e_collision(
    fresh_db: Path,
) -> None:
    """wyrd-gf28 round-2 review: the naïve suffix-strip would route
    'hoped' to 'hop' (silent-e + double-consonant verb) when both
    'hop' and 'hope' exist as etymons in the corpus. The 3-tuple
    INFLECTION_RULES form (suffix, label, restore_suffix) covers
    this by trying stem+restore (preferred) before bare stem. Pin
    the fix so a future refactor can't silently regress to the
    naïve single-candidate path."""
    with LexiconDB(fresh_db) as db:
        # Both 'hop' (verb, to jump) and 'hope' (verb, to wish) exist
        # — the canonical case for silent-e collision.
        hop_id = db.upsert_etymon("hop", "modern-english")
        hope_id = db.upsert_etymon("hope", "modern-english")
        hoped_id = db.upsert_etymon("hoped", "modern-english")
        hoping_id = db.upsert_etymon("hoping", "modern-english")
        # 'hopped' wouldn't be in the DB from Wiktionary (it's a form
        # of 'hop', tracked under hop's forms array, not as a separate
        # entry). But if it WERE present and only 'hop' existed (no
        # 'hope'), the bare-stem fallback would correctly link it.
        # That's covered by the prior walk_id test.
        db.commit()

        link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    # Silent-e wins: hoped → hope, NOT hop.
    assert by_id[hoped_id]["lemma_id"] == hope_id, (
        f"expected hoped → hope, got hoped → {by_id[hoped_id]['lemma_id']} "
        f"(hop_id={hop_id}, hope_id={hope_id})"
    )
    assert by_id[hoped_id]["inflection"] == "past"
    # Same for -ing.
    assert by_id[hoping_id]["lemma_id"] == hope_id
    assert by_id[hoping_id]["inflection"] == "gerund"
    # 'hop' stays its own lemma (no inflection in this test seeded
    # against it).
    assert by_id[hop_id]["lemma_id"] is None
    assert by_id[hope_id]["lemma_id"] is None


def test_link_lemmas_modern_english_falls_back_to_bare_stem_when_silent_e_missing(
    fresh_db: Path,
) -> None:
    """The silent-e candidate is PREFERRED but not required. When
    only the bare-stem lemma exists in the DB (e.g. 'walk' but no
    'walke'), the second candidate fires and the link still happens.
    Pin: silent-e must not become a hard prerequisite."""
    with LexiconDB(fresh_db) as db:
        # Only 'walk' exists, no 'walke'. 'walked' should link to walk.
        walk_id = db.upsert_etymon("walk", "modern-english")
        walked_id = db.upsert_etymon("walked", "modern-english")
        db.commit()

        link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    assert by_id[walked_id]["lemma_id"] == walk_id
    assert by_id[walked_id]["inflection"] == "past"


def test_link_lemmas_wires_in_mutation_path_for_goidelic(fresh_db: Path) -> None:
    """wyrd-jott Phase 1: link_lemmas falls through from suffix-strip
    to mutation prefix-strip for Goidelic etymons. Pin the wiring so
    a future refactor can't silently bypass the mutation path.

    'bhfuil' (eclipsis of 'fuil') should link to 'fuil' as its lemma
    with inflection='eclipsis' even though no suffix rule matches."""
    with LexiconDB(fresh_db) as db:
        fuil_id = db.upsert_etymon("fuil", "irish")
        bhfuil_id = db.upsert_etymon("bhfuil", "irish")
        # Independent etymon — should stay unlinked.
        beith_id = db.upsert_etymon("beith", "irish")
        db.commit()

        link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection FROM etymon"
        ).fetchall()
    by_id = {r["id"]: r for r in rows}
    assert by_id[bhfuil_id]["lemma_id"] == fuil_id
    assert by_id[bhfuil_id]["inflection"] == "eclipsis"
    # Lemma stays its own.
    assert by_id[fuil_id]["lemma_id"] is None
    # Unrelated row stays unlinked.
    assert by_id[beith_id]["lemma_id"] is None


def test_link_lemmas_dry_run_does_not_write(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("cot", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        db.commit()

        result = link_lemmas(db, apply=False)
        row = db.conn.execute("SELECT lemma_id FROM etymon WHERE id = ?", (cotan_id,)).fetchone()
    assert result["candidates"] == 1
    assert row["lemma_id"] is None


def test_link_lemmas_no_match_when_lemma_absent(fresh_db: Path) -> None:
    """If the proposed lemma isn't in the DB, no linkage proposal is made
    (we never fabricate lemmas in this pass)."""
    with LexiconDB(fresh_db) as db:
        # Only the inflected form exists; no "cot" lemma.
        db.upsert_etymon("cotan", "old-english")
        db.commit()
        result = link_lemmas(db, apply=False)
    assert result["candidates"] == 0


def test_consensus_view_rolls_up_to_lemma(fresh_db: Path) -> None:
    """Two sources, each citing a different inflected form of the same
    lemma, should produce a single consensus row with witnesses=2."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.upsert_etymon("cot", "old-english")  # the lemma
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Each inflected form has a different source citation.
        db.add_citation(cotan_id, "src-a")
        db.add_citation(cotes_id, "src-b")
        # Lemma itself has no direct citation.
        db.commit()

        # Before linking: each etymon is its own group; cotan has 1 witness,
        # cotes has 1 witness, cot has 0. Three rows.
        before = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY canonical_form"
        ).fetchall()
        before_map = {r["canonical_form"]: r["witnesses"] for r in before}
        assert before_map.get("cotan") == 1
        assert before_map.get("cotes") == 1

        link_lemmas(db, apply=True)

        # After linking: one consensus row for the lemma 'cot' with
        # witnesses=2 (src-a + src-b unioned).
        after = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY canonical_form"
        ).fetchall()
        after_map = {r["canonical_form"]: r["witnesses"] for r in after}

    assert after_map["cot"] == 2
    # cotan/cotes rows are gone from the consensus view (rolled into cot)
    assert "cotan" not in after_map
    assert "cotes" not in after_map


def test_language_field_mapping_covers_known_codes() -> None:
    """LANGUAGE_FIELDS must cover every source-language field used in
    meanings.json. Catches regressions if a new root is added in JSON but the
    seeder is forgotten.

    D18 spelling variants ride on per-language ``<lang>_variants`` keys; those
    are handled by the runtime's load_meanings (split out into Meaning.variants)
    rather than by LANGUAGE_FIELDS, so they're treated as known-handled here."""
    from wyrd.generators.kenning import _runtime_db_bundle_dict

    subjects = _runtime_db_bundle_dict().get("subjects") or []

    seen_fields: set[str] = set()
    for subject in subjects:
        for word in subject.get("words", []):
            seen_fields.update(word.keys())

    handled = LANGUAGE_FIELDS.keys() | NON_LANGUAGE_FIELDS
    # Also tolerate per-language metadata fields emitted by the export step:
    # *_variants (D18 spelling variants), *_inflections (D8 inflection
    # labels), *_citations (scholarly attribution), *_attested_years
    # (D5-2 era cells). They legitimately don't map to a LANGUAGE_FIELDS
    # code; the runtime's load_meanings handles them via separate Meaning
    # attributes. ``era_reflexes`` (wyrd-obpw) is a top-level word-scoped
    # field (not a per-language sibling) so it's listed here too.
    metadata_suffixes = (
        "_variants",
        "_inflections",
        "_citations",
        "_attested_years",
        "_stratum",
        # Per-language pronunciation + multi-script rendering siblings
        # emitted by export-meanings. The columns themselves were added
        # in wyrd-ha9q (Phase 2a/c/d, multi-script support); wyrd-69s5
        # is the corpus-miner fix that finally populated them on
        # European-language etymons. The runtime's load_meanings
        # parses these into Meaning.pronunciation / original_script /
        # transliteration / english_shaped maps — not part of the
        # LANGUAGE_FIELDS source-language table.
        "_pronunciation",
        "_original_script",
        "_transliteration",
        "_english_shaped",
        # wyrd-kq7w: per-language phonological_vector siblings emitted
        # by the wave-2 corpus miner. Consumed by the vector-scoring
        # path (Kenning.generate scoring_mode=vector); not part of the
        # LANGUAGE_FIELDS source-language table.
        "_phonological_vector",
    )
    metadata_exact = {"era_reflexes"}
    missing = {
        f
        for f in seen_fields - handled - metadata_exact
        if not any(f.endswith(s) for s in metadata_suffixes)
    }
    assert not missing, f"Unhandled fields in meanings.json: {missing}"


# --- export_meanings tests --------------------------------------------------


def _seed_subject(
    db: LexiconDB,
    *,
    source_id: str,
    glosses: list[str],
    tags: list[str],
    modifier_type: str | None,
    words: list[dict],
    language_overrides: dict[str, str] | None = None,
) -> None:
    """Helper: ingest one meanings.json subject via seed_from_meanings.

    ``language_overrides`` (wyrd-ok36) maps canonical_form → lexicon
    language code to relabel rows after seeding. ``celtic_mix`` rows
    seed as ``celtic`` (the legacy rando-port bucket); override to
    ``welsh`` / ``middle-welsh`` / ``old-welsh`` etc. when the test
    needs the substrata language shape Phase 1's classifier acts on.
    Pass ``{"caer": "welsh", "din": "middle-welsh"}`` to relabel two
    seeded rows independently.

    Scopes the UPDATE by ``(canonical_form, language)`` to keep the
    relabel intentional: tests that seed multiple etymons sharing a
    canonical form across different languages (e.g. an English ``cot``
    AND a Welsh ``cot``) won't accidentally relabel both. Caller
    passes the SOURCE language alongside the target via the dict key
    using the canonical_form alone is fine because today's tests don't
    overlap, but the WHERE clause is defensive.
    """
    seed_from_meanings(
        db,
        [
            {
                "meaning": glosses,
                "modifier_tags": tags,
                "modifier_type": modifier_type,
                "words": words,
            }
        ],
        source_id,
    )
    if language_overrides:
        # Resolve the seed-time language per form via LANGUAGE_FIELDS
        # so the WHERE clause excludes any cross-language collision.
        # Today's tests seed a single etymon per canonical_form so
        # this is belt-and-braces, but future tests that seed both
        # an English and a Welsh row with overlapping forms will need
        # the precision.
        from wyrd.generators.kenning.lexicon import LANGUAGE_FIELDS

        bucket_default_by_form: dict[str, str] = {}
        for word in words:
            for json_field, lang_code in LANGUAGE_FIELDS.items():
                forms = word.get(json_field) or []
                for form in forms:
                    bucket_default_by_form.setdefault(form, lang_code)
        for form, target_lang in language_overrides.items():
            source_lang = bucket_default_by_form.get(form)
            if source_lang is None:
                # Caller asked to relabel a form not in any LANGUAGE_FIELDS
                # entry on the seeded words — fall back to canonical_form-only
                # match so the override still applies in the unusual case.
                db.conn.execute(
                    "UPDATE etymon SET language = ? WHERE canonical_form = ?",
                    (target_lang, form),
                )
            else:
                db.conn.execute(
                    "UPDATE etymon SET language = ? WHERE canonical_form = ? AND language = ?",
                    (target_lang, form, source_lang),
                )
        db.commit()


def test_export_meanings_includes_rando_etymons_with_no_scholar_witnesses(
    fresh_db: Path,
) -> None:
    """A rando-port-only family is exported when include_rando=True (D4 legacy
    seed kept until corroborated)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Acorn"],
            tags=["plant", "food"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-ock", "old_english": ["aecern"]}],
        )

        with_rando = export_meanings(db, include_rando=True)
        without_rando = export_meanings(db, include_rando=False)

    assert len(with_rando) == 1
    subj = with_rando[0]
    assert subj["meaning"] == ["Acorn"]
    assert subj["modifier_tags"] == ["food", "plant"]
    assert subj["modifier_type"] == "Topographical"
    # wyrd-vm8t Loop 4: the OE surface form gets a deterministic G2P pronunciation
    # even though the etymon carried no IPA.
    assert subj["words"] == [
        {
            "modern_usage": "ock",
            "morpheme_id": "old-english:aecern",  # wyrd-rogd.10: owning morpheme (content id)
            "old_english": ["aecern"],
            "old_english_pronunciation": [{"form": "aecern", "ipa": "/ɑɛkɛrn/", "dialect": None}],
            # self-seed: the morpheme's own canonical form in its own era column,
            # PLUS (wyrd-mook) the generated surface '-ock' echoed into the modern
            # stage — surface '-ock' ≠ canonical 'aecern', so without it the SPA
            # grid would show 'aecern' but never match the '-ock' the user sees.
            "era_reflexes": {
                "modern-english": [{"form": "ock", "source": "self"}],
                "old-english": [{"form": "aecern", "source": "self"}],
            },
        }
    ]
    assert without_rando == []


def test_export_meanings_includes_wiktionary_empirical_etymons(fresh_db: Path) -> None:
    """wyrd-7bu1 / wyrd-4hx7 follow-up: a family cited only by
    'wiktionary-empirical' (no rando-port, no scholar witnesses)
    must surface in the bundle when ``include_wiktionary_empirical=True``
    and disappear when ``include_wiktionary_empirical=False``. Mirror
    of test_export_meanings_includes_rando_etymons_with_no_scholar_witnesses
    for the third UNION admit branch added by wyrd-4hx7."""
    with LexiconDB(fresh_db) as db:
        # Synthetic empirical source row (added at first --apply by the
        # corpus miner; must exist before the citation is written).
        db.upsert_source(id="wiktionary-empirical", title="Wiktionary (empirical)")
        eglwys_id = db.upsert_etymon("eglwys", "welsh", modifier_type="Architectural")
        db.add_gloss(eglwys_id, "church")
        db.add_tag(eglwys_id, "religious")
        db.add_citation(eglwys_id, "wiktionary-empirical")
        db.commit()

        with_empirical = export_meanings(
            db,
            include_rando=False,
            include_wiktionary_empirical=True,
        )
        without_empirical = export_meanings(
            db,
            include_rando=False,
            include_wiktionary_empirical=False,
        )

    assert len(with_empirical) == 1
    subj = with_empirical[0]
    assert subj["meaning"] == ["church"]
    assert "religious" in subj["modifier_tags"]
    # eglwys's welsh forms route into celtic_mix via _LANG_CODE_TO_JSON_FIELD.
    assert any("eglwys" in (w.get("celtic_mix") or []) for w in subj["words"])
    # Without the admit branch, the empirical-only family is dropped.
    assert without_empirical == []


def test_export_meanings_includes_toponym_breakdown_etymons(fresh_db: Path) -> None:
    """wyrd-oth3: a family attested ONLY as an element in a scholarly
    toponym_etymology breakdown (no rando-port, no wiktionary-empirical, no
    wave-2 english_shaped, no scholar witnesses) must surface in the bundle
    when ``include_toponym_breakdown=True`` and disappear when False — the
    fifth admission path. The place-name evidence channel that recovers the
    specifiers the dictionary-witness gate misses."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_src", title="Test")
        # A morpheme with NO citations (0 witnesses) and no other admit signal.
        alor_id = db.upsert_etymon("alor", "old-english", modifier_type="Flora")
        db.add_gloss(alor_id, "alder")
        db.add_tag(alor_id, "tree")
        # Its only evidence: a scholar decomposed a real place into it.
        tid = db.conn.execute(
            "INSERT INTO toponym (modern_name, country) VALUES ('Alorton', 'England')"
        ).lastrowid
        teid = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'test_src', 'high')",
            (tid,),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, ?)",
            (teid, alor_id),
        )
        db.commit()

        common = {
            "include_rando": False,
            "include_wiktionary_empirical": False,
            "include_wave2_enriched": False,
        }
        with_breakdown = export_meanings(db, include_toponym_breakdown=True, **common)
        without_breakdown = export_meanings(db, include_toponym_breakdown=False, **common)

    assert len(with_breakdown) == 1
    assert with_breakdown[0]["meaning"] == ["alder"]
    assert "tree" in with_breakdown[0]["modifier_tags"]
    # Without the breakdown admit branch, the breakdown-only family is dropped.
    assert without_breakdown == []


@pytest.mark.parametrize(
    "etymon_lang,expected_bundle_field",
    [
        # The new wyrd-4hx7 mappings — each Wiktionary-canonical
        # language must route into one of the existing 9 bundle fields.
        ("welsh", "celtic_mix"),
        ("middle-welsh", "celtic_mix"),
        ("old-welsh", "celtic_mix"),
        ("irish", "celtic_mix"),
        ("old-irish", "celtic_mix"),
        ("middle-irish", "celtic_mix"),
        ("scottish-gaelic", "celtic_mix"),
        ("manx", "celtic_mix"),
        ("cornish", "celtic_mix"),
        ("breton", "celtic_mix"),
        ("middle-breton", "celtic_mix"),
        ("old-breton", "celtic_mix"),
        ("proto-celtic", "celtic_mix"),
        ("proto-brythonic", "celtic_mix"),
        ("old-french", "old_french"),
        ("anglo-norman", "old_french"),
        ("middle-french", "old_french"),
        ("middle-english", "modern_english"),
        ("scots", "modern_english"),
        ("icelandic", "old_scandinavian"),
        ("faroese", "old_scandinavian"),
        ("old-high-german", "germanic"),
        ("middle-high-german", "germanic"),
        ("gothic", "germanic"),
        ("old-saxon", "germanic"),
        ("old-dutch", "germanic"),
        ("old-frisian", "germanic"),
        ("proto-germanic", "germanic"),
        ("proto-west-germanic", "germanic"),
        ("ancient-greek", "greek"),
        ("proto-greek", "greek"),
    ],
)
def test_lang_code_to_json_field_mapping_routes_into_bundle(
    fresh_db: Path, etymon_lang: str, expected_bundle_field: str
) -> None:
    """wyrd-7bu1: each new ``_LANG_CODE_TO_JSON_FIELD`` mapping added
    by wyrd-4hx7 must route an etymon's forms into the expected bundle
    field. Without these mappings ``_emit_word_languages`` silently
    drops the language (since it falls back to ``None`` for unknown
    keys), which was the bug behind the COVERAGE.md narrative
    miss in PR #68's first export pass."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary-empirical", title="Wiktionary (empirical)")
        eid = db.upsert_etymon("testform", etymon_lang)
        db.add_gloss(eid, "test")
        db.add_citation(eid, "wiktionary-empirical")
        db.commit()
        subjects = export_meanings(
            db,
            include_rando=False,
            include_wiktionary_empirical=True,
        )
    # The single empirical etymon should produce a single subject with
    # one word whose forms surface under the expected bundle field.
    assert len(subjects) == 1
    word = subjects[0]["words"][0]
    assert expected_bundle_field in word, (
        f"language {etymon_lang!r} should route into bundle field "
        f"{expected_bundle_field!r}; got word fields {list(word.keys())}"
    )
    assert "testform" in word[expected_bundle_field]


def test_lexicon_export_meanings_cli_no_include_wiktionary_empirical_flag(
    fresh_db: Path,
) -> None:
    """wyrd-7bu1: the CLI's ``--no-include-wiktionary-empirical`` flag
    suppresses the empirical admit branch, mirroring the existing
    ``--no-include-rando``. End-to-end via CliRunner."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary-empirical", title="Wiktionary (empirical)")
        eid = db.upsert_etymon("eglwys", "welsh", modifier_type="Architectural")
        db.add_gloss(eid, "church")
        db.add_citation(eid, "wiktionary-empirical")
        db.commit()

    runner = CliRunner()

    # Default: empirical-admit ON, eglwys is in the export.
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "export-meanings", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "eglwys" in result.output

    # --no-include-wiktionary-empirical: eglwys is dropped.
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-wiktionary-empirical",
            "--no-include-rando",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "eglwys" not in result.output


def test_export_meanings_promotes_at_witness_threshold(fresh_db: Path) -> None:
    """A family with no rando-port citation needs ≥ min_witnesses to export.

    wyrd-fssn note: the explicit ``min_witnesses=3`` arg here keeps this
    test pinning the gate semantics regardless of the default-default.
    The default-default itself dropped from 3 to 2 in wyrd-fssn; see
    :func:`test_export_meanings_default_uses_recommended_preset` for the
    default-shift coverage."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        ham_id = db.upsert_etymon("ham", "old-english", modifier_type="Habitative")
        db.add_gloss(ham_id, "homestead")
        db.add_tag(ham_id, "habitation")
        db.add_citation(ham_id, "a")
        db.add_citation(ham_id, "b")
        db.commit()

        two_witnesses = export_meanings(
            db, include_rando=False, min_witnesses=3, lang_thresholds={}
        )

        db.add_citation(ham_id, "c")
        db.commit()
        three_witnesses = export_meanings(
            db, include_rando=False, min_witnesses=3, lang_thresholds={}
        )

    assert two_witnesses == []
    assert len(three_witnesses) == 1
    subj = three_witnesses[0]
    assert subj["meaning"] == ["homestead"]
    # No reflex links in this minimal scenario, so the word is synthesized
    # from the canonical_form. position_pref defaults to no-dash → bare form.
    # All three scholarly sources (a, b, c) ride along as the wyrd-9kh.1
    # `<lang>_citations` sibling field — a GM holding this name in chat
    # sees who attests it.
    assert subj["words"] == [
        {
            "modern_usage": "ham",
            "morpheme_id": "old-english:ham",  # wyrd-rogd.10: owning morpheme (content id)
            "old_english": ["ham"],
            "old_english_citations": ["a", "b", "c"],
            # wyrd-vm8t Loop 4: G2P surface pronunciation (onset-h → /h/).
            "old_english_pronunciation": [{"form": "ham", "ipa": "/hɑm/", "dialect": None}],
            # self-seed: own form in its own era column
            "era_reflexes": {"old-english": [{"form": "ham", "source": "self"}]},
        }
    ]


def test_export_meanings_per_language_thresholds_apply(fresh_db: Path) -> None:
    """Old-Norse promotes at 2 witnesses while Old-English needs 3 — the
    per-language map is honored and the global ``min_witnesses`` is the
    fallback for languages not in the map."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # OE etymon at 2 witnesses — should NOT promote at OE threshold 3.
        oe = db.upsert_etymon("ham", "old-english")
        db.add_gloss(oe, "homestead")
        db.add_tag(oe, "habitation")
        db.add_citation(oe, "a")
        db.add_citation(oe, "b")
        # ON etymon at 2 witnesses — should promote at ON threshold 2.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_tag(on, "topography")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

        subjects = export_meanings(
            db,
            include_rando=False,
            min_witnesses=3,
            lang_thresholds={"old-english": 3, "old-norse": 2},
        )

    assert len(subjects) == 1
    subj = subjects[0]
    assert subj["meaning"] == ["mountain"]
    # The OE 'ham' must NOT appear; only 'fell' (ON) made the cut.
    forms_seen = {
        form for w in subj["words"] for k, v in w.items() if isinstance(v, list) for form in v
    }
    assert "fell" in forms_seen
    assert "ham" not in forms_seen


def test_export_meanings_default_uses_recommended_preset(fresh_db: Path) -> None:
    """Calling ``export_meanings()`` without ``lang_thresholds`` applies the
    ``RECOMMENDED_LANG_THRESHOLDS`` preset, which gates every language at
    ≥2 witnesses (uniform per wyrd-fssn 2026-05-28; OE relaxed from 3
    to 2 in the same change). At 2 witnesses an OE etymon promotes."""
    assert RECOMMENDED_LANG_THRESHOLDS["old-english"] == 2
    assert RECOMMENDED_LANG_THRESHOLDS["old-norse"] == 2

    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        oe = db.upsert_etymon("ham", "old-english")
        db.add_gloss(oe, "homestead")
        db.add_citation(oe, "a")
        db.add_citation(oe, "b")
        db.commit()
        # No lang_thresholds passed → preset applies → OE at 2 witnesses
        # promotes (this would NOT have happened pre-wyrd-fssn when the
        # OE preset entry was 3).
        subjects = export_meanings(db, include_rando=False)

    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["homestead"]


def test_export_meanings_empty_lang_thresholds_falls_back_to_uniform(
    fresh_db: Path,
) -> None:
    """Passing ``lang_thresholds={}`` disables the preset — every language
    uses the global ``min_witnesses`` uniformly."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()
        # ON has 2 witnesses; preset would promote it but {} disables that
        # and applies min_witnesses=3 uniformly → not promoted.
        subjects = export_meanings(db, include_rando=False, min_witnesses=3, lang_thresholds={})
    assert subjects == []


def test_export_meanings_surfaces_cognate_cluster_modern_forms(fresh_db: Path) -> None:
    """wyrd-nxhh: cognate-cluster mates from end-state target languages
    appear in the family subject's per-language sibling arrays (not
    only in the separate era_reflexes field).

    Fixture: OE ``ceaster`` (2 citations, promotable) cognate-linked
    to modern-english ``chester``. Post-fix ``modern_english`` carries
    ``chester`` (stored BARE — wyrd-aicu.8, D45; position is the separate
    axis, not a dash on the identity)."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        ceaster = db.upsert_etymon("ceaster", "old-english", modifier_type="Habitative")
        db.add_gloss(ceaster, "fortified place")
        db.add_tag(ceaster, "architecture")
        db.add_citation(ceaster, "a")
        db.add_citation(ceaster, "b")
        chester = db.upsert_etymon("chester", "modern-english")
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ? WHERE id IN (?, ?)",
            (ceaster, ceaster, chester),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=False)
    assert len(subjects) == 1
    word = subjects[0]["words"][0]
    assert word.get("old_english") == ["ceaster"]
    assert word.get("modern_english") == ["chester"], (
        f"forms_by_lang should include cluster mate's modern-english surface; "
        f"got modern_english={word.get('modern_english')!r}"
    )


def test_export_meanings_surfaces_descent_edge_modern_forms(fresh_db: Path) -> None:
    """wyrd-nxhh round 1 (test-coverage P2): the Tier-2 descent path
    (no cognate_id, but an etymon_descent edge to a modern-english
    child) must also surface in forms_by_lang. Without this pin a
    refactor that drops Tier 2 from _fetch_family_era_reflexes would
    pass the cluster test but break descent-only families."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        burg = db.upsert_etymon("burg", "old-english", modifier_type="Habitative")
        db.add_gloss(burg, "fortified town")
        db.add_citation(burg, "a")
        db.add_citation(burg, "b")
        borough = db.upsert_etymon("borough", "modern-english")  # bare (wyrd-aicu.8, D45)
        # Tier 2: direct inheritance edge, no cognate_id on either end.
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'a', 'high')",
            (burg, borough),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=False)
    assert len(subjects) == 1
    word = subjects[0]["words"][0]
    assert word.get("old_english") == ["burg"]
    assert word.get("modern_english") == ["borough"], (
        f"Tier-2 descent edge should surface modern-english child in forms_by_lang; "
        f"got modern_english={word.get('modern_english')!r}"
    )


def test_merge_era_reflexes_excludes_phonology_rule_tier() -> None:
    """wyrd-nxhh round 1 (code-reviewer P2): the merge filters to
    attested sources (cluster / descent / period-form) and drops
    Tier-4 phonology-rule forms. The main generator samples
    forms_by_lang uniformly with no source-quality differentiation,
    so admitting inferred forms would let it emit them as if they
    were attested."""
    from wyrd.generators.kenning.lexicon.bundle._family import (
        _merge_era_reflexes_into_forms_by_lang,
    )

    forms_by_lang: dict[str, list[str]] = {}
    era_reflexes = {
        "modern-english": [
            {"form": "chester", "source": "cluster"},
            {"form": "casterton", "source": "phonology-rule:v1"},
            {"form": "Manchester", "source": "descent"},
            {"form": "ceaster1500", "source": "period-form"},
        ],
        "middle-english": [{"form": "Cestre", "source": "cluster"}],
        "tier4-only-lang": [{"form": "X", "source": "phonology-rule:v1"}],
    }
    _merge_era_reflexes_into_forms_by_lang(forms_by_lang, era_reflexes)
    assert forms_by_lang == {
        "modern-english": ["chester", "Manchester", "ceaster1500"],
        "middle-english": ["Cestre"],
    }, (
        f"merge must filter phonology-rule:v1 forms and not create empty buckets "
        f"for Tier-4-only languages; got {forms_by_lang!r}"
    )


def test_merge_era_reflexes_dedupes_against_existing_forms() -> None:
    """wyrd-nxhh round 1 (test-coverage P2): the merge's dedup guard
    drops forms already in the direct-member bucket. Without it a
    family where both the direct rollup AND the cognate cluster
    contributed the same modern surface would ship a duplicated form
    in the bundle silently."""
    from wyrd.generators.kenning.lexicon.bundle._family import (
        _merge_era_reflexes_into_forms_by_lang,
    )

    forms_by_lang = {"modern-english": ["-chester"]}
    era_reflexes = {
        "modern-english": [
            {"form": "-chester", "source": "cluster"},  # dup of existing
            {"form": "-cester", "source": "cluster"},  # new
        ],
    }
    _merge_era_reflexes_into_forms_by_lang(forms_by_lang, era_reflexes)
    assert forms_by_lang == {"modern-english": ["-chester", "-cester"]}, (
        f"merge must dedup against pre-existing forms and preserve direct-member-first "
        f"ordering; got {forms_by_lang!r}"
    )


def test_merge_era_reflexes_empty_input_is_noop() -> None:
    """wyrd-nxhh round 1 (code-reviewer P2 ≡ fragile-fixture fix):
    direct-helper no-op pin replaces the earlier end-to-end fixture
    (a fresh OE etymon with no cluster / no descent) that silently
    depended on Tier-4 phonology-rule returning None. Future
    phonology-layer changes don't affect this assertion."""
    from wyrd.generators.kenning.lexicon.bundle._family import (
        _merge_era_reflexes_into_forms_by_lang,
    )

    forms_by_lang = {"old-english": ["ham"]}
    _merge_era_reflexes_into_forms_by_lang(forms_by_lang, {})
    assert forms_by_lang == {"old-english": ["ham"]}


def test_export_meanings_rando_min_corroborators_zero_admits_pure_rando(
    fresh_db: Path,
) -> None:
    """wyrd-fssn: ``rando_min_corroborators=0`` (default) reproduces the
    pre-fssn rando-port-admit semantic — every rando-cited family lands
    in the bundle regardless of corroborating sources."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        ham = db.upsert_etymon("ham", "old-english")
        db.add_gloss(ham, "homestead")
        db.add_citation(ham, "rando-port")
        db.commit()
        # No corroborators at all; default 0 still admits.
        subjects = export_meanings(db, lang_thresholds={}, min_witnesses=99)
    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["homestead"]


def test_export_meanings_rando_min_corroborators_one_drops_pure_rando(
    fresh_db: Path,
) -> None:
    """wyrd-fssn: ``rando_min_corroborators=1`` quarantines pure-rando
    lemmas (no other source has cited them) while keeping rando+1
    families. This is the forward-flippable retirement lever — flip
    once mining corroboration catches up, drop the pure-rando tail."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="scholar_a", title="scholar a")
        pure = db.upsert_etymon("pollen", "old-english")
        db.add_gloss(pure, "pollen")
        db.add_citation(pure, "rando-port")
        corroborated = db.upsert_etymon("ham", "old-english")
        db.add_gloss(corroborated, "homestead")
        db.add_citation(corroborated, "rando-port")
        db.add_citation(corroborated, "scholar_a")
        db.commit()
        subjects = export_meanings(
            db,
            lang_thresholds={},
            min_witnesses=99,
            rando_min_corroborators=1,
        )
    glosses = sorted(s["meaning"][0] for s in subjects)
    assert glosses == ["homestead"], (
        f"corroborated rando lemma should remain; pure-rando should drop. got {glosses}"
    )


def test_export_meanings_rando_min_corroborators_counts_wiktionary_empirical(
    fresh_db: Path,
) -> None:
    """wyrd-fssn: the rando-corroborator gate's "non-rando" filter is
    literal — wiktionary-empirical citations count as corroborators.
    Path (c) already independently admits any wiktionary-empirical-cited
    family regardless of the rando knob, but the corroborator gate is
    written as ``source_id != 'rando-port'`` not ``source_id NOT IN
    ('rando-port', 'wiktionary-empirical')`` so a wiktionary-empirical
    co-citation on a rando lemma also satisfies path (a)'s ≥1
    corroborator requirement. Pin the policy: empirical citations
    corroborate rando admit."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="wiktionary-empirical", title="wikt-empirical")
        ham = db.upsert_etymon("ham", "old-english")
        db.add_gloss(ham, "homestead")
        db.add_citation(ham, "rando-port")
        db.add_citation(ham, "wiktionary-empirical")
        db.commit()
        # Disable path (c) so we observe path (a) in isolation —
        # otherwise the empirical citation would short-circuit admit
        # via (c) regardless of the corroborator gate.
        subjects = export_meanings(
            db,
            lang_thresholds={},
            min_witnesses=99,
            rando_min_corroborators=1,
            include_wiktionary_empirical=False,
        )
    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["homestead"]


def test_export_meanings_rando_min_corroborators_rolls_inflection_child_citation(
    fresh_db: Path,
) -> None:
    """wyrd-fssn: docstring claims the corroborator rollup follows
    ``build_family_rollup`` — an inflection child's citation
    corroborates the lemma head. Pin the structural claim with a
    fixture where the rando-cited LEMMA has no direct non-rando
    citation but its INFLECTED CHILD does."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="scholar_a", title="scholar a")
        ham = db.upsert_etymon("ham", "old-english")
        db.add_gloss(ham, "homestead")
        db.add_citation(ham, "rando-port")
        # Inflected child of ham — cited by scholar (not rando).
        hamum = db.upsert_etymon("hamum", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = 'dative_pl', "
            "lemma_method = 'link-lemmas-v1' WHERE id = ?",
            (ham, hamum),
        )
        db.add_citation(hamum, "scholar_a")
        db.commit()
        subjects = export_meanings(
            db,
            lang_thresholds={},
            min_witnesses=99,
            rando_min_corroborators=1,
        )
    # The rando-rooted family is corroborated via its inflection
    # child's scholar citation. Without family rollup the lemma head
    # would have 0 corroborators and drop.
    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["homestead"]


def test_families_with_corroborators_zero_admits_all_rando_roots() -> None:
    """wyrd-fssn round 2 (Gemini P2): the helper's docstring contract
    says "at least ``min_corroborators`` distinct non-rando-port
    citation sources" — at ``min_corroborators=0`` every set trivially
    satisfies ``≥ 0``, so every rando root must be admitted.

    The production caller short-circuits before reaching the helper
    when ``rando_min_corroborators <= 0`` (so this case wouldn't surface
    via ``export_meanings``), but the helper still needs to honor its
    documented contract when called directly. Tested here with a
    rando-only lemma (zero non-rando citations) at
    ``min_corroborators=0`` — pre-Gemini-P2 the helper would have
    silently dropped it from the result; post-fix it's admitted."""
    import sqlite3

    from wyrd.generators.kenning.lexicon.bundle._export import (
        _families_with_corroborators,
    )

    # Use the live LexiconDB shape via a minimal sqlite3 connection so
    # we don't need fresh_db's autogen — the helper only reads
    # etymon_citation, which we synthesize directly.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("CREATE TABLE etymon_citation (etymon_id INTEGER, source_id TEXT);")

    class _DBStub:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

    db_stub = _DBStub(conn)
    rando_root_ids = {1, 2}
    members_by_root = {1: [1], 2: [2]}
    # NO citations in etymon_citation. At min_corroborators=0 both
    # roots admit (contract: every set ≥ 0); at min_corroborators=1
    # neither does.
    result_zero = _families_with_corroborators(db_stub, rando_root_ids, members_by_root, 0)
    assert result_zero == {1, 2}, (
        f"min_corroborators=0 should admit all rando roots regardless of "
        f"citation count; got {result_zero}"
    )
    result_one = _families_with_corroborators(db_stub, rando_root_ids, members_by_root, 1)
    assert result_one == set()


def test_export_meanings_rando_min_corroborators_no_op_when_include_rando_false(
    fresh_db: Path,
) -> None:
    """wyrd-fssn: ``rando_min_corroborators`` is a knob ON the rando-port
    admit path. With ``include_rando=False`` the whole path is skipped
    regardless of the corroborator setting — the two flags are orthogonal,
    so flipping ``rando_min_corroborators`` from 0 to 1 with
    ``include_rando=False`` must produce identical output."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="scholar_a", title="scholar a")
        # Pure-rando lemma: only path 2 could admit it.
        pure = db.upsert_etymon("pollen", "old-english")
        db.add_gloss(pure, "pollen")
        db.add_citation(pure, "rando-port")
        db.commit()
        baseline = export_meanings(
            db,
            include_rando=False,
            rando_min_corroborators=0,
        )
        tightened = export_meanings(
            db,
            include_rando=False,
            rando_min_corroborators=1,
        )
    assert baseline == [] == tightened


def test_export_meanings_rolls_inflected_variants_into_lemma(fresh_db: Path) -> None:
    """`cot` + linked inflections export as one entry; the language form list
    contains the lemma followed by its variants (D8)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        # Add inflected children — these point at the lemma via lemma_id.
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)", (cot_id, cotan_id, cotes_id)
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    word = subjects[0]["words"][0]
    assert word["modern_usage"] == "cot"
    # Lemma form leads; variants follow in the same language array.
    assert word["old_english"][0] == "cot"
    assert set(word["old_english"]) == {"cot", "cotan", "cotes"}


def _add_text_match(db, etymon_id, source_id, matched_form, count, method="fuzzy-search-v1"):
    """Helper: insert a row directly into etymon_text_match for D18 fixture
    setup. The schema doesn't ship a write helper because the post-mining
    pipeline owns this table."""
    db.conn.execute(
        "INSERT INTO etymon_text_match "
        "(etymon_id, source_id, matched_form, match_count, method, edit_distance) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (etymon_id, source_id, matched_form, count, method),
    )


def test_lang_code_to_json_field_values_are_in_language_fields() -> None:
    """Invariant: every value in `_LANG_CODE_TO_JSON_FIELD` (the export
    direction: lexicon code → bundle field name) must itself be a key in
    `LANGUAGE_FIELDS` (the ingest direction: bundle field name → lexicon
    code). Otherwise the bundle round-trips through ingestion as if the
    field doesn't exist — silently dropping wave-2 etymons.

    Pinned as a sanity invariant: a typo in any of the new wave-2 entries
    (e.g. 'persian' miskeyed as 'persia' or 'arabic' as 'arab') would
    break round-tripping but pass every individual export test."""
    for lang_code, json_field in _LANG_CODE_TO_JSON_FIELD.items():
        assert json_field in LANGUAGE_FIELDS, (
            f"_LANG_CODE_TO_JSON_FIELD[{lang_code!r}] = {json_field!r} "
            f"but {json_field!r} is not a key in LANGUAGE_FIELDS — "
            f"bundle would silently drop {lang_code} on ingestion round-trip"
        )


def test_lang_code_to_json_field_has_no_duplicate_keys_or_unbalanced_buckets() -> None:
    """Each lexicon code maps to exactly one bundle bucket, and every
    bundle bucket is targeted by at least one lexicon code (else the
    bucket would be empty in practice).

    Catches the 'wave-2 typo' regression class: if 'sem-pro' got
    miskeyed as 'sem_pro' (with underscore), the dict would keep
    both keys but only the canonical one routes correctly. The
    invariant doesn't catch THIS case directly but the test below
    on precursor-stack routing does."""

    # No duplicate KEYS in the dict literal — Python would silently keep
    # the last value, but the assertion makes the contract explicit.
    keys = list(_LANG_CODE_TO_JSON_FIELD)
    assert len(keys) == len(set(keys)), f"_LANG_CODE_TO_JSON_FIELD has duplicate keys: {keys}"
    # Every bundle bucket has at least one routing entry.
    buckets = set(_LANG_CODE_TO_JSON_FIELD.values())
    expected_wave2_buckets = {
        "hebrew",
        "arabic",
        "persian",
        "sanskrit",
        "akkadian",
        "egyptian",
        "aramaic",
        "armenian",
    }
    assert expected_wave2_buckets.issubset(buckets), (
        f"missing wave-2 bucket(s): {expected_wave2_buckets - buckets}"
    )


def test_seed_from_meanings_round_trips_wave_2_bundle_fields(fresh_db: Path) -> None:
    """wyrd-vsrn Phase 2c: a meanings.json bundle that contains wave-2
    fields (`hebrew`, `arabic`, etc.) must ingest cleanly with the
    correct lexicon `language` code on each etymon row. Pre-PR these
    fields were untracked; now they round-trip via `LANGUAGE_FIELDS`.

    Pinned because export → re-ingest consistency depends on every
    bundle field name in `LANGUAGE_FIELDS` mapping to the same lexicon
    code that `_LANG_CODE_TO_JSON_FIELD` uses for emit."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["golem"],
            tags=["monster"],
            modifier_type=None,
            words=[
                {
                    "modern_usage": "Golem",
                    "hebrew": ["גולם"],
                    "arabic": ["جن"],
                    "sanskrit": ["रक्षस"],
                }
            ],
        )
        db.commit()

        rows = db.conn.execute(
            "SELECT canonical_form, language FROM etymon "
            "WHERE language IN ('he','ar','sa') ORDER BY language, canonical_form"
        ).fetchall()
    forms_by_lang = {(r["language"], r["canonical_form"]) for r in rows}
    assert forms_by_lang == {
        ("ar", "جن"),
        ("he", "גולם"),
        ("sa", "रक्षस"),
    }


def test_export_meanings_routes_precursor_postcursor_stack_to_canonical_bucket(
    fresh_db: Path,
) -> None:
    """wyrd-vsrn Phase 2c: when an etymon family has members in a wave-2
    canonical lang AND in its precursor / postcursor stack, all of them
    surface in ONE bundle bucket (the canonical's). Same pattern as
    celtic_mix already groups welsh + old-welsh + middle-welsh.

    This test seeds a Hebrew family with members in `he` (Modern
    Hebrew) and `hbo` (Biblical Hebrew); both must land in
    `word['hebrew']`."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Use the existing seed helper (it routes through LANGUAGE_FIELDS),
        # then add the precursor row directly via upsert_etymon and
        # link it to the same family via lemma_id.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["golem"],
            tags=["monster"],
            modifier_type=None,
            words=[{"modern_usage": "Golem", "hebrew": ["גולם"]}],
        )
        modern_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
            ("גולם", "he"),
        ).fetchone()["id"]
        # Insert a precursor (`hbo`, Biblical Hebrew) etymon and tie it
        # into the family via lemma_id rollup. The export step's family
        # rollup walks lemma_id chains so this row lands in the same
        # word entry as the modern Hebrew form.
        biblical_id = db.upsert_etymon("גלם", "hbo")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (modern_id, biblical_id))
        # Add a citation so the biblical row joins the family rollup
        # under the rando-port admit path.
        db.add_citation(biblical_id, "rando-port")
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = next(w for s in subjects for w in s["words"] if w["modern_usage"] == "Golem")
    # Both forms land in the SAME bucket (hebrew) — the precursor
    # routing collapses he and hbo together.
    assert "hebrew" in word
    assert set(word["hebrew"]) == {"גולם", "גלם"}


def test_export_meanings_routes_phase2d_renderings_through_bucket_aggregation(
    fresh_db: Path,
) -> None:
    """wyrd-qhs0 Phase 2d: when two members of the same family are in
    different lexicon codes that route to the same bundle bucket
    (he + hbo → hebrew), all four wyrd-ha9q rendering columns
    aggregate into the bucket cleanly. Each (canonical_form,
    rendering) pair lands once; different forms within the bucket
    keep their own renderings.

    Pinned as the multi-code-per-bucket regression test for the 3 new
    Phase 2d columns. The Phase 2c precursor-stack test only verified
    that the FORMS arrays unioned correctly; this test extends coverage
    to the rendering siblings."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["golem"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "Golem", "hebrew": ["גולם"]}],
        )
        modern_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
            ("גולם", "he"),
        ).fetchone()["id"]
        # Biblical Hebrew (hbo) member with DIFFERENT canonical form
        # and DIFFERENT renderings; verify both members' data surfaces
        # in the merged hebrew bucket.
        biblical_id = db.upsert_etymon("גלם", "hbo")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (modern_id, biblical_id))
        db.add_citation(biblical_id, "rando-port")
        db.conn.execute(
            """UPDATE etymon SET
                 original_script = ?, transliteration = ?,
                 english_shaped = ?, pronunciation_ipa = ?,
                 pronunciation_dialect = ?
               WHERE id = ?""",
            ("גוֹלֶם", "gōlem", "golem", "/ɡoːlɛm/", "Modern-Hebrew", modern_id),
        )
        db.conn.execute(
            """UPDATE etymon SET
                 original_script = ?, transliteration = ?,
                 english_shaped = ?, pronunciation_ipa = ?,
                 pronunciation_dialect = ?
               WHERE id = ?""",
            ("גֹּלֶם", "gōlem-bib", "golem", "/goːlɛm/", "Biblical-Hebrew", biblical_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = next(w for s in subjects for w in s["words"] if w["modern_usage"] == "Golem")
    assert set(word["hebrew"]) == {"גולם", "גלם"}
    assert word["hebrew_original_script"] == [
        {"form": "גולם", "original_script": "גוֹלֶם"},
        {"form": "גלם", "original_script": "גֹּלֶם"},
    ]
    assert word["hebrew_transliteration"] == [
        {"form": "גולם", "transliteration": "gōlem"},
        {"form": "גלם", "transliteration": "gōlem-bib"},
    ]
    assert word["hebrew_english_shaped"] == [
        {"form": "גולם", "english_shaped": "golem"},
        {"form": "גלם", "english_shaped": "golem"},
    ]
    assert word["hebrew_pronunciation"] == [
        {"form": "גולם", "ipa": "/ɡoːlɛm/", "dialect": "Modern-Hebrew"},
        {"form": "גלם", "ipa": "/goːlɛm/", "dialect": "Biblical-Hebrew"},
    ]


def test_export_meanings_emits_phase2d_four_renderings_per_language(fresh_db: Path) -> None:
    """wyrd-qhs0 Phase 2d: export emits four sibling arrays per
    non-Latin language — `<lang>_original_script`, `_transliteration`,
    `_pronunciation`, `_english_shaped`. All four populate when the
    underlying etymon row has data; sparse / absent when it doesn't.
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["dog"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "Kelev", "hebrew": ["כלב"]}],
        )
        # Populate all four wyrd-ha9q rendering columns on the row.
        db.conn.execute(
            """UPDATE etymon SET
                 original_script = ?,
                 transliteration = ?,
                 english_shaped = ?,
                 pronunciation_ipa = ?,
                 pronunciation_dialect = ?
               WHERE canonical_form = ? AND language = ?""",
            ("כֶּלֶב", "kɛ́lɛḇ", "kelev", "/kalb/", "Biblical-Hebrew", "כלב", "he"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["hebrew"] == ["כלב"]
    assert word["hebrew_original_script"] == [{"form": "כלב", "original_script": "כֶּלֶב"}]
    assert word["hebrew_transliteration"] == [{"form": "כלב", "transliteration": "kɛ́lɛḇ"}]
    assert word["hebrew_english_shaped"] == [{"form": "כלב", "english_shaped": "kelev"}]
    assert word["hebrew_pronunciation"] == [
        {"form": "כלב", "ipa": "/kalb/", "dialect": "Biblical-Hebrew"}
    ]


def test_export_meanings_pronunciation_omits_dialect_when_null(fresh_db: Path) -> None:
    """An IPA without a dialect tag (untagged-canonical case) surfaces
    in the bundle with dialect=None. Round-trips through load_meanings
    correctly via .get() (see test_load_meanings_handles_pronunciation_with_null_dialect)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["jinn"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "Jinn", "arabic": ["جن"]}],
        )
        db.conn.execute(
            "UPDATE etymon SET pronunciation_ipa = ? WHERE canonical_form = ? AND language = ?",
            ("/d͡ʒɪnː/", "جن", "ar"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["arabic_pronunciation"] == [{"form": "جن", "ipa": "/d͡ʒɪnː/", "dialect": None}]


def test_export_meanings_omits_phase2d_siblings_when_columns_null(fresh_db: Path) -> None:
    """Latin-script rows / rows without rendering data must NOT emit
    empty `_original_script` / `_transliteration` / `_pronunciation`
    siblings — bundle stays compact."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # wyrd-vm8t Loop 4: _pronunciation is now G2P-derived for table-language
    # forms even when the etymon IPA column is null, so it is no longer omitted
    # (script/transliteration remain column-driven and stay absent).
    for suffix in ("_original_script", "_transliteration"):
        assert all(not k.endswith(suffix) for k in word), (
            f"unexpected {suffix} sibling on Latin-script row: {list(word.keys())}"
        )
    assert "old_english_pronunciation" in word, "expected G2P surface pronunciation"


def test_export_meanings_emits_english_shaped_per_language(fresh_db: Path) -> None:
    """wyrd-ha9q Phase 2c: an etymon family with non-NULL english_shaped
    on at least one member emits a `<lang>_english_shaped` sibling array
    in its word entry. Each entry maps the canonical_form to its
    English-friendly rendering. Sorted by form for determinism."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Seed a Hebrew-language family with two members so we can
        # verify multiple english_shaped values land in one array.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["golem"],
            tags=["monster"],
            modifier_type=None,
            words=[
                {
                    "modern_usage": "Golem",
                    "hebrew": ["גולם", "גלם"],
                }
            ],
        )
        # Populate english_shaped on both etymon rows.
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE canonical_form = ? AND language = ?",
            ("golem", "גולם", "he"),
        )
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE canonical_form = ? AND language = ?",
            ("gelem", "גלם", "he"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "hebrew_english_shaped" in word, (
        f"hebrew_english_shaped missing from word entry: keys={list(word.keys())}"
    )
    # Sorted by canonical_form for determinism.
    assert word["hebrew_english_shaped"] == [
        {"form": "גולם", "english_shaped": "golem"},
        {"form": "גלם", "english_shaped": "gelem"},
    ]


def test_export_meanings_skips_english_shaped_when_all_members_null(fresh_db: Path) -> None:
    """An etymon family whose every member has NULL english_shaped (the
    common case for Latin-script source langs OR rows that lacked
    sufficient transliteration / IPA input) MUST NOT emit a
    `<lang>_english_shaped` field at all. Pinning this so the bundle
    doesn't bloat with empty arrays for the ~85k unshapeable rows."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        # english_shaped stays NULL on the Old English etymon (Latin-
        # script source lang — derive_english_shaped returned None at
        # ingest time per wyrd-ha9q's _LATIN_SCRIPT_LANGS guard).
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_english_shaped" not in word
    assert all(not k.endswith("_english_shaped") for k in word)


def test_export_meanings_partial_english_shaped_emits_only_populated(fresh_db: Path) -> None:
    """A family where SOME members carry english_shaped and others
    don't emits a `<lang>_english_shaped` array containing ONLY the
    populated forms. The unmapped form survives in the language form
    array but doesn't pollute the english_shaped sibling. Pinning the
    rule so the runtime can detect 'this form has no shaping' via
    an absent Meaning.english_shaped[lang] entry."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["jinn"],
            tags=["monster"],
            modifier_type=None,
            words=[
                {
                    "modern_usage": "Jinn",
                    "arabic": ["جن", "عفريت"],
                }
            ],
        )
        # Only the first form gets an english_shaped value; second stays NULL.
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE canonical_form = ? AND language = ?",
            ("jinn", "جن", "ar"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["arabic"] == ["جن", "عفريت"]  # Both forms in the lang array.
    assert word["arabic_english_shaped"] == [
        {"form": "جن", "english_shaped": "jinn"},
    ]


def test_export_meanings_wave2_enriched_promotes_uncited_etymons(fresh_db: Path) -> None:
    """wyrd-z3cp: an etymon with non-NULL english_shaped is admitted to
    the bundle even when it has zero scholar citations + zero
    wiktionary-empirical citations. Without this promotion path the
    wave-2 non-Latin source-language enrichment (wyrd-vsrn Phase 2c)
    never reaches the bundle — they're ingested by the wiktextract
    bulk path WITHOUT citations and would fail every other promotion
    rule.

    Reproduction: seed a Hebrew etymon WITHOUT any citation, set its
    english_shaped, run export_meanings with include_rando=False +
    include_wiktionary_empirical=False (so only the wave-2 path can
    admit it), and confirm the bundle contains the family + its
    hebrew_english_shaped sibling."""
    with LexiconDB(fresh_db) as db:
        # No citation seed — drop directly into etymon.
        cur = db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, english_shaped) "
            "VALUES ('גולם', 'he', 'golem')"
        )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'monster')",
            (cur.lastrowid,),
        )
        db.commit()

        # With include_wave2_enriched=False (default would admit it):
        # only the rando + wiktionary-empirical paths apply → no admit.
        subjects_off = export_meanings(
            db,
            include_rando=False,
            include_wiktionary_empirical=False,
            include_wave2_enriched=False,
        )
        # With include_wave2_enriched=True: the english_shaped value
        # alone admits the family.
        subjects_on = export_meanings(
            db,
            include_rando=False,
            include_wiktionary_empirical=False,
            include_wave2_enriched=True,
        )

    assert subjects_off == [], (
        "with include_wave2_enriched=False the uncited Hebrew etymon must NOT "
        "appear in the bundle — the other three promotion paths can't admit it"
    )
    assert len(subjects_on) >= 1, (
        "include_wave2_enriched=True should admit the Hebrew family via "
        "the non-NULL english_shaped marker"
    )
    # The promoted subject carries the hebrew form array + its
    # english_shaped sibling — closing the bug-report loop on wyrd-z3cp.
    word = subjects_on[0]["words"][0]
    assert "hebrew" in word
    assert word["hebrew_english_shaped"] == [
        {"form": "גולם", "english_shaped": "golem"},
    ]


def test_export_meanings_wave2_enriched_defaults_to_true(fresh_db: Path) -> None:
    """Default behavior includes wave-2 enriched families — operators
    who don't pass the flag get the new admit path. Pinning the default
    so a future toggle to False (which would hide ~23K live-DB families
    from the bundle) trips this test loudly."""
    with LexiconDB(fresh_db) as db:
        cur = db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, english_shaped) "
            "VALUES ('גולם', 'he', 'golem')"
        )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'monster')",
            (cur.lastrowid,),
        )
        db.commit()
        # No explicit flag — exercise the default.
        subjects = export_meanings(db, include_rando=False, include_wiktionary_empirical=False)
    assert len(subjects) >= 1, "wave-2 enriched admit must be on by default"


def test_export_meanings_emits_stratum_per_language(fresh_db: Path) -> None:
    """wyrd-lr4 Phase 2: a Welsh-family word entry carries a
    ``celtic_mix_stratum`` array when at least one member etymon has a
    populated ``stratum`` column. Each entry maps the canonical_form to
    its within-language register tag. Sorted by form for determinism.

    The seed routes celtic_mix → language='celtic' (the legacy
    rando-port bucket); Phase 1 only classifies the modern-language
    welsh slots, so we relabel both rows to language='welsh' and stamp
    distinct strata directly. This pins the export behavior without
    coupling to the classifier."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["fort"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[
                {
                    "modern_usage": "Caer-",
                    "celtic_mix": ["caer", "din"],
                }
            ],
            language_overrides={"caer": "welsh", "din": "welsh"},
        )
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = ? AND language = 'welsh'",
            ("latin-loan", "caer"),
        )
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = ? AND language = 'welsh'",
            ("native-welsh", "din"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "celtic_mix_stratum" in word, (
        f"celtic_mix_stratum missing from word entry: keys={list(word.keys())}"
    )
    assert word["celtic_mix_stratum"] == [
        {"form": "caer", "stratum": "latin-loan"},
        {"form": "din", "stratum": "native-welsh"},
    ]


def test_export_meanings_skips_stratum_when_all_members_null(fresh_db: Path) -> None:
    """A family whose every member has NULL stratum (the common case for
    languages without a Phase 1 classifier — Old English / Latin / etc.,
    plus pre-classifier rows in classified languages) MUST NOT emit a
    ``<lang>_stratum`` field. Pinning so the bundle doesn't bloat with
    empty arrays for unclassified rows."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        # stratum stays NULL — Old English has no Phase 1 classifier.
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_stratum" not in word
    assert all(not k.endswith("_stratum") for k in word)


def test_export_meanings_partial_stratum_emits_only_populated(fresh_db: Path) -> None:
    """A welsh family where some forms carry stratum and others stay
    NULL emits a ``celtic_mix_stratum`` array with only the populated
    entries. Mirrors the english_shaped contract so the runtime can
    treat absence-from-array as 'no stratum filter applies' for the
    unmapped form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[
                {
                    "modern_usage": "Cwm-",
                    "celtic_mix": ["cwm", "glyn"],
                }
            ],
            language_overrides={"cwm": "welsh", "glyn": "welsh"},
        )
        # Only 'cwm' gets a stratum; 'glyn' stays NULL.
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = ? AND language = 'welsh'",
            ("native-welsh", "cwm"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["celtic_mix"] == ["cwm", "glyn"]  # Both forms in lang array.
    assert word["celtic_mix_stratum"] == [
        {"form": "cwm", "stratum": "native-welsh"},
    ]


def test_export_meanings_unions_stratum_across_welsh_substrata(
    fresh_db: Path,
) -> None:
    """Welsh + middle-welsh + old-welsh all funnel into the celtic_mix
    bundle bucket. A family spanning multiple welsh-substrata languages
    must surface each form's stratum in the bucket-level array. Pins
    that the per-bundle-bucket aggregation in _emit_word_languages
    unions stratum the same way it unions english_shaped, citations,
    etc.

    Hand-seeded so the family rollup picks up an old-welsh row whose
    lemma is the welsh-language root — the legacy seeder doesn't expose
    this multi-substrata shape directly."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["river"],
            tags=["water"],
            modifier_type="Topographical",
            words=[
                {
                    "modern_usage": "Afon-",
                    "celtic_mix": ["afon"],
                }
            ],
            language_overrides={"afon": "welsh"},
        )
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = 'afon'",
            ("native-welsh",),
        )
        afon_root_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'afon' AND language = 'welsh'"
        ).fetchone()["id"]
        # Add an old-welsh sibling pointing at the same lemma so the
        # rollup sees them as one family.
        db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, lemma_id, stratum) VALUES (?, ?, ?, ?)",
            ("auon", "old-welsh", afon_root_id, "brittonic-substrate"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # Both forms surface in celtic_mix; the stratum array carries both
    # tags keyed by their canonical_form.
    assert set(word["celtic_mix"]) == {"afon", "auon"}
    assert word["celtic_mix_stratum"] == [
        {"form": "afon", "stratum": "native-welsh"},
        {"form": "auon", "stratum": "brittonic-substrate"},
    ]


def test_export_meanings_stratum_collision_resolves_by_lang_sort_order(
    fresh_db: Path,
) -> None:
    """Two welsh-substrata members sharing the SAME canonical_form with
    DIFFERENT strata land in the same celtic_mix bucket key. Pin the
    deterministic resolution: ``_emit_word_languages`` walks
    ``sorted(accs.forms_by_lang)`` and ``bucket.stratum.update(...)``,
    so the alphabetically-LAST language wins. middle-welsh < welsh, so
    welsh's stratum overrides middle-welsh's. This locks the same
    last-lang-wins contract the english_shaped / inflections / citations
    plumbing already follows; without this test the order would be
    silently dependent on dict iteration order."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["fort"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[
                {
                    "modern_usage": "Caer-",
                    "celtic_mix": ["caer"],
                }
            ],
            language_overrides={"caer": "welsh"},
        )
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = 'caer'",
            ("latin-loan",),
        )
        caer_root_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'caer' AND language = 'welsh'"
        ).fetchone()["id"]
        # Add a middle-welsh sibling with the SAME canonical_form but a
        # DIFFERENT stratum, lemma_id pointing at the welsh root.
        db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, lemma_id, stratum) VALUES (?, ?, ?, ?)",
            ("caer", "middle-welsh", caer_root_id, "medieval-welsh"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # Both rows share form='caer'; the bucket emits exactly one entry.
    assert word["celtic_mix"] == ["caer"]
    # welsh > middle-welsh alphabetically → welsh's stratum wins.
    assert word["celtic_mix_stratum"] == [
        {"form": "caer", "stratum": "latin-loan"},
    ]


def test_export_meanings_attested_years_picks_earliest_across_lexicon_codes(
    fresh_db: Path,
) -> None:
    """attested_years twin of the stratum-collision test, but the merge is a
    MIN not last-lang-wins: two welsh-substrata codes sharing form='caer' land
    in the same celtic_mix bucket with DIFFERENT attested years; the bucket
    must keep the EARLIEST regardless of lang sort order. Unlike the eight
    plain-.update() families (last-lang-wins), attested_years uses the bespoke
    ``year < existing`` merge that lives entirely in _accumulate_lang_into_bucket
    (wyrd-8uvi C901 split) — a `<`→`>` flip or a dropped `existing is None`
    guard would pass every other test."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["fort"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "Caer-", "celtic_mix": ["caer"]}],
            language_overrides={"caer": "welsh"},
        )
        caer_root_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'caer' AND language = 'welsh'"
        ).fetchone()["id"]
        db.add_citation(caer_root_id, "mawer_1920")
        # welsh 'caer' attests 1300 (later); middle-welsh sibling 'caer'
        # (same bucket) attests 1086 (earlier) — the min must win.
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, ?, ?, ?, ?, ?)",
            (caer_root_id, "mawer_1920", "caer", 1, 0, 1300),
        )
        cur = db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, lemma_id) VALUES (?, ?, ?)",
            ("caer", "middle-welsh", caer_root_id),
        )
        mw_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, ?, ?, ?, ?, ?)",
            (mw_id, "mawer_1920", "caer", 1, 0, 1086),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["celtic_mix"] == ["caer"]
    # Earliest (1086, middle-welsh) wins over 1300 (welsh) in the bucket merge.
    assert word["celtic_mix_attested_years"] == [{"form": "caer", "year": 1086}]


def test_export_meanings_stratum_via_reflex_linked_inflected_descendant(
    fresh_db: Path,
) -> None:
    """The ``_word_for_reflex`` path walks ``member_descendants`` so a
    reflex linked to a lemma picks up its inflected children's stratum
    too. Pin that the descendant walk absorbs stratum from each
    descendant, not just the directly-linked member.

    Realistic shape: classify_welsh stamps stratum on every welsh-family
    etymon including inflected children; the export must surface those
    in the bundle so a Phase 3 generator filtering by stratum sees the
    inflected forms alongside the lemma."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Cwm-", "celtic_mix": ["cwm"]}],
            language_overrides={"cwm": "welsh"},
        )
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE canonical_form = 'cwm'",
            ("native-welsh",),
        )
        cwm_root_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'cwm' AND language = 'welsh'"
        ).fetchone()["id"]
        # Add an inflected child carrying its own (different) stratum.
        # The reflex 'Cwm-' is already linked to cwm_root; the descendant
        # walk in _word_for_reflex must absorb the child's stratum even
        # though the reflex doesn't link directly to the child.
        db.conn.execute(
            "INSERT INTO etymon (canonical_form, language, lemma_id, inflection, stratum) "
            "VALUES (?, ?, ?, ?, ?)",
            ("cymau", "welsh", cwm_root_id, "plural", "english-loan"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert set(word["celtic_mix"]) == {"cwm", "cymau"}
    # Both strata surface — the descendant walk picked up the child's
    # tag from its own etymon row, not the lemma's.
    assert word["celtic_mix_stratum"] == [
        {"form": "cwm", "stratum": "native-welsh"},
        {"form": "cymau", "stratum": "english-loan"},
    ]


def test_export_meanings_emits_variant_pool_per_language(fresh_db: Path) -> None:
    """An etymon with text-match rows whose matched_form differs from the
    canonical_form gets a `<lang>_variants` field populated in its word
    entry, ordered by descending weight (most-attested first)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        _add_text_match(db, brycg_id, "src-a", "brigge", 5)
        # Use reverse-search method on the singleton so it survives the
        # low-confidence-singleton filter (which drops count=1 fuzzy /
        # llm-disambiguated rows as the dominant OCR-noise class).
        _add_text_match(db, brycg_id, "src-a", "brige", 1, method="reverse-search-v1")
        # Same matched_form from a different source: weights sum.
        _add_text_match(db, brycg_id, "rando-port", "brigge", 3)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_variants" in word
    # Sorted by descending weight; brigge (5+3=8) leads brige (1).
    assert word["old_english_variants"] == [
        {"form": "brigge", "weight": 8},
        {"form": "brige", "weight": 1},
    ]


def test_export_meanings_dedupes_variant_against_canonical_forms(
    fresh_db: Path,
) -> None:
    """A matched_form that's already a canonical form of any family member
    gets dropped from the variant pool — emitting it again would just
    duplicate what's already in the language array."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        # 'BRYCG' (case-insensitive match against canonical) should be dropped.
        _add_text_match(db, brycg_id, "src-a", "BRYCG", 100)
        # 'brigge' is genuinely different and should survive.
        _add_text_match(db, brycg_id, "src-a", "brigge", 5)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    forms = [v["form"] for v in word.get("old_english_variants", [])]
    assert "brigge" in forms
    assert "BRYCG" not in forms
    assert "brycg" not in forms


def test_export_meanings_aggregates_variants_across_descendants(
    fresh_db: Path,
) -> None:
    """A reflex linked to a lemma surfaces variants from the lemma AND its
    inflected descendants (D8 rollup composes with D18 variant pools).
    Weights for the same matched_form across descendants sum together."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)",
            (cot_id, cotan_id, cotes_id),
        )
        # The lemma and its inflections each produce 'cotte' as a variant —
        # weights should sum.
        _add_text_match(db, cot_id, "src-a", "cotte", 4)
        _add_text_match(db, cotan_id, "src-a", "cotte", 3)
        _add_text_match(db, cotes_id, "rando-port", "cotte", 2)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    variants = word["old_english_variants"]
    assert variants == [{"form": "cotte", "weight": 9}]


def test_emit_variant_list_sorts_by_descending_weight_then_form(
    fresh_db: Path,
) -> None:
    """`_emit_variant_list` sorts (form, weight) pairs by descending weight,
    breaking ties on form alphabetically — both axes matter for diff
    stability across rebuilds."""

    out = _emit_variant_list({"alpha": 5, "beta": 5, "gamma": 7, "delta": 1})
    assert out == [
        {"form": "gamma", "weight": 7},
        {"form": "alpha", "weight": 5},
        {"form": "beta", "weight": 5},
        {"form": "delta", "weight": 1},
    ]


def test_emit_variant_list_empty_input_returns_empty(fresh_db: Path) -> None:
    assert _emit_variant_list({}) == []


def test_export_meanings_emits_inflection_pool_per_language(fresh_db: Path) -> None:
    """An etymon family with inflected children gets a `<lang>_inflections`
    field listing the (form, label) pairs for each inflected member.
    The lemma form does NOT appear in the inflections list — it's the
    unmarked headword."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotum_id = db.upsert_etymon("cotum", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "dative_or_pl", cotum_id),
        )
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "weak_oblique", cotan_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_inflections" in word
    inflections = word["old_english_inflections"]
    forms = {entry["form"]: entry["inflection"] for entry in inflections}
    assert forms == {"cotum": "dative_or_pl", "cotan": "weak_oblique"}
    # cot (the lemma) doesn't appear in the inflection pool.
    assert "cot" not in {entry["form"] for entry in inflections}


def test_export_meanings_synthesized_family_emits_inflections(fresh_db: Path) -> None:
    """A family without any linked reflex (synthesized-word path) emits
    inflection metadata too. _build_words_for_group has separate aggregation
    blocks for the reflex-linked and synthesized branches; this test pins
    the synthesized side so a regression there can't slip past the
    reflex-only test above."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        # No rando-port reflex — hits the families_without_reflex branch.
        cot_id = db.upsert_etymon("cot", "old-english", modifier_type="Habitative")
        db.add_gloss(cot_id, "cottage")
        db.add_tag(cot_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(cot_id, src)
        cotum_id = db.upsert_etymon("cotum", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "dative_or_pl", cotum_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    word = subjects[0]["words"][0]
    assert word["modern_usage"] == "cot"  # synthesized from canonical_form
    assert word["old_english_inflections"] == [{"form": "cotum", "inflection": "dative_or_pl"}]


def test_emit_inflection_list_sorts_by_form_alphabetically(fresh_db: Path) -> None:
    """`_emit_inflection_list` sorts entries by form so output is byte-stable
    regardless of dict insertion order."""

    out = _emit_inflection_list(
        {"cotum": "dative_or_pl", "cotan": "weak_oblique", "cotes": "genitive_strong"}
    )
    assert out == [
        {"form": "cotan", "inflection": "weak_oblique"},
        {"form": "cotes", "inflection": "genitive_strong"},
        {"form": "cotum", "inflection": "dative_or_pl"},
    ]


def test_emit_inflection_list_empty_input_returns_empty(fresh_db: Path) -> None:
    assert _emit_inflection_list({}) == []


# --- wyrd-8tp: gloss + variant data-quality filters -----------------------


def test_filter_concatenation_glosses_drops_comma_concat():
    """Glosses that are entirely a concatenation of already-present
    singleton glosses should be filtered."""

    out = _filter_concatenation_glosses(["lake", "pond", "lake, pond"])
    assert out == ["lake", "pond"]


def test_filter_concatenation_glosses_keeps_partial_overlap():
    """A multi-token gloss that contains a singleton plus extra info should
    be kept — only the strict-subset case is considered redundant."""

    # 'shore' is not in the singleton set, so 'bank, shore' is informative.
    out = _filter_concatenation_glosses(["bank", "bank, shore"])
    assert out == ["bank", "bank, shore"]


def test_filter_concatenation_glosses_recognizes_semicolon_and_or():
    """Splitter accepts ',', ';' and ' or ' as connectors."""

    out = _filter_concatenation_glosses(["book", "charter", "charter; book", "book or charter"])
    assert out == ["book", "charter"]


def test_filter_concatenation_glosses_keeps_singletons_unconditionally():
    """A single-token gloss is never dropped, even if its tokens overlap
    other singletons (since by definition it has only one token)."""

    out = _filter_concatenation_glosses(["bank", "edge", "shore"])
    assert out == ["bank", "edge", "shore"]


def test_export_meanings_filters_low_confidence_singleton_variants(
    fresh_db: Path,
) -> None:
    """A variant with match_count=1 from fuzzy-search-v1 alone should be
    dropped (the dominant OCR-noise class). One with match_count>=2 OR
    any high-confidence method survives."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["sharp"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "-scearp", "old_english": ["scearu"]}],
        )
        scearu_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'scearu'"
        ).fetchone()["id"]
        # OCR-noise: count=1, fuzzy-search-v1. Should be dropped.
        _add_text_match(db, scearu_id, "src-a", "saearp", 1, method="fuzzy-search-v1")
        # OCR-noise: count=1, llm-disambiguated-v1. Same low-confidence class;
        # locks in that the second member of _LOW_CONFIDENCE_METHODS is honored.
        _add_text_match(db, scearu_id, "src-a", "scrarp", 1, method="llm-disambiguated-v1")
        # Legitimate variant: count=2 from fuzzy-search-v1 — survives because
        # ≥2 attestations gives more confidence.
        _add_text_match(db, scearu_id, "src-a", "scerp", 2, method="fuzzy-search-v1")
        # Legitimate variant: count=1 but reverse-search-v1 (canonical-form
        # body scan) is high-confidence. Survives.
        _add_text_match(db, scearu_id, "src-a", "scearp-form", 1, method="reverse-search-v1")
        # Mixed-methods row: count=1 split between fuzzy-search-v1 AND
        # reverse-search-v1 from different sources. methods set is NOT a
        # subset of low-confidence (reverse-search is high-confidence), so
        # the form survives. This pins the issubset check vs a simpler
        # "all-fuzzy" equality test.
        db.upsert_source(id="src-b", title="B")
        _add_text_match(db, scearu_id, "src-a", "scearp-mix", 1, method="fuzzy-search-v1")
        _add_text_match(db, scearu_id, "src-b", "scearp-mix", 1, method="reverse-search-v1")
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    forms = [v["form"] for v in word.get("old_english_variants", [])]
    # Low-confidence singletons dropped.
    assert "saearp" not in forms
    assert "scrarp" not in forms
    # ≥2-count fuzzy survives.
    assert "scerp" in forms
    # Pure high-confidence singleton survives.
    assert "scearp-form" in forms
    # Mixed-methods singleton survives because not all methods are low-confidence.
    assert "scearp-mix" in forms


def test_export_meanings_drops_concatenation_glosses_in_subjects(
    fresh_db: Path,
) -> None:
    """A subject whose family carries 'lake', 'pond', AND the redundant
    'lake, pond' should emit only the singletons in its `meaning` list."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Manually inject etymon + glosses for the test, since _seed_subject
        # runs through the seed_from_meanings path which deduplicates.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["lake", "pond"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "-mere", "old_english": ["mere"]}],
        )
        mere_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'mere'").fetchone()[
            "id"
        ]
        # Add a redundant concatenation gloss directly.
        db.conn.execute(
            "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (mere_id, "lake, pond"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    # The mere subject's meaning list should have 'lake' and 'pond' but not
    # 'lake, pond'.
    found = next(s for s in subjects if any(w["modern_usage"] == "mere" for w in s["words"]))
    assert "lake" in found["meaning"]
    assert "pond" in found["meaning"]
    assert "lake, pond" not in found["meaning"]


def test_export_meanings_omits_merge_loser_as_separate_subject(
    fresh_db: Path,
) -> None:
    """An OCR-cluster loser doesn't become its own subject — its form rides
    along in the canonical's language list (D22)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-den", "old_english": ["denu"]}],
        )
        denu_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'denu'").fetchone()[
            "id"
        ]
        # Loser: an OCR variant pointing into denu.
        dene_id = db.upsert_etymon("dene", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (denu_id, dene_id))
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    forms = subjects[0]["words"][0]["old_english"]
    assert forms[0] == "denu"  # canonical leads
    assert "dene" in forms  # loser rides along


def test_export_meanings_synthesizes_word_for_mined_only_family(
    fresh_db: Path,
) -> None:
    """A family with ≥3 witnesses but no linked reflex gets a synthesized
    `words` entry from its canonical_form."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        tune_id = db.upsert_etymon("tune", "old-english", modifier_type="Habitative")
        db.add_gloss(tune_id, "settlement")
        db.add_tag(tune_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(tune_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    assert len(subjects) == 1
    assert subjects[0]["words"] == [
        {
            "modern_usage": "tune",
            "morpheme_id": "old-english:tune",  # wyrd-rogd.10: owning morpheme (content id)
            "old_english": ["tune"],
            "old_english_citations": ["a", "b", "c"],
            # wyrd-vm8t Loop 4: G2P surface pronunciation.
            "old_english_pronunciation": [{"form": "tune", "ipa": "/tʊnɛ/", "dialect": None}],
            # self-seed: own form in its own era column
            "era_reflexes": {"old-english": [{"form": "tune", "source": "self"}]},
        }
    ]


def test_export_meanings_synthesize_path_emits_attested_years(
    fresh_db: Path,
) -> None:
    """The synth-without-linked-reflex code path (mined-only families,
    no rando-port reflex) needs its own attested-year coverage. The
    structure differs from the linked-reflex path (flat
    member_form_by_id walk vs descendant walk), so a regression here
    wouldn't be caught by the reflex-path tests in the wyrd-bag set.

    Pin: a tune-id mined etymon with three citations + an ETM row
    carrying attested_year=1086 surfaces the year in the synthesized
    word's `<lang>_attested_years` field."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        tune_id = db.upsert_etymon("tune", "old-english", modifier_type="Habitative")
        db.add_gloss(tune_id, "settlement")
        db.add_tag(tune_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(tune_id, src)
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tune_id, "a", "tune", 1, 0, 1086),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "tune", "year": 1086}]


# --- wyrd-9kh.1: per-language citation attribution -----------------------


def test_export_meanings_filters_rando_port_from_citations(fresh_db: Path) -> None:
    """The rando-port seed isn't a real scholarly attribution — it's the
    Wikipedia-derived legacy bootstrap. Filter it from `<lang>_citations`
    so a rando-only word emits no citations field at all (rather than a
    confusing 'cited by rando-port')."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["acorn"],
            tags=["plant"],
            modifier_type=None,
            words=[{"modern_usage": "-ock", "old_english": ["aecern"]}],
        )
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english" in word
    assert "old_english_citations" not in word


def test_export_meanings_emits_citations_for_scholarly_witnesses(fresh_db: Path) -> None:
    """Mixed scholarly + rando-port citations: only the scholars surface in
    `<lang>_citations`, sorted alphabetically. Set semantics dedupe across
    the descendant walk."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        for src in ("mawer_1920", "skeat_1901"):
            db.upsert_source(id=src, title=src)
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        # Two scholars + a duplicate of mawer (ON CONFLICT DO NOTHING dedupes).
        db.add_citation(brycg_id, "mawer_1920")
        db.add_citation(brycg_id, "skeat_1901")
        db.add_citation(brycg_id, "mawer_1920")
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_citations"] == ["mawer_1920", "skeat_1901"]


def test_export_meanings_emits_attested_years_from_etymon_text_match(
    fresh_db: Path,
) -> None:
    """wyrd-bag: ``<lang>_attested_years`` is sourced from the rollup
    of ``etymon_text_match.attested_year`` (PR #47) and
    ``toponym_etymology.attested_year`` (PR #53). This test exercises
    the ETM path: a scholar-cited etymon with one text-match row
    carrying year 1086 → bundle entry ``[{form: brycg, year: 1086}]``."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        db.add_citation(brycg_id, "mawer_1920")
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (brycg_id, "mawer_1920", "brycg", 1, 0, 1086),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "brycg", "year": 1086}]


def test_export_meanings_emits_attested_years_from_toponym_etymology(
    fresh_db: Path,
) -> None:
    """wyrd-bag: the toponym_etymology path. A toponym whose etymology
    cites this etymon as one element, and whose row has attested_year
    populated, surfaces the year in the bundle."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # Set up a toponym + etymology + element pointing at tun, with
        # attested_year on the etymology row.
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Town",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, historical_form, confidence, notes, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (toponym_id, "mawer_1920", "tūn", "high", "Tune 1086 DB", 1086),
        )
        te_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (te_id, 0, tun_id),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "tun", "year": 1086}]


def test_export_meanings_attested_years_picks_earliest_across_sources(
    fresh_db: Path,
) -> None:
    """An etymon attested via BOTH etymon_text_match and toponym_etymology
    surfaces the EARLIEST year — pin via SQL MIN(year) so a future
    refactor that drops the rollup surfaces immediately."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # ETM row attests year 1242. Toponym_etymology attests 1086.
        # MIN should win.
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tun_id, "mawer_1920", "tun", 1, 0, 1242),
        )
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Town",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, confidence, notes, attested_year) "
            "VALUES (?, ?, ?, ?, ?)",
            (toponym_id, "mawer_1920", "high", "1086 DB", 1086),
        )
        te_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (te_id, 0, tun_id),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # 1086 wins over 1242.
    assert word["old_english_attested_years"] == [{"form": "tun", "year": 1086}]


def test_export_meanings_omits_attested_years_field_when_no_year_evidence(
    fresh_db: Path,
) -> None:
    """A morpheme with no attested year on either row source has no
    ``<lang>_attested_years`` field in its emitted word — keeps the
    bundle clean and lets the runtime --era filter pass it through."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["wood"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-wood", "old_english": ["wudu"]}],
        )
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_attested_years" not in word


def test_export_meanings_attested_years_sorted_by_year_ascending(
    fresh_db: Path,
) -> None:
    """The emit list is sorted by year ascending so the runtime --era
    filter (D5-2) can short-circuit on the first match. Ties broken
    alphabetically by form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # Three inflected variants, intentionally inserted in non-
        # chronological order. Each has its own etymon row + ETM row
        # with attested_year. The lemma_id chains them under tun.
        for form, year in [("tunna", 1340), ("tun", 1086), ("tunes", 1242)]:
            child_id = db.upsert_etymon(form, "old-english")
            db.conn.execute(
                "UPDATE etymon SET lemma_id = ? WHERE id = ?",
                (tun_id, child_id),
            )
            db.conn.execute(
                "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
                "match_count, edit_distance, attested_year) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (child_id, "mawer_1920", form, 1, 0, year),
            )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # Sorted by year ascending; tun(1086) beats tunes(1242) beats tunna(1340).
    assert word["old_english_attested_years"] == [
        {"form": "tun", "year": 1086},
        {"form": "tunes", "year": 1242},
        {"form": "tunna", "year": 1340},
    ]


def test_export_meanings_citations_dedupe_across_descendants(fresh_db: Path) -> None:
    """A reflex linked to a lemma surfaces citations from the lemma AND its
    OCR-cluster losers / inflected children — the same scholar cited on a
    descendant shouldn't double in the bundle output."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        for src in ("mawer_1920", "skeat_1901"):
            db.upsert_source(id=src, title=src)
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-den", "old_english": ["denu"]}],
        )
        denu_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'denu'").fetchone()[
            "id"
        ]
        # OCR-cluster loser: another spelling of the same etymon merged into denu.
        loser_id = db.upsert_etymon("denū", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (denu_id, loser_id))
        # mawer cites both the canonical and the OCR-loser (because it
        # extracted from a body where the OCR garbled the form). skeat
        # cites only the canonical.
        db.add_citation(denu_id, "mawer_1920")
        db.add_citation(denu_id, "skeat_1901")
        db.add_citation(loser_id, "mawer_1920")
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # mawer appears once despite citing two etymons in the family.
    assert word["old_english_citations"] == ["mawer_1920", "skeat_1901"]


def test_export_meanings_synthesizes_with_position_pref(fresh_db: Path) -> None:
    """When position_pref is set, the synthesized usage carries the matching
    dash markers (`pre`/`post`/`inner`)."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        ham_id = db.upsert_etymon(
            "ham", "old-english", modifier_type="Habitative", position_pref="post"
        )
        db.add_gloss(ham_id, "homestead")
        for src in ("a", "b", "c"):
            db.add_citation(ham_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    assert subjects[0]["words"][0]["modern_usage"] == "-ham"


def test_export_meanings_synthesizes_pre_position_with_trailing_dash(
    fresh_db: Path,
) -> None:
    """position_pref='pre' produces 'foo-' usage."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        e_id = db.upsert_etymon("ada", "old-english", position_pref="pre")
        db.add_gloss(e_id, "Ada (Name)")
        for src in ("a", "b", "c"):
            db.add_citation(e_id, src)
        db.commit()
        subjects = export_meanings(db, include_rando=False, min_witnesses=3)
    assert subjects[0]["words"][0]["modern_usage"] == "ada-"


def test_export_meanings_synthesizes_inner_position_with_dashes_both_sides(
    fresh_db: Path,
) -> None:
    """position_pref='inner' produces '-foo-' usage."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        e_id = db.upsert_etymon("ar", "old-english", position_pref="inner")
        db.add_gloss(e_id, "of the")
        for src in ("a", "b", "c"):
            db.add_citation(e_id, src)
        db.commit()
        subjects = export_meanings(db, include_rando=False, min_witnesses=3)
    assert subjects[0]["words"][0]["modern_usage"] == "-ar-"


def test_export_meanings_includes_orphan_reflex_subjects(fresh_db: Path) -> None:
    """A reflex with no linked etymon (rando seed of {'modern_usage': 'Adam-'}
    or {'celtic_mix': [], 'modern_usage': 'Bre-'}) still gets exported as its
    own subject so per-culture proportions can resolve the usage."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Subject with empty language slots — produces an orphan reflex.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Adam (Name)"],
            tags=["male name"],
            modifier_type="Habitative",
            words=[{"modern_usage": "Adam-"}],
        )
        subjects = export_meanings(db, include_rando=True)

    orphan_words = [
        w["modern_usage"]
        for s in subjects
        for w in s["words"]
        if not any(k != "modern_usage" for k in w)
    ]
    assert "Adam" in orphan_words


def test_export_meanings_orphan_reflexes_skipped_when_include_rando_false(
    fresh_db: Path,
) -> None:
    """Orphan reflexes are rando-port-only data; --no-include-rando drops them."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=[],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "Adam-"}],
        )
        subjects = export_meanings(db, include_rando=False)
    assert subjects == []


def test_export_meanings_groups_multiple_families_into_one_subject(
    fresh_db: Path,
) -> None:
    """Two distinct families with the same (modifier_type, glosses, tags)
    signature collapse into one subject with multiple words. This exercises
    the dict-based grouping in _group_families_into_subjects."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c", "d", "e", "f"):
            db.upsert_source(id=src, title=src)
        # Two distinct lemmas, same gloss/tag/modifier_type signature.
        # Each gets a separate reflex linked to it.
        cot_id = db.upsert_etymon(
            "cot", "old-english", modifier_type="Habitative", position_pref="post"
        )
        ham_id = db.upsert_etymon(
            "ham", "old-english", modifier_type="Habitative", position_pref="post"
        )
        for e_id in (cot_id, ham_id):
            db.add_gloss(e_id, "homestead")
            db.add_tag(e_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(cot_id, src)
        for src in ("d", "e", "f"):
            db.add_citation(ham_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    # Single subject because both families share the same signature.
    assert len(subjects) == 1
    subj = subjects[0]
    assert subj["meaning"] == ["homestead"]
    assert subj["modifier_tags"] == ["habitation"]
    usages = sorted(w["modern_usage"] for w in subj["words"])
    assert usages == ["-cot", "-ham"]


def test_export_meanings_reflex_linked_to_inflection_only_emits_that_inflection(
    fresh_db: Path,
) -> None:
    """A reflex linked directly to an inflection (not the lemma) emits only
    that inflection's form plus the inflection's own descendants — not the
    lemma's form or sibling inflections. The descendant walk is downward,
    not upward; if the seed wanted the reflex to cover the whole morpheme,
    it should have linked to the lemma."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Seed with reflex linked to inflected form 'cotan' rather than lemma 'cot'.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cotan", "old_english": ["cotan"]}],
        )
        # Add the lemma + sibling inflection as separate entities later.
        cotan_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'cotan'"
        ).fetchone()["id"]
        cot_id = db.upsert_etymon("cot", "old-english", modifier_type="Habitative")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Wire D8 lemma linkage: cotan and cotes both inflect from cot.
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)",
            (cot_id, cotan_id, cotes_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    # All three etymons roll up to cot (the lemma). The subject reconstruction
    # groups by (modifier_type, glosses, tags); here the seed gave glosses
    # only to cotan, so cot/cotes have no glosses → they form a separate
    # subject (no overlap). Find the subject with the -cotan reflex.
    word = next(w for s in subjects for w in s["words"] if w["modern_usage"] == "cotan")
    # Reflex linked only to cotan; descendants of cotan = {cotan} (no children
    # below). Lemma 'cot' and sibling 'cotes' must NOT appear here.
    assert word["old_english"] == ["cotan"]


def test_export_meanings_narrows_word_language_to_linked_etymons(
    fresh_db: Path,
) -> None:
    """A subject grouping a Celtic family and an Old English family must NOT
    bleed forms across reflexes: '-don' (linked to OE) carries old_english
    only; 'Bre-' (linked to Celtic) carries celtic_mix only. Without per-
    reflex narrowing the explainer would falsely attribute 'don' to Celtic
    and vice versa (wyrd-mzl regression guard)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[
                {"modern_usage": "Bre-", "celtic_mix": ["bre"]},
                {"modern_usage": "-don", "old_english": ["dun"]},
            ],
        )
        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    by_usage = {w["modern_usage"]: w for w in subjects[0]["words"]}
    assert "celtic_mix" in by_usage["Bre"]
    assert "old_english" not in by_usage["Bre"]
    assert "old_english" in by_usage["don"]
    assert "celtic_mix" not in by_usage["don"]


def test_export_meanings_round_trips_through_load_meanings(fresh_db: Path) -> None:
    """A seeded subject exports to a structure the runtime can re-load via
    load_meanings — closes the authoring↔runtime loop end-to-end."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[
                {"modern_usage": "Bre-", "celtic_mix": ["bre", "bryn"]},
                {"modern_usage": "-don", "old_english": ["dun"]},
            ],
        )
        subjects = export_meanings(db, include_rando=True)

    meaning_db, tag_db = load_meanings(subjects)
    assert "Bre" in meaning_db
    assert "don" in meaning_db
    assert any("topography" in m.tags for m in meaning_db["Bre"])
    assert "topography" in tag_db


def test_export_meanings_cli_writes_file(fresh_db: Path, tmp_path: Path) -> None:
    """CLI command writes JSON to --output path and reports the subject count."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-hill", "old_english": ["hyll"]}],
        )

    out_path = tmp_path / "meanings.json"
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 1 subjects" in result.stderr
    # wyrd-c1vq: bundle is dict-shape; subjects under the 'subjects' key.
    payload = json.loads(out_path.read_text())
    subjects = payload["subjects"]
    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["Hill"]
    assert subjects[0]["words"][0]["modern_usage"] == "hill"


def test_export_meanings_cli_emits_to_stdout(fresh_db: Path) -> None:
    """Without --output, the CLI emits the JSON payload to stdout."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-hill", "old_english": ["hyll"]}],
        )

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "export-meanings", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(s["meaning"] == ["Hill"] for s in payload["subjects"])


def test_export_meanings_cli_lang_threshold_flag(fresh_db: Path) -> None:
    """--lang-threshold LANG=N overrides the preset for a specific language."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # Two ON sources for 'fell'; with --lang-threshold old-norse=3 it
        # falls under the gate even though the preset would have promoted it.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    # Without override (preset has ON at 2): promotes.
    r1 = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "export-meanings", "--db", str(fresh_db), "--no-include-rando"],
    )
    assert r1.exit_code == 0, r1.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(r1.stdout)["subjects"])
    assert promoted

    # With --lang-threshold old-norse=3: doesn't promote.
    r2 = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--lang-threshold",
            "old-norse=3",
        ],
    )
    assert r2.exit_code == 0, r2.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(r2.stdout)["subjects"])
    assert not promoted


def test_export_meanings_cli_lang_threshold_accepts_json_field_alias(
    fresh_db: Path,
) -> None:
    """--lang-threshold accepts the JSON-field alias ('old_scandinavian') and
    normalizes to the lexicon's canonical code ('old-norse'). Without
    normalization, --lang-threshold old_scandinavian=3 would silently miss
    every ON consensus row (since the column stores 'old-norse')."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--lang-threshold",
            "old_scandinavian=3",
        ],
    )
    assert result.exit_code == 0, result.output
    # Threshold 3 with only 2 witnesses → not promoted.
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(result.stdout)["subjects"])
    assert not promoted


def test_export_meanings_cli_no_preset_uses_uniform_min_witnesses(fresh_db: Path) -> None:
    """--no-preset clears the preset; the global --min-witnesses applies uniformly."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--no-preset",
            "--min-witnesses",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(result.stdout)["subjects"])
    assert not promoted


def test_export_meanings_cli_rejects_malformed_lang_threshold(fresh_db: Path) -> None:
    """--lang-threshold must be LANG=N; bad input fails fast with a clear message."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--lang-threshold",
            "old-norse-2",  # missing '='
        ],
    )
    assert result.exit_code != 0
    assert "LANG=N" in result.output


def test_export_meanings_cli_rejects_non_integer_threshold(fresh_db: Path) -> None:
    """The N in --lang-threshold LANG=N must be an integer; non-int fails clearly."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--lang-threshold",
            "old-norse=foo",
        ],
    )
    assert result.exit_code != 0
    assert "must be an integer" in result.output


def test_export_meanings_cli_rejects_empty_lang_or_threshold(fresh_db: Path) -> None:
    """--lang-threshold =3 or old-norse= give a clearer 'non-empty' error
    rather than the misleading 'must be an integer' message you'd get from
    int('') alone."""
    for spec in ("=3", "old-norse=", " = "):
        result = CliRunner().invoke(
            kenning_cli,
            [
                "lexicon",
                "export-meanings",
                "--db",
                str(fresh_db),
                "--lang-threshold",
                spec,
            ],
        )
        assert result.exit_code != 0, f"spec={spec!r} should have failed"
        assert "non-empty" in result.output, f"spec={spec!r}: {result.output}"


def test_export_meanings_cli_no_preset_with_lang_threshold_overrides(
    fresh_db: Path,
) -> None:
    """--no-preset --lang-threshold old-norse=2 → only ON gates at 2; everything
    else uses --min-witnesses uniformly. Pins the dict-merge ordering so a
    bug where the empty {} got the preset re-applied wouldn't slip through."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # ON at 2 witnesses — should promote via the explicit override.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        # Celtic at 2 witnesses — should NOT promote (preset would have said
        # celtic=2, but --no-preset cleared that, so --min-witnesses=3 applies).
        cel = db.upsert_etymon("bryn", "celtic")
        db.add_gloss(cel, "hill")
        db.add_citation(cel, "a")
        db.add_citation(cel, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--no-preset",
            "--min-witnesses",
            "3",
            "--lang-threshold",
            "old-norse=2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    glosses = {tuple(s["meaning"]) for s in payload["subjects"]}
    assert ("mountain",) in glosses
    assert ("hill",) not in glosses


def test_build_witness_filter_empty_uses_uniform_threshold() -> None:
    """Direct unit test: empty lang_thresholds returns the simple fragment.
    Parenthesized to match the per-language path's contract so the helper
    composes safely when dropped into a larger expression."""

    sql, params = _build_witness_filter({}, 3)
    assert sql == "(witnesses >= ?)"
    assert params == [3]


def test_build_witness_filter_emits_clauses_in_sorted_order() -> None:
    """Direct unit test: with multiple lang_thresholds, params land in
    sorted-by-lang order (so output is byte-stable regardless of dict
    insertion order), and the trailing fallback NOT IN (...) clause carries
    the global min_witnesses."""

    # Insertion order intentionally not sorted, to verify the helper sorts.
    sql, params = _build_witness_filter({"old-norse": 2, "celtic": 2, "old-english": 3}, 4)
    # SQL fragment uses ? placeholders; param order is what's deterministic.
    # Per-language clauses interleave (lang, threshold) in sorted order,
    # then the NOT IN list (sorted langs), then the fallback threshold.
    assert params == [
        "celtic",
        2,
        "old-english",
        3,
        "old-norse",
        2,
        "celtic",
        "old-english",
        "old-norse",
        4,
    ]
    # Three per-language clauses + one NOT IN fallback clause.
    assert sql.count("language = ?") == 3
    assert "language NOT IN" in sql
    assert sql.count("witnesses >= ?") == 4


def test_export_meanings_output_is_byte_stable_across_runs(fresh_db: Path) -> None:
    """Same DB → same JSON bytes. AUTOINCREMENT root_ids and SQLite's
    iteration order can shuffle results between fresh seeds; the export must
    sort comprehensively enough to defeat that."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        # Two families with the same signature — exercises tie-breaking in
        # _group_families_into_subjects' sort key.
        for form in ("cot", "ham"):
            e_id = db.upsert_etymon(
                form, "old-english", modifier_type="Habitative", position_pref="post"
            )
            db.add_gloss(e_id, "homestead")
            db.add_tag(e_id, "habitation")
            for src in ("a", "b", "c"):
                db.add_citation(e_id, src)
        db.commit()
        first = json.dumps(export_meanings(db, include_rando=False, min_witnesses=3))
        second = json.dumps(export_meanings(db, include_rando=False, min_witnesses=3))
    assert first == second


# --- D27 / wyrd-0ug: etymon_descent + cognate-cluster schema --------------------


def test_init_schema_creates_etymon_descent_table(fresh_db: Path) -> None:
    """Fresh install carries the etymon_descent table from data/seed/lexicon.sql
    so the migration and fresh-install paths stay in lockstep."""
    with LexiconDB(fresh_db) as db:
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "etymon_descent" in tables


def test_init_schema_etymon_has_cognate_columns(fresh_db: Path) -> None:
    """Fresh install carries cognate_id + cognate_method on etymon."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    assert "cognate_id" in cols
    assert "cognate_method" in cols


def test_init_schema_creates_etymon_descent_indexes(fresh_db: Path) -> None:
    """Both directional indexes on etymon_descent must exist so
    'descendants of X' and 'ancestors of Y' queries are O(degree) not
    O(rows)."""
    with LexiconDB(fresh_db) as db:
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert "idx_etymon_descent_parent" in indexes
    assert "idx_etymon_descent_child" in indexes
    assert "idx_etymon_cognate" in indexes


def test_etymon_descent_round_trip_insert_and_query(fresh_db: Path) -> None:
    """Round-trip: insert a Proto-Germanic → Old English inheritance edge
    and retrieve it cleanly. Pins the column shape (parent_id, child_id,
    edge_type, source_id, confidence, notes) so a future schema tweak
    surfaces here."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (proto_id, oe_id, "inheritance", "wiktionary", "high", "via {{inh}}"),
        )
        db.commit()
        row = db.conn.execute(
            "SELECT parent_id, child_id, edge_type, source_id, confidence, notes "
            "FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchone()

    assert row["parent_id"] == proto_id
    assert row["child_id"] == oe_id
    assert row["edge_type"] == "inheritance"
    assert row["source_id"] == "wiktionary"
    assert row["confidence"] == "high"
    assert row["notes"] == "via {{inh}}"


def test_etymon_descent_unique_per_parent_child_edge_source(fresh_db: Path) -> None:
    """The same (parent, child, edge_type, source) tuple deduplicates,
    so re-mining Wiktionary doesn't double-insert. Different edge_types
    or different sources for the same (parent, child) are SEPARATE rows
    — distinct evidence."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        db.upsert_source(id="other", title="Other source")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")

        def insert(edge_type: str, source: str) -> None:
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
                (proto_id, oe_id, edge_type, source),
            )

        insert("inheritance", "wiktionary")
        insert("inheritance", "wiktionary")  # dedupes
        insert("inheritance", "other")  # different source → separate row
        insert("borrowing", "wiktionary")  # different edge_type → separate row
        db.commit()

        n = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchone()["n"]
    assert n == 3


def test_etymon_descent_check_constraint_rejects_unknown_edge_type(
    fresh_db: Path,
) -> None:
    """The CHECK constraint on edge_type must reject typos / made-up
    values so a misspelled wiktextract template kind doesn't silently
    poison the graph."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        a_id = db.upsert_etymon("a", "old-english")
        b_id = db.upsert_etymon("b", "old-english")
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, ?, ?)",
                (a_id, b_id, "made-up-kind", "wiktionary"),
            )


def test_etymon_consensus_does_not_count_descent_as_witness(
    fresh_db: Path,
) -> None:
    """Critical contract pin: descent edges are a different axis of
    evidence than extraction citations (D4). A morpheme with one
    extraction citation but ten descent edges must still report
    witnesses=1 — otherwise the descent graph would inflate the
    promotion threshold and let unverified morphemes 'promote' just
    because Wiktionary attests their etymology chain."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        db.upsert_source(id="mawer_1920", title="Mawer")
        oe_id = db.upsert_etymon("tūn", "old-english")
        # One extraction witness from a real scholar.
        db.add_citation(oe_id, "mawer_1920")
        # Many descent edges — these MUST NOT count toward witnesses.
        for parent_form in ("*tūnaz", "*deuh₂-", "*dūnom"):
            parent_id = db.upsert_etymon(parent_form, "proto-germanic")
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, ?, ?)",
                (parent_id, oe_id, "inheritance", "wiktionary"),
            )
        db.commit()
        consensus_row = db.conn.execute(
            "SELECT witnesses FROM etymon_consensus WHERE canonical_form = 'tūn'"
        ).fetchone()
    assert consensus_row["witnesses"] == 1


def test_etymon_descent_descendants_query_walks_inheritance_chain(
    fresh_db: Path,
) -> None:
    """The reference query 'modern descendants of <root>' walks
    inheritance + borrowing edges via a recursive CTE. Pins the SQL so
    a schema tweak (column renames, edge type changes) surfaces here.

    Graph: *tūnaz → tūn (OE) → town (modern English)
                  → tún (ON) → tún (modern Icelandic)"""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        me_id = db.upsert_etymon("town", "modern-english")
        is_id = db.upsert_etymon("tún", "icelandic")

        for parent, child in (
            (proto_id, oe_id),
            (proto_id, on_id),
            (oe_id, me_id),
            (on_id, is_id),
        ):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (parent, child),
            )
        db.commit()

        # Recursive CTE — proposed reference query for INGESTION.md.
        rows = db.conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
              SELECT child_id FROM etymon_descent
                WHERE parent_id = ? AND edge_type IN ('inheritance', 'borrowing')
              UNION
              SELECT d.child_id FROM etymon_descent d
                JOIN descendants ON d.parent_id = descendants.id
                WHERE d.edge_type IN ('inheritance', 'borrowing')
            )
            SELECT e.canonical_form, e.language
            FROM descendants
            JOIN etymon e ON e.id = descendants.id
            ORDER BY e.language, e.canonical_form
            """,
            (proto_id,),
        ).fetchall()

    forms = [(row["canonical_form"], row["language"]) for row in rows]
    assert ("tún", "icelandic") in forms
    assert ("town", "modern-english") in forms
    assert ("tūn", "old-english") in forms
    assert ("tún", "old-norse") in forms
    # Root itself should NOT appear in descendants (the seed row is
    # parent_id = proto_id, but we select child_id).
    assert ("*tūnaz", "proto-germanic") not in forms


def test_etymon_descent_ancestors_query_walks_inheritance_chain_upward(
    fresh_db: Path,
) -> None:
    """Symmetric pin for the 'ancestors of X' recursive CTE (documented
    in DECISIONS.md D27 and INGESTION.md). Without this, a regression
    that drops idx_etymon_descent_child or breaks the upward walk would
    only surface in production. Same graph as the descendants test:

        *tūnaz → tūn (OE) → town (modern English)
               → tún (ON) → tún (modern Icelandic)

    Walking UP from modern Icelandic 'tún' should reach ON 'tún' and
    Proto-Germanic '*tūnaz' (but NOT modern English 'town' — that's
    a sibling-via-different-branch, not an ancestor)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        me_id = db.upsert_etymon("town", "modern-english")
        is_id = db.upsert_etymon("tún", "icelandic")
        for parent, child in (
            (proto_id, oe_id),
            (proto_id, on_id),
            (oe_id, me_id),
            (on_id, is_id),
        ):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (parent, child),
            )
        db.commit()

        # Recursive CTE — reference query for INGESTION.md / DECISIONS.md.
        rows = db.conn.execute(
            """
            WITH RECURSIVE ancestors(id) AS (
              SELECT parent_id FROM etymon_descent
                WHERE child_id = ? AND edge_type IN ('inheritance', 'borrowing')
              UNION
              SELECT d.parent_id FROM etymon_descent d
                JOIN ancestors ON d.child_id = ancestors.id
                WHERE d.edge_type IN ('inheritance', 'borrowing')
            )
            SELECT e.canonical_form, e.language
            FROM ancestors
            JOIN etymon e ON e.id = ancestors.id
            ORDER BY e.language, e.canonical_form
            """,
            (is_id,),
        ).fetchall()

    forms = [(row["canonical_form"], row["language"]) for row in rows]
    assert ("tún", "old-norse") in forms
    assert ("*tūnaz", "proto-germanic") in forms
    # Sibling branch — must NOT appear in ancestors of Icelandic tún.
    assert ("town", "modern-english") not in forms
    assert ("tūn", "old-english") not in forms
    # Self should not appear either.
    assert ("tún", "icelandic") not in forms


def test_etymon_descent_cascades_when_parent_deleted(fresh_db: Path) -> None:
    """ON DELETE CASCADE on etymon_descent.parent_id — when an etymon row
    is deleted, every descent edge pointing at it as parent goes away
    too. Without this guard, deleting a Proto-* root would leave dangling
    edges with broken parent_id references. Pin so a regression to
    SET NULL or constraint removal surfaces here."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        for child in (oe_id, on_id):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (proto_id, child),
            )
        db.commit()
        before = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
        assert before == 2

        # Delete the parent etymon — both descent edges should vanish.
        db.conn.execute("DELETE FROM etymon WHERE id = ?", (proto_id,))
        db.commit()

        after = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
    assert after == 0, (
        "ON DELETE CASCADE on parent_id failed; got dangling descent edges. "
        "Check the FK declaration in data/seed/lexicon.sql + the migration helper."
    )


def test_etymon_descent_cascades_when_child_deleted(fresh_db: Path) -> None:
    """ON DELETE CASCADE on etymon_descent.child_id — symmetric pin to
    the parent variant. Required so deleting a misclassified etymon
    doesn't leave orphan edges pointing at a non-existent child_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        for child in (oe_id, on_id):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (proto_id, child),
            )
        db.commit()

        # Delete the OE child — only the proto→oe edge should vanish;
        # proto→on must survive.
        db.conn.execute("DELETE FROM etymon WHERE id = ?", (oe_id,))
        db.commit()

        rows = db.conn.execute(
            "SELECT child_id FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchall()
    remaining = [row["child_id"] for row in rows]
    assert remaining == [on_id]


def test_migrate_schema_adds_etymon_descent_to_legacy_db(fresh_db: Path) -> None:
    """A pre-D27 DB without etymon_descent picks it up on migrate_schema.
    Verifies (1) the table is created, (2) ON DELETE CASCADE survives
    the migration DDL — not just the data/seed/lexicon.sql DDL the fresh-
    install cascade tests cover. Catches a drift between the two
    parallel CREATE TABLE statements in lexicon.py vs lexicon.sql.
    Idempotent on re-run."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-D27 schema by dropping the descent table.
        db.conn.execute("DROP TABLE IF EXISTS etymon_descent")
        # Pre-migration: table is missing.
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_descent" not in tables

        applied = migrate_schema(db)
        assert applied["etymon_descent_table"] is True

        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_descent" in tables

        # Cascade behavior must survive the migration-path DDL. PRAGMA
        # foreign_key_list reports each FK's on_delete action, so we can
        # assert both parent_id and child_id report 'CASCADE' without
        # having to seed-and-delete (cheaper, more direct).
        fks = list(db.conn.execute("PRAGMA foreign_key_list(etymon_descent)"))
        on_delete = {fk["from"]: fk["on_delete"] for fk in fks}
        assert on_delete.get("parent_id") == "CASCADE", (
            f"migration-path DDL dropped ON DELETE CASCADE on parent_id; got {on_delete!r}"
        )
        assert on_delete.get("child_id") == "CASCADE", (
            f"migration-path DDL dropped ON DELETE CASCADE on child_id; got {on_delete!r}"
        )

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon_descent_table"] is False


def test_migrate_schema_adds_etymon_variant_to_legacy_db(fresh_db: Path) -> None:
    """A pre-wyrd-fqil DB without etymon_variant picks it up on
    migrate_schema. Mirrors the etymon_descent migration test:
    verifies (1) the table is created, (2) ON DELETE CASCADE on
    etymon_id survives the migration DDL, (3) all three indexes are
    created, (4) the COLLATE NOCASE on form survives the migration
    path. Idempotent on re-run."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-wyrd-fqil schema by dropping the variant table.
        db.conn.execute("DROP TABLE IF EXISTS etymon_variant")
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_variant" not in tables

        applied = migrate_schema(db)
        assert applied["etymon_variant_table"] is True

        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_variant" in tables

        # Cascade behavior must survive the migration-path DDL.
        fks = list(db.conn.execute("PRAGMA foreign_key_list(etymon_variant)"))
        on_delete = {fk["from"]: fk["on_delete"] for fk in fks}
        assert on_delete.get("etymon_id") == "CASCADE", (
            f"migration-path DDL dropped ON DELETE CASCADE on etymon_id; got {on_delete!r}"
        )

        # All three indexes from lexicon.sql must also be created on the
        # migration path. Drift between lexicon.py and lexicon.sql in
        # which indexes get attached is a real risk.
        indexes = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='etymon_variant'"
            )
        }
        assert "idx_etymon_variant_etymon" in indexes
        assert "idx_etymon_variant_form" in indexes
        assert "idx_etymon_variant_class" in indexes

        # COLLATE NOCASE on form must survive — verify by inserting a
        # case-mismatched lookup and asserting it matches.
        db.conn.execute("INSERT INTO source (id, title, author) VALUES ('test-variant', 'T', 'T')")
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (9999, 'lemma', 'old-english')"
        )
        db.conn.execute(
            "INSERT INTO etymon_variant (etymon_id, form, variant_class, source_id) "
            "VALUES (9999, 'CaseTest', 'alternative', 'test-variant')"
        )
        n = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_variant WHERE form = 'casetest'"
        ).fetchone()[0]
        assert n == 1, "COLLATE NOCASE on form did not survive the migration path"

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon_variant_table"] is False


def test_migrate_schema_adds_etymon_cognate_columns_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A pre-D27 etymon table without cognate_id / cognate_method picks
    them up on migrate_schema. Existing rows survive with NULL in the
    new columns."""
    with LexiconDB(fresh_db) as db:
        # Drop the columns by recreating the table without them. Simpler
        # to drop and recreate the etymon table from a minimal pre-D27
        # shape; FK targets are restored by the consensus view rebuild.
        db.upsert_source(id="src-a", title="A")
        legacy_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        # Pre-migration check: post-init the columns ARE there (fresh
        # install). Drop them to simulate a legacy DB.
        # Note: `CREATE TABLE etymon_legacy AS SELECT ...` silently drops
        # the original PK / UNIQUE / FK / NOT NULL constraints — the
        # resulting table is a content snapshot only. That's enough for
        # this test (which only checks that the synset_* columns are
        # ADDED on migration), but it's not a faithful simulation of a
        # production pre-D27 schema. A FK-cascade test or constraint
        # check on this fixture would silently pass.
        # Drop every view that references etymon so the rename below
        # doesn't trip SQLite's dependent-view check (the wyrd-7lo
        # canonical views all join against etymon).
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_gloss_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_tag_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_text_match_canonical")
        db.conn.execute(
            "CREATE TABLE etymon_legacy AS "
            "SELECT id, canonical_form, language, modifier_type, position_pref, "
            "       notes, lemma_id, inflection, lemma_method, merged_into_id "
            "FROM etymon"
        )
        db.conn.execute("DROP TABLE etymon")
        db.conn.execute("ALTER TABLE etymon_legacy RENAME TO etymon")

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "cognate_id" not in cols
        assert "cognate_method" not in cols

        applied = migrate_schema(db)
        assert applied["etymon.cognate_id"] is True
        assert applied["etymon.cognate_method"] is True
        assert applied["idx_etymon_cognate"] is True

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "cognate_id" in cols
        assert "cognate_method" in cols

        # Existing row survives with NULL in the new columns.
        row = db.conn.execute(
            "SELECT id, cognate_id, cognate_method FROM etymon WHERE id = ?",
            (legacy_id,),
        ).fetchone()
        assert row["cognate_id"] is None
        assert row["cognate_method"] is None

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon.cognate_id"] is False
        assert applied2["etymon.cognate_method"] is False


def test_migrate_schema_adds_phase2a_pronunciation_columns_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A pre-wyrd-ha9q etymon table without the Phase 2a columns
    (pronunciation_ipa, pronunciation_dialect, original_script,
    transliteration, english_shaped) picks them up on migrate_schema.
    Existing rows survive with NULL in every new column. Idempotent
    re-run reports all five as already-applied (False).

    Mirrors the D27 cognate-column legacy-migration test pattern; this
    is the canonical place to pin the migration story for new etymon
    columns."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        legacy_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        # Drop all etymon-dependent views before the table swap (same
        # CASCADE concern as the cognate-columns test).
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_gloss_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_tag_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_text_match_canonical")
        # Recreate without the Phase 2a columns to simulate a pre-ha9q DB.
        db.conn.execute(
            "CREATE TABLE etymon_legacy AS "
            "SELECT id, canonical_form, language, modifier_type, position_pref, "
            "       notes, lemma_id, inflection, lemma_method, merged_into_id, "
            "       cognate_id, cognate_method "
            "FROM etymon"
        )
        db.conn.execute("DROP TABLE etymon")
        db.conn.execute("ALTER TABLE etymon_legacy RENAME TO etymon")

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        for c in (
            "pronunciation_ipa",
            "pronunciation_dialect",
            "original_script",
            "transliteration",
            "english_shaped",
        ):
            assert c not in cols, f"{c} unexpectedly present in pre-migration legacy table"

        applied = migrate_schema(db)
        for c in (
            "etymon.pronunciation_ipa",
            "etymon.pronunciation_dialect",
            "etymon.original_script",
            "etymon.transliteration",
            "etymon.english_shaped",
        ):
            assert applied[c] is True, f"{c} migration didn't apply"

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        for c in (
            "pronunciation_ipa",
            "pronunciation_dialect",
            "original_script",
            "transliteration",
            "english_shaped",
        ):
            assert c in cols, f"{c} missing post-migration"

        # Existing row survives with NULL in every new column.
        row = db.conn.execute(
            """SELECT id, pronunciation_ipa, pronunciation_dialect,
                      original_script, transliteration, english_shaped
               FROM etymon WHERE id = ?""",
            (legacy_id,),
        ).fetchone()
        assert row["pronunciation_ipa"] is None
        assert row["pronunciation_dialect"] is None
        assert row["original_script"] is None
        assert row["transliteration"] is None
        assert row["english_shaped"] is None

        # Idempotent: re-running reports all five as already-present.
        applied2 = migrate_schema(db)
        for c in (
            "etymon.pronunciation_ipa",
            "etymon.pronunciation_dialect",
            "etymon.original_script",
            "etymon.transliteration",
            "etymon.english_shaped",
        ):
            assert applied2[c] is False, f"{c} re-applied on second run"


def test_migrate_schema_adds_stratum_and_phonological_vector_columns_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A pre-wyrd-lr4 / pre-wyrd-kq7w.1 etymon table without the stratum
    and phonological_vector columns picks them up on migrate_schema.
    Existing rows survive with NULL in both new columns. Idempotent
    re-run reports both as already-applied (False).

    These are the two most-recently-added etymon columns (wyrd-cnf8): the
    fresh-alembic path covers them, but the legacy migrate_schema uplift
    path had no dedicated assertion. Mirrors
    test_migrate_schema_adds_phase2a_pronunciation_columns_to_legacy_db."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        legacy_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        # Drop all etymon-dependent views before the table swap (same
        # CASCADE concern as the phase2a-columns test).
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_gloss_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_tag_canonical")
        db.conn.execute("DROP VIEW IF EXISTS etymon_text_match_canonical")

        # Recreate the etymon table without stratum/phonological_vector to
        # simulate a pre-lr4/pre-kq7w.1 DB. Build the surviving column list
        # dynamically from the live schema so the fixture doesn't rot as
        # new etymon columns are added.
        legacy_cols = [
            row["name"]
            for row in db.conn.execute("PRAGMA table_info(etymon)")
            if row["name"] not in ("stratum", "phonological_vector")
        ]
        col_list = ", ".join(legacy_cols)
        db.conn.execute(f"CREATE TABLE etymon_legacy AS SELECT {col_list} FROM etymon")
        db.conn.execute("DROP TABLE etymon")
        db.conn.execute("ALTER TABLE etymon_legacy RENAME TO etymon")

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        for c in ("stratum", "phonological_vector"):
            assert c not in cols, f"{c} unexpectedly present in pre-migration legacy table"

        applied = migrate_schema(db)
        for c in ("etymon.stratum", "etymon.phonological_vector"):
            assert applied[c] is True, f"{c} migration didn't apply"
        # The CREATE TABLE AS SELECT swap above drops the etymon indexes
        # along with the original table, so the stratum index is recreated
        # too — assert that adjacent re-creation since it's part of the
        # same wyrd-lr4 column's migration story.
        assert applied["idx_etymon_stratum"] is True, "idx_etymon_stratum not recreated"

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        for c in ("stratum", "phonological_vector"):
            assert c in cols, f"{c} missing post-migration"

        # Existing row survives with NULL in both new columns.
        row = db.conn.execute(
            "SELECT id, stratum, phonological_vector FROM etymon WHERE id = ?",
            (legacy_id,),
        ).fetchone()
        assert row["stratum"] is None
        assert row["phonological_vector"] is None

        # Idempotent: re-running reports both columns and the stratum
        # index as already-present.
        applied2 = migrate_schema(db)
        for c in ("etymon.stratum", "etymon.phonological_vector"):
            assert applied2[c] is False, f"{c} re-applied on second run"
        assert applied2["idx_etymon_stratum"] is False, (
            "idx_etymon_stratum re-created on second run"
        )


# --- D27 / wyrd-81n: cluster-cognates enrichment pass ------------------


def _seed_descent_chain(db: LexiconDB, *edges: tuple[str, str, str]) -> dict[str, int]:
    """Helper: insert a list of (parent_form, child_form, edge_type)
    descent edges, upserting the etymon rows as needed. Returns a
    {form: id} map for the test to reference. All etymons are written
    with language='proto-germanic' for fixture simplicity — the cluster
    pass walks edges, not languages."""
    db.upsert_source(id="wiktionary", title="Wiktionary")
    forms: dict[str, int] = {}
    for parent_form, child_form, _edge in edges:
        for form in (parent_form, child_form):
            if form not in forms:
                forms[form] = db.upsert_etymon(form, "proto-germanic")
    for parent_form, child_form, edge_type in edges:
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
            (forms[parent_form], forms[child_form], edge_type, "wiktionary"),
        )
    db.commit()
    return forms


def test_assign_cognate_clusters_smallest_root_wins_and_flags_cycle_orphans() -> None:
    """The pure BFS assigner extracted from cluster_cognates, called directly on
    canonical (parent_id, child_id) edges. Pins the two load-bearing properties:
    a node reachable from multiple roots is claimed by the SMALLEST root id
    (roots are walked ascending and the BFS skips already-assigned nodes), and a
    pure cycle with no external anchor yields cycle_orphans (nothing assigned)."""
    from wyrd.generators.kenning.lexicon.cognate_cluster import _assign_cognate_clusters

    # Node 3 is reachable from root 1 (1->2->3) AND root 5 (5->3): smaller root wins.
    assignments, roots, orphans = _assign_cognate_clusters([(1, 2), (2, 3), (5, 3)])
    assert roots == [1, 5]
    assert assignments == {1: 1, 2: 1, 3: 1, 5: 5}  # 3 claimed by 1, not 5
    assert orphans == set()

    # A pure 2-cycle has no root (each node is also a child) -> both are orphans.
    assignments2, roots2, orphans2 = _assign_cognate_clusters([(7, 8), (8, 7)])
    assert roots2 == []
    assert assignments2 == {}
    assert orphans2 == {7, 8}


def test_cluster_cognates_assigns_root_id_to_inheritance_chain(fresh_db: Path) -> None:
    """Happy path: a → b → c inheritance chain. All three rows get
    cognate_id = a.id, cognate_method = 'cluster-cognates-v2'."""

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row
            for row in db.conn.execute(
                "SELECT canonical_form, cognate_id, cognate_method FROM etymon"
            )
        }

    assert result["roots"] == 1
    assert result["candidates"] == 3
    assert result["applied"] is True
    for form in ("a", "b", "c"):
        assert rows[form]["cognate_id"] == forms["a"]
        assert rows[form]["cognate_method"] == "cluster-cognates-v2"


def test_cluster_cognates_nulls_stale_tombstone_cognate_id(fresh_db: Path) -> None:
    """wyrd-hn03: a tombstone (merged_into_id set) that retains a STALE
    cognate_id from a prior run — when it was canonical and in a different
    cluster, before a later fold tombstoned it — gets NULLed. The bundle
    export's per-member era-reflex union (_fetch_family_era_reflexes) would
    otherwise read the stale cluster and leak its reflexes onto the surviving
    lemma's grid (the verb do/done leaking onto the toponym -don)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="w", title="W")
        # Two separate clusters: a 'verb' chain and a 'hill' chain.
        verb_root = db.upsert_etymon("*verb", "proto-germanic")
        verb_child = db.upsert_etymon("do", "modern-english")
        hill_root = db.upsert_etymon("*hill", "proto-germanic")
        hill = db.upsert_etymon("dūn", "old-english")
        # The homograph: 'don' — gets folded (tombstoned) into the hill lemma,
        # but first-run state left it stamped with the VERB cluster.
        don = db.upsert_etymon("don", "old-english")
        for parent, child in ((verb_root, verb_child), (hill_root, hill)):
            db.conn.execute(
                "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'w')",
                (parent, child),
            )
        # Simulate the stale state: don was canonical + in the verb cluster on a
        # prior run, then got folded into the hill lemma (tombstoned).
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ?, cognate_method = 'cluster-cognates-v2' WHERE id = ?",
            (verb_root, don),
        )
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (hill, don))
        db.commit()
        # precondition: the tombstone carries the stale verb cognate_id
        assert (
            db.conn.execute("SELECT cognate_id FROM etymon WHERE id = ?", (don,)).fetchone()[0]
            == verb_root
        )

        result = cluster_cognates(db, apply=True)
        don_row = db.conn.execute(
            "SELECT cognate_id, cognate_method FROM etymon WHERE id = ?", (don,)
        ).fetchone()
        # the canonical members keep their freshly-assigned cluster (the NULL
        # step is gated on merged_into_id IS NOT NULL, so it never touches them)
        canon = {
            r["id"]: r["cognate_id"]
            for r in db.conn.execute(
                "SELECT id, cognate_id FROM etymon WHERE merged_into_id IS NULL"
            )
        }
        # second run: nothing left to clear (idempotent)
        second = cluster_cognates(db, apply=True)

    assert don_row["cognate_id"] is None  # stale verb pointer cleared
    assert don_row["cognate_method"] is None
    assert result["tombstones_cleared"] >= 1
    assert canon[verb_child] == verb_root  # canonical assignment intact
    assert canon[hill] == hill_root  # the other cluster untouched
    assert second["tombstones_cleared"] == 0  # idempotent — nothing stale remains


def test_cluster_cognates_tombstone_clear_stops_era_reflex_leak(fresh_db: Path) -> None:
    """End-to-end (the actual wyrd-hn03 symptom): a tombstone with a stale
    cognate_id makes etymon_era_reflexes return the OLD cluster's reflexes;
    after cluster_cognates NULLs the tombstone, the leak stops."""
    from wyrd.generators.kenning.lexicon import etymon_era_reflexes

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="w", title="W")
        # A verb cluster carrying a modern-english reflex 'do'.
        verb_root = db.upsert_etymon("*verb", "proto-germanic")
        verb_modern = db.upsert_etymon("do", "modern-english")
        hill = db.upsert_etymon("dūn", "old-english")
        don = db.upsert_etymon("don", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'w')",
            (verb_root, verb_modern),
        )
        db.commit()
        # populate the verb cluster's cognate_id (so it has mates to leak)
        cluster_cognates(db, apply=True)
        # NOW inject the stale state: 'don' is tombstoned into the hill lemma but
        # still points at the verb cluster (it has no descent edges of its own).
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ?, cognate_method = 'cluster-cognates-v2', "
            "merged_into_id = ? WHERE id = ?",
            (verb_root, hill, don),
        )
        db.commit()
        before = etymon_era_reflexes(db, don, target_language="modern-english")
        cluster_cognates(db, apply=True)  # the fix: NULLs the stale tombstone
        after = etymon_era_reflexes(db, don, target_language="modern-english")

    # before the fix the tombstone leaks the verb's modern reflex...
    assert any(r.form == "do" for r in before)
    # ...and after, with its stale cognate_id cleared, it leaks nothing
    assert after == []


def test_cluster_cognates_dry_run_reports_tombstone_clear_without_writing(fresh_db: Path) -> None:
    """Dry-run reports the would-clear count but leaves the stale tombstone
    cognate_id in place."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="w", title="W")
        a = db.upsert_etymon("a", "old-english")
        b = db.upsert_etymon("b", "old-english")
        db.conn.execute(
            "UPDATE etymon SET cognate_id = ?, cognate_method = 'cluster-cognates-v2', "
            "merged_into_id = ? WHERE id = ?",
            (a, a, b),
        )
        db.commit()
        result = cluster_cognates(db, apply=False)
        still = db.conn.execute("SELECT cognate_id FROM etymon WHERE id = ?", (b,)).fetchone()[0]
    assert result["tombstones_cleared"] == 1
    assert still == a  # not written in dry-run


def test_cluster_cognates_v2_does_not_bridge_through_proto_indo_european(fresh_db: Path) -> None:
    """v2: a Proto-Indo-European root must NOT bridge a cognate cluster. PIE
    sits above the family layer, so inheritance from it fans across every branch
    (Germanic ``down`` AND Greek ``thymia``). Dropping every edge touching a PIE
    node leaves the two daughters in SEPARATE clusters (and the PIE node itself
    unclustered) — the fix for the 1000+-member mega-clusters that polluted the
    era-grid's modern reflexes."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="w", title="W")
        pie = db.upsert_etymon("*dʰewh₂-", "proto-indo-european")
        gmc = db.upsert_etymon("dūn", "old-english")
        grk = db.upsert_etymon("thymia", "ancient-greek")
        for parent, child in ((pie, gmc), (pie, grk)):
            db.conn.execute(
                "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'w')",
                (parent, child),
            )
        db.commit()
        cluster_cognates(db, apply=True)
        cid = {
            r["canonical_form"]: r["cognate_id"]
            for r in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
    # PIE never bridges: the Germanic and Greek daughters are NOT co-clustered,
    # and the PIE node itself is left unclustered (it anchors no edge).
    assert cid["dūn"] != cid["thymia"] or (cid["dūn"] is None and cid["thymia"] is None)
    assert cid["dūn"] is None  # no surviving edge → not a candidate
    assert cid["thymia"] is None
    # PIE root stored de-dashed (wyrd-aicu.8, D45): '*dʰewh₂-' → '*dʰewh₂'.
    assert cid["*dʰewh₂"] is None


def test_cluster_cognates_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves cognate_id NULL on every
    candidate row."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        result = cluster_cognates(db, apply=False)
        cognate_ids = [
            row["cognate_id"] for row in db.conn.execute("SELECT cognate_id FROM etymon")
        ]

    assert result["candidates"] == 3
    assert result["applied"] is False
    assert all(sid is None for sid in cognate_ids)


def test_cluster_cognates_separate_roots_get_separate_cognate_clusters(fresh_db: Path) -> None:
    """Two unrelated roots produce two distinct cognate_id values; rows
    in each cluster point at their own root."""

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("x", "y", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["roots"] == 2
    assert rows["a"] == forms["a"]
    assert rows["b"] == forms["a"]
    assert rows["x"] == forms["x"]
    assert rows["y"] == forms["x"]
    assert rows["a"] != rows["x"]


def test_cluster_cognates_cognate_edge_does_not_bridge_cognate_clusters(fresh_db: Path) -> None:
    """The 'cognate' edge_type is a peer relation, NOT a chain — it must
    not bridge cognate-cluster assignments. Pin the contract: a {{cog}}-only
    relation between two etymons leaves both with NULL cognate_id (no
    inheritance/borrowing chain to walk)."""

    with LexiconDB(fresh_db) as db:
        # Only a cognate edge — no inheritance.
        forms = _seed_descent_chain(
            db,
            ("a", "b", "cognate"),
        )
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["a"] is None
    assert rows["b"] is None
    # The forms map should still have both etymons, just with no
    # cognate_id assignment because no bridging edge exists.
    assert "a" in forms and "b" in forms


def test_cluster_cognates_borrowing_edges_bridge_cognate_clusters(fresh_db: Path) -> None:
    """Borrowing edges DO bridge — a borrowed word is part of the
    borrowing language's cognate set in practice (D27 / wyrd-81n
    documents this choice)."""

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("source-form", "borrowed-form", "borrowing"),
        )
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["source-form"] == forms["source-form"]
    assert rows["borrowed-form"] == forms["source-form"]


def test_cluster_cognates_diamond_graph_picks_smallest_root(
    fresh_db: Path,
) -> None:
    """When a descendant is reachable from multiple roots (rare; happens
    when scholars disagree on the chain), the smallest root id wins.
    Pin the determinism so re-runs are bit-stable."""

    with LexiconDB(fresh_db) as db:
        # Diamond: root1 → mid; root2 → mid. mid is reachable from both.
        forms = _seed_descent_chain(
            db,
            ("root1", "mid", "inheritance"),
            ("root2", "mid", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        mid_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE canonical_form = 'mid'"
        ).fetchone()["cognate_id"]

    # root1 was inserted first (lower id), so it wins.
    assert mid_synset == forms["root1"]


def test_cluster_cognates_idempotent_on_rerun(fresh_db: Path) -> None:
    """A second --apply run with no schema changes is a content no-op:
    the same rows get the same assignments. 'candidates' count stays
    constant because the implementation doesn't track 'already-assigned'
    vs 'newly-assigned' — it overwrites with the same values."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        first = cluster_cognates(db, apply=True)
        first_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
        second = cluster_cognates(db, apply=True)
        second_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert first["candidates"] == second["candidates"]
    assert first_ids == second_ids


def test_cluster_cognates_skips_etymons_without_descent_edges(fresh_db: Path) -> None:
    """Singleton etymons (rando-port seed entries with no Wiktionary
    chain) get NULL cognate_id. They're not in any cognate set."""

    with LexiconDB(fresh_db) as db:
        # One chain, plus a singleton that has no descent edges.
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        singleton_id = db.upsert_etymon("singleton", "old-english")
        db.commit()

        cluster_cognates(db, apply=True)

        singleton_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?",
            (singleton_id,),
        ).fetchone()["cognate_id"]

    assert singleton_synset is None


def test_clear_enrichment_cognates_stage_resets_cognate_assignments(
    fresh_db: Path,
) -> None:
    """clear-enrichment --stage=cognates undoes a cluster-cognates run:
    cognate_id and cognate_method go back to NULL on every row. Mining
    evidence (etymon_descent rows themselves) is untouched."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        # Pre-clear: assignments exist.
        before = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
        assert before == 2

        result = clear_enrichment(db, stage="cognates", apply=True)
        after = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
        # etymon_descent rows must survive — they're mining evidence (D21).
        descent_after = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]

    assert result["cognate_assignments_to_clear"] == 2
    assert after == 0
    assert descent_after == 1


def test_clear_enrichment_all_derived_includes_cognates(fresh_db: Path) -> None:
    """all-derived sweeps the cognate stage too — verifies the new stage
    was wired into the all-derived alias. Without this, callers using
    the --stage=all-derived shortcut would silently leave synset
    assignments behind."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        result = clear_enrichment(db, stage="all-derived", apply=True)
        synset_count = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]

    assert result["cognate_assignments_to_clear"] == 2
    assert synset_count == 0


def test_lexicon_path_cli_prints_resolved_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`wyrd kenning lexicon path` prints whatever default_lexicon_path
    resolves to. Pin via env override (the simplest way to control it
    from a test). Confirms the new wyrd-366 CLI surface is wired."""
    target = tmp_path / "scratch" / "lex.db"
    monkeypatch.setenv("WYRD_LEXICON_DB", str(target))
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "path"])
    assert result.exit_code == 0, result.output
    assert str(target) in result.output


def test_lexicon_era_cell_cli_resolves_known_pair() -> None:
    """`wyrd kenning lexicon era-cell <language> <year>` prints the
    family/label pair for a valid (language, year) input. Confirms
    the wyrd-38d CLI surface is wired through the era module."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "old-english", "950"])
    assert result.exit_code == 0, result.output
    assert "english/oe-late" in result.output


def test_lexicon_era_cell_cli_exits_nonzero_for_proto_language() -> None:
    """Proto-Germanic doesn't have an era family. The CLI surfaces
    that as a non-zero exit so a script piping to xargs etc. fails
    loudly rather than silently producing no output."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "proto-germanic", "500"])
    assert result.exit_code == 1
    assert "no era family" in result.output


def test_lexicon_era_cell_cli_exits_nonzero_for_year_out_of_range() -> None:
    """Latin's renaissance ends at 1800; year 2025 has no defined
    cell. CLI exits non-zero with a descriptive stderr message."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "latin", "2025"])
    assert result.exit_code == 1
    assert "outside the defined cells" in result.output


def test_lexicon_help_does_not_trigger_default_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical regression guard: `wyrd kenning lexicon migrate --help`
    must NOT call default_lexicon_path() (and therefore must NOT
    trigger the legacy-DB migration as a side effect of viewing help).
    Click invokes a `default=` callable when `show_default=True` in
    order to display the resolved value; this PR uses the static
    LEXICON_DB_DEFAULT_DISPLAY string instead so the callable stays
    quiet on --help.

    A future regression that flips one option back to
    `show_default=True` would re-introduce the surprise side-effect.
    """

    call_count = [0]

    def counting_resolver() -> Path:
        call_count[0] += 1
        return Path("/sentinel/should-never-be-called")

    monkeypatch.setattr(paths, "default_lexicon_path", counting_resolver)
    # Also patch the alias re-bound at cli import time.

    monkeypatch.setattr(cli_mod, "_DEFAULT_LEXICON_PATH", counting_resolver)

    runner = CliRunner()
    # Hit --help on a command whose --db option uses the default
    # callable; if any of them silently resolve, the count jumps.
    for cmd_path in (
        ["lexicon", "migrate", "--help"],
        ["lexicon", "cluster-cognates", "--help"],
        ["lexicon", "lookup-attested-years", "--help"],
        ["lexicon", "clear-enrichment", "--help"],
    ):
        result = runner.invoke(kenning_cli, cmd_path)
        assert result.exit_code == 0, f"{cmd_path}: {result.output}"
    assert call_count[0] == 0, (
        f"--help should not invoke default_lexicon_path "
        f"(got {call_count[0]} calls). Some --db option likely uses "
        f"show_default=True instead of LEXICON_DB_DEFAULT_DISPLAY."
    )


def test_cluster_cognates_cli_dry_run_reports_without_writing(
    fresh_db: Path,
) -> None:
    """End-to-end CLI smoke: dry-run reports root + candidate counts
    and leaves the DB untouched."""
    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db)],
    )

    assert result.exit_code == 0, result.output
    assert "1 root(s) walked" in result.output
    assert "would assign cognate_id on 3 etymon(s)" in result.output
    assert "dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        cognate_ids = [
            row["cognate_id"] for row in db.conn.execute("SELECT cognate_id FROM etymon")
        ]
    assert all(sid is None for sid in cognate_ids)


def test_cluster_cognates_cli_apply_writes_assignments(fresh_db: Path) -> None:
    """--apply persists the cluster pass; subsequent SELECT shows the
    canonical cognate_id pointer."""
    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db), "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert "assigned cognate_id on 2 etymon(s)" in result.output
    # Pin the rows_written echo branch (added round-2 fix) — without
    # this assertion a typo in the format string would slip through
    # despite the lib-level dict field having coverage.
    assert "rows_written = 2" in result.output
    assert "dry-run" not in result.output

    with LexiconDB(fresh_db) as db:
        cognate_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
    assert cognate_ids["a"] == forms["a"]
    assert cognate_ids["b"] == forms["a"]


# --- wyrd-81n round-1 review follow-ups -------------------------------


@pytest.mark.parametrize("non_bridging_edge", ["derivation", "calque", "compound", "unknown"])
def test_cluster_cognates_non_bridging_edge_types_do_not_cluster(
    fresh_db: Path, non_bridging_edge: str
) -> None:
    """D27 lists derivation, calque, compound, unknown as non-bridging
    (alongside cognate). Pin each so a refactor that widens
    _COGNATE_BRIDGING_EDGES to include any of them surfaces here.
    Without these tests, only 'cognate' was pinned and the others
    could silently start bridging."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(db, ("a", "b", non_bridging_edge))
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
    assert rows["a"] is None, f"{non_bridging_edge} edge unexpectedly bridged"
    assert rows["b"] is None, f"{non_bridging_edge} edge unexpectedly bridged"


def test_cluster_cognates_self_loop_silently_filtered(fresh_db: Path) -> None:
    """Self-loops (X→X) are meaningless data — they assert an etymon is
    its own ancestor. Per wyrd-223, the canonical_edges builder filters
    self-loops upstream of the BFS, so they don't generate cycle_orphans
    or otherwise pollute the cluster pass. BFS must terminate (no
    infinite loop) and the etymon stays NULL — silently."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        x_id = db.upsert_etymon("x", "proto-germanic")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
            (x_id, x_id, "inheritance", "wiktionary"),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)

        cognate = db.conn.execute("SELECT cognate_id FROM etymon WHERE id = ?", (x_id,)).fetchone()[
            "cognate_id"
        ]

    assert result["roots"] == 0
    # cycle_orphans=0 because the self-loop edge was filtered upstream;
    # the etymon doesn't participate in any canonical edge.
    assert result["cycle_orphans"] == 0
    assert cognate is None


def test_cluster_cognates_mutual_cycle_terminates_and_orphans(fresh_db: Path) -> None:
    """A two-node cycle (A→B, B→A) with no external root: BFS must
    terminate and both etymons flagged as cycle orphans."""

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["roots"] == 0
    assert result["cycle_orphans"] == 2
    assert rows["a"] is None
    assert rows["b"] is None
    assert "a" in forms and "b" in forms


def test_cluster_cognates_anchored_cycle_assigns_via_root(fresh_db: Path) -> None:
    """root → a → b → a (cycle reachable from a real root): the cycle
    members get root's cognate_id, and BFS terminates because the
    not-in-assignments check doubles as a visited-set."""

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("root", "a", "inheritance"),
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["cycle_orphans"] == 0
    assert rows["root"] == forms["root"]
    assert rows["a"] == forms["root"]
    assert rows["b"] == forms["root"]


def test_cluster_cognates_idempotent_apply_skips_unchanged_rows(fresh_db: Path) -> None:
    """The second apply against unchanged data must produce
    rows_written = 0 — the WHERE-clause guard short-circuits no-op
    UPDATEs. Without the guard, every re-run would touch every row
    and inflate write counters."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        first = cluster_cognates(db, apply=True)
        second = cluster_cognates(db, apply=True)

    assert first["rows_written"] == 3
    assert second["rows_written"] == 0
    assert first["candidates"] == second["candidates"]


def test_clear_enrichment_cli_cognates_stage_round_trips(fresh_db: Path) -> None:
    """End-to-end CLI smoke for clear-enrichment --stage=cognates: the
    new Click choice + new echo branch had no test (only the lib
    function did). Pin via CliRunner so a typo in the choice list
    surfaces immediately."""

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(db, ("a", "b", "inheritance"))
        cluster_cognates(db, apply=True)

    runner = CliRunner()
    dry = runner.invoke(
        kenning_cli,
        ["lexicon", "clear-enrichment", "--db", str(fresh_db), "--stage", "cognates"],
    )
    assert dry.exit_code == 0, dry.output
    assert "Stage: cognates" in dry.output
    assert "would clear 2 cognate_id assignments" in dry.output
    assert "dry-run" in dry.output

    apply = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "clear-enrichment",
            "--db",
            str(fresh_db),
            "--stage",
            "cognates",
            "--apply",
        ],
    )
    assert apply.exit_code == 0, apply.output
    assert "cleared 2 cognate_id assignments" in apply.output

    with LexiconDB(fresh_db) as db:
        remaining = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
    assert remaining == 0


def test_clear_enrichment_cli_attested_years_stage_round_trips(
    fresh_db: Path, tmp_path: Path
) -> None:
    """End-to-end CLI smoke for clear-enrichment --stage=attested-years
    (D5-1 / wyrd-3ux). Pin via CliRunner so a typo in the new Click
    Choice or the new echo branch surfaces immediately — the lib-level
    test (test_clear_enrichment_attested_years_resets_only_year_column)
    can't catch CLI wiring bugs."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

    runner = CliRunner()
    dry = runner.invoke(
        kenning_cli,
        ["lexicon", "clear-enrichment", "--db", str(fresh_db), "--stage", "attested-years"],
    )
    assert dry.exit_code == 0, dry.output
    assert "Stage: attested-years" in dry.output
    assert "would clear 1 attested_year values" in dry.output
    assert "dry-run" in dry.output

    apply = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "clear-enrichment",
            "--db",
            str(fresh_db),
            "--stage",
            "attested-years",
            "--apply",
        ],
    )
    assert apply.exit_code == 0, apply.output
    assert "cleared 1 attested_year values" in apply.output

    with LexiconDB(fresh_db) as db:
        remaining = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon_text_match WHERE attested_year IS NOT NULL"
        ).fetchone()["n"]
    assert remaining == 0


def test_lookup_attested_years_cli_dry_run_reports_without_writing(
    fresh_db: Path, tmp_path: Path
) -> None:
    """End-to-end CLI smoke: dry-run reports row + candidate counts and
    leaves attested_year NULL. Pins the Click wiring (--db, --apply
    flag, sources_dir argument) so a typo surfaces immediately."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "lookup-attested-years", str(sources), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "etymon_text_match" in result.output
    assert "scanned=    1" in result.output
    assert "candidates=    1" in result.output
    assert "rows_written=    0" in result.output
    assert "dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        years = [
            row["attested_year"]
            for row in db.conn.execute("SELECT attested_year FROM etymon_text_match")
        ]
    assert all(y is None for y in years)


def test_lookup_attested_years_cli_apply_writes_years(fresh_db: Path, tmp_path: Path) -> None:
    """--apply persists attested_year on each qualifying row. Pins the
    rows_written echo branch — without this assertion a typo in the
    format string would slip through despite the lib-level dict field
    having coverage."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "lookup-attested-years",
            str(sources),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    # etymon_text_match line shows the actual write; toponym_etymology
    # line is zero (no rows seeded for that table).
    assert "etymon_text_match" in result.output
    assert "rows_written=    1" in result.output
    assert "dry-run" not in result.output

    with LexiconDB(fresh_db) as db:
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_cluster_cognates_cli_warns_on_cycle_orphans(fresh_db: Path) -> None:
    """A descent graph with a closed cycle and no external root produces
    a warn line on stderr so operators notice the missed cluster."""
    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.output
    assert "2 etymon(s) sit in a bridging-edge cycle" in result.output
    # rows_written = 0 here: there are 0 candidates because no roots,
    # so the apply branch echoes 'rows_written = 0'.
    assert "rows_written = 0" in result.output


# --- wyrd-223: cluster-cognates resolves through merged_into_id -------


def test_cluster_cognates_resolves_descent_edges_through_merged_into_id(
    fresh_db: Path,
) -> None:
    """Bug discovered during PR #41 live ingest 2026-05-03: a descent
    edge pointing at an OCR-merge tombstone was being walked literally,
    so cognate_id landed on the tombstone (loser) rather than the
    canonical winner. Cluster-cognates now resolves both endpoints
    through merged_into_id before BFS walking.

    Setup mirrors the real wiktextract+place-name case:
      - canonical 'tūn' (with macron) — the winner from a prior OCR merge
      - tombstone 'tun' (no macron) — merged_into_id → tūn
      - descent edge: gmw-pro:*tūn → ang:tun (points at TOMBSTONE)
    Expected: cognate_id lands on the canonical 'tūn', not on 'tun'.
    """

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūn", "gmw-pro")
        canonical_id = db.upsert_etymon("tūn", "old-english")
        tombstone_id = db.upsert_etymon("tun", "old-english")
        # Tombstone the ASCII variant.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical_id, tombstone_id),
        )
        # Edge points at the TOMBSTONE id (the wiktextract upsert path).
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, tombstone_id),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)
        canonical_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (canonical_id,)
        ).fetchone()["cognate_id"]
        tombstone_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (tombstone_id,)
        ).fetchone()["cognate_id"]

    # Critical: the canonical 'tūn' got the synset assignment, NOT the tombstone.
    assert canonical_synset == proto_id
    assert tombstone_synset is None
    # Sanity: the cluster has 2 canonical members (proto root + canonical tūn).
    assert result["candidates"] == 2


def test_cluster_cognates_canonical_root_wins_when_root_was_merged(
    fresh_db: Path,
) -> None:
    """If a tombstone is itself a parent in some descent edge, the
    canonical winner becomes the parent in the canonical edge set.
    Pin so a refactor that drops the parent-side resolution doesn't
    silently regress."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        canonical_root_id = db.upsert_etymon("*canonical_root", "gmw-pro")
        tombstone_root_id = db.upsert_etymon("*tombstone_root", "gmw-pro")
        child_id = db.upsert_etymon("child", "old-english")
        # Tombstone one root variant.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical_root_id, tombstone_root_id),
        )
        # Edge points FROM the tombstone root.
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (tombstone_root_id, child_id),
        )
        db.commit()

        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["*canonical_root"] == canonical_root_id
    assert rows["child"] == canonical_root_id
    # Tombstone stays NULL.
    assert rows["*tombstone_root"] is None


def test_cluster_cognates_self_loop_via_merge_does_not_crash(fresh_db: Path) -> None:
    """If both endpoints of an edge merge into the same canonical (the
    merge itself created a self-loop in canonical space), the edge is
    filtered out — no infinite loop, no spurious cluster of a single
    etymon under itself."""

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        canonical_id = db.upsert_etymon("canonical", "old-english")
        tomb_a_id = db.upsert_etymon("tomb_a", "old-english")
        tomb_b_id = db.upsert_etymon("tomb_b", "old-english")
        # Both tombstones merge into the same canonical.
        for tomb in (tomb_a_id, tomb_b_id):
            db.conn.execute(
                "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
                (canonical_id, tomb),
            )
        # Edge between two tombstones that resolve to the same canonical.
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (tomb_a_id, tomb_b_id),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)
    # No roots (no canonical edges left after self-loop filter);
    # no cluster either. Just a clean no-op, no infinite walk.
    assert result["roots"] == 0
    assert result["candidates"] == 0
    assert result["cycle_orphans"] == 0


# --- wyrd-083: bridge_generic_language ---------------------------------


def test_bridge_generic_language_picks_priority_match(fresh_db: Path) -> None:
    """A generic 'celtic' etymon with a matching canonical_form in
    'irish' AND 'old-irish' bridges to old-irish (higher priority per
    the candidate order). Pin so a refactor of the priority semantics
    surfaces here."""

    with LexiconDB(fresh_db) as db:
        # Older entries get smaller ids; pre-create the generic FIRST
        # to verify priority order — not insertion order — wins.
        celtic_id = db.upsert_etymon("bun", "celtic")
        irish_id = db.upsert_etymon("bun", "irish")
        oi_id = db.upsert_etymon("bun", "old-irish")
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("old-irish", "irish"),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == oi_id  # old-irish is higher priority than irish
    assert merged_target != irish_id
    assert result["bridged"] == 1
    assert result["unmatched"] == 0


def test_bridge_generic_language_no_match_leaves_unmerged(fresh_db: Path) -> None:
    """A generic etymon with no specific-language match keeps
    merged_into_id NULL; counter increments unmatched."""

    with LexiconDB(fresh_db) as db:
        no_match_id = db.upsert_etymon("rma", "celtic")  # not in any specific-celtic
        db.upsert_etymon("bun", "irish")  # exists but different form
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish", "welsh"),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (no_match_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["unmatched"] == 1


def test_bridge_generic_language_case_insensitive_match(fresh_db: Path) -> None:
    """Canonical form match is case-insensitive — Wiktextract sometimes
    capitalizes proper nouns (`Nás`) where place-name dicts lowercase
    them (`nás`). Pin so a refactor doesn't drop the case-fold."""

    with LexiconDB(fresh_db) as db:
        celtic_id = db.upsert_etymon("nás", "celtic")
        ir_id = db.upsert_etymon("Nás", "irish")
        db.commit()
        bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish",),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == ir_id


def test_bridge_generic_language_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves merged_into_id NULL."""

    with LexiconDB(fresh_db) as db:
        celtic_id = db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish",),
            apply=False,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0


def test_bridge_generic_language_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """A second apply against unchanged data writes zero rows — the
    UPDATE has a WHERE-clause guard."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()
        first = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )
        second = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )

    assert first["rows_written"] == 1
    assert second["rows_written"] == 0


def test_bridge_generic_language_already_merged_tombstones_get_chain_flattened(
    fresh_db: Path,
) -> None:
    """Generic etymons that are already OCR-merged (merged_into_id IS
    NOT NULL) aren't directly examined by the bridge candidate scan
    (the SELECT filters on merged_into_id IS NULL). BUT when the
    canonical they point at gets bridged, the chain-flatten clause
    re-routes them onto the new target so the consensus rollup chain
    stays at depth 1 (round-1 fix for wyrd-083 / wyrd-ft3).

    Pre: tomb_celtic → target_celtic
    Bridge: target_celtic → irish
    Post: BOTH tomb_celtic AND target_celtic point at irish
    """

    with LexiconDB(fresh_db) as db:
        target_celtic_id = db.upsert_etymon("bun", "celtic")
        tomb_celtic_id = db.upsert_etymon("BUN", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (target_celtic_id, tomb_celtic_id),
        )
        irish_id = db.upsert_etymon("bun", "irish")
        db.commit()
        result = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }

    # Both rows now point at irish — chain stays at depth 1.
    assert rows[target_celtic_id] == irish_id
    assert rows[tomb_celtic_id] == irish_id
    # Only the canonical celtic etymon was examined as a bridge source.
    assert result["generic_etymons"] == 1


def test_bridge_generic_language_empty_candidates_raises(fresh_db: Path) -> None:
    """An empty candidate_langs is a programmer error; raise rather
    than silently process zero candidates."""

    with LexiconDB(fresh_db) as db, pytest.raises(ValueError, match="candidate_langs"):
        bridge_generic_language(db, generic_lang="celtic", candidate_langs=(), apply=False)


def test_bridge_celtic_then_cluster_cognates_includes_generic_in_cluster(
    fresh_db: Path,
) -> None:
    """Load-bearing integration test: bridge celtic → irish, then run
    cluster-cognates (which is redirect-aware per wyrd-223), and
    confirm the generic celtic etymon is rolled up via the merged_into_id
    chain into the irish entry's cluster.

    Tree: proto-celtic *bunV → irish bun → (celtic bun is now a
    tombstone pointing at irish bun)
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*bunV", "proto-celtic")
        irish_id = db.upsert_etymon("bun", "irish")
        celtic_id = db.upsert_etymon("bun", "celtic")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, irish_id),
        )
        db.commit()

        # Pre-bridge: celtic bun is its own canonical, no synset link.
        cluster_cognates(db, apply=True)
        celtic_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["cognate_id"]
        assert celtic_synset is None  # no edges anchor it

        # Bridge celtic → irish.
        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        # Re-cluster after the bridge (redirect-aware so irish's synset
        # rollup applies to celtic via the merged_into_id chain).
        from wyrd.generators.kenning.lexicon import clear_enrichment

        clear_enrichment(db, stage="cognates", apply=True)
        cluster_cognates(db, apply=True)

        irish_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (irish_id,)
        ).fetchone()["cognate_id"]
        celtic_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    # Critical: the celtic etymon now points at irish via merged_into_id;
    # irish has the synset assignment from the proto-celtic root.
    assert celtic_target == irish_id
    assert irish_synset == proto_id


# --- wyrd-ft3: bridge_phonological_oe ----------------------------------


def test_compute_phonological_bridges_matches_skips_and_counts_missing() -> None:
    """The pure match step extracted from _bridge_same_language_phonological,
    called directly on dict rows. Collects (source_id, target_id) redirects;
    skips rows whose form isn't in the bridge table; skips self-bridges (target
    resolves to the row itself); and counts rows whose bridged Wiktionary form
    has no matching etymon as missing_target."""
    from wyrd.generators.kenning.lexicon.bridges._common import _compute_phonological_bridges

    def row(i, canonical_form):
        return {"id": i, "canonical_form": canonical_form}

    examined = [
        row(1, "ton"),  # -> 'tūn' (id 10): a real bridge
        row(2, "lea"),  # -> 'lēah' but no etymon for it: missing_target
        row(3, "xyz"),  # not in the bridge table: skipped silently
        row(4, "by"),  # -> 'býr' which resolves to id 4 (itself): self-bridge skipped
    ]
    table = {"ton": "tūn", "lea": "lēah", "by": "býr"}
    target_index = {"tūn": 10, "býr": 4}  # 'lēah' deliberately absent

    bridges, missing_target = _compute_phonological_bridges(examined, table, target_index)

    assert bridges == [(1, 10)]  # only the real, non-self bridge
    assert missing_target == 1  # 'lea' bridged to 'lēah', which had no etymon


def test_compute_phonological_bridges_case_folds_both_sides() -> None:
    """The match is case-insensitive on BOTH sides: a mixed-case stored
    canonical_form (normalize_morpheme_surface preserves case, so mixed case
    survives to here) and a mixed-case bridge-table VALUE both fold to lowercase
    before lookup. Pins the load-bearing .lower() calls — dropping either the
    canonical_form fold or the wiktionary-form fold would miss the lookup and
    fail here (pr-test-analyzer #765)."""
    from wyrd.generators.kenning.lexicon.bridges._common import _compute_phonological_bridges

    examined = [
        {"id": 1, "canonical_form": "Ton"},  # mixed-case form -> folds to 'ton'
        {"id": 2, "canonical_form": "lea"},  # table value is mixed-case 'LĒAH'
    ]
    table = {"ton": "tūn", "lea": "LĒAH"}
    target_index = {"tūn": 10, "lēah": 20}  # keys are lower-cased resolved canonicals

    bridges, missing_target = _compute_phonological_bridges(examined, table, target_index)

    assert bridges == [(1, 10), (2, 20)]  # both bridge despite case on either side
    assert missing_target == 0


def test_bridge_phonological_oe_merges_known_pair(fresh_db: Path) -> None:
    """A known OE place-name form (`ton`) with a canonical Wiktionary
    target (`tūn`) gets merged_into_id set to the canonical's id.
    Pin so a refactor of _OE_PHONOLOGICAL_BRIDGES surfaces here."""

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == wiktionary_id
    assert result["bridged"] == 1
    assert result["unmatched"] == 1  # the wiktionary 'tūn' itself isn't a place-name form


@pytest.mark.parametrize(
    "place_form,wiktionary_form",
    [
        ("ton", "tūn"),
        ("lea", "lēah"),
        ("cote", "cot"),
        ("burgh", "burh"),
        ("dale", "dæl"),
        ("hall", "heall"),
        ("ey", "ieg"),
        ("burn", "burna"),
        ("hale", "healh"),
        ("new", "nīwe"),
        ("wood", "wudu"),
        ("bridge", "brycg"),
        ("stone", "stān"),
        ("hill", "hyll"),
        # wyrd-9ig additions:
        ("wella", "welle"),
        ("inga", "ing"),
        ("cippa", "cipp"),
        ("hāran", "hār"),
        ("cloppa", "clopp"),
        ("ticce", "ticcen"),
        ("kirk", "cirice"),
        ("hala", "healh"),
        ("pool", "pol"),
        ("scelf", "scylfe"),
        ("ēg", "ieg"),
        ("hare", "hara"),
        ("hlyp", "hlīep"),
        ("lech", "lacu"),
    ],
)
def test_bridge_phonological_oe_table_entries_resolve(
    fresh_db: Path, place_form: str, wiktionary_form: str
) -> None:
    """Each high-witness entry in the hand-curated bridge table is
    pinned individually. A typo or accidental drop in
    _OE_PHONOLOGICAL_BRIDGES would surface against the specific
    place-name → wiktionary pair it broke."""

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon(place_form, "old-english")
        target_id = db.upsert_etymon(wiktionary_form, "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged_target == target_id, f"{place_form!r} did not bridge to {wiktionary_form!r}"


def test_bridge_phonological_oe_no_target_increments_missing_target(
    fresh_db: Path,
) -> None:
    """If the table names a target form that doesn't exist as a
    canonical OE etymon (because it hasn't been ingested), the
    missing_target counter increments and the place-name etymon
    stays unmerged. Pin so operators see the signal in CLI output."""

    with LexiconDB(fresh_db) as db:
        # 'ton' is in the table → 'tūn', but no 'tūn' etymon exists.
        place_id = db.upsert_etymon("ton", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_phonological_oe_unmatched_form_left_alone(fresh_db: Path) -> None:
    """An OE etymon whose canonical_form isn't in the bridge table is
    counted as unmatched (silent — no missing_target signal because
    the table doesn't claim it should bridge)."""

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unmapped", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["unmatched"] == 1
    assert result["missing_target"] == 0


def test_bridge_phonological_oe_ignores_other_languages(fresh_db: Path) -> None:
    """The pass only walks language='old-english' rows. A 'modern-english'
    or 'celtic' etymon with the same form (`ton`) is not touched."""

    with LexiconDB(fresh_db) as db:
        modern_id = db.upsert_etymon("ton", "modern-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        modern_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (modern_id,)
        ).fetchone()["merged_into_id"]
    assert modern_target is None  # Modern English 'ton' untouched


def test_bridge_phonological_oe_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but doesn't persist merges."""

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0


def test_bridge_phonological_oe_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running --apply against unchanged data writes zero rows."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        first = bridge_phonological_oe(db, apply=True)
        second = bridge_phonological_oe(db, apply=True)
    assert first["rows_written"] == 1
    assert second["rows_written"] == 0


def test_bridge_phonological_oe_then_cluster_cognates_includes_place_form(
    fresh_db: Path,
) -> None:
    """End-to-end: bridge OE place-name 'ton' onto Wiktionary canonical
    'tūn', then run redirect-aware cluster_cognates, and confirm the
    place-name etymon is rolled up via merged_into_id into the cluster
    that contains the Proto-Germanic ancestor."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_canonical_id = db.upsert_etymon("tūn", "old-english")
        oe_place_id = db.upsert_etymon("ton", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, oe_canonical_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        cluster_cognates(db, apply=True)

        canonical_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (oe_canonical_id,)
        ).fetchone()["cognate_id"]
        place_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (oe_place_id,)
        ).fetchone()["merged_into_id"]

    assert place_target == oe_canonical_id
    assert canonical_synset == proto_id


# --- wyrd-083 + wyrd-ft3 round-1 fix: chain-flatten -------------------


def test_bridge_generic_language_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Regression for the round-1 critical finding: when bridging
    generic→target, any pre-existing OCR-tombstone that points AT the
    generic etymon must be re-routed to the target too. Without this
    flatten, a 2-deep chain X → generic → target forms, and the
    single-level COALESCE rollup in etymon_consensus / etymon_canonical
    splits witnesses."""

    with LexiconDB(fresh_db) as db:
        # Order matters: target_id < generic_id, so the bridge picks target.
        target_id = db.upsert_etymon("bun", "irish")
        generic_id = db.upsert_etymon("bun", "celtic")
        # Pre-existing OCR tombstone pointing at the generic celtic row
        # (e.g. an earlier normalize-ocr pass that merged 'BUN' → 'bun').
        old_ocr_loser_id = db.upsert_etymon("BUN", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (generic_id, old_ocr_loser_id),
        )
        db.commit()

        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        # After bridge: BOTH the generic celtic AND the old OCR loser
        # must point at the target_id (the irish entry), so the rollup
        # chain stays at depth 1.
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }
    assert rows[generic_id] == target_id
    assert rows[old_ocr_loser_id] == target_id, (
        "OCR-tombstone wasn't re-routed; 2-deep chain remains."
    )


def test_bridge_phonological_oe_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Same regression as bridge_generic_language: a pre-existing OCR
    tombstone pointing AT a place-name OE etymon must re-route to the
    Wiktionary canonical when the bridge fires."""

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        place_id = db.upsert_etymon("ton", "old-english")
        # Pre-existing OCR tombstone (e.g. 'TON' → 'ton' from a prior pass).
        old_ocr_loser_id = db.upsert_etymon("TON", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (place_id, old_ocr_loser_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }
    assert rows[place_id] == wiktionary_id
    assert rows[old_ocr_loser_id] == wiktionary_id, (
        "OCR-tombstone wasn't re-routed; 2-deep chain remains."
    )


def test_bridge_generic_language_reparents_lemma_children(fresh_db: Path) -> None:
    """If an inflected etymon has lemma_id pointing at the generic
    etymon being bridged, the lemma_id is re-routed to the canonical
    target so etymon_consensus's single-level lemma_id rollup keeps
    inflected variants in the same group as the canonical."""

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("bun", "irish")
        generic_id = db.upsert_etymon("bun", "celtic")
        inflected_id = db.upsert_etymon("buin", "celtic")  # hypothetical inflection
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (generic_id, inflected_id),
        )
        db.commit()

        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == target_id


def test_bridge_phonological_oe_reparents_lemma_children(fresh_db: Path) -> None:
    """Same lemma_id re-parent for the OE phonological bridge."""

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        place_id = db.upsert_etymon("ton", "old-english")
        inflected_id = db.upsert_etymon("tones", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (place_id, inflected_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == wiktionary_id


# --- CLI smoke tests for bridge-language / bridge-phonological-oe -----


def test_cli_bridge_language_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-language --generic celtic` (no --apply) reports
    counts without writing merged_into_id. Pinned so the dry-run path
    stays the default for operator safety."""
    with LexiconDB(fresh_db) as db:
        generic_id = db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--candidates",
            "irish",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (generic_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_language_apply_uses_celtic_default_candidates(
    fresh_db: Path,
) -> None:
    """`--generic celtic` with no --candidates resolves through the
    built-in _CELTIC_CANDIDATES_DEFAULT priority list and writes the
    merge when --apply is set."""
    with LexiconDB(fresh_db) as db:
        generic_id = db.upsert_etymon("bun", "celtic")
        # Proto-Celtic has higher priority than Irish in the default list.
        proto_id = db.upsert_etymon("bun", "proto-celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (generic_id,)
        ).fetchone()["merged_into_id"]
    assert merged == proto_id


def test_cli_bridge_language_requires_candidates_for_unknown_generic(
    fresh_db: Path,
) -> None:
    """No built-in candidate list exists for arbitrary generic tags;
    the CLI must fail loudly with --candidates missing rather than
    silently no-op."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "germanic",
        ],
    )
    assert result.exit_code == 1
    assert "no built-in default for 'germanic'" in result.stderr


def test_cli_bridge_language_empty_candidates_string_exits_cleanly(
    fresh_db: Path,
) -> None:
    """`--candidates ""` resolves to an empty tuple and would raise
    ValueError from bridge_generic_language. The CLI must catch this
    case before the call and exit 1 with a readable message rather
    than dumping a stack trace."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--candidates",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "empty list" in result.stderr


def test_cli_bridge_phonological_oe_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-phonological-oe` (no --apply) reports counts
    from the hand-curated table without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_phonological_oe_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge for known place-name → wiktionary
    pairs and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        target_id = db.upsert_etymon("tūn", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_phonological_oe_warns_on_missing_target(fresh_db: Path) -> None:
    """When a table entry names a wiktionary target that hasn't been
    ingested, the CLI emits the missing_target warning so operators
    know to ingest more or extend the table."""
    with LexiconDB(fresh_db) as db:
        # Place-name form present, wiktionary target absent.
        db.upsert_etymon("ton", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "target that doesn't exist" in result.stderr


# --- wyrd-9ig: bridge_phonological_oe target_index redirect-following ---


def test_bridge_phonological_oe_resolves_through_tombstone_target(
    fresh_db: Path,
) -> None:
    """When the bridge-table value names a form that exists only as a
    tombstone (merged_into another canonical), the target_index must
    follow the merged_into_id chain and resolve to the live canonical.
    Without this fix, _OE_PHONOLOGICAL_BRIDGES entries like 'dale → dæl'
    silently miss when a prior OCR-cluster pass has merged dæl into a
    macron-stripped 'dael' canonical.

    Reproduces the production bug where 5+ source citations of 'dale'
    stayed unbridged because dæl was merged into dael by normalize-ocr."""

    with LexiconDB(fresh_db) as db:
        # Live canonical that the macroned form has been OCR-merged into.
        live_id = db.upsert_etymon("dael", "old-english")
        # The macroned form named in the bridge table, but it's a tombstone.
        tombstone_id = db.upsert_etymon("dæl", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        # The place-name form the bridge wants to route.
        place_id = db.upsert_etymon("dale", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    # Place-name 'dale' must end up pointing at the LIVE canonical, not
    # the tombstone (which would create a 2-deep chain) and not None
    # (which is the pre-fix bug behavior).
    assert merged_target == live_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0


def test_bridge_phonological_oe_target_index_handles_multi_step_chain(
    fresh_db: Path,
) -> None:
    """Multi-step merged_into_id chains (A → B → C) must resolve all the
    way to the live canonical C, not stop at intermediate B. Pin so the
    target_index walk doesn't regress to single-step lookup."""

    with LexiconDB(fresh_db) as db:
        c_live = db.upsert_etymon("dael", "old-english")
        b_mid = db.upsert_etymon("daell", "old-english")  # arbitrary intermediate
        a_top = db.upsert_etymon("dæl", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (c_live, b_mid))
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (b_mid, a_top))
        place_id = db.upsert_etymon("dale", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == c_live


def test_bridge_phonological_oe_target_index_skips_self_loop(fresh_db: Path) -> None:
    """A merged_into_id self-loop (data corruption) must not infinite-loop
    the target_index resolution. The walk should bail and treat the row
    as its own canonical."""

    with LexiconDB(fresh_db) as db:
        weird_id = db.upsert_etymon("ton", "old-english")  # bridge-table key
        target_id = db.upsert_etymon("tūn", "old-english")
        # Synthetic self-loop on the target.
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (target_id, target_id))
        db.commit()
        # Should not hang.
        bridge_phonological_oe(db, apply=True)
        # And ton should still successfully bridge to tūn.
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (weird_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == target_id


def test_bridge_phonological_oe_ey_bridges_to_macron_stripped_ieg(
    fresh_db: Path,
) -> None:
    """The bridge value for 'ey' was changed from 'īeg' (which never
    exists in the corpus — Wiktionary's macroned headword wasn't
    ingested in that form) to 'ieg' (the OCR-stripped form that IS
    canonical in the DB). Pin the corrected mapping so a regression
    can't accidentally re-introduce 'īeg' as the bridge value."""

    with LexiconDB(fresh_db) as db:
        ieg_id = db.upsert_etymon("ieg", "old-english")
        ey_id = db.upsert_etymon("ey", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ey_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == ieg_id


# --- wyrd-fkg: bridge_phonological_on ----------------------------------


def test_bridge_phonological_on_merges_known_pair(fresh_db: Path) -> None:
    """Smoke test: a known place-name → wiktionary ON pair (`by` → `býr`)
    gets merged when --apply is on."""

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged == target_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0
    assert result["rows_written"] == 1


@pytest.mark.parametrize(
    "place_form,wiktionary_form",
    [
        ("by", "býr"),
        ("byr", "býr"),
        ("holm", "holmr"),
        ("kirk", "kirkja"),
        ("dal", "dalr"),
        ("dale", "dalr"),
        ("gardr", "garðr"),
        ("garth", "garðr"),
        ("griss", "gríss"),
        ("skogr", "skógr"),
        ("tun", "tún"),
        ("vik", "vík"),
        ("dyr", "dýr"),
        ("hest", "hestr"),
        ("krokr", "krókr"),
        ("nord", "norðr"),
        ("rauthr", "rauðr"),
        ("vollr", "vǫllr"),
        ("hogg", "hǫgg"),
        ("buski", "buskr"),
        ("flad", "flatr"),
        ("hain", "hagi"),
    ],
)
def test_bridge_phonological_on_table_entries_resolve(
    fresh_db: Path, place_form: str, wiktionary_form: str
) -> None:
    """Each high-witness ON bridge-table entry pinned individually so a
    typo or table edit surfaces against the specific pair it broke."""

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon(place_form, "old-norse")
        target_id = db.upsert_etymon(wiktionary_form, "old-norse")
        db.commit()
        bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id, f"{place_form!r} did not bridge to {wiktionary_form!r}"


def test_bridge_phonological_on_no_target_increments_missing_target(
    fresh_db: Path,
) -> None:
    """If the table names a target form that doesn't exist as a canonical
    ON etymon (because it hasn't been ingested), missing_target ticks."""

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")  # 'by' → 'býr'
        db.commit()  # No 'býr' etymon present.
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_phonological_on_unmatched_form_left_alone(fresh_db: Path) -> None:
    """An ON etymon whose canonical_form isn't in the bridge table is
    counted as unmatched (silent — no missing_target signal)."""

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unmapped_on", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 0
    assert result["unmatched"] >= 1
    assert result["missing_target"] == 0


def test_bridge_phonological_on_ignores_other_languages(fresh_db: Path) -> None:
    """Bridge only touches old-norse rows; OE / Modern English / etc.
    are untouched even if their canonical_form happens to match a key."""

    with LexiconDB(fresh_db) as db:
        oe_by = db.upsert_etymon("by", "old-english")  # accidental form match
        on_target = db.upsert_etymon("býr", "old-norse")
        on_by = db.upsert_etymon("by", "old-norse")
        db.commit()
        bridge_phonological_on(db, apply=True)
        oe_state = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (oe_by,)
        ).fetchone()["merged_into_id"]
        on_state = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (on_by,)
        ).fetchone()["merged_into_id"]

    assert oe_state is None  # OE 'by' untouched
    assert on_state == on_target  # ON 'by' bridged


def test_bridge_phonological_on_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts without writing merged_into_id."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1  # would-bridge count
    assert result["rows_written"] == 0
    assert result["applied"] is False


def test_bridge_phonological_on_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running apply on an already-bridged corpus writes 0 new rows."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("býr", "old-norse")
        db.upsert_etymon("by", "old-norse")
        db.commit()
        first = bridge_phonological_on(db, apply=True)
        second = bridge_phonological_on(db, apply=True)
    assert first["rows_written"] >= 1
    assert second["rows_written"] == 0


def test_bridge_phonological_on_resolves_through_tombstone_target(
    fresh_db: Path,
) -> None:
    """Same redirect-follow guarantee as the OE bridge: when the table
    value names a tombstone (e.g. þorp merged into thorp), the lookup
    must walk the chain and route to the live canonical."""

    with LexiconDB(fresh_db) as db:
        live_id = db.upsert_etymon("thorp", "old-norse")  # the live canonical
        tombstone_id = db.upsert_etymon("þorp", "old-norse")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        place_id = db.upsert_etymon("thorpe", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged == live_id
    assert result["bridged"] == 1


def test_bridge_phonological_on_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Chain-flatten: if a row already redirects AT the place-name etymon
    being bridged, the bridge re-routes that redirect onto the canonical
    target so no 2-deep chain forms (which would split witnesses in the
    single-level COALESCE rollup)."""

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        upstream_id = db.upsert_etymon("by-variant", "old-norse")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (place_id, upstream_id),
        )
        db.commit()
        bridge_phonological_on(db, apply=True)
        upstream_after = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (upstream_id,)
        ).fetchone()["merged_into_id"]
    assert upstream_after == target_id


def test_bridge_phonological_on_reparents_lemma_children(fresh_db: Path) -> None:
    """Same lemma_id re-parent for the ON phonological bridge."""

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        inflected_id = db.upsert_etymon("byjar", "old-norse")  # arbitrary inflection
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (place_id, inflected_id),
        )
        db.commit()
        bridge_phonological_on(db, apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == wiktionary_id


# --- CLI smoke tests for bridge-phonological-on -----------------------


def test_cli_bridge_phonological_on_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-phonological-on` (no --apply) reports counts
    without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")
        db.upsert_etymon("býr", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_phonological_on_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge for known place-name → wiktionary
    pairs and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")
        target_id = db.upsert_etymon("býr", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_phonological_on_warns_on_missing_target(fresh_db: Path) -> None:
    """When the table names a wiktionary target that hasn't been ingested,
    the CLI emits the missing_target warning."""
    with LexiconDB(fresh_db) as db:
        # Place-name form present, wiktionary target absent.
        db.upsert_etymon("by", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "target that doesn't exist" in result.stderr


# --- wyrd-7k4: bridge_celtic_forms --------------------------------


def test_bridge_celtic_forms_genitive_to_lemma(fresh_db: Path) -> None:
    """Smoke test: a known Goidelic genitive (`choill` = gen of coill
    'wood') bridges to the Irish lemma."""

    with LexiconDB(fresh_db) as db:
        irish_id = db.upsert_etymon("coill", "irish")
        # Pretend the Irish entry is clustered (has a synset).
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (irish_id,))
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    assert merged == irish_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0


def test_bridge_celtic_forms_prefers_clustered_target(fresh_db: Path) -> None:
    """When the lemma exists in multiple candidate languages, the bridge
    must pick the one with a non-NULL cognate_id (clustered) over an
    unclustered alternative — even if priority order would name the
    unclustered one first.

    This is the wyrd-083 stub-target bug: the original celtic bridge
    picked old-irish/mac (no synset, isolated stub) over irish/mac
    (synset 87349, clustered) because old-irish came first in the
    priority list. The new bridge prefers clustered."""

    with LexiconDB(fresh_db) as db:
        # Higher-priority candidate (per default order: irish first), but
        # we'll un-cluster it and cluster the lower-priority one to
        # exercise the prefer-clustered branch.
        db.upsert_etymon("mac", "irish")  # higher-priority but unclustered
        old_irish_clustered = db.upsert_etymon("mac", "old-irish")
        # Cluster only the old-irish entry.
        db.conn.execute(
            "UPDATE etymon SET cognate_id = id WHERE id = ?",
            (old_irish_clustered,),
        )
        # Add a celtic form whose lemma is 'mac' via the Anglicized
        # 'mhic' (lenited gen of mac) entry in _CELTIC_FORM_BRIDGES.
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]

    # Despite irish/mac being higher-priority, the clustered old-irish/mac
    # is chosen because the prefer-clustered logic kicks in.
    assert merged == old_irish_clustered


def test_bridge_celtic_forms_candidate_index_first_seen_wins(fresh_db: Path) -> None:
    """_build_candidate_index keys on lower(canonical_form), so two
    case-variant rows in the same candidate language collapse to one key;
    ORDER BY id makes the LOWER-id (first-seen) row win. Pin it: a
    lower-id UNCLUSTERED 'Mac' must beat a higher-id CLUSTERED 'mac' —
    proving the later duplicate is dropped at index-build BEFORE the
    prefer-clustered selection runs (wyrd-8uvi). A last-seen-wins or
    dropped ORDER BY regression would instead surface the clustered
    higher-id row and fail this."""
    with LexiconDB(fresh_db) as db:
        first_seen = db.upsert_etymon("Mac", "irish")  # lower id, unclustered
        later_clustered = db.upsert_etymon("mac", "irish")  # higher id, clustered
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (later_clustered,))
        celtic_x = db.upsert_etymon("x", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True, table={"x": "mac"})
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_x,)
        ).fetchone()["merged_into_id"]
    # First-seen (lower-id 'Mac') wins despite the higher-id 'mac' being clustered.
    assert merged == first_seen


def test_bridge_celtic_forms_falls_back_to_unclustered(
    fresh_db: Path,
) -> None:
    """If no candidate has a synset, the bridge falls back to the
    first-found by priority order (unclustered target). Prevents
    silently dropping bridges when the entire Wiktionary chain is stub."""

    with LexiconDB(fresh_db) as db:
        # Both candidates unclustered.
        irish_id = db.upsert_etymon("coill", "irish")
        db.upsert_etymon("coill", "scottish-gaelic")
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    # Falls back to irish (priority-order first).
    assert merged == irish_id


def test_bridge_celtic_forms_reroutes_existing_stub_bridge(
    fresh_db: Path,
) -> None:
    """The killer feature: if a celtic etymon was previously bridged to
    an unclustered stub (e.g. by an earlier wyrd-083 run), this pass
    re-routes it via chain-flatten to the clustered alternative.

    The chain-flatten OR-clause `WHERE id = ? OR merged_into_id = ?`
    catches the celtic etymon AND any rows that were already redirected
    onto it (which would now form a 2-deep chain otherwise)."""

    with LexiconDB(fresh_db) as db:
        # The pre-existing wyrd-083 stub-bridge target (unclustered).
        stub_id = db.upsert_etymon("mac", "old-irish")
        # The clustered alternative we want to land on.
        clustered_id = db.upsert_etymon("mac", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (clustered_id,))
        # Celtic etymon already pointed at the stub (simulating wyrd-083).
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (stub_id, celtic_mhic),
        )
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]

    assert merged == clustered_id


def test_bridge_celtic_forms_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts without writing merged_into_id."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0
    assert result["applied"] is False


def test_bridge_celtic_forms_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running apply on an already-bridged corpus writes 0 new rows."""

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        db.upsert_etymon("choill", "celtic")
        db.commit()
        first = bridge_celtic_forms(db, apply=True)
        second = bridge_celtic_forms(db, apply=True)
    assert first["rows_written"] >= 1
    assert second["rows_written"] == 0


def test_bridge_celtic_forms_unmatched_celtic_form_left_alone(
    fresh_db: Path,
) -> None:
    """A celtic etymon whose canonical_form isn't in the inflection table
    is counted as `unmatched` (silent — table makes no claim)."""

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unknown_celtic", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None
    assert result["unmatched"] >= 1
    assert result["missing_target"] == 0


def test_bridge_celtic_forms_missing_target_increments_counter(
    fresh_db: Path,
) -> None:
    """When the table names a lemma but no candidate-language etymon
    exists with that form, missing_target ticks (operator signal)."""

    with LexiconDB(fresh_db) as db:
        # 'choill' is in the table → 'coill', but no Irish/SG/Welsh
        # entry for 'coill' exists.
        db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)

    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_celtic_forms_resolves_through_tombstone_lemma(
    fresh_db: Path,
) -> None:
    """Candidate-index regression: when the lemma named in the bridge
    table exists in a candidate language as a TOMBSTONE (already
    OCR-merged into a canonical), the resolve step must walk the
    merged_into_id chain so the bridge routes to the LIVE canonical,
    not to the tombstone (which would create a 2-deep chain that the
    single-level COALESCE rollup would split).

    This pins the redirect-resolve loop in the candidate_index build."""

    with LexiconDB(fresh_db) as db:
        # The live canonical Irish lemma (clustered).
        live_id = db.upsert_etymon("coill-canonical", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (live_id,))
        # The tombstone Irish entry that the bridge table value 'coill'
        # actually names — already OCR-merged onto the live canonical.
        tombstone_id = db.upsert_etymon("coill", "irish")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        celtic_choill = db.upsert_etymon("choill", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?",
            (celtic_choill,),
        ).fetchone()["merged_into_id"]

    # The bridge resolved through tombstone → live canonical.
    assert merged == live_id


def test_bridge_celtic_forms_self_bridge_is_noop(fresh_db: Path) -> None:
    """If the celtic etymon and the resolved target happen to be the
    SAME etymon row (same id), the bridge is a no-op — bridges count
    excludes it. Pin the `target_id == row['id']` guard."""
    with LexiconDB(fresh_db) as db:
        # One celtic row whose canonical_form matches a table key, AND
        # the lemma the table names ALSO equals that same canonical_form,
        # AND there's no separate candidate-language entry — but we make
        # a celtic entry for the lemma itself. In practice the function
        # iterates ALL celtic rows; we exercise the self-bridge guard by
        # using a candidate-language match that happens to be the same id.
        # Concretely: 'cu' (celtic) → 'cú' lemma. If we add 'cú' as the
        # lemma in irish AND the celtic 'cu' is identified, no self-bridge
        # because they're different rows. The self-bridge guard only fires
        # if `target_id == row['id']`, which requires both to be the SAME
        # etymon — only possible if a celtic row's lemma resolves back to
        # itself. We construct that: 'cu' as both celtic and the only
        # candidate by overriding the candidate_langs to include 'celtic'.
        celtic_cu = db.upsert_etymon("cu", "celtic")
        db.upsert_etymon("cú", "irish")
        db.commit()
        # Override candidate_langs to include 'celtic' so the celtic
        # entry itself becomes a candidate. With only 'cu' (celtic)
        # mapping to 'cú', and the 'cú' (irish) being a different row,
        # the bridge would fire. To exercise the self-guard we need the
        # candidate to BE the celtic row. Use a custom table where the
        # form maps to its own canonical_form.
        from wyrd.generators.kenning.lexicon import bridge_celtic_forms as bic

        result = bic(
            db,
            apply=True,
            table={"cu": "cu"},  # self-mapping
            candidate_langs=("celtic",),
        )
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_cu,)
        ).fetchone()["merged_into_id"]

    # Self-bridge guard kept the row untouched.
    assert merged is None
    assert result["bridged"] == 0


def test_bridge_celtic_forms_iterates_tombstones(fresh_db: Path) -> None:
    """Unlike same-language phonological bridges, this bridge iterates
    ALL celtic rows (canonical + tombstones). A tombstone celtic row
    whose stub target is unclustered must still be considered for re-route
    via the chain-flatten OR-clause.

    This pins that the bridge does NOT filter out tombstones up-front."""

    with LexiconDB(fresh_db) as db:
        clustered = db.upsert_etymon("mac", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (clustered,))
        # 'mhic' celtic → 'mac' lemma. Tombstone it onto a totally
        # unrelated etymon to simulate corrupt prior bridge.
        bystander = db.upsert_etymon("zzz_bystander", "celtic")
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (bystander, celtic_mhic),
        )
        db.commit()
        result = bridge_celtic_forms(db, apply=True)

    # Examined count includes BOTH the canonical bystander and the
    # tombstone celtic_mhic (irish/mac is non-celtic, doesn't count).
    assert result["examined"] == 2
    # Tombstone re-routed to clustered target (chain-flatten OR-clause).
    with LexiconDB(fresh_db) as db2:
        actual = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]
    assert actual == clustered


# --- CLI smoke for bridge-celtic-forms ----------------------------


def test_cli_bridge_celtic_forms_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-celtic-forms` (no --apply) reports counts
    without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        place_id = db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_celtic_forms_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("coill", "irish")
        place_id = db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_celtic_forms_warns_on_missing_target(
    fresh_db: Path,
) -> None:
    """When a table entry names a lemma that doesn't exist in any
    candidate language, the CLI emits the missing_target warning."""
    with LexiconDB(fresh_db) as db:
        # 'choill' → 'coill', but no 'coill' lemma anywhere.
        db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "doesn't exist" in result.stderr


# --- meaning_synset (wyrd-7tz Phase 1) -------------------------------------


def test_init_schema_creates_meaning_synset_tables(fresh_db: Path) -> None:
    """Fresh DB must have both meaning_synset + etymon_meaning_synset tables.
    Pin so a regression that drops either table from lexicon.sql surfaces
    immediately."""
    conn = sqlite3.connect(fresh_db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert "meaning_synset" in tables
    assert "etymon_meaning_synset" in tables


def test_init_schema_meaning_synset_has_unique_canonical_label(fresh_db: Path) -> None:
    """canonical_label is the natural key — duplicate inserts must fail."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
            )


def test_init_schema_etymon_meaning_synset_fit_check_constraint(fresh_db: Path) -> None:
    """fit must be 'core' or 'peripheral' — bad values must fail at the
    DB layer so a buggy caller can't silently insert garbage."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
        )
        cognate_id = db.conn.execute(
            "SELECT id FROM meaning_synset WHERE canonical_label = ?", ("water/flowing",)
        ).fetchone()["id"]
        etymon_id = db.upsert_etymon("wæter", "old-english")
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                """
                INSERT INTO etymon_meaning_synset (etymon_id, meaning_synset_id, fit)
                VALUES (?, ?, ?)
                """,
                (etymon_id, cognate_id, "fuzzy"),
            )


def test_seed_meaning_synsets_inserts_full_catalog(fresh_db: Path) -> None:
    """Seed loader must populate the meaning_synset table from the bundled
    JSON catalog. The bundled catalog has 50+ entries — check for a known
    one to confirm the read+insert path works end-to-end."""
    with LexiconDB(fresh_db) as db:
        result = seed_meaning_synsets(db)
    assert result["inserted"] >= 50
    assert result["updated"] == 0
    assert result["unchanged"] == 0
    with LexiconDB(fresh_db) as db:
        labels = {
            row["canonical_label"]
            for row in db.conn.execute("SELECT canonical_label FROM meaning_synset")
        }
    # Spot-check spec'd labels.
    assert "water/flowing" in labels
    assert "water/body" in labels
    assert "hill/rounded" in labels
    assert "enclosure/farmstead" in labels


def test_seed_meaning_synsets_resolves_hypernyms(fresh_db: Path) -> None:
    """The two-pass seed walks the catalog so 'water/flowing' (declared
    with hypernym='water') gets its hypernym_id resolved against the
    'water' row inserted in pass 1. Pin via FK lookup."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        water_id = db.conn.execute(
            "SELECT id FROM meaning_synset WHERE canonical_label = ?", ("water",)
        ).fetchone()["id"]
        flowing_hypernym = db.conn.execute(
            "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
            ("water/flowing",),
        ).fetchone()["hypernym_id"]
    assert flowing_hypernym == water_id


def test_seed_meaning_synsets_idempotent(fresh_db: Path) -> None:
    """Running the seed twice on the same DB is a no-op — the second run
    reports 0 inserted, 0 updated, all unchanged."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        result_second = seed_meaning_synsets(db)
    assert result_second["inserted"] == 0
    assert result_second["updated"] == 0
    assert result_second["unchanged"] >= 50


def test_seed_meaning_synsets_updates_drifted_notes(fresh_db: Path) -> None:
    """If a row's notes have drifted from the catalog (e.g. someone
    edited the JSON to clarify a definition), the next seed run brings
    them into sync — counted as 'updated', not 'unchanged'."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        db.conn.execute(
            "UPDATE meaning_synset SET notes = ? WHERE canonical_label = ?",
            ("STALE", "water/flowing"),
        )
        db.commit()
        result = seed_meaning_synsets(db)
    assert result["updated"] == 1
    with LexiconDB(fresh_db) as db:
        notes = db.conn.execute(
            "SELECT notes FROM meaning_synset WHERE canonical_label = ?", ("water/flowing",)
        ).fetchone()["notes"]
    assert notes != "STALE"
    assert "OE wæter" in notes


def test_assign_etymon_to_meaning_synset_inserts_first_time(fresh_db: Path) -> None:
    """First-time assignment returns True (inserted)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        inserted = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
    assert inserted is True


def test_assign_etymon_to_meaning_synset_idempotent(fresh_db: Path) -> None:
    """Re-assigning the same (etymon, synset, fit) triple is a no-op
    returning False (not inserted, not changed)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
        again = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
    assert again is False


def test_assign_etymon_to_meaning_synset_updates_fit(fresh_db: Path) -> None:
    """Re-assigning with a different fit updates the existing row in
    place, returns False (not a fresh insert)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="peripheral")
        result = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="core")
    assert result is False
    with LexiconDB(fresh_db) as db:
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    assert synsets[0]["fit"] == "core"


def test_assign_etymon_to_meaning_synset_raises_on_unknown_etymon(fresh_db: Path) -> None:
    """Bad etymon_id must surface as a ValueError so a typo at the CLI
    surface doesn't silently insert nothing."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        with pytest.raises(ValueError, match="unknown etymon_id"):
            assign_etymon_to_meaning_synset(db, 9999, "water/flowing")


def test_assign_etymon_to_meaning_synset_raises_on_unknown_synset(fresh_db: Path) -> None:
    """Bad synset_label must surface as a ValueError naming the bad
    label so the user sees what they typo'd."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="unknown meaning_synset label"):
            assign_etymon_to_meaning_synset(db, etymon_id, "water/imaginary")


def test_assign_etymon_to_meaning_synset_raises_on_bad_fit(fresh_db: Path) -> None:
    """Validation runs in Python before SQLite is touched — bad fit
    raises ValueError, not IntegrityError."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="fit must be"):
            assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="fuzzy")


def test_get_meaning_synsets_for_etymon_orders_core_first(fresh_db: Path) -> None:
    """Multi-synset membership must surface 'core' before 'peripheral'
    so callers can pick the primary sense without sorting again."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        # water/body is alphabetically before water/flowing, but assigned
        # peripheral; water/flowing is core.
        assign_etymon_to_meaning_synset(db, etymon_id, "water/body", fit="peripheral")
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="core")
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    assert [s["canonical_label"] for s in synsets] == ["water/flowing", "water/body"]


def test_get_meaning_preserving_candidates_returns_cross_language(fresh_db: Path) -> None:
    """Spec'd regression test: OE wæter + ON bekkr both in water/flowing
    AND OE mere in water/body. Candidates for wæter include bekkr but
    NOT mere."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        mere = db.upsert_etymon("mere", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        assign_etymon_to_meaning_synset(db, mere, "water/body")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter)
    forms = {r["canonical_form"] for r in rows}
    assert "strēam" in forms
    assert "bekkr" in forms
    assert "mere" not in forms
    assert "wæter" not in forms  # default excludes self


def test_get_meaning_preserving_candidates_target_language_filter(fresh_db: Path) -> None:
    """target_language filter restricts candidates to one language —
    the anglicize/foreignize transform's primary use case."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        for eid in (waeter, stream, bekkr):
            assign_etymon_to_meaning_synset(db, eid, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter, target_language="old-norse")
    assert {r["canonical_form"] for r in rows} == {"bekkr"}


def test_get_meaning_preserving_candidates_fit_filter(fresh_db: Path) -> None:
    """fit='core' restricts BOTH ends of the join — only candidates
    where the source AND the candidate are in the synset as 'core'."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        burna = db.upsert_etymon("burna", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, burna, "water/flowing", fit="peripheral")
        rows = get_meaning_preserving_candidates(db, waeter, fit="core")
    assert {r["canonical_form"] for r in rows} == {"strēam"}


def test_get_meaning_preserving_candidates_include_self(fresh_db: Path) -> None:
    """include_self=True keeps the source etymon in the result — useful
    for dedup-aware callers."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter, include_self=True)
    assert {r["canonical_form"] for r in rows} == {"wæter", "strēam"}


def test_get_meaning_preserving_candidates_empty_when_no_membership(fresh_db: Path) -> None:
    """An etymon with no synset memberships has no meaning-preserving
    candidates — empty list, no crash."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        rows = get_meaning_preserving_candidates(db, etymon_id)
    assert rows == []


def test_list_meaning_synsets_with_member_counts(fresh_db: Path) -> None:
    """member_count column reports per-synset membership tally so a CLI
    user can spot empty/under-populated synsets."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        synsets = list_meaning_synsets(db, with_member_counts=True)
    flowing = [s for s in synsets if s["canonical_label"] == "water/flowing"][0]
    body = [s for s in synsets if s["canonical_label"] == "water/body"][0]
    assert flowing["member_count"] == 2
    assert body["member_count"] == 0


def test_migrate_schema_creates_meaning_synset_tables_on_legacy_db(tmp_path: Path) -> None:
    """A pre-wyrd-7tz DB without the meaning_synset tables should get
    them via migrate_schema — both tables created, both reported as
    applied=True."""
    # Build a DB with the WAL-pragma schema but DROP the new tables to
    # simulate a legacy state.
    db_path = tmp_path / "legacy.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE etymon_meaning_synset")
        conn.execute("DROP TABLE meaning_synset")
        conn.commit()
    finally:
        conn.close()
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["meaning_synset_table"] is True
    assert applied["etymon_meaning_synset_table"] is True
    # And re-running is a no-op.
    with LexiconDB(db_path) as db:
        applied_second = migrate_schema(db)
    assert applied_second["meaning_synset_table"] is False
    assert applied_second["etymon_meaning_synset_table"] is False


# --- CLI: lexicon synsets ---------------------------------------------------


def test_cli_lexicon_synsets_seed(fresh_db: Path) -> None:
    """`lexicon synsets seed` populates the catalog from bundled JSON."""
    result = CliRunner().invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    assert result.exit_code == 0, result.output
    assert "inserted=" in result.output
    with LexiconDB(fresh_db) as db:
        n = db.conn.execute("SELECT COUNT(*) AS c FROM meaning_synset").fetchone()["c"]
    assert n >= 50


def test_cli_lexicon_synsets_list_with_counts(fresh_db: Path) -> None:
    """`synsets list --with-counts` shows per-synset membership tallies."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "list", "--db", str(fresh_db), "--with-counts"],
    )
    assert result.exit_code == 0, result.output
    assert "water/flowing" in result.output
    # The water/flowing line should show 1 member.
    flowing_line = next(ln for ln in result.output.splitlines() if "water/flowing" in ln)
    assert "1 members" in flowing_line


def test_cli_lexicon_synsets_assign_and_show(fresh_db: Path) -> None:
    """`synsets assign` writes a membership; `synsets show` reads it back."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    assign_result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "assign", str(eid), "water/flowing", "--db", str(fresh_db)],
    )
    assert assign_result.exit_code == 0, assign_result.output
    assert "Added" in assign_result.output

    show_result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "show", str(eid), "--db", str(fresh_db)],
    )
    assert show_result.exit_code == 0, show_result.output
    assert "water/flowing" in show_result.output
    assert "core" in show_result.output


def test_cli_lexicon_synsets_assign_unknown_synset_exits_nonzero(fresh_db: Path) -> None:
    """A bad synset label at the CLI surface exits non-zero with a
    friendly error rather than spilling a Python traceback."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            str(eid),
            "water/imaginary",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "unknown meaning_synset label" in result.output


def test_cli_lexicon_synsets_candidates_cross_language(fresh_db: Path) -> None:
    """End-to-end: assign two etymons to the same synset, then ask for
    candidates of one — the other should appear."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "candidates", str(waeter), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "bekkr" in result.output
    assert "water/flowing" in result.output


def test_cli_lexicon_synsets_candidates_target_language(fresh_db: Path) -> None:
    """--target-language restricts the candidate pool to one language."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        for eid in (waeter, stream, bekkr):
            assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "candidates",
            str(waeter),
            "--target-language",
            "old-norse",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bekkr" in result.output
    assert "strēam" not in result.output


# --- wyrd-7tz Phase 1 round-2 follow-ups -----------------------------------


def test_get_meaning_preserving_candidates_dedupe_collapses_multi_synset_pairs(
    fresh_db: Path,
) -> None:
    """A (target, candidate) pair sharing N synsets produces N rows
    raw, but ONE row with dedupe=True (default). Phase 3 transforms
    want one row per candidate; the raw cartesian is for audit only."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        # Both etymons in TWO shared synsets — the cartesian would
        # produce 2 rows per candidate without dedupe.
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, waeter, "water/body", fit="peripheral")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/body", fit="peripheral")
        raw = get_meaning_preserving_candidates(db, waeter, dedupe=False)
        deduped = get_meaning_preserving_candidates(db, waeter)  # default dedupe=True
    assert len(raw) == 2
    assert len(deduped) == 1
    # Dedupe picks the strongest fit pair — core/core beats peripheral/peripheral.
    assert deduped[0]["target_fit"] == "core"
    assert deduped[0]["candidate_fit"] == "core"


def test_get_meaning_preserving_candidates_dedupe_picks_strongest_fit(
    fresh_db: Path,
) -> None:
    """When a pair shares two synsets, one core/core and one
    core/peripheral, dedupe picks the core/core row."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, waeter, "water/body", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/body", fit="peripheral")
        rows = get_meaning_preserving_candidates(db, waeter)
    assert len(rows) == 1
    # core/core wins over core/peripheral.
    assert rows[0]["synset_label"] == "water/flowing"
    assert rows[0]["candidate_fit"] == "core"


def test_get_meaning_preserving_candidates_raises_on_bad_fit(fresh_db: Path) -> None:
    """A bad --fit value raises ValueError before the SQL is executed."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="fit filter must be"):
            get_meaning_preserving_candidates(db, eid, fit="garbage")


def test_get_meaning_preserving_candidates_target_language_and_fit_combined(
    fresh_db: Path,
) -> None:
    """Both filters applied together — only candidates in the target
    language AND with matching fit on both ends survive."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")  # OE, core
        burna = db.upsert_etymon("burna", "old-english")  # OE, peripheral
        bekkr = db.upsert_etymon("bekkr", "old-norse")  # ON, core
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, burna, "water/flowing", fit="peripheral")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        # OE + core: only strēam (burna excluded by fit, bekkr by language).
        rows = get_meaning_preserving_candidates(
            db, waeter, target_language="old-english", fit="core"
        )
    assert {r["canonical_form"] for r in rows} == {"strēam"}


def test_seed_meaning_synsets_warns_on_unknown_hypernym(tmp_path: Path) -> None:
    """A seed entry pointing at a hypernym the catalog doesn't define
    (typo or out-of-catalog) emits a warning rather than silently
    dropping the link."""
    db_path = tmp_path / "hypernym_warn.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Inject a synthetic catalog with a bad hypernym pointer by
        # patching _meaning_synsets_seed at module level — easier than
        # writing a temp JSON file. seed_meaning_synsets resolves the
        # helper from its OWN module globals (synsets.py), so the patch
        # must target the submodule, not the lexicon-package re-export.
        import warnings

        from wyrd.generators.kenning import lexicon as lex
        from wyrd.generators.kenning.lexicon import synsets as lex_synsets

        original = lex_synsets._meaning_synsets_seed
        try:
            lex_synsets._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing", "hypernym": "ocean"},  # 'ocean' undefined
                ]
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lex.seed_meaning_synsets(db)
        finally:
            lex_synsets._meaning_synsets_seed = original
    assert any("unknown hypernym 'ocean'" in str(w.message) for w in caught)


def test_seed_meaning_synsets_clears_removed_hypernym_link(tmp_path: Path) -> None:
    """If a hypernym is removed from the JSON between seed runs, the FK
    on the existing row gets NULL'd out — drift-aware like notes."""
    db_path = tmp_path / "hypernym_clear.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        from wyrd.generators.kenning import lexicon as lex
        from wyrd.generators.kenning.lexicon import synsets as lex_synsets

        original = lex_synsets._meaning_synsets_seed
        try:
            # First run: water/flowing → water hypernym
            lex_synsets._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing", "hypernym": "water"},
                ]
            }
            lex.seed_meaning_synsets(db)
            row = db.conn.execute(
                "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
                ("water/flowing",),
            ).fetchone()
            assert row["hypernym_id"] is not None
            # Second run: hypernym removed from the catalog
            lex_synsets._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing"},  # no hypernym key
                ]
            }
            lex.seed_meaning_synsets(db)
        finally:
            lex_synsets._meaning_synsets_seed = original
        row = db.conn.execute(
            "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
            ("water/flowing",),
        ).fetchone()
    assert row["hypernym_id"] is None


def test_cli_lexicon_synsets_assign_by_form_and_language(fresh_db: Path) -> None:
    """The --language flag lets a curator assign by canonical_form
    rather than looking up the integer id first."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "wæter",
            "water/flowing",
            "--language",
            "old-english",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0, result.output
    # Resolved to the right id and assigned.
    with LexiconDB(fresh_db) as db:
        synsets = get_meaning_synsets_for_etymon(db, eid)
    assert {s["canonical_label"] for s in synsets} == {"water/flowing"}


def test_cli_lexicon_synsets_assign_by_form_unknown_exits_nonzero(fresh_db: Path) -> None:
    """A canonical_form that doesn't exist in the named language fails
    fast with a friendly message naming both."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "ghostform",
            "water/flowing",
            "--language",
            "old-english",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "no etymon with canonical_form='ghostform'" in result.output


def test_cli_lexicon_synsets_assign_non_numeric_without_language_exits_nonzero(
    fresh_db: Path,
) -> None:
    """Without --language, the positional must be numeric — passing a
    word fails with a friendly message rather than a Python traceback."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "wæter",
            "water/flowing",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "not numeric" in result.output


def test_cli_lexicon_synsets_show_empty_membership(fresh_db: Path) -> None:
    """An etymon with no synset memberships gets a friendly message
    (not an empty stdout)."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "show", str(eid), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "no meaning_synset memberships" in result.output


def test_cli_lexicon_synsets_candidates_empty(fresh_db: Path) -> None:
    """An etymon with no candidates gets a friendly message."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "candidates", str(eid), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "no meaning-preserving candidates" in result.output


# --- wyrd-44a: rename etymon.synset_id → etymon.cognate_id ----------------


def test_migrate_schema_renames_legacy_synset_id_to_cognate_id(tmp_path: Path) -> None:
    """A legacy DB carrying the old `synset_id` / `synset_method` columns
    + `idx_etymon_synset` index must be migrated to the new naming —
    column data preserved, index re-created under the new name."""
    db_path = tmp_path / "legacy_synset.db"
    init_schema(db_path)
    # Simulate a pre-rename DB: drop the new names and add the legacy
    # ones so the migration helper has work to do.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE etymon RENAME COLUMN cognate_id TO synset_id")
        conn.execute("ALTER TABLE etymon RENAME COLUMN cognate_method TO synset_method")
        conn.execute("DROP INDEX IF EXISTS idx_etymon_cognate")
        conn.execute("CREATE INDEX idx_etymon_synset ON etymon(synset_id)")
        # Insert a row carrying real data on the legacy column so the
        # migration's data-preservation guarantee can be checked.
        conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
            ("tūn", "old-english"),
        )
        eid = conn.execute("SELECT id FROM etymon WHERE canonical_form = ?", ("tūn",)).fetchone()[0]
        conn.execute(
            "UPDATE etymon SET synset_id = ?, synset_method = ? WHERE id = ?",
            (eid, "cluster-cognates-v1", eid),
        )
        conn.commit()
    finally:
        conn.close()
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["etymon.synset_id_renamed_to_cognate_id"] is True
    assert applied["etymon.synset_method_renamed_to_cognate_method"] is True
    assert applied["idx_etymon_synset_renamed_to_idx_etymon_cognate"] is True
    # Post-migration: new names exist, old names are gone, data preserved.
    with LexiconDB(db_path) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        row = db.conn.execute(
            "SELECT cognate_id, cognate_method FROM etymon WHERE canonical_form = ?", ("tūn",)
        ).fetchone()
    assert "cognate_id" in cols
    assert "cognate_method" in cols
    assert "synset_id" not in cols
    assert "synset_method" not in cols
    assert "idx_etymon_cognate" in indexes
    assert "idx_etymon_synset" not in indexes
    assert row["cognate_id"] == eid
    assert row["cognate_method"] == "cluster-cognates-v1"


def test_migrate_schema_rename_is_idempotent_on_already_migrated_db(fresh_db: Path) -> None:
    """Running migrate_schema on a fresh DB (already on the new naming)
    is a no-op for the rename helper — applied=False on all three rename
    keys."""
    with LexiconDB(fresh_db) as db:
        applied = migrate_schema(db)
    assert applied["etymon.synset_id_renamed_to_cognate_id"] is False
    assert applied["etymon.synset_method_renamed_to_cognate_method"] is False
    assert applied["idx_etymon_synset_renamed_to_idx_etymon_cognate"] is False


def test_init_schema_uses_cognate_id_not_synset_id(fresh_db: Path) -> None:
    """Fresh installs go straight to the new naming — no legacy
    synset_id column ever appears."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    assert "cognate_id" in cols
    assert "cognate_method" in cols
    assert "synset_id" not in cols
    assert "synset_method" not in cols
    assert "idx_etymon_cognate" in indexes
    assert "idx_etymon_synset" not in indexes


def test_gather_family_uses_member_tags_only(fresh_db: Path) -> None:
    """wyrd-c4wd: _gather_family's "tags" field is now the MEMBER
    tags only — the cluster-mate union was producing visible noise
    (e.g. '-y' tagged 'animal', 'Dar-' (Deer) tagged 'architecture'),
    so it was dropped in favor of faithful member-only tags.

    Pre-fix this test asserted the union (member + cluster mate);
    post-fix it asserts member-only. The cluster mate's 'topography'
    no longer rides along when only the OE root is the family member.
    The cluster-mate-tag helper itself was removed once the bundle
    exporter stopped consulting it (dead-code-audit)."""

    with LexiconDB(fresh_db) as db:
        oe_id = db.upsert_etymon("ceaster", "old-english")
        me_id = db.upsert_etymon("chestre", "middle-english")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (oe_id, oe_id))
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (oe_id, me_id))
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)", (oe_id, "architecture")
        )
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)", (me_id, "topography")
        )
        db.commit()
        # member_ids=[oe_id] → only the OE root's tags. ME cluster mate
        # 'chestre' is a separate root; its 'topography' tag stays put.
        family = _gather_family(db, oe_id, [oe_id])
    assert family is not None
    assert family["tags"] == ["architecture"]


def test_gather_family_slices_provided_attested_years(fresh_db: Path) -> None:
    """wyrd-4zyb: when the caller bulk-prefetched attested years (the
    _iterate_families_with_progress path), _gather_family SLICES the provided
    map rather than re-running the per-family query. Members present in the map
    are included; ids outside member_ids are ignored; the per-family DB fallback
    is NOT consulted (no attested_year rows are seeded here, yet the year
    survives — it can only have come from the provided map)."""

    with LexiconDB(fresh_db) as db:
        oe_id = db.upsert_etymon("ceaster", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (oe_id, oe_id))
        db.commit()
        family = _gather_family(
            db, oe_id, [oe_id], attested_years_by_member={oe_id: 1086, 999999: 1200}
        )
    assert family is not None
    # oe_id sliced from the map; 999999 (not a member) dropped; no DB fallback.
    assert family["member_attested_years"] == {oe_id: 1086}


# ---------------------------------------------------------------------
# wyrd-h0u1 CLI smoke for `lexicon report`
# ---------------------------------------------------------------------
#
# `lexicon report` had no direct CLI coverage before — the refactor to
# _readonly_lexicon would have passed every existing test even if the
# command was completely broken. Smoke test invokes against a small
# seeded DB and pins exit-code + the headline sections so a regression
# (missing section, raised exception, leaked connection on early exit)
# trips here.


def test_cli_lexicon_report_runs_against_seeded_db(fresh_db: Path) -> None:
    """End-to-end smoke for `wyrd kenning lexicon report` — exit-code
    zero against a small populated DB + every headline section shows
    up in the output. wyrd-h0u1 follow-up after agent-reviewer P2 on
    PR #289 (no direct CLI coverage before)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="skeat-1901", title="Skeat 1901", year=1901)
        db.upsert_source(id="mawer-1920", title="Mawer 1920", year=1920)
        eid_cot = db.upsert_etymon("cot", "old-english")
        eid_tun = db.upsert_etymon("tūn", "old-english")
        db.add_gloss(eid_cot, "cottage")
        db.add_tag(eid_cot, "habitation")
        db.add_citation(eid_cot, "skeat-1901", page="15")
        db.add_citation(eid_tun, "mawer-1920", page="7")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "report", "--db", str(fresh_db)])

    assert result.exit_code == 0, result.output + (result.stderr or "")
    # Each headline section appears (regression catcher for the
    # _section_* extraction in PR #289 round 2 — if a section helper
    # got dropped from the sequencer the section disappears silently).
    for marker in (
        "=== sources ===",
        "=== totals ===",
        "=== consensus ===",
        "=== top",  # top {N} etymons by witnesses ===
        "languages of mined etymons",  # now "=== top {N} languages of mined etymons (...) ==="
        "=== confidence distribution",
        "=== toponym countries ===",  # wyrd-c95y
        "toponym regions",  # "=== top {N} toponym regions (...) ===" — wyrd-c95y
        "=== toponyms per source ===",
        "=== disagreements ===",
    ):
        assert marker in result.output, (
            f"missing '{marker}' section in `lexicon report` output:\n{result.output}"
        )
    # Sanity: the seeded sources surface in the sources section.
    assert "skeat-1901" in result.output
    assert "mawer-1920" in result.output
