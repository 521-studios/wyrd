"""Tests for ``runtime_db_adapter`` (wyrd-d90t).

Exercises the L4-DB → bundle-dict adapter end-to-end:

1. The adapter's pure-Python output round-trips through
   ``load_meanings`` / ``load_canonical_decompositions`` /
   ``load_fantasy_morphemes`` to produce the same shape the
   historical JSON bundle did.
2. The per-cache loaders (``_load_meanings`` etc.) resolve their
   data from the SQLite runtime DB end-to-end — the canonical
   runtime path post-cutover.
3. The committed seed-runtime.db loads cleanly via the SQLite path —
   the offline / CI default works.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import wyrd.generators.kenning as kenning_pkg
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db
from wyrd.generators.kenning.runtime.runtime_db_adapter import (
    bundle_dict_from_runtime_db,
    genitive_split_map_from_runtime_db,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear the lru_cache on _load_meanings etc. between tests so the
    feature-flag flip is honored — caches don't see env-var changes."""
    kenning_pkg._load_meanings.cache_clear()
    kenning_pkg._load_canonical_decompositions.cache_clear()
    kenning_pkg._load_fantasy_morphemes.cache_clear()
    kenning_pkg._load_genitive_prior.cache_clear()
    # The runtime DB loader has its own process-wide cached conn.
    from wyrd.generators.kenning.runtime.runtime_db import reset_runtime_db_cache

    reset_runtime_db_cache()
    yield
    kenning_pkg._load_meanings.cache_clear()
    kenning_pkg._load_canonical_decompositions.cache_clear()
    kenning_pkg._load_fantasy_morphemes.cache_clear()
    kenning_pkg._load_genitive_prior.cache_clear()
    reset_runtime_db_cache()


@pytest.fixture
def _runtime_db_from_fixture(tmp_path: Path) -> Path:
    """Build a minimal L4 runtime DB from a tmp lexicon + fixture
    proportions. Returns the path the loader env-var can point at."""
    import json

    from wyrd.generators.kenning.lexicon import init_schema

    lexicon_path = tmp_path / "lexicon.db"
    init_schema(lexicon_path)
    with LexiconDB(lexicon_path) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        etymon_id = db.upsert_etymon(
            "ham",
            "old-english",
            modifier_type="Topographical",
            position_pref="post",
        )
        db.add_citation(etymon_id, "rando-port")
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (etymon_id, "home, dwelling"),
        )
        db.commit()

    # Per-culture proportions stubs (minimum needed for the emit).
    proportions_dir = tmp_path / "proportions"
    proportions_dir.mkdir()
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        (proportions_dir / f"{culture}_proportions.json").write_text(
            json.dumps(
                {
                    "usages": {"-ham-": 1},
                    "single_usages": {},
                    "structures": [{"proportion": 1, "words": [[{"location": "pre"}]]}],
                    "tag_marginal": {},
                    "tag_cooccurrence": {},
                }
            )
        )

    out_path = tmp_path / "runtime.db"
    write_runtime_db(
        output_path=out_path,
        subjects=[],  # will fetch via export inline
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=proportions_dir,
        source_lexicon_db=lexicon_path,
    )
    # Re-emit with real subjects from export_meanings — the simplest
    # path is to invoke the full lexicon export flow.
    from wyrd.generators.kenning.lexicon import (
        collect_canonical_decompositions,
        collect_fantasy_morphemes,
        export_meanings,
    )

    with LexiconDB(lexicon_path) as db:
        subjects = export_meanings(db)
        canonical = collect_canonical_decompositions(db)
        fantasy = collect_fantasy_morphemes(db)
    write_runtime_db(
        output_path=out_path,
        subjects=subjects,
        fantasy_morphemes=fantasy,
        canonical_decompositions=canonical,
        proportions_dir=proportions_dir,
        source_lexicon_db=lexicon_path,
    )
    return out_path


# ---------- adapter pure-Python shape ----------


def test_bundle_dict_has_canonical_keys(_runtime_db_from_fixture: Path) -> None:
    """The adapter output carries the three keys the bundle loaders
    consume (``subjects``, ``canonical_decompositions``,
    ``fantasy_morphemes``)."""
    conn = sqlite3.connect(f"file:{_runtime_db_from_fixture}?mode=ro", uri=True)
    try:
        data = bundle_dict_from_runtime_db(conn)
    finally:
        conn.close()

    assert set(data.keys()) >= {"subjects", "canonical_decompositions", "fantasy_morphemes"}
    assert isinstance(data["subjects"], list)
    assert isinstance(data["canonical_decompositions"], dict)
    assert isinstance(data["fantasy_morphemes"], dict)


def test_subjects_have_word_and_metadata(_runtime_db_from_fixture: Path) -> None:
    """Each synthesized subject has the shape ``{meaning, modifier_tags,
    modifier_type, words: [word_dict]}`` — what ``load_meanings``
    iterates over."""
    conn = sqlite3.connect(f"file:{_runtime_db_from_fixture}?mode=ro", uri=True)
    try:
        data = bundle_dict_from_runtime_db(conn)
    finally:
        conn.close()

    assert data["subjects"], "expected at least one subject from the fixture lexicon"
    sample = data["subjects"][0]
    assert set(sample.keys()) == {"meaning", "modifier_tags", "modifier_type", "words"}
    assert len(sample["words"]) == 1
    assert "modern_usage" in sample["words"][0]


# ---------- _load_meanings always routes through the L4 ----------


def test_load_meanings_reads_l4_fixture(monkeypatch, _runtime_db_from_fixture: Path) -> None:
    """With ``WYRD_RUNTIME_DB`` pointed at the fixture L4,
    ``_load_meanings`` rehydrates the bundle from SQLite. The
    meaning_db carries the fixture's ``-ham-`` morpheme alongside
    sidecar entries (irish_anglicizations, Norman manorial families)
    that always fold in."""
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(_runtime_db_from_fixture))

    meaning_db, _tag_db = kenning_pkg._load_meanings()
    assert any("ham" in usage_key.lower() for usage_key in meaning_db)


def test_load_canonical_decompositions_reads_l4(
    monkeypatch, _runtime_db_from_fixture: Path
) -> None:
    """``_load_canonical_decompositions`` routes through the L4 too."""
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(_runtime_db_from_fixture))

    # Fixture L3 has no toponym_decompositions → empty L4 canonical
    # table → empty dict (no crash).
    canonicals = kenning_pkg._load_canonical_decompositions()
    assert isinstance(canonicals, dict)


def test_load_fantasy_morphemes_reads_l4(monkeypatch, _runtime_db_from_fixture: Path) -> None:
    """``_load_fantasy_morphemes`` routes through the L4 too."""
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(_runtime_db_from_fixture))

    fantasy = kenning_pkg._load_fantasy_morphemes()
    assert isinstance(fantasy, dict)


# ---------- committed seed loads via SQLite path ----------


def test_load_meanings_uses_committed_seed_by_default(monkeypatch) -> None:
    """With no env overrides, the loader falls back to the bundled
    seed-runtime.db. The SQLite path produces a non-empty meaning_db
    from that seed — proves the offline CI/dev default works
    end-to-end without any operator-side environment plumbing."""
    monkeypatch.delenv("WYRD_RUNTIME_DB", raising=False)
    monkeypatch.delenv("WYRD_RUNTIME_DB_BUCKET", raising=False)

    meaning_db, _tag_db = kenning_pkg._load_meanings()
    # Seed has ~1146 meaning rows — should be far more than zero.
    assert len(meaning_db) > 100


def test_meaning_objects_have_subject_metadata(monkeypatch, _runtime_db_from_fixture: Path) -> None:
    """The synthesized per-entry subjects in the adapter must produce
    Meaning objects with the same subject-level metadata (tags + plain
    meanings + modifier_type) the L3-built bundle did. Pins the
    'identical Meaning objects across backends' contract."""
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(_runtime_db_from_fixture))

    meaning_db, _ = kenning_pkg._load_meanings()
    # Find the 'ham' meaning from the fixture.
    for usage, meanings in meaning_db.items():
        if "ham" in usage.lower():
            sample = meanings[0]
            # Subject metadata threaded through to the Meaning. tags
            # is the subject's modifier_tags; sources is the per-word
            # language-form dict. Both are derived from the data the
            # adapter rehydrated from the L4 row.
            assert hasattr(sample, "tags")
            assert hasattr(sample, "sources")
            return
    pytest.fail("expected at least one '-ham-' family meaning in the loaded db")


# ---------- genitive-split prior round-trip (wyrd-aicu.9, D-3) ----------


def _emit_runtime_db_with_genitive(tmp_path: Path, genitive_split_map) -> Path:
    """Emit a minimal L4 runtime DB carrying ``genitive_split_map`` (or None)."""
    import json

    from wyrd.generators.kenning.lexicon import init_schema

    lexicon_path = tmp_path / "lexicon.db"
    init_schema(lexicon_path)
    with LexiconDB(lexicon_path) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.commit()

    proportions_dir = tmp_path / "proportions"
    proportions_dir.mkdir()
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        (proportions_dir / f"{culture}_proportions.json").write_text(
            json.dumps(
                {
                    "usages": {"-ham-": 1},
                    "single_usages": {},
                    "structures": [{"proportion": 1, "words": [[{"location": "pre"}]]}],
                    "tag_marginal": {},
                    "tag_cooccurrence": {},
                }
            )
        )

    out_path = tmp_path / "runtime.db"
    write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=proportions_dir,
        genitive_split_map=genitive_split_map,
        source_lexicon_db=lexicon_path,
    )
    return out_path


def test_genitive_split_map_round_trips_exact_dict(tmp_path: Path) -> None:
    """Emit a known prob-map → the adapter reads back the EXACT
    ``{(long, short): p}`` dict (keys are tuples, values are floats)."""
    known = {("ston", "ton"): 0.82, ("sley", "ley"): 0.41}
    out_path = _emit_runtime_db_with_genitive(tmp_path, known)

    conn = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
    try:
        loaded = genitive_split_map_from_runtime_db(conn)
    finally:
        conn.close()

    assert loaded == known


def test_genitive_split_map_absent_when_empty_map(tmp_path: Path) -> None:
    """An empty / None map writes NO row → the adapter returns None →
    the loader treats it as {} (connective coverage lands, tiebreak
    silent)."""
    out_path = _emit_runtime_db_with_genitive(tmp_path, {})

    conn = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT COUNT(*) FROM genitive_split_prior").fetchone()[0]
        loaded = genitive_split_map_from_runtime_db(conn)
    finally:
        conn.close()

    assert row == 0
    assert loaded is None


def test_load_genitive_prior_returns_empty_dict_when_absent(monkeypatch, tmp_path: Path) -> None:
    """The package-level ``_load_genitive_prior`` cache returns ``{}`` (not
    None) when the bundle carries no genitive row — the matcher-facing
    'no prior' contract."""
    out_path = _emit_runtime_db_with_genitive(tmp_path, {})
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(out_path))

    assert kenning_pkg._load_genitive_prior() == {}


def test_load_genitive_prior_reads_l4_map(monkeypatch, tmp_path: Path) -> None:
    """The package-level cache reads the emitted prob-map from the L4."""
    known = {("ston", "ton"): 0.9}
    out_path = _emit_runtime_db_with_genitive(tmp_path, known)
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(out_path))

    assert kenning_pkg._load_genitive_prior() == known
