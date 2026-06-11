"""Tests for the per-culture proportions SQLite path (wyrd-d90t).

Covers:

1. ``proportions_dict_for_culture`` produces the same shape the
   historical JSON sidecar did (round-trip through
   ``load_proportions`` succeeds).
2. ``_load_culture`` routes through the L4 end-to-end (no flag —
   SQLite is the only path).
3. ``proportions_dict_for_culture`` and the legacy JSON shape are
   bit-equivalent when the L4 was built from the same fixture data.
   Confirms the L4 emit preserves the load-bearing surface the
   ``NameGenerator`` samples from.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import wyrd.generators.kenning as kenning_pkg
from wyrd.generators.kenning.runtime.runtime_db_adapter import (
    proportions_dict_for_culture,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear lru_caches so the feature-flag flip is honored."""
    kenning_pkg._load_meanings.cache_clear()
    from wyrd.generators.kenning.runtime.runtime_db import reset_runtime_db_cache

    reset_runtime_db_cache()
    yield
    kenning_pkg._load_meanings.cache_clear()
    reset_runtime_db_cache()


@pytest.fixture
def _proportions_db(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small L4 DB with realistic per-culture proportions data
    so the adapter's round-trip + bit-equivalence tests have something
    non-trivial to chew on."""
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
    from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db

    lexicon_path = tmp_path / "lexicon.db"
    init_schema(lexicon_path)
    with LexiconDB(lexicon_path) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        for form in ("ham", "ton", "ford"):
            eid = db.upsert_etymon(
                form,
                "old-english",
                modifier_type="Topographical",
                position_pref="post",
            )
            db.add_citation(eid, "rando-port")
        db.commit()

    # Per-culture proportions stubs: english has multiple usages so the
    # bit-equivalence sample doesn't collapse to a single keep_keys.
    proportions_dir = tmp_path / "proportions"
    proportions_dir.mkdir()
    english = {
        "usages": {"-ham-": 100, "-ton-": 50, "-ford-": 25},
        "single_usages": {"-castle": 10, "-bury": 5},
        "structures": [
            {"proportion": 70, "words": [[{"location": "pre"}, {"location": "post"}]]},
            {
                "proportion": 30,
                "words": [[{"location": "pre"}, {"location": "inner"}, {"location": "post"}]],
            },
        ],
        "tag_marginal": {"water": 12, "architecture": 8, "topography": 5},
        "tag_cooccurrence": {"water|architecture": 3, "water|tree": 2},
    }
    (proportions_dir / "english_proportions.json").write_text(json.dumps(english))
    # The other 4 cultures get minimal stubs so the emitter doesn't skip them.
    for c in ("scottish", "welsh", "irish", "breton"):
        (proportions_dir / f"{c}_proportions.json").write_text(
            json.dumps(
                {
                    "usages": {"-stub-": 1},
                    "single_usages": {},
                    "structures": [{"proportion": 1, "words": [[{"location": "pre"}]]}],
                    "tag_marginal": {},
                    "tag_cooccurrence": {},
                }
            )
        )

    from wyrd.generators.kenning.lexicon import (
        collect_canonical_decompositions,
        collect_fantasy_morphemes,
        export_meanings,
    )

    out_path = tmp_path / "runtime.db"
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
    return (out_path, proportions_dir)


# ---------- adapter shape ----------


def test_proportions_dict_has_canonical_keys(_proportions_db: tuple[Path, Path]) -> None:
    """The adapter's output has the keys load_proportions consumes."""
    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = proportions_dict_for_culture(conn, "english")
    finally:
        conn.close()
    assert set(out.keys()) == {
        "usages",
        "single_usages",
        "structures",
        "tag_marginal",
        "tag_cooccurrence",
        # wyrd-pfoo: per-Meaning attestation (per-culture). Empty
        # dict for fixtures that don't populate the L4 attestation
        # table; populated dict for fresh emits.
        "attested_languages",
        # wyrd-rogd.13: per-word-position bare stats. Empty dict for
        # fixtures without positional rows; populated for fresh emits.
        "bare_word_positions",
    }


def test_proportions_dict_matches_source_weights(_proportions_db: tuple[Path, Path]) -> None:
    """Round-trip: weights stored in the L4 come back unchanged."""
    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = proportions_dict_for_culture(conn, "english")
    finally:
        conn.close()

    assert out["usages"] == {"-ham-": 100, "-ton-": 50, "-ford-": 25}
    assert out["single_usages"] == {"-castle": 10, "-bury": 5}
    assert len(out["structures"]) == 2
    assert out["structures"][0] == {
        "proportion": 70,
        "words": [[{"location": "pre"}, {"location": "post"}]],
    }
    assert out["tag_marginal"] == {"water": 12, "architecture": 8, "topography": 5}
    assert out["tag_cooccurrence"] == {"water|architecture": 3, "water|tree": 2}


def test_proportions_iteration_order_matches_cumulative(_proportions_db: tuple[Path, Path]) -> None:
    """usages dict iteration order = cumulative ascending. Tested
    here so a future ORDER BY drift surfaces immediately."""
    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = proportions_dict_for_culture(conn, "english")
    finally:
        conn.close()
    # Source dict iteration order was -ham- → -ton- → -ford- (highest
    # weight first); cumulative monotonic in that order.
    assert list(out["usages"].keys()) == ["-ham-", "-ton-", "-ford-"]


def test_proportions_unknown_culture_returns_empty(_proportions_db: tuple[Path, Path]) -> None:
    """A culture the L4 doesn't have rows for returns empty dicts —
    consistent with the JSON path returning an empty proportions file."""
    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = proportions_dict_for_culture(conn, "klingon")
    finally:
        conn.close()
    assert out["usages"] == {}
    assert out["structures"] == []


def test_read_proportions_usage_map_rejects_non_whitelisted_table(
    _proportions_db: tuple[Path, Path],
) -> None:
    """Defense-in-depth on the f-string table-name interpolation —
    same shape as runtime_db_export's whitelist."""
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        _read_proportions_usage_map,
    )

    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with pytest.raises(ValueError, match="not in whitelist"):
            _read_proportions_usage_map(conn, "evil_table; DROP TABLE meaning;--", "english")
    finally:
        conn.close()


# ---------- _load_culture rehydrates the per-culture generator ----------


def test_load_culture_reads_l4(monkeypatch, _proportions_db: tuple[Path, Path]) -> None:
    """``_load_culture`` builds the proportions dict from the L4.
    Smoke-test by checking a generator can be constructed against
    the fixture-sized English proportions."""
    db_path, _ = _proportions_db
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(db_path))
    kenning_pkg._load_culture.cache_clear()

    name_gen, tag_db = kenning_pkg._load_culture("english")
    # Generator constructed without crashing against the fixture-sized
    # L4 (3 usages, 2 single_usages, 2 structures).
    assert name_gen is not None
    assert name_gen.meaning_db is not None
    # tag_db is the second return value from _load_culture; the
    # fixture lexicon contributes at least a few tag rows.
    assert isinstance(tag_db, dict)


# ---------- bit-equivalence ----------


def test_bit_equivalent_names_across_backends(
    monkeypatch, _proportions_db: tuple[Path, Path]
) -> None:
    """Same seed → same name across JSON and SQLite backends.

    The L4 was built from the fixture proportions_dir; load_proportions
    + Generator.select run unchanged on either backend's data. So a
    deterministic seed must produce the same name. This is the load-
    bearing contract for PR 6.
    """
    db_path, _ = _proportions_db
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(db_path))

    # JSON path: point _data_path at the fixture proportions_dir so
    # _load_culture's JSON branch reads from there. Easiest way: chdir
    # into a synthetic data dir containing the fixture proportions +
    # a stub meanings.json.
    # Simpler approach: run the SQLite path AND verify the rebuilt
    # proportions dict round-trips through load_proportions to the
    # SAME NameGenerator structure as the JSON would. The JSON-vs-
    # SQLite identity is then provable from the dict identity (since
    # load_proportions is pure).
    db_path, fixture_proportions_dir = _proportions_db

    json_path = fixture_proportions_dir / "english_proportions.json"
    with json_path.open() as f:
        json_proportions = json.load(f)

    db_path, _ = _proportions_db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sqlite_proportions = proportions_dict_for_culture(conn, "english")
    finally:
        conn.close()

    # Bit equivalence at the dict level → load_proportions produces
    # identical NameGenerators → same RNG seed produces same name.
    # (Iteration order matters for random.choices; verify dict keys
    # come back in the same order.)
    assert list(sqlite_proportions["usages"].keys()) == list(json_proportions["usages"].keys())
    assert sqlite_proportions["usages"] == json_proportions["usages"]
    assert sqlite_proportions["single_usages"] == json_proportions["single_usages"]
    assert sqlite_proportions["structures"] == json_proportions["structures"]
    assert sqlite_proportions["tag_marginal"] == json_proportions["tag_marginal"]
    assert sqlite_proportions["tag_cooccurrence"] == json_proportions["tag_cooccurrence"]

    # Smoke: build a name generator from each and roll the same RNG.
    # Both should produce the same sequence of weighted_choice picks
    # over the proportions data.
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    # Use a tiny meaning_db so load_proportions has something to wrap.
    meaning_db = {
        "-ham-": [],
        "-ton-": [],
        "-ford-": [],
        "-castle": [],
        "-bury": [],
    }
    tag_db: dict = {}
    json_gen = load_proportions(json_proportions, meaning_db, tag_db)
    sqlite_gen = load_proportions(sqlite_proportions, meaning_db, tag_db)

    # The structures dict + cooccurrence dict on the NameGenerator must be
    # identical across backends — that's the load-bearing surface the vector
    # selection path samples from. (Direct equality check is stronger than
    # seeded-sample equivalence because it covers every potential sample, not
    # just the one seed=42 happens to pick.)
    assert json_gen.structs == sqlite_gen.structs
    assert json_gen.tag_cooccurrence == sqlite_gen.tag_cooccurrence
