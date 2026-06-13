"""D45 (wyrd-aicu.5) data-gate: no dash in any stored morpheme identity.

Two layers of enforcement, both pinned here:

1. **Exporter guard** — ``_verify_no_dashed_identity`` runs inside
   ``write_runtime_db`` on every emit and RAISES if any guarded
   stored-identity surface carries a dash, so a regression fails the
   BUILD loudly (the structural, un-regressable enforcement of D45).
2. **Committed-artifact gate** — the shipped ``seed-runtime.db`` is
   asserted dash-free directly, so a stale / hand-edited committed seed
   that bypassed a fresh export is still caught.

Guarded surfaces (D45): ``meaning.usage_key``, the four proportions
``usage_key`` columns, and the meaning/morpheme blob ``modern_usage``
values. ``morpheme_id`` is deliberately EXEMPT — it is the L3-derived
content key (``language:form``), de-dashed by wyrd-aicu.3; this gate
gains it then.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _DASH_GUARD_BLOB_TABLES,
    _DASH_GUARD_KEY_COLUMNS,
    _verify_no_dashed_identity,
)
from wyrd.generators.kenning.runtime.runtime_db import _bundled_seed_path

pytestmark = pytest.mark.no_lexicon_isolation


# ---- committed-artifact gate ----------------------------------------------


def test_committed_seed_has_no_dashed_identity():
    """The shipped seed-runtime.db carries no dash in any guarded
    stored-identity surface. Catches a stale / hand-edited committed seed
    that skipped a fresh export (the exporter guard only fires on emit)."""
    conn = sqlite3.connect(f"file:{_bundled_seed_path()}?mode=ro", uri=True)
    try:
        # Key columns — exact (short bare surfaces).
        for table, col in _DASH_GUARD_KEY_COLUMNS:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE '%-%'").fetchone()[0]
            assert n == 0, f"{table}.{col} has {n} dash-marked rows in the committed seed"
        # Blob modern_usage — JSON parse (a greedy LIKE false-positives on
        # dashes in other fields, e.g. compound headwords like lēac-tūn).
        for table in _DASH_GUARD_BLOB_TABLES:
            dashed = [
                entry.get("word", {}).get("modern_usage")
                for (data,) in conn.execute(f"SELECT data FROM {table}")
                for entry in json.loads(data).get("entries", [])
                if "-" in (entry.get("word", {}).get("modern_usage") or "")
            ]
            assert not dashed, f"{table} blob has dash-marked modern_usage: {dashed[:5]}"
    finally:
        conn.close()


# ---- exporter guard -------------------------------------------------------


def _guard_db() -> sqlite3.Connection:
    """An in-memory DB with the guarded tables, all initially clean."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meaning (usage_key TEXT, data BLOB)")
    conn.execute("CREATE TABLE proportions_usage (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_single_usage (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_attested_language (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_bare_word_position (usage_key TEXT)")
    conn.execute("CREATE TABLE morpheme (data BLOB)")
    bare = json.dumps({"entries": [{"word": {"modern_usage": "ton"}}]})
    conn.execute("INSERT INTO meaning VALUES ('ton', ?)", (bare,))
    conn.execute("INSERT INTO proportions_usage VALUES ('ton')")
    conn.execute("INSERT INTO morpheme VALUES (?)", (bare,))
    return conn


def test_guard_passes_on_clean_db():
    conn = _guard_db()
    _verify_no_dashed_identity(conn)  # no raise
    conn.close()


def test_guard_raises_on_dashed_usage_key():
    conn = _guard_db()
    conn.execute("INSERT INTO proportions_usage VALUES ('-ton')")
    with pytest.raises(RuntimeError, match=r"proportions_usage.usage_key: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_raises_on_dashed_meaning_key():
    conn = _guard_db()
    conn.execute(
        "INSERT INTO meaning VALUES ('-ham', ?)",
        (json.dumps({"entries": []}),),
    )
    with pytest.raises(RuntimeError, match=r"meaning.usage_key: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_raises_on_dashed_blob_modern_usage():
    """The blob check JSON-parses (not a greedy LIKE): a dashed modern_usage
    inside the data blob is caught even though the row's key is bare."""
    conn = _guard_db()
    dashed = json.dumps({"entries": [{"word": {"modern_usage": "-ham-"}}]})
    conn.execute("INSERT INTO morpheme VALUES (?)", (dashed,))
    with pytest.raises(RuntimeError, match=r"morpheme blob modern_usage: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_ignores_internal_hyphen_in_other_blob_fields():
    """A hyphen in a NON-identity field (a compound source form like
    ``lēac-tūn``) must not trip the guard — only modern_usage is the
    identity. Guards against a greedy-LIKE false positive."""
    conn = _guard_db()
    ok = json.dumps(
        {"entries": [{"word": {"modern_usage": "leighton", "old_english": ["lēac-tūn"]}}]}
    )
    conn.execute("INSERT INTO meaning VALUES ('leighton', ?)", (ok,))
    _verify_no_dashed_identity(conn)  # no raise — the dash is in old_english, not the identity
    conn.close()
