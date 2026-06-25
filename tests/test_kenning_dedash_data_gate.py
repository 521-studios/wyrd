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
``usage_key`` columns, the meaning/morpheme blob ``modern_usage`` values,
and — since wyrd-aicu.8 — the ``morpheme_id`` content key's FORM part
(``language:form``; boundary dash only, because the language prefix
legitimately carries a dash like ``old-english:`` and an interior in-word
hyphen is legitimate like ``al-Quadim``).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _ANY_DASH_GLOB,
    _DASH_CHARS,
    _DASH_GUARD_BLOB_TABLES,
    _DASH_GUARD_KEY_COLUMNS,
    _MORPHEME_ID_FORM_DASH_SQL,
    _verify_no_dashed_identity,
)
from wyrd.generators.kenning.runtime.runtime_db import _bundled_seed_path

pytestmark = pytest.mark.no_lexicon_isolation


# ---- committed-artifact gate ----------------------------------------------


def test_committed_seed_has_no_dashed_identity():
    """The shipped seed-runtime.db carries no dash in any guarded
    stored-identity surface. Catches a stale / hand-edited committed seed
    that skipped a fresh export (the exporter guard only fires on emit)."""
    db_uri = f"{_bundled_seed_path().absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        # Key columns — exact (short bare surfaces). Any dash variant (GLOB).
        for table, col in _DASH_GUARD_KEY_COLUMNS:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} GLOB ?", (_ANY_DASH_GLOB,)
            ).fetchone()[0]
            assert n == 0, f"{table}.{col} has {n} dash-marked rows in the committed seed"
        # morpheme_id FORM part — boundary dash only (the language prefix +
        # interior hyphens are legitimate; wyrd-aicu.8).
        n = conn.execute(_MORPHEME_ID_FORM_DASH_SQL).fetchone()[0]
        assert n == 0, f"morpheme.morpheme_id has {n} boundary-dashed forms in the committed seed"
        # Blob modern_usage — JSON parse (a greedy LIKE false-positives on
        # dashes in other fields, e.g. compound headwords like lēac-tūn). Any variant.
        for table in _DASH_GUARD_BLOB_TABLES:
            dashed = [
                (entry.get("word") or {}).get("modern_usage")
                for (data,) in conn.execute(f"SELECT data FROM {table}")
                for entry in json.loads(data).get("entries", [])
                if _DASH_CHARS.intersection((entry.get("word") or {}).get("modern_usage") or "")
            ]
            assert not dashed, f"{table} blob has dash-marked modern_usage: {dashed[:5]}"
    finally:
        conn.close()


# ---- exporter guard -------------------------------------------------------


def _blob(payload: dict) -> bytes:
    """Serialize a blob EXACTLY as the exporter does — json.dumps(...) then
    ``.encode("utf-8")``. Inserting bytes (not str) is load-bearing: the
    ``data`` column is BLOB, so the guard must work against the *blob* storage
    class. A str insert lands as TEXT and would let a storage-class bug (e.g. a
    SQLite ``LIKE`` prefilter that silently skips blobs) pass undetected."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _guard_db() -> sqlite3.Connection:
    """An in-memory DB with the guarded tables, all initially clean. Blob rows
    are inserted as BYTES to mirror the exporter's real BLOB storage class."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meaning (usage_key TEXT, data BLOB NOT NULL)")
    conn.execute("CREATE TABLE proportions_usage (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_single_usage (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_attested_language (usage_key TEXT)")
    conn.execute("CREATE TABLE proportions_bare_word_position (usage_key TEXT)")
    conn.execute("CREATE TABLE morpheme (morpheme_id TEXT, data BLOB NOT NULL)")
    bare = _blob({"entries": [{"word": {"modern_usage": "ton"}}]})
    conn.execute("INSERT INTO meaning VALUES ('ton', ?)", (bare,))
    conn.execute("INSERT INTO proportions_usage VALUES ('ton')")
    conn.execute("INSERT INTO morpheme (morpheme_id, data) VALUES ('old-english:ton', ?)", (bare,))
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
        (_blob({"entries": []}),),
    )
    with pytest.raises(RuntimeError, match=r"meaning.usage_key: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_raises_on_dashed_blob_modern_usage():
    """The blob check JSON-parses (not a greedy LIKE): a dashed modern_usage
    inside the data blob is caught even though the row's key is bare."""
    conn = _guard_db()
    dashed = _blob({"entries": [{"word": {"modern_usage": "-ham-"}}]})
    conn.execute("INSERT INTO morpheme (data) VALUES (?)", (dashed,))
    with pytest.raises(RuntimeError, match=r"morpheme blob identity: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_catches_dash_in_blob_storage_class():
    """Regression (round-3 finding): the guard must catch a dashed modern_usage
    even though ``data`` is the BLOB storage class. A SQLite ``LIKE`` prefilter
    silently matches ZERO blob rows (LIKE with a TEXT pattern never matches a
    blob), which would neuter the guard while every str-inserting test stayed
    green. Pin it: insert BYTES, assert the row really is stored as 'blob', and
    assert the guard still raises."""
    conn = _guard_db()
    conn.execute(
        "INSERT INTO morpheme (data) VALUES (?)",
        (_blob({"entries": [{"word": {"modern_usage": "-ton"}}]}),),
    )
    stored_classes = {row[0] for row in conn.execute("SELECT typeof(data) FROM morpheme")}
    assert stored_classes == {"blob"}, f"test must exercise blob storage, got {stored_classes}"
    with pytest.raises(RuntimeError, match=r"morpheme blob identity: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_ignores_internal_hyphen_in_other_blob_fields():
    """An INTERIOR hyphen on a language-form surface (a compound like ``lēac-tūn``)
    must not trip the guard — D45 forbids only the affix-position BOUNDARY dash; an
    in-word hyphen (``al-Quadim``) is legitimate identity. Guards against a greedy
    false positive."""
    conn = _guard_db()
    ok = _blob({"entries": [{"word": {"modern_usage": "leighton", "old_english": ["lēac-tūn"]}}]})
    conn.execute("INSERT INTO meaning VALUES ('leighton', ?)", (ok,))
    _verify_no_dashed_identity(conn)  # no raise — interior hyphen, not a boundary dash
    conn.close()


def test_guard_raises_on_boundary_dashed_blob_form():
    """Regression: the blob IDENTITY is not just ``modern_usage`` — each word's
    language-form ``form`` / ``canonical_form`` is a stored morpheme surface too.
    A BOUNDARY-dashed form (an affix etymon like ``-tun`` that became reflex-reachable
    and exported into a word blob) must trip the guard, even when modern_usage is
    bare. Before this, a dashed identity could ship to prod with the guard silent."""
    conn = _guard_db()
    dashed = _blob(
        {"entries": [{"word": {"modern_usage": "ton", "old-english": [{"form": "-tun"}]}}]}
    )
    conn.execute("INSERT INTO morpheme (data) VALUES (?)", (dashed,))
    with pytest.raises(RuntimeError, match=r"morpheme blob identity: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_raises_on_boundary_dashed_morpheme_id_form():
    """A boundary (leading/trailing) dash on the morpheme_id FORM part — the
    affix-position decoration D45 forbids — trips the guard (wyrd-aicu.8)."""
    conn = _guard_db()
    conn.execute(
        "INSERT INTO morpheme (morpheme_id, data) VALUES ('celtic:-ach', ?)",
        (_blob({"entries": []}),),
    )
    with pytest.raises(RuntimeError, match=r"morpheme.morpheme_id form: 1 boundary-dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_allows_dashed_language_and_interior_hyphen_in_morpheme_id():
    """The morpheme_id guard checks only the FORM's boundary: a dash in the
    LANGUAGE prefix (``old-english:`` — the clean fixture row) and an interior
    in-word hyphen (``arabic:al-Quadim``) are legitimate and must NOT trip it
    (wyrd-aicu.8). A blanket ``morpheme_id LIKE '%-%'`` would wrongly flag both."""
    conn = _guard_db()  # clean row already carries 'old-english:ton' (dashed language)
    conn.execute(
        "INSERT INTO morpheme (morpheme_id, data) VALUES ('arabic:al-Quadim', ?)",
        (_blob({"entries": []}),),
    )
    conn.execute(
        "INSERT INTO morpheme (morpheme_id, data) VALUES ('proto-indo-european:tūn', ?)",
        (_blob({"entries": []}),),
    )
    _verify_no_dashed_identity(conn)  # no raise — only the form boundary matters
    conn.close()


# ---- Unicode-dash variants (#782): the de-dash this guard backstops is
# Unicode-aware (#769), so an en-dash / U+2010 boundary marker that slipped past
# a missed de-dash must ALSO trip the guard. Pre-fix (ASCII-only LIKE / "-")
# these all escaped it — the guard's whole point is to fail the build on exactly
# this regression class.


def test_guard_raises_on_unicode_dashed_usage_key():
    """An en-dash (U+2013) boundary marker on a key column trips the guard —
    the old ``LIKE '%-%'`` (ASCII hyphen only) silently missed it."""
    conn = _guard_db()
    conn.execute("INSERT INTO proportions_usage VALUES ('–ton')")  # en-dash
    with pytest.raises(RuntimeError, match=r"proportions_usage.usage_key: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_raises_on_unicode_boundary_dashed_morpheme_id_form():
    """A U+2010 boundary dash on the morpheme_id FORM part trips the guard;
    an interior Unicode dash (legitimate, like ``al‑Quadim``) does NOT."""
    conn = _guard_db()
    conn.execute(
        "INSERT INTO morpheme (morpheme_id, data) VALUES ('celtic:‐ach', ?)",  # U+2010 lead
        (_blob({"entries": []}),),
    )
    with pytest.raises(RuntimeError, match=r"morpheme.morpheme_id form: 1 boundary-dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()


def test_guard_allows_interior_unicode_hyphen_in_morpheme_id_form():
    """An INTERIOR Unicode dash (a typographic ``al‑Quadim``, U+2010) is a
    legitimate in-word hyphen, not boundary decoration — it must NOT trip the
    guard (the boundary GLOBs match only the ends)."""
    conn = _guard_db()
    conn.execute(
        "INSERT INTO morpheme (morpheme_id, data) VALUES ('arabic:al‐Quadim', ?)",
        (_blob({"entries": []}),),
    )
    _verify_no_dashed_identity(conn)  # no raise — interior, not boundary
    conn.close()


def test_guard_raises_on_unicode_dashed_blob_modern_usage():
    """An en-dash in a blob ``modern_usage`` identity trips the guard even
    though the prefilter on the RAW bytes must match the UTF-8 dash encoding
    (``b"-"`` alone would skip the row and neuter the guard)."""
    conn = _guard_db()
    dashed = _blob({"entries": [{"word": {"modern_usage": "–ham–"}}]})  # en-dash both ends
    conn.execute("INSERT INTO morpheme (data) VALUES (?)", (dashed,))
    # The raw blob really carries no ASCII hyphen — the old prefilter would skip it.
    assert b"-" not in dashed
    with pytest.raises(RuntimeError, match=r"morpheme blob identity: 1 dashed"):
        _verify_no_dashed_identity(conn)
    conn.close()
