"""D53 (wyrd-x5y4) data-gate: no uppercase in any stored morpheme identity key.

Two layers, mirroring the D45 dash gate (``test_kenning_dedash_data_gate.py``):

1. **Exporter guard** — ``_verify_no_capitalized_identity`` runs inside
   ``write_runtime_db`` on every emit and RAISES if any guarded surface-identity
   key carries an uppercase letter, so a mint regression that stopped
   case-folding fails the BUILD (the structural enforcement of D53). It is active
   and PASSING now — the mint (wyrd-x5y4.2/.3) lower-cases every surface key.

2. **Committed-artifact gate** — the shipped ``seed-runtime.db`` is asserted
   uppercase-free. This currently **XFAILs (strict)**: the committed seed
   predates the case-fold reseed (wyrd-x5y4.5) and still carries ~657 capitalized
   ``meaning`` keys + the ``proportions`` caps. When the reseed lands and the
   seed conforms, the test XPASSes → the strict marker FAILS the suite, the
   signal to drop the xfail and make this a live gate.

Guarded surfaces (D53): the same key columns as the dash gate —
``meaning.usage_key`` + the four ``proportions`` ``usage_key`` columns
(``fantasy_morpheme.usage_key`` is a whole-input-NAME lookup key, outside
D45/D53 just as for dashes).
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _CASE_GUARD_KEY_COLUMNS,
    _verify_no_capitalized_identity,
)
from wyrd.generators.kenning.runtime.runtime_db import _bundled_seed_path

pytestmark = pytest.mark.no_lexicon_isolation


# ---- committed-artifact gate (xfail-strict until the wyrd-x5y4.5 reseed) ----


@pytest.mark.xfail(
    strict=True,
    reason="committed seed predates the wyrd-x5y4.5 case-fold reseed (~657 "
    "capitalized meaning keys + proportions caps). It XPASSes once the reseed "
    "makes the committed data conform — that strict-xfail failure is the signal "
    "to drop this marker and let the gate go live.",
)
def test_committed_seed_has_no_capitalized_identity():
    """The shipped seed-runtime.db carries no uppercase in any guarded
    surface-identity key. Unicode-aware (``key != key.lower()``, matching the
    mint's ``str.lower()``) so an OE capital thorn could not slip an ASCII-only
    check."""
    db_uri = f"{_bundled_seed_path().absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        for table, col in _CASE_GUARD_KEY_COLUMNS:
            n = sum(
                1
                for (key,) in conn.execute(f"SELECT {col} FROM {table}")
                if key and key != key.lower()
            )
            assert n == 0, f"{table}.{col} has {n} capitalized rows in the committed seed"
    finally:
        conn.close()


# ---- exporter guard (active + gating now) ----------------------------------


def _empty_guard_tables(conn: sqlite3.Connection) -> None:
    """Create the five guarded key tables so the guard's per-table SELECT runs."""
    for table, _col in _CASE_GUARD_KEY_COLUMNS:
        conn.execute(f"CREATE TABLE {table} (usage_key TEXT)")


def test_exporter_guard_raises_on_capitalized_key():
    """The guard refuses to emit a capitalized surface-identity key (a 'Abbey'
    that escaped the mint fold), naming the offending column."""
    conn = sqlite3.connect(":memory:")
    try:
        _empty_guard_tables(conn)
        conn.execute("INSERT INTO meaning (usage_key) VALUES ('abbey'), ('Abbey')")
        with pytest.raises(RuntimeError, match=r"D53.*meaning\.usage_key: 1 capitalized"):
            _verify_no_capitalized_identity(conn)
    finally:
        conn.close()


def test_exporter_guard_passes_on_folded_keys():
    """A fully lower-cased pool emits clean — including a Unicode lower (þ) and an
    interior-space multi-word surface ('mac leod')."""
    conn = sqlite3.connect(":memory:")
    try:
        _empty_guard_tables(conn)
        conn.execute("INSERT INTO meaning (usage_key) VALUES ('abbey'), ('þorn'), ('mac leod')")
        conn.execute("INSERT INTO proportions_usage (usage_key) VALUES ('ton')")
        _verify_no_capitalized_identity(conn)  # must not raise
    finally:
        conn.close()


def test_exporter_guard_catches_unicode_uppercase():
    """An OE capital thorn (Þ) leak is caught. SQLite ``LOWER`` / ``GLOB [A-Z]``
    only fold ASCII, so an ASCII-only guard would miss it — the Python
    ``key != key.lower()`` check does not."""
    conn = sqlite3.connect(":memory:")
    try:
        _empty_guard_tables(conn)
        conn.execute("INSERT INTO proportions_usage (usage_key) VALUES ('Þorn')")
        with pytest.raises(RuntimeError, match="capitalized"):
            _verify_no_capitalized_identity(conn)
    finally:
        conn.close()
