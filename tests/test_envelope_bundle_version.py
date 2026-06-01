"""Envelope + registry bundle_version stamp (wyrd-dsl5).

The API envelope carries a top-level ``bundle_version`` ONLY when the
generator supplies one via ``runtime_version()`` — keeping the response
shape uniform for generators with no versioned data layer (which return
None by default).
"""

from __future__ import annotations

from typing import Any

from wyrd.envelope import envelope
from wyrd.registry import GenerationResult, Generator


def _results():
    return [GenerationResult(result="Foo")]


def test_envelope_omits_bundle_version_when_none():
    out = envelope(generator="g", parameters={}, seed=1, results=_results())
    assert "bundle_version" not in out


def test_envelope_includes_bundle_version_when_supplied():
    bv = {"built_at": "2026-05-30T00:00:00Z", "schema_version": "2"}
    out = envelope(generator="g", parameters={}, seed=1, results=_results(), bundle_version=bv)
    assert out["bundle_version"] == bv


def test_generator_runtime_version_defaults_to_none():
    class _Dummy(Generator):
        name = "dummy"
        display_name = "Dummy"
        description = "test"

        def input_schema(self) -> dict[str, Any]:
            return {}

        def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
            return GenerationResult(result="x")

    assert _Dummy().runtime_version() is None


# --- kenning bundle_version() reader + end-to-end (wyrd-dsl5) ---------------


def test_bundle_version_reads_metadata(monkeypatch):
    import sqlite3

    from wyrd.generators.kenning.runtime import runtime_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE bundle_metadata (key TEXT, value TEXT)")
    conn.executemany(
        "INSERT INTO bundle_metadata VALUES (?, ?)",
        [("schema_version", "2"), ("built_at", "2026-05-30T00:00:00Z")],
    )
    conn.commit()
    monkeypatch.setattr(runtime_db, "get_runtime_db", lambda: conn)
    runtime_db.reset_runtime_db_cache()
    try:
        bv = runtime_db.bundle_version()
        assert bv["schema_version"] == "2"
        assert bv["built_at"] == "2026-05-30T00:00:00Z"
        # Cache hit — same object, no re-query.
        assert runtime_db.bundle_version() is bv
    finally:
        runtime_db.reset_runtime_db_cache()  # don't leak the stub cache


def test_bundle_version_missing_table_returns_empty(monkeypatch):
    import sqlite3

    from wyrd.generators.kenning.runtime import runtime_db

    conn = sqlite3.connect(":memory:")  # no bundle_metadata table
    monkeypatch.setattr(runtime_db, "get_runtime_db", lambda: conn)
    runtime_db.reset_runtime_db_cache()
    try:
        assert runtime_db.bundle_version() == {}
    finally:
        runtime_db.reset_runtime_db_cache()


def test_bundle_version_cold_does_not_deadlock(tmp_path, monkeypatch):
    # Regression: bundle_version() must not call get_runtime_db() while holding
    # the non-reentrant _conn_lock. Exercises the REAL lock path on a cold
    # connection (no get_runtime_db monkeypatch). Runs in a daemon thread with
    # a join timeout so a re-entrancy regression fails the assertion instead of
    # hanging the whole suite.
    import sqlite3
    import threading

    from wyrd.generators.kenning.runtime import runtime_db

    db = tmp_path / "rt.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE bundle_metadata (key TEXT, value TEXT)")
    conn.executemany(
        "INSERT INTO bundle_metadata VALUES (?, ?)",
        [
            ("schema_version", "2"),  # must match _EXPECTED_SCHEMA_VERSION
            ("built_at", "2026-05-30T00:00:00Z"),
            ("emitter_version", "test"),
            ("source_lexicon_db", "test"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("WYRD_RUNTIME_DB", str(db))
    monkeypatch.delenv("WYRD_RUNTIME_DB_BUCKET", raising=False)
    runtime_db.reset_runtime_db_cache()  # force a cold _conn so get_runtime_db re-resolves

    result: dict = {}

    def _call():
        result["bv"] = runtime_db.bundle_version()

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=10)
    try:
        assert not t.is_alive(), "bundle_version() deadlocked on a cold connection"
        assert result["bv"]["schema_version"] == "2"
    finally:
        runtime_db.reset_runtime_db_cache()


def test_kenning_roll_stamps_bundle_version():
    # End-to-end: the kenning runtime_version binding + bundle_version() reader
    # surface on a real roll envelope (uses the committed dev seed DB).
    from wyrd.app import create_app

    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/kenning", json={"culture": "english", "count": 1})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    bv = resp.get_json().get("bundle_version")
    assert isinstance(bv, dict) and "schema_version" in bv


def test_dispatch_swallows_runtime_version_error(monkeypatch):
    # A generator whose runtime_version() raises must not sink the roll; the
    # envelope just omits bundle_version.
    from wyrd import registry
    from wyrd.app import create_app

    app = create_app()
    gen = registry.get("kenning")

    def _boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(gen), "runtime_version", _boom)
    with app.test_client() as client:
        resp = client.post("/api/kenning", json={"culture": "english", "count": 1})
    assert resp.status_code == 200
    assert "bundle_version" not in resp.get_json()
