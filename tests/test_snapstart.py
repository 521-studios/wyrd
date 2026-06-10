"""wyrd-g1wp: SnapStart preload + after-restore connection handling."""

from __future__ import annotations

import pytest

from wyrd import snapstart
from wyrd.generators.kenning.runtime import runtime_db


@pytest.fixture(autouse=True)
def _restore_runtime_db_cache():
    """Each test mutates the process-global runtime_db caches; reset after so
    later tests get a clean lazy-resolve."""
    yield
    runtime_db.reset_runtime_db_cache()


def test_preload_runtime_noop_unless_snapstart(monkeypatch):
    """preload only runs when AWS_LAMBDA_INITIALIZATION_TYPE=snap-start — a
    normal (on-demand) init or local/test run is a no-op, so the lazy
    per-request load is unchanged and nothing is warmed."""
    runtime_db.reset_runtime_db_cache()
    monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
    snapstart.preload_runtime()
    assert runtime_db._conn is None  # nothing loaded

    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "on-demand")
    snapstart.preload_runtime()
    assert runtime_db._conn is None  # still a no-op for on-demand


def test_preload_runtime_warms_caches_under_snapstart(monkeypatch):
    """Under snap-start init, preload opens the DB + warms the parsed caches so
    they land in the snapshot."""
    runtime_db.reset_runtime_db_cache()
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "snap-start")
    snapstart.preload_runtime()
    assert runtime_db._conn is not None
    assert runtime_db._bundle_version_cache is not None  # version stamp warmed


def test_register_snapstart_hooks_noop_without_runtime_lib():
    """Off SnapStart the snapshot_restore_py lib isn't importable; hook
    registration must be a silent no-op, not an error."""
    snapstart.register_snapstart_hooks()  # must not raise


def test_reset_runtime_db_connection_keeps_version_cache():
    """The after-restore reset nulls the stale connection but KEEPS the parsed
    version cache — so a restored container serves from memory and never
    re-downloads the DB (the whole point of the SnapStart connection handling)."""
    runtime_db.get_runtime_db()
    runtime_db.bundle_version()
    assert runtime_db._conn is not None
    assert runtime_db._bundle_version_cache is not None

    runtime_db.reset_runtime_db_connection()
    assert runtime_db._conn is None  # stale connection dropped
    assert runtime_db._bundle_version_cache is not None  # parsed cache kept
