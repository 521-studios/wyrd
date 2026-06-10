"""wyrd-g1wp: SnapStart preload + after-restore connection handling."""

from __future__ import annotations

import sys

import pytest

from wyrd import snapstart
from wyrd.generators.kenning.runtime import runtime_db


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Each test mutates process-global state — the runtime_db caches, the
    snapstart idempotency flag, and snapshot_restore_py's hook registry. Reset
    all three after every test so they're order-independent and don't leak."""
    yield
    runtime_db.reset_runtime_db_cache()
    snapstart._hooks_registered = False
    try:
        import snapshot_restore_py

        snapshot_restore_py._after_restore_registry.clear()
    except ImportError:
        pass


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


def test_preload_runtime_warms_every_default_path_cache(monkeypatch):
    """Under snap-start, preload warms the DB connection, version stamp, AND
    every lru-cache the default generate() path touches — including
    _load_empirical_priors, which calls get_runtime_db() directly (B1)."""
    from wyrd.generators import kenning as kg

    runtime_db.reset_runtime_db_cache()
    kg._load_empirical_priors.cache_clear()
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "snap-start")
    snapstart.preload_runtime()
    assert runtime_db._conn is not None
    assert runtime_db._bundle_version_cache is not None
    # the loader that re-resolves the DB directly must be warm
    assert kg._load_empirical_priors.cache_info().currsize == 1


def test_post_restore_generate_does_not_reresolve_db(monkeypatch):
    """B1 — the property that actually matters: after preload + the after-restore
    connection drop, a full default generate() must trigger ZERO DB path
    re-resolves (each of which is a 46MB S3 re-download on Lambda). Catches any
    default-path loader that bypasses the warm caches and hits get_runtime_db()."""
    runtime_db.reset_runtime_db_cache()
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "snap-start")
    snapstart.preload_runtime()  # warm the snapshot
    runtime_db.reset_runtime_db_connection()  # simulate the after-restore hook

    resolves: list[int] = []
    orig = runtime_db._resolve_db_path
    monkeypatch.setattr(runtime_db, "_resolve_db_path", lambda: (resolves.append(1), orig())[1])

    from wyrd.generators.kenning.generators.kenning import Kenning

    Kenning().generate({"culture": "english"}, seed=1)
    assert resolves == [], "post-restore generate re-resolved the DB (would re-download on Lambda)"


def test_register_snapstart_hooks_noop_without_runtime_lib(monkeypatch):
    """Off SnapStart, snapshot_restore_py isn't importable; registration must be
    a silent no-op. The lib IS a real dep here, so force the ImportError."""
    monkeypatch.setitem(sys.modules, "snapshot_restore_py", None)  # → ImportError on import
    snapstart._hooks_registered = False
    snapstart.register_snapstart_hooks()  # must not raise, must register nothing
    assert snapstart._hooks_registered is False


def test_register_snapstart_hooks_registers_once_and_is_idempotent():
    """Registers exactly one after-restore hook, and repeat calls (create_app may
    run more than once) don't append duplicates."""
    import snapshot_restore_py

    snapshot_restore_py._after_restore_registry.clear()
    snapstart._hooks_registered = False

    snapstart.register_snapstart_hooks()
    snapstart.register_snapstart_hooks()  # idempotent
    assert len(snapshot_restore_py.get_after_restore()) == 1


def test_after_restore_hook_drops_connection_keeps_caches():
    """Invoke the actually-registered after-restore callback and confirm its body
    nulls the stale connection while KEEPING the version cache — so a restored
    container serves from memory, no re-download."""
    import snapshot_restore_py

    snapshot_restore_py._after_restore_registry.clear()
    snapstart._hooks_registered = False
    snapstart.register_snapstart_hooks()

    runtime_db.get_runtime_db()
    runtime_db.bundle_version()
    assert runtime_db._conn is not None and runtime_db._bundle_version_cache is not None

    for fn, args, kwargs in snapshot_restore_py.get_after_restore():
        fn(*args, **kwargs)

    assert runtime_db._conn is None  # stale connection dropped
    assert runtime_db._bundle_version_cache is not None  # parsed cache kept


def test_reset_runtime_db_connection_keeps_version_cache():
    """Unit: the connection-only reset nulls _conn but keeps _bundle_version_cache
    (distinct from reset_runtime_db_cache, which drops both)."""
    runtime_db.get_runtime_db()
    runtime_db.bundle_version()
    assert runtime_db._conn is not None
    assert runtime_db._bundle_version_cache is not None

    runtime_db.reset_runtime_db_connection()
    assert runtime_db._conn is None
    assert runtime_db._bundle_version_cache is not None
