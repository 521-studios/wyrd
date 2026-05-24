"""Tests for ``runtime/runtime_db.py`` loader (wyrd-d90t PR 4).

Covers the four-step resolution order (env var → /tmp cache → S3
download → bundled seed), lazy + cached connection behavior, and the
read-only sqlite3 contract.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from wyrd.generators.kenning.runtime import runtime_db as mod
from wyrd.generators.kenning.runtime.runtime_db import (
    ENV_AWS_PROFILE,
    ENV_BUCKET,
    ENV_LOCAL_PATH,
    get_runtime_db,
    reset_runtime_db_cache,
)

TEST_BUCKET = "kenning-runtime-test"


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Every test starts with a fresh module-cached connection — the
    process-wide cache would otherwise leak across tests and produce
    confusing 'why did this test see the previous test's DB' bugs."""
    reset_runtime_db_cache()
    yield
    reset_runtime_db_cache()


@pytest.fixture
def _clear_env(monkeypatch):
    """Strip every WYRD_RUNTIME_DB* env var the loader cares about so
    tests start from a known state. Individual tests opt into specific
    overrides on top of this baseline."""
    for name in (ENV_LOCAL_PATH, ENV_BUCKET, ENV_AWS_PROFILE):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _tiny_db(tmp_path: Path) -> Path:
    """A minimal valid sqlite3 DB the loader can open read-only."""
    path = tmp_path / "fixture.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("CREATE TABLE meaning(k TEXT); INSERT INTO meaning VALUES('hi');")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def fake_s3(monkeypatch):
    """moto S3 fake + fake AWS creds so boto3's credential resolution
    doesn't hit operator SSO."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=TEST_BUCKET)
        yield s3


# ---------- env var resolution ----------


def test_env_var_path_wins(_clear_env, _tiny_db, monkeypatch) -> None:
    """WYRD_RUNTIME_DB is the highest-priority resolution path —
    completely bypasses S3 + the bundled seed."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    conn = get_runtime_db()
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"


def test_env_var_missing_file_raises(_clear_env, tmp_path, monkeypatch) -> None:
    """Operator typo on WYRD_RUNTIME_DB should fail loudly at the
    loader, not produce a confusing 'unable to open database' deep in
    sqlite."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(tmp_path / "missing.db"))
    with pytest.raises(FileNotFoundError, match=ENV_LOCAL_PATH):
        get_runtime_db()


# ---------- bundled seed fallback ----------


def test_falls_back_to_bundled_seed(_clear_env) -> None:
    """With no env var + no bucket, the loader resolves to the
    committed seed-runtime.db. Validates the offline-dev path that
    the test suite + local dev rely on."""
    conn = get_runtime_db()
    # The bundled seed has the L4 schema; meaning table exists.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


# ---------- read-only contract ----------


def test_connection_is_read_only(_clear_env, _tiny_db, monkeypatch) -> None:
    """The connection MUST reject writes — the L4 DB is treated as
    immutable for the container's lifetime."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    conn = get_runtime_db()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO meaning VALUES('write')")


# ---------- caching ----------


def test_returns_same_connection_across_calls(_clear_env, _tiny_db, monkeypatch) -> None:
    """Lazy + idempotent contract: two calls return the same handle.
    Warm-Lambda reuse depends on this."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    a = get_runtime_db()
    b = get_runtime_db()
    assert a is b


def test_reset_cache_returns_fresh_connection(_clear_env, _tiny_db, monkeypatch) -> None:
    """reset_runtime_db_cache drops the cached handle so the next
    call re-opens. Tests + a future hot-reload hook both depend on
    this."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    a = get_runtime_db()
    reset_runtime_db_cache()
    b = get_runtime_db()
    assert a is not b


# ---------- S3 download ----------


def test_downloads_from_s3_when_bucket_set(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """With WYRD_RUNTIME_DB_BUCKET set + no env-var local path, the
    loader fetches current.json and downloads the referenced
    versioned key to /tmp under the etag-named filename."""
    # Seed S3 with a versioned key + pointer.
    version_key = "v/2026-05-24T20-30Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=_tiny_db.read_bytes())
    pointer = json.dumps({"key": version_key, "etag": "abc123"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    # Direct the cache dir into tmp_path so concurrent test runs don't
    # collide on /tmp.
    cache_dir = Path(_tiny_db).parent
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_DIR", str(cache_dir))

    conn = get_runtime_db()
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"
    # File is named after the etag.
    assert (cache_dir / "wyrd-runtime-abc123.db").is_file()


def test_etag_match_skips_download(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """When the etag-named cache file already exists, no download
    happens — existence IS the etag-match check now that the filename
    encodes the etag."""
    version_key = "v/2026-05-24T20-30Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=b"stale-from-s3")
    pointer = json.dumps({"key": version_key, "etag": "cached-etag"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    cache_dir = Path(_tiny_db).parent
    # Pre-populate the etag-named cache file with a valid sqlite DB.
    (cache_dir / "wyrd-runtime-cached-etag.db").write_bytes(_tiny_db.read_bytes())

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_DIR", str(cache_dir))

    conn = get_runtime_db()
    # If we accidentally re-downloaded, the etag-named file would now
    # hold "stale-from-s3" which isn't a valid sqlite file. Since the
    # query succeeds, the short-circuit worked.
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"


def test_etag_mismatch_redownloads(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """A new pointer etag should produce a new etag-named cache file —
    the old one stays untouched (so concurrent readers with
    immutable=1 connections on it don't see bytes change under
    them)."""
    version_key = "v/2026-05-24T21-00Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=_tiny_db.read_bytes())
    pointer = json.dumps({"key": version_key, "etag": "fresh-etag"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    cache_dir = Path(_tiny_db).parent
    # Stale cache under a different etag name.
    stale = cache_dir / "wyrd-runtime-stale-etag.db"
    stale.write_bytes(b"stale bytes")

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_DIR", str(cache_dir))

    conn = get_runtime_db()
    # New download lands under the fresh-etag filename; the stale
    # file stays untouched (no overwrite, no race for any reader
    # that had it open).
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"
    assert (cache_dir / "wyrd-runtime-fresh-etag.db").is_file()
    # Stale file untouched — that's the race-avoidance property.
    assert stale.read_bytes() == b"stale bytes"


def test_s3_failure_falls_back_to_bundled_seed(_clear_env, fake_s3, monkeypatch) -> None:
    """If WYRD_RUNTIME_DB_BUCKET is set but the S3 call fails (no
    creds / wrong bucket / corrupt pointer), the loader should fall
    back to the bundled seed rather than hard-crashing the runtime.

    Uses moto + a nonexistent key in the test bucket to produce a
    deterministic ClientError without depending on real network
    behavior. (A previous version of this test relied on
    nonexistent-bucket-name DNS failures, which is offline-CI hostile
    + DNS-hang prone.)"""
    # Bucket exists (fake_s3 created it) but current.json doesn't, so
    # get_object raises NoSuchKey → ClientError → fallback path.
    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)

    conn = get_runtime_db()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


def test_malformed_pointer_falls_back_to_bundled_seed(_clear_env, fake_s3, monkeypatch) -> None:
    """A current.json that JSON-parses to a non-dict (null, list,
    scalar) triggers TypeError on the subsequent ['key'] access.
    Should be caught + fall through to the bundled seed, not crash
    the runtime."""
    fake_s3.put_object(
        Bucket=TEST_BUCKET,
        Key="current.json",
        Body=b"null",  # valid JSON, wrong shape
    )
    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)

    conn = get_runtime_db()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


def test_pointer_missing_required_keys_falls_back_to_bundled_seed(
    _clear_env, fake_s3, monkeypatch
) -> None:
    """A current.json missing the 'key' or 'etag' field triggers
    KeyError; same fallback path as the malformed-JSON case."""
    fake_s3.put_object(
        Bucket=TEST_BUCKET,
        Key="current.json",
        Body=b'{"only-key": "value"}',
    )
    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)

    conn = get_runtime_db()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


# ---------- string-type validation on pointer ----------


def test_pointer_etag_with_path_traversal_rejected(
    _clear_env, fake_s3, monkeypatch
) -> None:
    """An adversarial S3 supplying etag='../../etc/passwd' must NOT
    be allowed to escape /tmp via the cache-filename interpolation.
    The path-traversal guard catches it; falls back to bundled seed."""
    pointer = json.dumps({"key": "v/x.db", "etag": "../../etc/passwd"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)
    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)

    conn = get_runtime_db()
    # Fell through to bundled seed (no crash, no escaped write).
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


def test_pointer_non_string_key_falls_back_to_bundled_seed(
    _clear_env, fake_s3, monkeypatch
) -> None:
    """current.json with a numeric 'key' (or 'etag') triggers the
    explicit string-type check that we added so the failure surfaces
    in the immediate except block instead of crashing further down
    in download_file."""
    fake_s3.put_object(
        Bucket=TEST_BUCKET,
        Key="current.json",
        Body=b'{"key": 42, "etag": "abc"}',
    )
    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)

    conn = get_runtime_db()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables
