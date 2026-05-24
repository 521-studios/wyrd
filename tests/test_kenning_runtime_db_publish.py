"""Tests for ``runtime_db_publish`` + ``bin/publish-runtime-db.sh``
(wyrd-d90t PR 3).

Uses moto for offline S3 fakery — no real AWS calls. Covers:

* MD5 / version key generation
* Upload ORDER (versioned key must land before current.json so a
  reader between the two PUTs never sees a broken pointer)
* Refuses to overwrite an existing versioned key (collision = same-
  minute double-push, almost always a mistake)
* current.json atomic-flip semantics + cache headers
* --dry-run computes everything but issues no PUTs
* CLI entry point smoke + error paths
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from wyrd.generators.kenning.lexicon.runtime_db_publish import (
    CURRENT_POINTER_KEY,
    ENV_PROFILES,
    _bucket_for_env,
    _generate_version_key,
    _md5_of_file,
    _resolve_profile,
    format_summary,
    main,
    publish_runtime_db,
)

TEST_BUCKET = "kenning-runtime-test"


@pytest.fixture
def fake_s3(monkeypatch):
    """Spin up a moto S3 fake + pre-create the test bucket. Sets fake
    AWS env vars so boto3.Session() in the SUT doesn't try to refresh
    real SSO tokens — moto's mock works at the client layer but the
    session still goes through credential resolution before that. Also
    clears AWS_PROFILE so the operator's local profile (with possibly
    expired SSO) doesn't bleed in."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    # Point AWS config / credentials at empty paths so the default
    # profile's SSO config in ~/.aws/config doesn't trigger token
    # refresh under moto.
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=TEST_BUCKET)
        yield s3


@pytest.fixture
def source_db(tmp_path: Path) -> Path:
    """A small file standing in for a real L4 DB. The publish path
    doesn't crack the contents — it MD5s the bytes and uploads — so
    a few KB of arbitrary content is enough."""
    path = tmp_path / "runtime.db"
    path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 1024)
    return path


# ---------- pure helpers ----------


def test_bucket_for_env_known() -> None:
    assert _bucket_for_env("staging") == "521studios-staging-kenning-runtime"
    assert _bucket_for_env("production") == "521studios-production-kenning-runtime"


def test_bucket_for_env_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown env"):
        _bucket_for_env("not-an-env")


def test_generate_version_key_uses_utc_minute_precision() -> None:
    fixed = _dt.datetime(2026, 5, 24, 20, 30, 45, tzinfo=_dt.UTC)
    assert _generate_version_key(now=fixed) == "v/2026-05-24T20-30Z.db"


def test_generate_version_key_default_now_yields_v_prefix() -> None:
    """Default ``now=None`` uses wall-clock UTC; smoke-check the shape."""
    key = _generate_version_key()
    assert key.startswith("v/") and key.endswith(".db") and "T" in key


def test_md5_of_file_matches_known_content(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"hello world")
    # md5sum -- known value for "hello world"
    assert _md5_of_file(p) == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_generate_version_key_normalizes_naive_datetime_to_utc() -> None:
    """A naive (tzinfo=None) datetime is treated as UTC, not silently
    stamped with a misleading Z suffix on a non-UTC value."""
    naive = _dt.datetime(2026, 5, 24, 15, 0, 0)  # no tzinfo
    assert _generate_version_key(now=naive) == "v/2026-05-24T15-00Z.db"


def test_generate_version_key_converts_non_utc_to_utc() -> None:
    """A tz-aware datetime in a non-UTC zone is converted to UTC
    before stamping. PST 15:00 → UTC 23:00."""
    pst = _dt.timezone(_dt.timedelta(hours=-8))
    moment = _dt.datetime(2026, 5, 24, 15, 0, 0, tzinfo=pst)
    assert _generate_version_key(now=moment) == "v/2026-05-24T23-00Z.db"


def test_resolve_profile_none_uses_env_mapped_default() -> None:
    """profile=None falls through to ENV_PROFILES[env]."""
    assert _resolve_profile(None, "staging") == ENV_PROFILES["staging"]
    assert _resolve_profile(None, "production") == ENV_PROFILES["production"]


def test_resolve_profile_empty_string_means_no_profile() -> None:
    """profile='' is the 'explicit no-profile' sentinel."""
    assert _resolve_profile("", "staging") == ""


def test_resolve_profile_named_passes_through() -> None:
    """profile='some-name' is returned as-is."""
    assert _resolve_profile("custom-profile", "staging") == "custom-profile"


def test_resolve_profile_unknown_env_returns_empty() -> None:
    """An env not in ENV_PROFILES + profile=None falls through to the
    no-profile chain rather than raising. Useful for future envs added
    to ENV_BUCKETS but not yet ENV_PROFILES."""
    assert _resolve_profile(None, "nonexistent-env") == ""


# ---------- publish_runtime_db: real upload path ----------


def test_publish_uploads_versioned_key_then_pointer(fake_s3, source_db: Path) -> None:
    """End-to-end: both PUTs land, pointer references the versioned key,
    pointer's etag equals the local MD5."""
    fixed = _dt.datetime(2026, 5, 24, 20, 30, 0, tzinfo=_dt.UTC)
    result = publish_runtime_db(
        source=source_db,
        env="staging",
        bucket=TEST_BUCKET,
        profile="",  # explicit no-profile for moto
        now=fixed,
    )

    expected_key = "v/2026-05-24T20-30Z.db"
    assert result["version_key"] == expected_key
    assert result["bucket"] == TEST_BUCKET
    assert result["dry_run"] is False

    # Versioned key landed.
    versioned = fake_s3.get_object(Bucket=TEST_BUCKET, Key=expected_key)
    assert versioned["Body"].read() == source_db.read_bytes()

    # Pointer landed with correct payload + cache header.
    pointer = fake_s3.get_object(Bucket=TEST_BUCKET, Key=CURRENT_POINTER_KEY)
    parsed = json.loads(pointer["Body"].read())
    assert parsed == {"key": expected_key, "etag": _md5_of_file(source_db)}
    assert "must-revalidate" in pointer.get("CacheControl", "")


def test_publish_refuses_to_overwrite_existing_versioned_key(fake_s3, source_db: Path) -> None:
    """Same-minute double-push collision = operator mistake. Must fail
    loudly rather than silently shadowing the prior push."""
    fixed = _dt.datetime(2026, 5, 24, 20, 30, 0, tzinfo=_dt.UTC)
    # Seed S3 with a colliding object.
    fake_s3.put_object(
        Bucket=TEST_BUCKET,
        Key="v/2026-05-24T20-30Z.db",
        Body=b"prior push",
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        publish_runtime_db(
            source=source_db,
            env="staging",
            bucket=TEST_BUCKET,
            profile="",
            now=fixed,
        )
    # Pointer not touched — collision aborts before the current.json PUT.
    with pytest.raises(ClientError):
        fake_s3.head_object(Bucket=TEST_BUCKET, Key=CURRENT_POINTER_KEY)


def test_publish_dry_run_issues_no_puts(fake_s3, source_db: Path) -> None:
    """--dry-run computes the version key + ETag + pointer payload but
    must not touch S3."""
    fixed = _dt.datetime(2026, 5, 24, 20, 30, 0, tzinfo=_dt.UTC)
    result = publish_runtime_db(
        source=source_db,
        env="staging",
        bucket=TEST_BUCKET,
        profile="",
        now=fixed,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["version_key"] == "v/2026-05-24T20-30Z.db"
    # Neither key should exist on S3.
    with pytest.raises(ClientError):
        fake_s3.head_object(Bucket=TEST_BUCKET, Key="v/2026-05-24T20-30Z.db")
    with pytest.raises(ClientError):
        fake_s3.head_object(Bucket=TEST_BUCKET, Key=CURRENT_POINTER_KEY)


def test_publish_missing_source_raises(tmp_path: Path) -> None:
    """Operator passing --source pointing at a non-existent file should
    fail before any AWS calls."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        publish_runtime_db(
            source=tmp_path / "missing.db",
            env="staging",
            bucket=TEST_BUCKET,
            profile="",
        )


def test_publish_bucket_override(fake_s3, source_db: Path, tmp_path: Path) -> None:
    """--bucket overrides the env-derived bucket — supports non-standard
    deployments + tests."""
    custom = "custom-bucket"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=custom)

    fixed = _dt.datetime(2026, 5, 24, 21, 0, 0, tzinfo=_dt.UTC)
    publish_runtime_db(
        source=source_db,
        env="staging",
        bucket=custom,
        profile="",
        now=fixed,
    )
    s3.head_object(Bucket=custom, Key="v/2026-05-24T21-00Z.db")
    s3.head_object(Bucket=custom, Key=CURRENT_POINTER_KEY)


def test_publish_versioned_upload_precedes_pointer(fake_s3, source_db: Path, monkeypatch) -> None:
    """Order matters: the versioned-key PUT MUST land before
    current.json. Reverse it and a reader between the two PUTs sees a
    pointer to a missing key.

    Verify by intercepting put_object and recording call order. The
    stub below routes upload_file through put_object so a single
    call-order list captures both PUTs — this test ONLY checks order,
    NOT upload_file behavior. End-to-end coverage of upload_file
    (multipart on >8MB files, content-type, etc.) lives in
    test_publish_uploads_versioned_key_then_pointer where the real
    moto upload_file path runs."""
    call_order: list[str] = []
    real_put = fake_s3.put_object

    def tracking_put(**kwargs):
        call_order.append(kwargs["Key"])
        return real_put(**kwargs)

    # Patch the boto3 client factory so the SUT sees our tracking client.
    import wyrd.generators.kenning.lexicon.runtime_db_publish as mod

    class _TrackingSession:
        def client(self, _name):
            class _Client:
                pass

            c = _Client()
            c.put_object = tracking_put
            c.upload_file = lambda local, bucket, key, **kw: tracking_put(
                Bucket=bucket, Key=key, Body=Path(local).read_bytes()
            )
            c.head_object = fake_s3.head_object
            return c

    monkeypatch.setattr(mod.boto3, "Session", lambda **kw: _TrackingSession())

    publish_runtime_db(
        source=source_db,
        env="staging",
        bucket=TEST_BUCKET,
        profile="",
        now=_dt.datetime(2026, 5, 24, 22, 0, 0, tzinfo=_dt.UTC),
    )

    assert call_order == [
        "v/2026-05-24T22-00Z.db",
        CURRENT_POINTER_KEY,
    ], f"expected versioned-key PUT before pointer PUT; got {call_order}"


# ---------- format_summary ----------


def test_format_summary_real_run() -> None:
    result = {
        "bucket": "b",
        "version_key": "v/x.db",
        "etag": "abc",
        "current_pointer": {"key": "v/x.db", "etag": "abc"},
        "dry_run": False,
    }
    out = format_summary(result)
    assert "[DRY RUN]" not in out
    assert "s3://b/v/x.db" in out
    assert "abc" in out


def test_format_summary_dry_run_prefixes() -> None:
    result = {
        "bucket": "b",
        "version_key": "v/x.db",
        "etag": "abc",
        "current_pointer": {"key": "v/x.db", "etag": "abc"},
        "dry_run": True,
    }
    out = format_summary(result)
    assert out.startswith("[DRY RUN]")


# ---------- main() argparse entry point ----------


def test_main_dry_run_returns_zero(fake_s3, source_db: Path, capsys) -> None:
    rc = main(
        [
            "--env",
            "staging",
            "--bucket",
            TEST_BUCKET,
            "--source",
            str(source_db),
            "--dry-run",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "[DRY RUN]" in err


def test_main_missing_source_returns_nonzero(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "--env",
            "staging",
            "--bucket",
            TEST_BUCKET,
            "--source",
            str(tmp_path / "missing.db"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_main_collision_exit_returns_nonzero(fake_s3, source_db: Path, capsys) -> None:
    """main()'s second except-branch (RuntimeError / ClientError /
    ValueError) is exercised by inducing a versioned-key collision.
    Pre-seed S3 with the timestamped key the upload will compute, then
    run main() and confirm exit=1 + error message reaches stderr."""
    # Pre-seed so the upload's head_object finds an existing object.
    # Use a fixed --version-key would be nicer but main() doesn't
    # surface that; seed the key the wall-clock minute will produce
    # instead by intercepting _generate_version_key.
    import wyrd.generators.kenning.lexicon.runtime_db_publish as mod

    fixed_key = "v/2026-05-24T22-22Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=fixed_key, Body=b"prior")
    # Patch the version-key generator so main() lands on the seeded key.
    orig = mod._generate_version_key
    mod._generate_version_key = lambda *, now=None: fixed_key
    try:
        rc = main(
            [
                "--env",
                "staging",
                "--bucket",
                TEST_BUCKET,
                "--profile",
                "",
                "--source",
                str(source_db),
            ]
        )
    finally:
        mod._generate_version_key = orig

    assert rc == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err


def test_main_unknown_env_rejected_by_argparse(tmp_path: Path, capsys) -> None:
    """argparse choices= rejects bad --env values before our code runs."""
    p = tmp_path / "x.db"
    p.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--env", "preprod", "--source", str(p)])


# ---------- bin/publish-runtime-db.sh wrapper ----------


def test_shell_wrapper_forwards_dry_run(
    fake_s3, source_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """The bash wrapper at bin/publish-runtime-db.sh must dispatch into
    the Python module; smoke-test via subprocess with --dry-run so we
    don't need moto inside the subprocess."""
    script = Path(__file__).resolve().parents[1] / "bin" / "publish-runtime-db.sh"
    assert script.is_file(), f"expected script at {script}"

    # Subprocess won't share the moto context, so use --dry-run which
    # short-circuits before any AWS call. The dry-run computes + prints
    # the version key, so we can assert on the output shape. PYTHONPATH
    # points at the worktree so the subprocess picks up the local
    # wyrd.generators.kenning.lexicon.runtime_db_publish module rather
    # than the installed (older) wyrd package in the venv.
    worktree_root = script.parent.parent
    env = {
        **os.environ,
        # WYRD_PYTHON points at the test's known-good python, bypassing
        # the wrapper's .venv lookup (worktrees don't have their own
        # .venv — the parent repo's venv was set up before the worktree
        # was created).
        "WYRD_PYTHON": sys.executable,
        "PYTHONPATH": str(worktree_root),
        # Bypass any operator SSO that would 401 on import.
        "AWS_ACCESS_KEY_ID": "moto-test",
        "AWS_SECRET_ACCESS_KEY": "moto-test",
    }
    env.pop("AWS_PROFILE", None)
    result = subprocess.run(
        [str(script), "--env", "staging", "--source", str(source_db), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        # Run from a temp cwd to avoid picking up dev's runtime.db.
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "[DRY RUN]" in result.stderr
    assert "521studios-staging-kenning-runtime" in result.stderr


def test_shell_wrapper_errors_loudly_when_venv_missing(source_db: Path, tmp_path: Path) -> None:
    """Without .venv/bin/python AND without WYRD_PYTHON/the opt-in
    fallback flag, the wrapper must error with a useful message
    instead of silently calling python3 and producing a confusing
    ModuleNotFoundError downstream."""
    script = Path(__file__).resolve().parents[1] / "bin" / "publish-runtime-db.sh"

    # Point REPO_ROOT at a fresh empty tmpdir so .venv/bin/python is
    # definitively missing. The shell wrapper resolves REPO_ROOT
    # relative to the script location, so we need to copy the script
    # into the fake tree to point its lookup at the empty .venv.
    fake_root = tmp_path / "fake-repo"
    fake_bin = fake_root / "bin"
    fake_bin.mkdir(parents=True)
    fake_script = fake_bin / "publish-runtime-db.sh"
    fake_script.write_text(script.read_text())
    fake_script.chmod(0o755)

    env = {**os.environ}
    env.pop("WYRD_PYTHON", None)
    env.pop("WYRD_PUBLISH_ALLOW_SYSTEM_PYTHON", None)

    result = subprocess.run(
        [str(fake_script), "--env", "staging", "--source", str(source_db), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 1
    assert ".venv/bin/python is missing" in result.stderr
    assert "WYRD_PYTHON" in result.stderr


def test_shell_wrapper_honors_wyrd_python_override(source_db: Path, tmp_path: Path) -> None:
    """WYRD_PYTHON lets the operator point the wrapper at a non-default
    interpreter (CI runners, non-standard installs)."""
    script = Path(__file__).resolve().parents[1] / "bin" / "publish-runtime-db.sh"
    worktree_root = script.parent.parent

    env = {
        **os.environ,
        "WYRD_PYTHON": sys.executable,  # known-good python with our deps
        "PYTHONPATH": str(worktree_root),
        "AWS_ACCESS_KEY_ID": "moto-test",
        "AWS_SECRET_ACCESS_KEY": "moto-test",
    }
    env.pop("AWS_PROFILE", None)

    # Use a fake REPO_ROOT (without .venv) to force the WYRD_PYTHON
    # branch — otherwise the test would just exercise the normal
    # .venv-found path.
    fake_root = tmp_path / "fake-repo"
    fake_bin = fake_root / "bin"
    fake_bin.mkdir(parents=True)
    fake_script = fake_bin / "publish-runtime-db.sh"
    fake_script.write_text(script.read_text())
    fake_script.chmod(0o755)

    result = subprocess.run(
        [str(fake_script), "--env", "staging", "--source", str(source_db), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "[DRY RUN]" in result.stderr
