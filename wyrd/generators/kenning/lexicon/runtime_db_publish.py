"""Publish the L4 runtime SQLite DB to S3 (wyrd-d90t PR 3).

Pushes a built L4 DB (``wyrd kenning lexicon export-runtime-db --output
runtime.db``) to ``s3://521studios-{env}-kenning-runtime/v/<key>.db``,
then updates ``current.json`` to point at the new key. The runtime
loader reads ``current.json`` on cold start and downloads the
referenced versioned key.

S3 layout:

::

    s3://521studios-{env}-kenning-runtime/
    ├── current.json                          # {"key": "v/2026-05-24T20-30Z.db", "etag": "..."}
    └── v/
        ├── 2026-05-24T20-30Z.db              # immutable versioned (UTC, minute precision)
        ├── 2026-05-25T11-00Z.db
        └── ...

Versioned keys give: rollback = re-write current.json to a prior key;
A/B = staging on v+1 while production stays on v; history queryable
from S3 ListObjects.

Two correctness properties:

* **Upload order matters.** The versioned-key PUT MUST land before the
  ``current.json`` PUT. Reverse the order and a reader between the two
  writes sees a pointer to a key that doesn't exist yet.
* **The pointer flip is atomic at the key level** (S3 PutObject is
  strongly consistent since Dec 2020 — once it returns, every
  subsequent GET on that key sees the new value). This is the only
  atomicity the publish path relies on; nothing requires the
  versioned-key + pointer pair to flip together.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

# Force single-part S3 upload — without this, boto3's upload_file
# auto-switches to multipart on files larger than ~8MB, and multipart
# uploads produce a composite ETag (``"<md5>-<chunkcount>"``) rather
# than the file's raw MD5. That would mean the etag in current.json
# (computed locally as MD5) wouldn't match the S3 object's HTTP ETag,
# defeating the runtime loader's "redownload only on mismatch"
# optimization. 5GB is the S3 hard limit for single-object PutObject;
# the L4 DB is well under that today (~80MB on full corpus).
_SINGLE_PART_5GB_THRESHOLD = 5 * 1024 * 1024 * 1024
_SINGLE_PART_CONFIG = TransferConfig(multipart_threshold=_SINGLE_PART_5GB_THRESHOLD)

# Pattern matches the existing 521 conventions:
#   * S3 bucket names: 521studios-{env}-kenning-runtime
#   * AWS profiles: 521-{Env}-Admin
# Env strings are lowercase (s3 bucket) and Title-cased (profile).
ENV_BUCKETS = {
    "staging": "521studios-staging-kenning-runtime",
    "production": "521studios-production-kenning-runtime",
}
ENV_PROFILES = {
    "staging": "521-Staging-Admin",
    "production": "521-Production-Admin",
}

CURRENT_POINTER_KEY = "current.json"


def publish_runtime_db(
    *,
    source: Path,
    env: str,
    bucket: str | None = None,
    profile: str | None = None,
    dry_run: bool = False,
    version_key: str | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Upload the L4 DB at ``source`` to S3 + update current.json.

    Returns a dict describing the publish: ``{"bucket", "version_key",
    "etag", "current_pointer", "dry_run"}``. Operator-facing summary
    surfaces these.

    Arguments:
      * ``source`` — the L4 DB file to publish.
      * ``env`` — ``"staging"`` or ``"production"``; selects bucket +
        AWS profile from ENV_BUCKETS / ENV_PROFILES.
      * ``bucket`` — override the env-derived bucket name (for tests +
        non-standard deployments).
      * ``profile`` — override the env-derived AWS profile (for tests +
        local-dev with custom credentials).
      * ``dry_run`` — compute the version key + ETag + current pointer
        but don't issue any PUTs. Useful for verifying inputs before a
        production push.
      * ``version_key`` — override the auto-generated ``v/<utc>.db``
        key. Mostly for tests; operators should let the timestamp
        generator pick.
      * ``now`` — clock injection for tests.
    """
    if not source.is_file():
        raise FileNotFoundError(f"source L4 DB does not exist: {source}")

    resolved_bucket = bucket if bucket is not None else _bucket_for_env(env)
    resolved_profile = _resolve_profile(profile, env)
    resolved_version_key = (
        version_key if version_key is not None else _generate_version_key(now=now)
    )

    etag = _md5_of_file(source)
    pointer_payload = {"key": resolved_version_key, "etag": etag}
    pointer_bytes = json.dumps(pointer_payload, indent=2).encode("utf-8") + b"\n"

    if dry_run:
        return {
            "bucket": resolved_bucket,
            "version_key": resolved_version_key,
            "etag": etag,
            "current_pointer": pointer_payload,
            "dry_run": True,
        }

    # Named profile when resolved_profile is non-empty; otherwise fall
    # through to boto3's default credential chain (env vars + IMDS +
    # ~/.aws/credentials default profile).
    session = boto3.Session(profile_name=resolved_profile) if resolved_profile else boto3.Session()
    s3 = session.client("s3")

    # ORDER MATTERS — see module docstring. Upload the versioned DB
    # first, then current.json. Reverse would expose a window where
    # current.json points at a missing key.
    _upload_db(s3, str(source), resolved_bucket, resolved_version_key)
    _upload_pointer(s3, resolved_bucket, pointer_bytes)

    return {
        "bucket": resolved_bucket,
        "version_key": resolved_version_key,
        "etag": etag,
        "current_pointer": pointer_payload,
        "dry_run": False,
    }


def _bucket_for_env(env: str) -> str:
    """Map env → bucket via ENV_BUCKETS with a useful error message
    when the operator passes an unknown env."""
    if env not in ENV_BUCKETS:
        raise ValueError(f"unknown env {env!r}; expected one of {sorted(ENV_BUCKETS)}")
    return ENV_BUCKETS[env]


def _resolve_profile(profile: str | None, env: str) -> str:
    """Tri-state profile resolution:

    * ``profile=None`` (operator didn't pass --profile) → fall through
      to the env-mapped default (``ENV_PROFILES[env]`` or empty if env
      has no mapping).
    * ``profile=""`` → explicit no-profile; the caller wants boto3's
      default credential chain (env vars, IMDS) instead of any named
      profile. Used by tests + envs where ~/.aws/config has no entry
      for the operator.
    * ``profile="some-name"`` → use that named profile directly.

    Centralizing the convention here keeps the call sites readable and
    is the natural spot for tests to pin the behavior.
    """
    if profile is not None:
        return profile
    return ENV_PROFILES.get(env, "")


def _generate_version_key(*, now: _dt.datetime | None = None) -> str:
    """``v/<UTC ISO>.db`` — UTC-Z timestamp with minute precision.

    Minute precision is enough to dedupe legitimate same-second
    re-pushes without producing collisions in normal operation
    (publish frequency is operator-driven, not bursty), and keeps the
    key short enough to read in S3 listings.

    Always normalizes ``now`` to UTC before stamping. A test that
    passes a naive or non-UTC-aware datetime would otherwise produce a
    misleading ``...Z`` suffix on a non-UTC value.
    """
    moment = now if now is not None else _dt.datetime.now(_dt.UTC)
    if moment.tzinfo is None:
        # Naive datetime — treat as UTC. Most callers in this codebase
        # use timezone-aware datetimes; this branch protects against
        # the surprising case.
        moment = moment.replace(tzinfo=_dt.UTC)
    else:
        moment = moment.astimezone(_dt.UTC)
    return f"v/{moment.strftime('%Y-%m-%dT%H-%MZ')}.db"


def _md5_of_file(path: Path) -> str:
    """Compute the MD5 of a file.

    Recorded as the ``etag`` field in current.json. The publish path
    forces single-part upload via ``TransferConfig`` (see
    ``_upload_db``) so the S3 ETag matches this MD5 exactly; runtime
    loaders can use either ``current.json``'s etag or the S3 HEAD
    ETag interchangeably.

    ``usedforsecurity=False`` keeps the function callable on FIPS-
    compliant systems — MD5 here is a fingerprint, not a security
    primitive.
    """
    with path.open("rb") as f:
        return hashlib.file_digest(
            f,
            lambda: hashlib.md5(usedforsecurity=False),
        ).hexdigest()


def _upload_db(s3, local_path: str, bucket: str, key: str) -> None:
    """Versioned-key upload — fail loudly if the key already exists.

    The version key is timestamp-derived so a collision means the
    operator pushed two builds within the same minute, which is almost
    always a mistake (double-click, retry-after-success). Failing loud
    surfaces the duplicate instead of silently shadowing the prior
    push.
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        # 404 is the expected path — proceed with the upload.
    else:
        raise RuntimeError(
            f"refusing to overwrite existing versioned key s3://{bucket}/{key} "
            "(operator probably double-pushed within the same minute; wait one "
            "minute and retry to get a fresh version key)"
        )
    s3.upload_file(
        local_path,
        bucket,
        key,
        ExtraArgs={"ContentType": "application/x-sqlite3"},
        Config=_SINGLE_PART_CONFIG,
    )


def _upload_pointer(s3, bucket: str, pointer_bytes: bytes) -> None:
    """current.json overwrites unconditionally — the whole point is to
    flip the pointer atomically. S3 PutObject is the atomic primitive
    here; once it returns, every subsequent GET sees the new pointer.
    """
    s3.put_object(
        Bucket=bucket,
        Key=CURRENT_POINTER_KEY,
        Body=pointer_bytes,
        ContentType="application/json",
        # Short cache to keep stale current.json out of intermediate
        # caches (CloudFront/proxies). The loader does an explicit HEAD
        # on the versioned key per cold start regardless.
        CacheControl="max-age=60, must-revalidate",
    )


def format_summary(result: dict[str, Any]) -> str:
    """Operator-facing one-paragraph summary of a publish result.

    Single block of text suitable for ``click.echo(..., err=True)``
    or ``print()`` — same shape for dry_run + real runs so the
    operator can eyeball the inputs before flipping --dry-run off.
    """
    prefix = "[DRY RUN] " if result["dry_run"] else ""
    pointer = result["current_pointer"]
    return (
        f"{prefix}Published L4 DB to s3://{result['bucket']}/{result['version_key']} "
        f"(ETag={result['etag']}). "
        f"{prefix}current.json now points at key={pointer['key']!r} "
        f"etag={pointer['etag']!r}."
    )


def main(argv: list[str] | None = None) -> int:
    """Thin argparse entry point — the bash wrapper at
    ``bin/publish-runtime-db.sh`` calls into here.

    Returns the process exit code (0 = success, 1 = error). Uses
    argparse rather than click because this is operator-side tooling
    invoked outside the ``wyrd kenning`` CLI surface; keeping it off
    the click group means the operator can run it from anywhere
    boto3 is reachable without instantiating the kenning command
    tree.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="publish-runtime-db",
        description=(
            "Publish a built L4 runtime DB to S3 and atomically update "
            "current.json. Per-env (staging/production) with the 521 "
            "bucket + profile convention; --bucket / --profile override "
            "for tests + non-standard deployments."
        ),
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_BUCKETS),
        required=True,
        help="Target env (selects bucket + AWS profile).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the built L4 DB file to publish.",
    )
    parser.add_argument("--bucket", default=None, help="Override env-derived bucket.")
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Override env-derived AWS profile. Pass an empty string ('') "
            "to opt into boto3's default credential chain (env vars + "
            "IMDS) instead of any named profile in ~/.aws/config."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute version key + ETag + pointer but don't PUT anything.",
    )

    args = parser.parse_args(argv)

    try:
        result = publish_runtime_db(
            source=args.source,
            env=args.env,
            bucket=args.bucket,
            profile=args.profile,
            dry_run=args.dry_run,
        )
    # OSError covers FileNotFoundError (missing --source) plus the
    # broader file-system error class (PermissionError on a read-protected
    # source, etc.) so the operator gets a clean error message instead
    # of a traceback. ValueError = bad --env (or whatever
    # publish_runtime_db raises on input shape); RuntimeError =
    # collision-overwrite refusal; ClientError = anything boto3 surfaces
    # from the actual PUTs.
    except (OSError, ValueError, RuntimeError, ClientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
