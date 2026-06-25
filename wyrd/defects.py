"""Defective-name reports: DynamoDB I/O shared by the Lambda write path
and the operator triage CLI (wyrd-dsl5).

A *defect report* is a user-submitted "this generated name is bad" record.
It carries the full reproduction context (generator, parameters, seed, the
rendered name, morpheme breakdown, explanation, and the runtime-bundle
version it was generated against) plus the user's required free-text reason.

The design is **generator-agnostic**: ``generator`` is a first-class,
filterable attribute, so every present and future generator gets defect
reporting for free.

Storage: one DynamoDB table per env (``521studios-{env}-wyrd-defects``).
Query keys are stored as native attributes; nested blobs are stored as JSON
strings under ``payload`` to dodge DynamoDB's float / empty-string typing
pitfalls. Status lifecycle: ``new → accepted | dismissed``.

boto3 is imported lazily inside the client factory so non-defect code paths
(and ``wyrd --help``) pay no import cost — mirrors
``wyrd.generators.kenning.bulk_sources._boto3_client``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import uuid
from typing import Any

_logger = logging.getLogger(__name__)

# Status lifecycle. ``new`` reports are untriaged; the operator CLI moves
# each to ``accepted`` (real defect, optionally linked to a filed bd ticket)
# or ``dismissed`` (bogus / duplicate / wontfix, with a reason).
STATUS_NEW = "new"
STATUS_ACCEPTED = "accepted"
STATUS_DISMISSED = "dismissed"
STATUSES = (STATUS_NEW, STATUS_ACCEPTED, STATUS_DISMISSED)

# GSI that lets the CLI pull "all new reports, newest first" without a full
# table scan. Defined in terraform (terraform/main.tf, aws_dynamodb_table
# "defects"); the name here must match.
STATUS_INDEX = "status-created_at-index"

# Env var the Lambda + CLI read for the table name. Set on the Lambda by
# terraform (WYRD_DEFECTS_TABLE = aws_dynamodb_table.defects.name); the CLI
# falls back to the per-env naming convention when it's unset.
ENV_TABLE = "WYRD_DEFECTS_TABLE"
# Optional AWS profile override for the CLI triage path (the Lambda uses its
# execution role and leaves this unset).
ENV_PROFILE = "WYRD_DEFECTS_AWS_PROFILE"

# Region matches the terraform default (terraform/variables.tf aws_region).
DEFAULT_REGION = "us-east-2"

# Blob fields stored JSON-encoded under ``payload`` rather than as native
# DynamoDB attributes. Keeping them out of the top level avoids Decimal
# coercion (floats in parameters/components) and empty-string edge cases,
# and keeps the queryable attributes small.
_PAYLOAD_FIELDS = (
    "parameters",
    "explanation",
    "components",
    "morphemes_by_word",
    "bundle_version",
)


class DefectsError(RuntimeError):
    """Raised when a defect cannot be recorded or triaged (table unset,
    AWS error, unknown id). The Flask handler maps this to a 503; the CLI
    surfaces it as a clean error message."""


def resolve_table_name(env: str | None = None) -> str:
    """Resolve the DynamoDB table name.

    ``WYRD_DEFECTS_TABLE`` (set on the Lambda by terraform) wins. Otherwise
    fall back to the per-env naming convention so the CLI can target
    staging / production without the env var set locally.
    """
    explicit = os.environ.get(ENV_TABLE)
    if explicit:
        return explicit
    if not env:
        raise DefectsError(
            f"defects table not configured: set ${ENV_TABLE} or pass an env (staging/production)"
        )
    return f"521studios-{env}-wyrd-defects"


def _now_iso() -> str:
    """UTC timestamp, second precision, sortable as a string — the GSI
    range key relies on lexicographic order matching chronological order."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _defects_table(
    *,
    env: str | None = None,
    table_name: str | None = None,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
) -> Any:
    """Return a boto3 DynamoDB ``Table`` resource for the defects table.

    boto3 imported lazily (see module docstring). ``profile=None`` uses the
    default credential chain — the Lambda execution role in production, the
    operator's admin profile when the CLI passes one explicitly.

    The boto3 ``Session`` (whose construction does the credential resolution)
    is cached per ``(profile, region)`` so a warm Lambda reuses it across
    invocations. The ``resource``/``Table`` is built fresh each call (cheap,
    and keeps test isolation clean under moto's per-test mock context).
    """
    name = table_name or resolve_table_name(env)
    session = _session(profile, region)
    return session.resource("dynamodb").Table(name)


# Session cache keyed by (profile, region). Sessions are config-only (no live
# network binding), so caching is safe across moto test contexts; clients are
# created fresh per call from the cached session. reset_session_cache() drops
# it (tests / operator reload).
_session_cache: dict[tuple[str | None, str], Any] = {}


def _session(profile: str | None, region: str) -> Any:
    key = (profile, region)
    cached = _session_cache.get(key)
    if cached is None:
        import boto3

        cached = boto3.Session(profile_name=profile, region_name=region)
        _session_cache[key] = cached
    return cached


def reset_session_cache() -> None:  # noqa: V103 — test/ops cache-reset helper
    """Drop the cached boto3 sessions. For tests and operator-driven reload."""
    _session_cache.clear()


def _to_item(report: dict[str, Any]) -> dict[str, Any]:
    """Shape a report dict into a DynamoDB item. Query keys land as native
    attributes; the nested blobs go to a single JSON-encoded ``payload``."""
    item: dict[str, Any] = {
        "id": report["id"],
        "status": report["status"],
        "created_at": report["created_at"],
        "generator": report.get("generator") or "unknown",
        "name": report.get("result") or report.get("name") or "",
        "reason": report["reason"],
    }
    seed = report.get("seed")
    if seed is not None:
        # Stored as a string so a JS-safe-int seed round-trips without
        # DynamoDB Number → Decimal coercion on read.
        item["seed"] = str(seed)
    payload = {k: report[k] for k in _PAYLOAD_FIELDS if report.get(k) is not None}
    if payload:
        item["payload"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return item


def _from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_to_item` — re-expand ``payload`` JSON back into
    top-level keys so callers see a flat report dict."""
    out = {k: v for k, v in item.items() if k != "payload"}
    raw = item.get("payload")
    if raw:
        try:
            out.update(json.loads(raw))
        except (TypeError, ValueError):
            # Corrupt payload shouldn't sink a list/show; surface raw + log so
            # the storage-layer corruption isn't silent in CloudWatch.
            _logger.warning("corrupt defect payload id=%s; surfacing raw", out.get("id"))
            out["payload_raw"] = raw
    return out


def record_defect(
    report: dict[str, Any],
    *,
    env: str | None = None,
    table_name: str | None = None,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
) -> dict[str, Any]:
    """Persist a new defect report. Mints ``id`` / ``created_at`` and stamps
    ``status='new'``. Returns ``{id, status, created_at}``.

    Raises :class:`DefectsError` on a missing reason or any AWS failure.
    """
    # A non-STRING reason (number, list, object) is as invalid as a blank one:
    # treat it as empty so it raises the same DefectsError below, rather than an
    # AttributeError from `.strip()` on a non-string (which would violate this
    # function's documented "Raises DefectsError on a missing reason" contract).
    raw_reason = report.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if not reason:
        raise DefectsError("defect report requires a non-empty 'reason'")

    record = {
        **report,
        "id": uuid.uuid4().hex,
        "status": STATUS_NEW,
        "created_at": _now_iso(),
        "reason": reason,
    }
    try:
        table = _defects_table(env=env, table_name=table_name, profile=profile, region=region)
        table.put_item(Item=_to_item(record))
    except DefectsError:
        raise
    except Exception as exc:  # boto3 / botocore errors
        raise DefectsError(f"failed to record defect: {exc}") from exc
    return {"id": record["id"], "status": record["status"], "created_at": record["created_at"]}


def list_defects(
    *,
    env: str | None = None,
    table_name: str | None = None,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
    status: str = STATUS_NEW,
    generator: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List defect reports, newest first.

    ``status='all'`` scans the whole table; any other value queries the
    ``status-created_at`` GSI (cheaper, returned newest-first). ``generator``
    filters in-memory (a low-cardinality field not worth its own index).

    Both paths paginate (following ``LastEvaluatedKey``); DynamoDB's ``Limit``
    is a per-page evaluation cap, not a total-results cap, and the generator
    filter is applied after the fetch, so a single page can yield fewer than
    ``limit`` matches. The GSI-query path returns rows newest-first, so it stops
    once ``limit`` matches are collected. The ``status='all'`` scan returns rows
    in ARBITRARY order, so it must exhaust the table before the newest-first sort
    below — stopping early could drop a newer row on an unscanned page.
    """
    if status != "all" and status not in STATUSES:
        raise DefectsError(f"unknown status {status!r}; expected one of {STATUSES} or 'all'")
    try:
        table = _defects_table(env=env, table_name=table_name, profile=profile, region=region)
        reports: list[dict[str, Any]] = []
        start_key: dict | None = None
        while True:
            if status == "all":
                kwargs: dict[str, Any] = {}
            else:
                kwargs = {
                    "IndexName": STATUS_INDEX,
                    "KeyConditionExpression": "#s = :s",
                    "ExpressionAttributeNames": {"#s": "status"},
                    "ExpressionAttributeValues": {":s": status},
                    "ScanIndexForward": False,  # newest first
                }
            if start_key:
                kwargs["ExclusiveStartKey"] = start_key
            resp = table.scan(**kwargs) if status == "all" else table.query(**kwargs)
            for item in resp.get("Items", []):
                report = _from_item(item)
                if generator and report.get("generator") != generator:
                    continue
                reports.append(report)
            start_key = resp.get("LastEvaluatedKey")
            if not start_key:
                break
            # The GSI query returns rows newest-first, so once we hold `limit`
            # matches the remaining pages are strictly older — stop. A scan
            # (status == 'all') yields rows in ARBITRARY order, so stopping early
            # would let a newer row on an unscanned page be missed by the
            # newest-first sort below: we MUST exhaust the table first.
            if status != "all" and len(reports) >= limit:
                break
    except DefectsError:
        raise
    except Exception as exc:
        raise DefectsError(f"failed to list defects: {exc}") from exc

    if status == "all":
        # Scan order is arbitrary; impose newest-first like the query path.
        reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return reports[:limit]


def get_defect(
    defect_id: str,
    *,
    env: str | None = None,
    table_name: str | None = None,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
) -> dict[str, Any] | None:
    """Fetch one defect by id, or None if it doesn't exist."""
    try:
        table = _defects_table(env=env, table_name=table_name, profile=profile, region=region)
        resp = table.get_item(Key={"id": defect_id})
    except DefectsError:
        raise
    except Exception as exc:
        raise DefectsError(f"failed to fetch defect {defect_id}: {exc}") from exc
    item = resp.get("Item")
    return _from_item(item) if item else None


def update_status(
    defect_id: str,
    status: str,
    *,
    ticket: str | None = None,
    note: str | None = None,
    env: str | None = None,
    table_name: str | None = None,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
) -> dict[str, Any]:
    """Move a defect to ``accepted`` or ``dismissed`` (triage). Stamps
    ``triaged_at`` and optionally records a linked bd ``ticket`` and a free
    ``triage_note``. Raises :class:`DefectsError` if the id is unknown.
    """
    if status not in (STATUS_ACCEPTED, STATUS_DISMISSED):
        raise DefectsError(
            f"can only set status to {STATUS_ACCEPTED!r} or {STATUS_DISMISSED!r}, got {status!r}"
        )
    set_parts = ["#s = :s", "triaged_at = :t"]
    names = {"#s": "status"}
    values: dict[str, Any] = {":s": status, ":t": _now_iso()}
    if ticket:
        set_parts.append("ticket = :tk")
        values[":tk"] = ticket
    if note:
        set_parts.append("triage_note = :n")
        values[":n"] = note

    from botocore.exceptions import ClientError

    try:
        table = _defects_table(env=env, table_name=table_name, profile=profile, region=region)
        resp = table.update_item(
            Key={"id": defect_id},
            UpdateExpression="SET " + ", ".join(set_parts),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
            ConditionExpression="attribute_exists(id)",
            ReturnValues="ALL_NEW",
        )
    except DefectsError:
        raise
    except ClientError as exc:
        # The ConditionExpression fails (stable AWS error code) when no row
        # with this id exists — match on the documented code, not the
        # exception class name.
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise DefectsError(f"no defect with id {defect_id!r}") from exc
        raise DefectsError(f"failed to update defect {defect_id}: {exc}") from exc
    except Exception as exc:
        raise DefectsError(f"failed to update defect {defect_id}: {exc}") from exc
    return _from_item(resp.get("Attributes", {}))
