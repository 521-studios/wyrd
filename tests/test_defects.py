"""Tests for the defective-name reporting subsystem (wyrd-dsl5): the
``wyrd.defects`` DynamoDB module, the ``/api/defects`` endpoint, and the
``wyrd defects`` triage CLI. Uses moto for offline DynamoDB — no real AWS,
no live DB (project rule: tests use fixtures/fakes).
"""

from __future__ import annotations

import boto3
import pytest
from click.testing import CliRunner
from moto import mock_aws

from wyrd import defects
from wyrd.app import create_app
from wyrd.cli_defects import defects as defects_cli

TABLE = "521studios-test-wyrd-defects"

_SAMPLE = {
    "generator": "kenning",
    "reason": "ungrammatical compound — 'Ton North' reads backwards",
    "result": "Ton North",
    "seed": 1234567890,
    "parameters": {"culture": "english", "count": 5},
    "explanation": "Ton + North",
    "components": [{"word": "Ton North", "parts": []}],
    "morphemes_by_word": [[{"usage": "Ton", "meanings": ["enclosure"]}]],
    "bundle_version": {"built_at": "2026-05-30T00:00:00Z", "schema_version": "2"},
}


@pytest.fixture
def defects_table(monkeypatch):
    """moto DynamoDB fake + the defects table (hash id + status GSI).

    Fake AWS creds + /dev/null config so boto3.Session() doesn't try to
    refresh the operator's SSO under moto (mirrors the bulk-sources test
    fixture). Sets WYRD_DEFECTS_TABLE so resolve_table_name finds it without
    threading an env through every call."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv(defects.ENV_TABLE, TABLE)
    defects.reset_session_cache()  # don't reuse a session from a prior test's context
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-2")
        ddb.create_table(
            TableName=TABLE,
            AttributeDefinitions=[
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": defects.STATUS_INDEX,
                    "KeySchema": [
                        {"AttributeName": "status", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb.get_waiter("table_exists").wait(TableName=TABLE)
        yield TABLE


# --- module: validation + serialization ------------------------------------


def test_record_defect_rejects_blank_reason():
    with pytest.raises(defects.DefectsError):
        defects.record_defect({**_SAMPLE, "reason": "   "}, table_name=TABLE)


def test_record_defect_rejects_non_string_reason():
    """A non-string reason (number, list, object) is as invalid as a blank one —
    it must raise DefectsError (honoring the documented contract), not an
    AttributeError from calling .strip() on a non-string. The reason check runs
    before any DynamoDB call, so this needs no table."""
    for bad in (5, ["a", "b"], {"x": 1}, None):
        with pytest.raises(defects.DefectsError, match="non-empty 'reason'"):
            defects.record_defect({**_SAMPLE, "reason": bad}, table_name=TABLE)


def test_resolve_table_name_prefers_env(monkeypatch):
    monkeypatch.setenv(defects.ENV_TABLE, "explicit-table")
    assert defects.resolve_table_name(env="staging") == "explicit-table"


def test_resolve_table_name_falls_back_to_env_convention(monkeypatch):
    monkeypatch.delenv(defects.ENV_TABLE, raising=False)
    assert defects.resolve_table_name(env="production") == "521studios-production-wyrd-defects"


def test_resolve_table_name_unset_without_env_raises(monkeypatch):
    monkeypatch.delenv(defects.ENV_TABLE, raising=False)
    with pytest.raises(defects.DefectsError):
        defects.resolve_table_name()


def test_to_item_blobs_go_to_payload_seed_is_string():
    item = defects._to_item({**_SAMPLE, "id": "abc", "status": "new", "created_at": "t"})
    # Query keys are native attributes.
    assert item["id"] == "abc"
    assert item["generator"] == "kenning"
    assert item["name"] == "Ton North"
    assert item["seed"] == "1234567890"  # stringified to dodge Decimal coercion
    # Nested blobs are JSON-encoded under payload, not top-level.
    assert "parameters" not in item
    assert "payload" in item
    assert "morphemes_by_word" in item["payload"]


def test_from_item_reexpands_payload():
    record = {**_SAMPLE, "id": "abc", "status": "new", "created_at": "t"}
    flat = defects._from_item(defects._to_item(record))
    assert flat["parameters"] == _SAMPLE["parameters"]
    assert flat["morphemes_by_word"] == _SAMPLE["morphemes_by_word"]
    assert flat["bundle_version"] == _SAMPLE["bundle_version"]
    assert "payload" not in flat


# --- module: round-trip against moto ----------------------------------------


def test_record_then_get_roundtrip(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    assert created["status"] == defects.STATUS_NEW
    assert created["id"]
    assert created["created_at"]

    fetched = defects.get_defect(created["id"], env="staging")
    assert fetched is not None
    assert fetched["generator"] == "kenning"
    assert fetched["name"] == "Ton North"
    assert fetched["reason"].startswith("ungrammatical")
    assert fetched["parameters"] == _SAMPLE["parameters"]
    assert fetched["bundle_version"]["schema_version"] == "2"


def test_get_missing_returns_none(defects_table):
    assert defects.get_defect("does-not-exist", env="staging") is None


def test_list_defaults_to_new_newest_first(defects_table):
    a = defects.record_defect({**_SAMPLE, "result": "A"}, env="staging")
    b = defects.record_defect({**_SAMPLE, "result": "B"}, env="staging")
    rows = defects.list_defects(env="staging")
    ids = [r["id"] for r in rows]
    assert a["id"] in ids and b["id"] in ids
    assert all(r["status"] == "new" for r in rows)


def test_list_filters_by_generator(defects_table):
    defects.record_defect({**_SAMPLE, "generator": "kenning"}, env="staging")
    defects.record_defect({**_SAMPLE, "generator": "other-gen"}, env="staging")
    only_other = defects.list_defects(env="staging", generator="other-gen")
    assert only_other and all(r["generator"] == "other-gen" for r in only_other)


def test_accept_sets_status_ticket_and_triaged_at(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    updated = defects.update_status(
        created["id"], defects.STATUS_ACCEPTED, ticket="wyrd-zzzz", note="real bug", env="staging"
    )
    assert updated["status"] == "accepted"
    assert updated["ticket"] == "wyrd-zzzz"
    assert updated["triage_note"] == "real bug"
    assert updated["triaged_at"]
    # No longer surfaces under the default 'new' list.
    assert created["id"] not in [r["id"] for r in defects.list_defects(env="staging")]
    assert created["id"] in [
        r["id"] for r in defects.list_defects(env="staging", status="accepted")
    ]


def test_dismiss_sets_status(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    updated = defects.update_status(
        created["id"], defects.STATUS_DISMISSED, note="duplicate", env="staging"
    )
    assert updated["status"] == "dismissed"


def test_update_unknown_id_raises(defects_table):
    with pytest.raises(defects.DefectsError):
        defects.update_status("nope", defects.STATUS_ACCEPTED, env="staging")


def test_update_to_new_is_rejected(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    with pytest.raises(defects.DefectsError):
        defects.update_status(created["id"], "new", env="staging")


# --- /api/defects endpoint --------------------------------------------------


def test_endpoint_rejects_missing_reason(defects_table):
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json={"generator": "kenning", "result": "X"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_report"


def test_endpoint_rejects_missing_generator(defects_table):
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json={"reason": "bad", "result": "X"})
    assert resp.status_code == 400


def test_endpoint_records_valid_report(defects_table):
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json=_SAMPLE)
    assert resp.status_code == 201, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["status"] == "new"
    stored = defects.get_defect(body["id"], env="staging")
    assert stored["name"] == "Ton North"
    assert stored["reason"].startswith("ungrammatical")


def test_endpoint_503_when_table_unconfigured(monkeypatch):
    # No WYRD_DEFECTS_TABLE and no env arg → resolve_table_name raises →
    # the handler maps DefectsError to 503 rather than 500.
    monkeypatch.delenv(defects.ENV_TABLE, raising=False)
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json=_SAMPLE)
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "defects_unavailable"


# --- wyrd defects CLI -------------------------------------------------------
# --profile "" → default credential chain (no named profile) so moto works;
# WYRD_DEFECTS_TABLE (set by the fixture) overrides the per-env table name.


def test_cli_list_shows_new_report(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(defects_cli, ["list", "--profile", "", "--json"])
    assert result.exit_code == 0, result.output
    assert created["id"] in result.output


def test_cli_accept_moves_to_accepted(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(
        defects_cli, ["accept", created["id"], "--profile", "", "--ticket", "wyrd-aaaa"]
    )
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output
    assert defects.get_defect(created["id"], env="staging")["status"] == "accepted"


def test_cli_dismiss_moves_to_dismissed(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(
        defects_cli, ["dismiss", created["id"], "--profile", "", "--reason", "dup"]
    )
    assert result.exit_code == 0, result.output
    assert defects.get_defect(created["id"], env="staging")["status"] == "dismissed"


def test_cli_show_renders_reproduce_block(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(defects_cli, ["show", created["id"], "--profile", ""])
    assert result.exit_code == 0, result.output
    assert "reproduce" in result.output
    assert "kenning" in result.output


def test_cli_show_renders_optional_fields_and_gates_absent_ones(defects_table):
    # Locks the data-driven show rendering: a populated optional field
    # (explanation → "etymology:") must render under its display label, and
    # optionals absent from a fresh report (triaged_at / ticket / note) must be
    # suppressed by the `if value:` gate. Guards against a typo in the
    # _SHOW_OPTIONAL_FIELDS (label, key) tuples silently emitting None.
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(defects_cli, ["show", created["id"], "--profile", ""])
    assert result.exit_code == 0, result.output
    assert "etymology:  Ton + North" in result.output
    assert "triaged_at:" not in result.output
    assert "ticket:" not in result.output
    assert "\nnote:" not in result.output


def test_cli_show_unknown_id_errors(defects_table):
    runner = CliRunner()
    result = runner.invoke(defects_cli, ["show", "no-such-id", "--profile", ""])
    assert result.exit_code != 0
    assert "no defect" in result.output.lower()


def test_cli_show_json(defects_table):
    created = defects.record_defect(_SAMPLE, env="staging")
    runner = CliRunner()
    result = runner.invoke(defects_cli, ["show", created["id"], "--profile", "", "--json"])
    assert result.exit_code == 0, result.output
    import json as _json

    assert _json.loads(result.output)["id"] == created["id"]


# --- additional coverage from the PR #412 review round ----------------------


def test_list_status_all_returns_every_status_newest_first(defects_table, monkeypatch):
    # Distinct, increasing created_at so ordering is deterministic (the real
    # _now_iso is second-precision and two rapid records could collide).
    counter = {"n": 0}

    def _stamp():
        counter["n"] += 1
        return f"2026-01-01T00:00:{counter['n']:02d}Z"

    monkeypatch.setattr(defects, "_now_iso", _stamp)
    a = defects.record_defect({**_SAMPLE, "result": "A"}, env="staging")
    b = defects.record_defect({**_SAMPLE, "result": "B"}, env="staging")
    defects.update_status(b["id"], defects.STATUS_DISMISSED, env="staging")
    rows = defects.list_defects(env="staging", status="all")
    ids = [r["id"] for r in rows]
    assert a["id"] in ids and b["id"] in ids  # both statuses present
    assert rows[0]["id"] == b["id"]  # newest first


def test_list_new_orders_newest_first(defects_table, monkeypatch):
    counter = {"n": 0}

    def _stamp():
        counter["n"] += 1
        return f"2026-01-01T00:00:{counter['n']:02d}Z"

    monkeypatch.setattr(defects, "_now_iso", _stamp)
    defects.record_defect({**_SAMPLE, "result": "older"}, env="staging")
    b = defects.record_defect({**_SAMPLE, "result": "newer"}, env="staging")
    rows = defects.list_defects(env="staging")
    assert rows[0]["id"] == b["id"]


def test_list_unknown_status_raises(defects_table):
    with pytest.raises(defects.DefectsError):
        defects.list_defects(env="staging", status="bogus")


def test_record_omits_seed_when_none(defects_table):
    created = defects.record_defect({**_SAMPLE, "seed": None}, env="staging")
    fetched = defects.get_defect(created["id"], env="staging")
    assert "seed" not in fetched  # _to_item drops a None seed entirely


def test_non_ascii_reason_and_name_roundtrip(defects_table):
    created = defects.record_defect(
        {**_SAMPLE, "result": "Æthelfrith", "reason": "reads as naïve — wrong"},
        env="staging",
    )
    fetched = defects.get_defect(created["id"], env="staging")
    assert fetched["name"] == "Æthelfrith"
    assert fetched["reason"] == "reads as naïve — wrong"


def test_from_item_corrupt_payload_surfaces_raw():
    flat = defects._from_item(
        {"id": "x", "status": "new", "created_at": "t", "payload": "not-json"}
    )
    assert flat["payload_raw"] == "not-json"
    assert "payload" not in flat


def test_record_aws_error_raises_defects_error(defects_table):
    # A live boto3 error (PutItem against a table that doesn't exist) must
    # surface as DefectsError, not a raw botocore exception.
    with pytest.raises(defects.DefectsError):
        defects.record_defect(_SAMPLE, table_name="absent-table-xyz")


def test_endpoint_503_on_aws_error(defects_table, monkeypatch):
    # Table env points at a non-existent table → put_item raises under moto →
    # DefectsError → 503 (not a raw 500).
    monkeypatch.setenv(defects.ENV_TABLE, "absent-table-xyz")
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json=_SAMPLE)
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "defects_unavailable"


def test_endpoint_rejects_non_object_body(defects_table):
    app = create_app()
    with app.test_client() as client:
        resp = client.post("/api/defects", json=["not", "an", "object"])
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_report"


def test_resolve_profile_precedence(monkeypatch):
    from wyrd.cli_defects import _resolve_profile

    monkeypatch.delenv(defects.ENV_PROFILE, raising=False)
    # Explicit --profile wins (even empty → default chain / None).
    assert _resolve_profile("staging", "custom") == "custom"
    assert _resolve_profile("staging", "") is None
    # No explicit + no env → per-env admin default.
    assert _resolve_profile("staging", None) == "521-Staging-Admin"
    assert _resolve_profile("production", None) == "521-Production-Admin"
    # Env var is the middle tier; empty env value → None.
    monkeypatch.setenv(defects.ENV_PROFILE, "from-env")
    assert _resolve_profile("staging", None) == "from-env"
    monkeypatch.setenv(defects.ENV_PROFILE, "")
    assert _resolve_profile("staging", None) is None
