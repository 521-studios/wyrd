"""Tests for wyrd-0gou SPA feature-flag resolution + manifest wiring.

resolve_feature_config is pure over its env mapping, so most coverage is
synthetic-env unit tests; one integration test confirms the resolved block
rides on /api/manifest and reflects env at request time.
"""

from __future__ import annotations

from wyrd.app import create_app
from wyrd.feature_flags import resolve_feature_config


def test_default_env_all_flags_off():
    """No env set → master override off, no flags, no default overrides.
    This is the production-safe default: the SPA hides every gated option."""
    cfg = resolve_feature_config({})
    assert cfg == {"all": False, "flags": {}, "defaults": {}}


def test_all_override_does_not_leak_into_flags():
    """WYRD_FF_ALL sets the master override and is NOT echoed as a flag."""
    cfg = resolve_feature_config({"WYRD_FF_ALL": "true"})
    assert cfg["all"] is True
    assert cfg["flags"] == {}


def test_per_flag_parsed_by_envified_suffix():
    """WYRD_FF_<NAME> lands in flags keyed by the envified suffix; the SPA
    computes the same suffix from the flag name, so namespaced flags
    (culture.welsh → CULTURE_WELSH) round-trip without reverse-parsing."""
    cfg = resolve_feature_config(
        {
            "WYRD_FF_NOVELTY": "true",
            "WYRD_FF_SCORING_MODE": "false",
            "WYRD_FF_CULTURE_WELSH": "1",
        }
    )
    assert cfg["all"] is False
    assert cfg["flags"] == {"NOVELTY": True, "SCORING_MODE": False, "CULTURE_WELSH": True}


def test_bool_parsing_accepts_common_truthy_strings():
    cfg = resolve_feature_config(
        {
            "WYRD_FF_A": "TRUE",
            "WYRD_FF_B": "yes",
            "WYRD_FF_C": "on",
            "WYRD_FF_D": "0",
            "WYRD_FF_E": "off",
            "WYRD_FF_F": "",
        }
    )
    assert cfg["flags"] == {"A": True, "B": True, "C": True, "D": False, "E": False, "F": False}


def test_default_value_overrides_lowercased_and_kept_as_string():
    """WYRD_DEFAULT_<OPTION> overrides an option's default VALUE, keyed by
    the lower-cased option name (matching the SPA's snake_case field keys);
    value stays a raw string for the SPA to coerce per the field's type."""
    cfg = resolve_feature_config({"WYRD_DEFAULT_CULTURE": "english", "WYRD_DEFAULT_COUNT": "5"})
    assert cfg["defaults"] == {"culture": "english", "count": "5"}
    assert cfg["flags"] == {}


def test_flags_and_defaults_partition_cleanly():
    """A mixed env splits into flags vs defaults by prefix; unrelated env
    vars (no WYRD_FF_/WYRD_DEFAULT_ prefix) are ignored."""
    cfg = resolve_feature_config(
        {
            "WYRD_FF_ALL": "false",
            "WYRD_FF_ERA": "true",
            "WYRD_DEFAULT_SCORING_MODE": "proportions",
            "PATH": "/usr/bin",
            "WYRD_RUNTIME_DB_BUCKET": "some-bucket",
        }
    )
    assert cfg["all"] is False
    assert cfg["flags"] == {"ERA": True}
    assert cfg["defaults"] == {"scoring_mode": "proportions"}


def test_manifest_includes_config_block_default_off(monkeypatch):
    """/api/manifest carries the resolved config; with no flag env set the
    master override is off and flags are empty (prod-safe default)."""
    for key in list(__import__("os").environ):
        if key.startswith(("WYRD_FF_", "WYRD_DEFAULT_")):
            monkeypatch.delenv(key, raising=False)
    app = create_app()
    with app.test_client() as client:
        body = client.get("/api/manifest").get_json()
    assert "config" in body
    assert body["config"]["all"] is False
    assert body["config"]["flags"] == {}


def test_manifest_config_reflects_env_at_request_time(monkeypatch):
    """The block is resolved per request, so flipping a Lambda env var
    takes effect without a redeploy — staging's WYRD_FF_ALL=true surfaces
    as config.all."""
    monkeypatch.setenv("WYRD_FF_ALL", "true")
    monkeypatch.setenv("WYRD_DEFAULT_CULTURE", "english")
    app = create_app()
    with app.test_client() as client:
        config = client.get("/api/manifest").get_json()["config"]
    assert config["all"] is True
    assert config["defaults"]["culture"] == "english"
