"""Tests for wyrd-0gou SPA feature-flag resolution + manifest wiring.

resolve_feature_config is pure over its env mapping, so most coverage is
synthetic-env unit tests; one integration test confirms the resolved block
rides on /api/manifest and reflects env at request time.
"""

from __future__ import annotations

from wyrd.app import create_app
from wyrd.feature_flags import (
    _coerce_to_schema_type,
    apply_env_defaults,
    resolve_feature_config,
)

_NOVELTY_PROP = {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0}
_PROPS = {"novelty": _NOVELTY_PROP, "count": {"type": "integer", "default": 5}}


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


def test_scoring_mode_default_flips_to_vector_via_env():
    """wyrd-lnt6 (Phase 2): the per-environment default scoring mode flips to
    vector purely via WYRD_DEFAULT_SCORING_MODE (the terraform staging
    rollout), with NO code/schema-default change. Pins that the env var flows
    to the /api/manifest default the SPA seeds from — the mechanism the
    proportions→vector cutover rides on."""
    cfg = resolve_feature_config({"WYRD_DEFAULT_SCORING_MODE": "vector"})
    assert cfg["defaults"]["scoring_mode"] == "vector"


# wyrd-7f22: server-side application of WYRD_DEFAULT_<OPTION> for omitted params.
# The SPA omits any field still at its seeded default (wyrd-14hn), so a value-
# changing env default must be applied server-side or it never reaches the
# generator (the bug a holistic review caught: slider showed 0.1, generation ran 0.0).


def test_apply_env_defaults_fills_absent_option_coerced_to_type():
    params = {}
    apply_env_defaults(params, _PROPS, env={"WYRD_DEFAULT_NOVELTY": "0.1"})
    assert params["novelty"] == 0.1
    assert isinstance(params["novelty"], float)


def test_apply_env_defaults_does_not_override_present_param():
    # An explicit request value wins — even one equal to the schema default.
    params = {"novelty": 0.0}
    apply_env_defaults(params, _PROPS, env={"WYRD_DEFAULT_NOVELTY": "0.1"})
    assert params["novelty"] == 0.0


def test_apply_env_defaults_noop_without_env():
    # Bit-stable: no WYRD_DEFAULT_* set → nothing filled (schema default applies).
    params = {}
    apply_env_defaults(params, _PROPS, env={})
    assert params == {}


def test_apply_env_defaults_skips_unknown_and_uncoercible():
    params = {}
    apply_env_defaults(
        params,
        _PROPS,
        env={
            "WYRD_DEFAULT_NOT_AN_OPTION": "x",  # not in this generator's schema
            "WYRD_DEFAULT_COUNT": "abc",  # uncoercible integer → skipped
        },
    )
    assert params == {}


def test_apply_env_defaults_rejects_non_finite_number():
    # float() accepts "inf"/"nan"; they must NOT leak into params (would poison
    # the score blend / emit invalid-JSON NaN). Mirrors the SPA's isNaN guard.
    for bad in ("inf", "-inf", "Infinity", "nan", "NaN"):
        params = {}
        apply_env_defaults(params, _PROPS, env={"WYRD_DEFAULT_NOVELTY": bad})
        assert params == {}, f"{bad!r} should be rejected, not seeded"


def test_apply_env_defaults_integer_coercion():
    params = {}
    apply_env_defaults(params, _PROPS, env={"WYRD_DEFAULT_COUNT": "7"})
    assert params["count"] == 7
    assert isinstance(params["count"], int)


# The shared server/SPA integer-coercion parity vector (wyrd-8uuq). The SPA's
# featureFlags.test.js asserts the SAME (input → outcome) list against
# coerceToType, so both sides agree in BOTH directions: a slider seeded from a
# WYRD_DEFAULT never disagrees with what generation uses. ``None`` here == the
# SPA's ``undefined`` (fall back to the schema default).
_INTEGER_PARITY_VECTOR = {
    "5": 5,
    "5.0": 5,  # integral float → accepted (the wyrd-7f22 decimal-styled default)
    "5.5": None,  # non-integral → rejected (was 5 under the old lenient parseInt)
    "3abc": None,  # trailing garbage → rejected (was 3 under parseInt)
    "12px": None,  # trailing garbage → rejected (was 12)
    "-3.9": None,  # non-integral → rejected (was -3)
    "": None,
    "-2": -2,
    " 7 ": 7,
    "Infinity": None,
    "NaN": None,
    ".5": None,  # non-integral → rejected
    "١٢٣": None,  # Arabic-Indic 123: JS Number → NaN, so ASCII-reject to match
    "१२३": None,  # Devanagari 123: same ASCII-only parity
}


def test_integer_coercion_parity_vector():
    """Integer coercion is CONVERGED with the SPA's ``coerceToType`` (wyrd-8uuq):
    accept a plain integer or an integral float ("5"/"5.0" → 5), reject
    non-integers and trailing garbage ("5.5"/"3abc"/"12px" → None). Before this,
    BOTH sides were lenient and agreed — the server's ``_LEADING_INT_RE`` mirrored
    the SPA's ``parseInt`` — but both silently TRUNCATED a malformed integer
    ("3abc" → 3, "5.5" → 5); this converges both to STRICT rejection, so a
    malformed default falls back to the schema default on both sides instead of
    being truncated. ASCII-only, mirroring JS ``Number`` (Python int/float
    otherwise accept Unicode digits JS yields NaN for)."""
    intp = {"type": "integer"}
    for raw, expected in _INTEGER_PARITY_VECTOR.items():
        got = _coerce_to_schema_type(raw, intp)
        assert got == expected, f"{raw!r} → {got!r}, expected {expected!r}"


def test_number_coercion_rejects_unicode_digits():
    """wyrd-8uuq (Gemini): the ``number`` path is ASCII-guarded too. Python
    ``float()`` accepts Unicode digits (``float("١٢٣") == 123.0``) whereas JS
    ``Number("١٢٣")`` is NaN, so the server must reject them to keep number-default
    parity with the SPA. Plain ASCII floats are still accepted."""
    nump = {"type": "number"}
    assert _coerce_to_schema_type("1.5", nump) == 1.5  # ASCII float → accepted
    assert _coerce_to_schema_type("١٢٣", nump) is None  # Arabic-Indic → rejected
    assert _coerce_to_schema_type("١.٢", nump) is None  # Unicode digits, ASCII point
    assert _coerce_to_schema_type("१२३", nump) is None  # Devanagari → rejected


def test_apply_env_defaults_decimal_styled_integer_reaches_params():
    """End-to-end of the divergence: a decimal-styled integer default seeds the
    parsed int into params (pre-fix it was dropped, leaving the schema default
    while the SPA slider showed 5)."""
    params = {}
    apply_env_defaults(params, _PROPS, env={"WYRD_DEFAULT_COUNT": "5.0"})
    assert params["count"] == 5
    assert isinstance(params["count"], int)


def test_env_default_novelty_reaches_generator_via_dispatch(monkeypatch):
    """End-to-end: with WYRD_DEFAULT_NOVELTY set and the request OMITTING
    novelty (the SPA-at-default case), the dispatched generation runs with the
    env default — the echoed parameters carry novelty=0.1, not the 0.0 schema
    default. Pins the bug the no-op fix closes (wyrd-7f22)."""
    for key in list(__import__("os").environ):
        if key.startswith(("WYRD_FF_", "WYRD_DEFAULT_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WYRD_DEFAULT_NOVELTY", "0.1")
    client = create_app().test_client()
    resp = client.post("/api/kenning", json={"culture": "english", "count": 1, "seed": 1})
    assert resp.status_code == 200
    assert resp.get_json()["parameters"]["novelty"] == 0.1


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
