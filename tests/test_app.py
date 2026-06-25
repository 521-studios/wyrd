"""Unit tests for wyrd.app helpers (count coercion, query-param
flattening, LOG_LEVEL configuration, dispatch trace lines)."""

from __future__ import annotations

import logging

import pytest

from wyrd.app import _coerce_count, _coerce_query_params, _configure_logging, create_app
from wyrd.generators.kenning import CULTURES


def test_coerce_count_accepts_int_in_range():
    for n in (1, 5, 10):
        assert _coerce_count(n) == n


def test_coerce_count_accepts_numeric_string():
    assert _coerce_count("3") == 3


def test_coerce_count_below_minimum_rejected():
    with pytest.raises(ValueError, match="between 1 and 10"):
        _coerce_count(0)


def test_coerce_count_above_maximum_rejected():
    with pytest.raises(ValueError, match="between 1 and 10"):
        _coerce_count(11)


def test_coerce_count_non_integer_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_count("five")


def test_coerce_query_params_single_value_flattened():
    assert _coerce_query_params({"culture": ["english"]}) == {"culture": "english"}


def test_coerce_query_params_repeated_key_kept_as_list():
    assert _coerce_query_params({"tags": ["religion", "tree"]}) == {"tags": ["religion", "tree"]}


def test_coerce_query_params_mixed():
    out = _coerce_query_params(
        {"culture": ["english"], "tags": ["religion", "tree"], "seed": ["42"]}
    )
    assert out == {"culture": "english", "tags": ["religion", "tree"], "seed": "42"}


# --- LOG_LEVEL configuration ----------------------------------------------


def test_configure_logging_defaults_wyrd_to_info(monkeypatch):
    """Without LOG_LEVEL set, _configure_logging pins the wyrd
    package logger to INFO. Root stays at WARNING so botocore /
    urllib3 / s3transfer's chatty internal DEBUG logs never surface
    no matter what."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    _configure_logging()
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("wyrd").level == logging.INFO


def test_configure_logging_honors_debug_env_on_wyrd_only(monkeypatch):
    """LOG_LEVEL=DEBUG flips wyrd's logger to DEBUG so the per-phase
    timing logs Kenning.generate emits become visible. Root stays
    at WARNING — staging hit 10s timeouts when DEBUG also caught
    botocore's hundreds-of-lines-per-request chatter."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    _configure_logging()
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("wyrd").level == logging.DEBUG
    # Restore so other tests don't inherit DEBUG cadence.
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    _configure_logging()


def test_configure_logging_unknown_level_falls_back_to_info(monkeypatch):
    """Garbage in LOG_LEVEL (typo, empty, etc.) silently falls back to
    INFO on wyrd so a misconfigured env var never crashes Lambda init."""
    monkeypatch.setenv("LOG_LEVEL", "GARBAGE")
    _configure_logging()
    assert logging.getLogger("wyrd").level == logging.INFO


def test_configure_logging_strips_whitespace_and_lowercase(monkeypatch):
    """Operator-supplied LOG_LEVEL is .strip().upper()-normalized so
    ``debug`` / ``  DEBUG  `` both work."""
    monkeypatch.setenv("LOG_LEVEL", "  debug  ")
    _configure_logging()
    assert logging.getLogger("wyrd").level == logging.DEBUG
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    _configure_logging()


# --- dispatch trace lines (request-level timing logs) ---------------------


def test_dispatch_emits_ok_info_line_with_timing(caplog):
    """The INFO summary line per successful /api/<gen> request must
    carry ``dispatch ok``, the generator name, count, elapsed_ms, the
    resolved seed, and the operator's param dict — enough to triage
    a slow path from CloudWatch without a separate trace."""
    app = create_app()
    caplog.set_level(logging.INFO, logger="wyrd.app")
    with app.test_client() as client:
        response = client.get("/api/kenning?culture=english&count=1&seed=42")
    assert response.status_code == 200, response.data
    summary_lines = [rec for rec in caplog.records if "dispatch ok" in rec.getMessage()]
    assert summary_lines, (
        f"no dispatch-ok line emitted; got {[r.getMessage() for r in caplog.records]}"
    )
    msg = summary_lines[-1].getMessage()
    assert "generator=kenning" in msg
    assert "count=1" in msg
    assert "elapsed_ms=" in msg
    assert "seed=42" in msg
    assert "'culture': 'english'" in msg


def test_dispatch_emits_bad_params_info_line(caplog):
    """A ValueError-raising request (e.g. unknown culture) still
    surfaces a dispatch-trace line — the bad_params variant carries
    the detail string so CloudWatch shows the operator's exact bad
    input next to the elapsed time it cost."""
    app = create_app()
    caplog.set_level(logging.INFO, logger="wyrd.app")
    with app.test_client() as client:
        # 'count' must be 1-10; 11 raises ValueError inside _coerce_count.
        response = client.get("/api/kenning?culture=english&count=11&seed=42")
    assert response.status_code == 400
    error_lines = [rec for rec in caplog.records if "dispatch bad_params" in rec.getMessage()]
    assert error_lines, "bad_params dispatch line not emitted"
    msg = error_lines[-1].getMessage()
    assert "generator=kenning" in msg
    assert "elapsed_ms=" in msg
    assert "detail=" in msg


def test_dispatch_empty_eligible_pool_maps_to_422(caplog):
    """wyrd-hh2m: a valid request whose tag filters the eligible pool empty
    ('religion' isn't a real corpus tag — an API-only path, the SPA constrains
    tags to a valid enum) returns a clean 422 'empty_eligible_pool', not an
    opaque 500 — and emits its own dispatch-trace line."""
    app = create_app()
    caplog.set_level(logging.INFO, logger="wyrd.app")
    with app.test_client() as client:
        response = client.post(
            "/api/kenning",
            json={"culture": "english", "tags": ["religion"], "count": 1, "seed": 0},
        )
    assert response.status_code == 422
    body = response.get_json()
    assert body["error"] == "empty_eligible_pool"
    assert "no eligible name" in body["detail"]
    lines = [r for r in caplog.records if "dispatch empty_eligible_pool" in r.getMessage()]
    assert lines, "empty_eligible_pool dispatch line not emitted"


def test_empty_eligible_pool_exception_hierarchy_invariants():
    """wyrd-hh2m load-bearing invariant: EmptyEligiblePool must NOT subclass
    ValueError, or the dispatcher's generic ValueError->400 clause would shadow
    the 422 mapping. NoEligibleReplacementError (regenerate's per-slot flavor) is
    unified under it, so it maps to 422 too. A future `class
    KenningError(ValueError)` regression would pass the other tests silently;
    this pins it."""
    from wyrd.generators.kenning.errors import EmptyEligiblePool, KenningError
    from wyrd.generators.kenning.generators.kenning_regenerate import (
        NoEligibleReplacementError,
    )

    assert not issubclass(EmptyEligiblePool, ValueError)
    assert issubclass(EmptyEligiblePool, KenningError)
    assert issubclass(NoEligibleReplacementError, EmptyEligiblePool)


def test_dispatch_maps_empty_eligible_pool_to_422_via_monkeypatched_raise(monkeypatch):
    """Drives the 422 branch directly (decoupled from generator internals): any
    generator raising EmptyEligiblePool from generate() → 422, so the mapping
    stays exercised even if the 'religion' magic input ever becomes placeable."""
    from wyrd.generators.kenning.errors import EmptyEligiblePool

    def _boom(self, params, seed):
        raise EmptyEligiblePool("forced empty pool")

    monkeypatch.setattr(
        "wyrd.generators.kenning.generators.kenning.Kenning.generate", _boom, raising=True
    )
    app = create_app()
    with app.test_client() as client:
        response = client.post("/api/kenning", json={"culture": "english", "count": 1, "seed": 0})
    assert response.status_code == 422
    assert response.get_json()["error"] == "empty_eligible_pool"


def test_dispatch_debug_emits_per_result_breakdown(caplog):
    """When DEBUG is on, dispatch emits a per_result_ms=[…] line so
    a slow sub-seed within a count>1 batch can be localized."""
    app = create_app()
    caplog.set_level(logging.DEBUG, logger="wyrd.app")
    with app.test_client() as client:
        response = client.get("/api/kenning?culture=english&count=3&seed=42")
    assert response.status_code == 200
    debug_lines = [rec for rec in caplog.records if "per_result_ms=" in rec.getMessage()]
    assert debug_lines, "per_result_ms debug line missing under DEBUG"


def test_kenning_generate_debug_emits_phase_breakdown(caplog):
    """Kenning.generate's per-phase load/sample/render timing line
    fires under DEBUG. Splitting the phases is what tells an operator
    whether a slow request is cold-start load, sample loop, or
    rendering work."""
    app = create_app()
    caplog.set_level(logging.DEBUG, logger="wyrd.generators.kenning.generators.kenning")
    with app.test_client() as client:
        response = client.get("/api/kenning?culture=english&count=1&seed=42")
    assert response.status_code == 200
    phase_lines = [rec for rec in caplog.records if "kenning.generate" in rec.getMessage()]
    assert phase_lines, "kenning.generate phase-timing line missing under DEBUG"
    msg = phase_lines[-1].getMessage()
    assert "load_ms=" in msg
    assert "sample_ms=" in msg
    assert "render_ms=" in msg


# --- wyrd-t70w regression: SPA-shape vector dispatch ----------------------


# Mirrors Field.svelte's default-initialization logic in spa-next/src/
# components/Field.svelte (the ``$effect`` guarded by ``params[fieldKey]
# === undefined``): every schema field gets a default value, and fields
# with no schema default whose type isn't array/boolean fall back to
# ``''`` (empty string). For ``scoring_weights`` (type=object) and
# ``priors_path`` (type=string, no default) that means the SPA POSTs
# them as empty strings, not null/missing/{}.
_SPA_DEFAULT_PARAMS = {
    "tags": [],
    "count": 1,
    "spelling_variety": 0.0,
    "novelty": 0.0,
    "inflection_density": 0.0,
    "mood": [],
    "include_fiction": False,
    "harshness": 0.0,
    "era": "",
    "cohesion": 0.0,
    "stratum": "",
    "manorial_affix": 0.0,
    "joiner_density": 0.0,
    "scoring_mode": "vector",
    "priors_path": "",
    "scoring_weights": "",
    "packs": [],
    "seed": 42,
}


@pytest.mark.parametrize("culture", sorted(CULTURES))
def test_post_vector_with_spa_default_params_returns_a_name(culture):
    """wyrd-t70w regression: a POST to ``/api/kenning`` carrying the
    exact param shape Field.svelte produces for ``scoring_mode=vector``
    must return HTTP 200 with at least one rendered name across every
    shipped culture. Specifically: ``scoring_weights=''`` (text-input
    default for the type=object schema field) and ``priors_path=''``
    (text-input default) must NOT cause the vector dispatch to fall
    through to "no eligible name". The original ticket described an
    HTTP 400 with that message; the dispatch was hardened in #359
    (lemma_ref + uniform-tag default) so this test pins the working
    contract."""
    app = create_app()
    with app.test_client() as client:
        response = client.post(
            "/api/kenning",
            json={**_SPA_DEFAULT_PARAMS, "culture": culture},
        )
    assert response.status_code == 200, (
        f"vector dispatch regressed for culture={culture!r}: "
        f"HTTP {response.status_code} → {response.get_data(as_text=True)[:500]}"
    )
    body = response.get_json()
    results = body.get("results", [])
    assert results, f"empty results for culture={culture!r}: {body}"
    assert results[0].get("result"), (
        f"missing 'result' string for culture={culture!r}: {results[0]}"
    )


def test_manifest_exposes_kenning_era_stages():
    # wyrd-rogd.12: the per-family era axis is the single source of truth for the
    # col-3 grid columns + the Configure era control, exposed on the manifest.
    from wyrd.generators.kenning.era.cells import family_stage_order

    app = create_app()
    manifest = app.test_client().get("/api/manifest").get_json()
    kenning = next(g for g in manifest["generators"] if g["name"] == "kenning")
    assert kenning["era_stages"]["english"] == family_stage_order("english")
    assert kenning["era_stages"]["english"] == [
        "old-english",
        "middle-english",
        "modern-english",
    ]


def test_manifest_era_stages_optional_for_other_generators():
    # Generators without an era axis carry era_stages: null (getattr default).
    app = create_app()
    manifest = app.test_client().get("/api/manifest").get_json()
    for g in manifest["generators"]:
        assert "era_stages" in g  # field always present
        if g["name"] != "kenning":
            # other generators don't define the attribute → None
            assert g["era_stages"] is None or isinstance(g["era_stages"], dict)


def test_stripped_str_treats_non_string_as_absent():
    from wyrd.app import _stripped_str

    assert _stripped_str("  hi  ") == "hi"
    assert _stripped_str("hi") == "hi"
    # non-strings normalize to "" (absent) instead of raising on .strip()
    assert _stripped_str(5) == ""
    assert _stripped_str(None) == ""
    assert _stripped_str(["a", "b"]) == ""
    assert _stripped_str({"x": 1}) == ""


def test_record_defect_non_string_reason_is_400_not_500():
    """A malformed defect report whose reason/generator is a non-string (number,
    list, object) is a clean 400 'bad_report' — not a 500 from calling .strip()
    on a non-string (the old `(5 or "").strip()` AttributeError)."""
    app = create_app()
    with app.test_client() as client:
        for body in (
            {"reason": 5, "generator": "kenning"},
            {"reason": ["a", "b"], "generator": "kenning"},
            {"reason": {"x": 1}, "generator": "kenning"},
            {"reason": "ok", "generator": 7},
            {"reason": "ok", "generator": ["k"]},
        ):
            resp = client.post("/api/defects", json=body)
            assert resp.status_code == 400, body
            assert resp.get_json()["error"] == "bad_report"
