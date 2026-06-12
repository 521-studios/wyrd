"""wyrd-kqyf: the culture-agnostic ``present-day`` era token.

Era stages are per-culture (modern-english / welsh / irish), so a literal
stage can't be a global env default — ``WYRD_DEFAULT_ERA=modern-english``
would 4xx every non-english-family culture. The token translates to the
request culture's present-day stage and then behaves byte-identically to an
explicit stage pick (the stage's union year range as the inventory filter +
the force-modern render). Runs against the committed seed-runtime.db."""

from __future__ import annotations

import pytest

from wyrd.generators.kenning import (
    _CULTURE_TO_ERA_FAMILY,
    _resolve_era_param,
    _resolve_era_render_language,
)
from wyrd.generators.kenning.era.cells import family_stage_order
from wyrd.generators.kenning.generators import Kenning

pytestmark = pytest.mark.no_lexicon_isolation

CULTURES = sorted(_CULTURE_TO_ERA_FAMILY)


def _present_day_stage(culture: str) -> str:
    return family_stage_order(_CULTURE_TO_ERA_FAMILY[culture])[-1]


@pytest.mark.parametrize("culture", CULTURES)
def test_token_resolves_to_the_cultures_present_day_stage_range(culture):
    assert _resolve_era_param("present-day", culture) == _resolve_era_param(
        _present_day_stage(culture), culture
    )


@pytest.mark.parametrize("culture", CULTURES)
def test_token_renders_force_modern(culture):
    """Contemporary stages resolve to a None render language (wyrd-3vju.2
    force-modern); the token must inherit that exactly."""
    assert _resolve_era_render_language("present-day", culture) is None


@pytest.mark.parametrize("culture", CULTURES)
def test_generate_with_token_equals_explicit_stage(culture):
    """Same seed + token vs the explicit present-day stage must produce the
    same name — the token is a pure alias, not a parallel code path."""
    via_token = Kenning().generate({"culture": culture, "era": "present-day"}, seed=42)
    via_stage = Kenning().generate(
        {"culture": culture, "era": _present_day_stage(culture)}, seed=42
    )
    assert via_token.result == via_stage.result
    assert via_token.result_modern == via_stage.result_modern


def test_token_generation_is_deterministic():
    """Same params + same seed twice must produce the same name — the
    translation layer must not introduce any per-call variability."""
    params = {"culture": "welsh", "era": "present-day"}
    first = Kenning().generate(dict(params), seed=42)
    second = Kenning().generate(dict(params), seed=42)
    assert first.result == second.result
    assert first.result_modern == second.result_modern


def test_env_default_applies_to_requests_that_omit_era(monkeypatch):
    """The deployed contract (terraform feature_flag_defaults → WYRD_DEFAULT_ERA):
    a request that omits era gets the token via apply_env_defaults, for EVERY
    culture — including non-english families a literal stage default would 4xx."""
    from wyrd.app import create_app

    monkeypatch.setenv("WYRD_DEFAULT_ERA", "present-day")
    app = create_app()
    with app.test_client() as client:
        for culture in ("english", "welsh"):
            defaulted = client.post(
                "/api/kenning", json={"culture": culture, "count": 1, "seed": 42}
            )
            assert defaulted.status_code == 200
            explicit = client.post(
                "/api/kenning",
                json={"culture": culture, "count": 1, "seed": 42, "era": "present-day"},
            )
            assert (
                defaulted.get_json()["results"][0]["result"]
                == explicit.get_json()["results"][0]["result"]
            )


def test_explicit_empty_era_still_means_mixed(monkeypatch):
    """An explicit era="" (the SPA's Mixed Era pick) must NOT be overridden by
    the env default — apply_env_defaults only fills ABSENT options."""
    from wyrd.app import create_app

    monkeypatch.setenv("WYRD_DEFAULT_ERA", "present-day")
    app = create_app()
    with app.test_client() as client:
        mixed = client.post(
            "/api/kenning", json={"culture": "english", "count": 1, "seed": 42, "era": ""}
        )
        assert mixed.status_code == 200
        # Mixed-era (native per-morpheme) output differs from the force-modern
        # default for this seed; equality here would mean the "" was clobbered.
        defaulted = client.post("/api/kenning", json={"culture": "english", "count": 1, "seed": 42})
        assert (
            mixed.get_json()["results"][0]["result"] != defaulted.get_json()["results"][0]["result"]
        )
