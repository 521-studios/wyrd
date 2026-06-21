"""wyrd-3tvd: a bounded requested era leans the priors-baseline toward that
period's naming FASHION via era_midpoint (D36.7 per-era frequency cells), on top
of the D46 inventory cutoff. Open-ended / no era stays at the D44 wildcard (0), so
the deployed present-day default is unchanged.
"""

from __future__ import annotations

import wyrd.generators.kenning.runtime.proportions as proportions
from wyrd.generators.kenning import _resolve_era_param
from wyrd.generators.kenning.generators.kenning import Kenning, _request_era_midpoint


def test_request_era_midpoint_rules():
    # bounded → integer midpoint
    assert _request_era_midpoint((800, 1100)) == 950
    # single-bound → that bound (NOT the priors open-bound offset)
    assert _request_era_midpoint((800, None)) == 800
    assert _request_era_midpoint((None, 1100)) == 1100
    # no era / both-None → 0 (the D44 wildcard)
    assert _request_era_midpoint(None) == 0
    assert _request_era_midpoint((None, None)) == 0


def _capture_era_midpoint(monkeypatch, params):
    """Run one generate() and return the era_midpoint that reached
    NameGenerator.select_via_vector."""
    captured: dict[str, int | None] = {}
    orig = proportions.NameGenerator.select_via_vector

    def spy(self, rng, **kw):
        captured["era_midpoint"] = kw.get("era_midpoint")
        return orig(self, rng, **kw)

    monkeypatch.setattr(proportions.NameGenerator, "select_via_vector", spy)
    Kenning().generate(params, 12345)
    return captured["era_midpoint"]


def test_bounded_era_passes_its_midpoint_to_the_scorer(monkeypatch):
    expected = _request_era_midpoint(_resolve_era_param("me", "english"))
    assert expected > 0  # Middle English is a bounded cell
    got = _capture_era_midpoint(monkeypatch, {"culture": "english", "era": "me"})
    assert got == expected


def test_no_era_keeps_the_wildcard_zero(monkeypatch):
    # default (no era) — unchanged behavior
    assert _capture_era_midpoint(monkeypatch, {"culture": "english"}) == 0
    # explicit empty era is also 'no filter' → wildcard
    assert _capture_era_midpoint(monkeypatch, {"culture": "english", "era": ""}) == 0
