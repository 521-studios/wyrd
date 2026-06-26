"""KenningEraMap honors its `count` param over the API (region size), and
`_region_count` coerces/validates it.

Regression: the dispatcher POPped `count` for multi_result generators, pinning
era-map to its default 6 over the API — even though era-map sizes its output to
`count` (the CLI, which calls generate_all directly, never hit the pop).
"""

from __future__ import annotations

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning.generators.kenning_era_map import _region_count


@pytest.fixture
def client():
    return create_app().test_client()


def test_region_count_coerces_and_validates():
    # blank / missing → default 6
    assert _region_count({}) == 6
    assert _region_count({"count": None}) == 6
    assert _region_count({"count": ""}) == 6
    # a CLI/POST int and a GET numeric-string coerce identically
    assert _region_count({"count": 3}) == 3
    assert _region_count({"count": "5"}) == 5
    assert _region_count({"count": 1}) == 1  # schema min
    assert _region_count({"count": 50}) == 50  # schema max
    # non-coercible → ValueError (→ 400), never a bare TypeError → 500
    for bad in ([1, 2], {"x": 1}, "abc"):
        with pytest.raises(ValueError, match="must be an integer"):
            _region_count({"count": bad})
    # out of the schema's 1-50 range → ValueError (→ 400)
    for oor in (0, -1, 51, 100):
        with pytest.raises(ValueError, match="between 1 and 50"):
            _region_count({"count": oor})


def test_era_map_api_honors_count(client):
    """A fixed seed keeps the roll deterministic; the dedup makes the exact
    unique-name count seed-dependent, so assert the honored relationship rather
    than brittle equalities. Pre-fix EVERY count returned 6."""

    def n(count):
        resp = client.post(
            "/api/kenning-era-map",
            json={"culture": "english", "count": count, "seed": 42},
        )
        assert resp.status_code == 200, resp.data
        return len(resp.get_json()["results"])

    assert n(1) == 1  # a single roll → one name
    assert n(3) <= 3  # pre-fix this was pinned to 6 (> 3) — proves count is honored
    assert n(10) >= n(3)  # more rolls (same seed prefix) → at least as many unique
    assert n(10) <= 10  # never more than requested

    # unspecified count → the schema default of 6
    resp = client.post("/api/kenning-era-map", json={"culture": "english", "seed": 42})
    assert len(resp.get_json()["results"]) == 6


def test_era_map_api_bad_count_is_400(client):
    """A non-coercible or out-of-range count is a 400 bad_params, not a 500.
    (Reachable only now that the dispatcher passes `count` through.)"""
    for bad in ([1, 2], {"x": 1}, "abc", 0, 99):
        resp = client.post("/api/kenning-era-map", json={"culture": "english", "count": bad})
        assert resp.status_code == 400, (bad, resp.status_code)
        assert resp.get_json()["error"] == "bad_params"
