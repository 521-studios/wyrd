"""Smoke tests for the Kenning generator and the dispatcher around it."""

from __future__ import annotations

from wyrd.app import create_app
from wyrd.generators.kenning import CULTURES, Kenning, available_tags


def test_each_culture_generates_a_name():
    k = Kenning()
    for culture in CULTURES:
        result = k.generate({"culture": culture}, seed=12345)
        assert result.result, f"empty result for {culture}"
        assert result.explanation, f"empty explanation for {culture}"


def test_seed_is_reproducible():
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42)
    b = k.generate({"culture": "english"}, seed=42)
    assert a.result == b.result
    assert a.explanation == b.explanation


def test_different_seeds_diverge():
    k = Kenning()
    seen = {k.generate({"culture": "english"}, seed=s).result for s in range(10)}
    assert len(seen) > 1, "10 distinct seeds yielded one name — RNG isn't propagating"


def test_components_are_structured():
    result = Kenning().generate({"culture": "english"}, seed=7)
    assert isinstance(result.components, list)
    assert all("usage" in c and "location" in c for c in result.components)


def test_available_tags_excludes_internal():
    tags = available_tags()
    assert "male name" not in tags
    assert "female name" not in tags
    assert "saint" not in tags


def test_manifest_lists_kenning():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/manifest")
    assert resp.status_code == 200
    body = resp.get_json()
    names = [g["name"] for g in body["generators"]]
    assert "kenning" in names


def test_get_endpoint_returns_envelope():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/kenning?culture=english&seed=99&count=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["generator"] == "kenning"
    assert body["seed"] == 99
    assert len(body["results"]) == 1
    assert body["results"][0]["result"]
    assert body["parameters"]["culture"] == "english"


def test_post_endpoint_accepts_json_body():
    app = create_app()
    client = app.test_client()
    resp = client.post("/api/kenning", json={"culture": "welsh", "seed": 100, "count": 1})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["generator"] == "kenning"
    assert body["seed"] == 100
    assert len(body["results"]) == 1


def test_default_count_is_five():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/kenning?culture=english&seed=99")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["results"]) == 5


def test_count_returns_n_results():
    app = create_app()
    client = app.test_client()
    resp = client.post("/api/kenning", json={"culture": "english", "count": 5, "seed": 7})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["results"]) == 5
    assert all(r["result"] for r in body["results"])


def test_count_is_seed_reproducible():
    app = create_app()
    client = app.test_client()
    a = client.post(
        "/api/kenning", json={"culture": "english", "count": 4, "seed": 2026}
    ).get_json()
    b = client.post(
        "/api/kenning", json={"culture": "english", "count": 4, "seed": 2026}
    ).get_json()
    assert [r["result"] for r in a["results"]] == [r["result"] for r in b["results"]]


def test_count_out_of_range_is_400():
    app = create_app()
    client = app.test_client()
    resp = client.post("/api/kenning", json={"culture": "english", "count": 11})
    assert resp.status_code == 400


def test_unknown_culture_is_a_400():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/kenning?culture=martian")
    assert resp.status_code == 400


def test_unknown_generator_is_404():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/nope")
    assert resp.status_code == 404
