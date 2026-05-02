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


def test_default_novelty_zero_matches_unspecified():
    """A fresh Kenning() at default params (no novelty) and at explicit
    novelty=0.0 produce the same output. Guards the 'novelty=0 is identity'
    contract for D17."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "novelty": 0.0}, seed=42).result
    assert a == b


def test_high_novelty_shifts_distribution():
    """At novelty=1, the same seed produces a different name than novelty=0.
    Verifies the knob actually changes the distribution rather than being a
    no-op. Seed-stable so the assertion isn't flaky."""
    k = Kenning()
    canonical = k.generate({"culture": "english"}, seed=42).result
    novel = k.generate({"culture": "english", "novelty": 1.0}, seed=42).result
    assert canonical != novel


def test_novelty_is_seed_stable():
    """Same (params, seed) tuple yields the same name even with novelty>0 —
    the reproducibility contract holds across the new knob."""
    k = Kenning()
    a = k.generate({"culture": "english", "novelty": 0.7}, seed=42).result
    b = k.generate({"culture": "english", "novelty": 0.7}, seed=42).result
    assert a == b


def test_default_spelling_variety_zero_matches_unspecified():
    """A fresh Kenning() at default params and at explicit
    spelling_variety=0.0 produce the same output."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "spelling_variety": 0.0}, seed=42).result
    assert a == b


def test_spelling_variety_is_seed_stable():
    """Same (params, seed) tuple yields the same name even with
    spelling_variety>0 — the reproducibility contract holds across D18."""
    k = Kenning()
    a = k.generate({"culture": "english", "spelling_variety": 0.7}, seed=42).result
    b = k.generate({"culture": "english", "spelling_variety": 0.7}, seed=42).result
    assert a == b


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


def test_manifest_kenning_has_details_and_legend():
    app = create_app()
    client = app.test_client()
    body = client.get("/api/manifest").get_json()
    kenning = next(g for g in body["generators"] if g["name"] == "kenning")
    assert kenning["details"], "kenning should publish a details panel"
    assert "morphemes" in kenning["details"]
    legend_codes = {entry["code"] for entry in kenning["legend"]}
    assert legend_codes == {"EN", "SC", "FR", "CL", "LA", "GE", "GR"}
    for entry in kenning["legend"]:
        assert entry["name"], f"legend entry {entry['code']} missing name"


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
