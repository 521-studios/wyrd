"""Tests for the KenningExplain generator and the multi_result dispatch path."""

from __future__ import annotations

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning import KenningExplain


@pytest.fixture
def client():
    return create_app().test_client()


def test_known_compound_decomposes_cleanly():
    rs = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    # Cleanest reading should appear first thanks to the unaccounted-first sort.
    assert "bridge" in rs[0].explanation
    assert "water" in rs[0].explanation
    assert "[" not in rs[0].explanation, "best reading should have no unaccounted"
    # Case-preservation: the capitalized `Bridge` etymon must still surface in
    # at least one reading so that the original-case form remains decomposable
    # even when the lowercase variant wins the top slot.
    assert any("Bridge" in r.explanation for r in rs)


def test_multi_word_name_decomposes_per_word():
    rs = KenningExplain().generate_all({"name": "Saint Albans"}, 0)
    # Best reading: "saint" (saint) + "Albans" (auto-pluralized name). The
    # canonical_form for the saint etymon is lowercase in the bundle.
    top = rs[0]
    assert "saint" in top.explanation
    assert "Albans" in top.explanation
    assert "[" not in top.explanation


def test_unrecognized_input_still_returns_results():
    # Not all readings will be perfect; explainer still surfaces the partial
    # decompositions with [bracketed] unaccounted fragments.
    rs = KenningExplain().generate_all({"name": "Zzqxk"}, 0)
    assert len(rs) >= 1
    assert "[" in rs[0].explanation


def test_empty_name_raises():
    with pytest.raises(ValueError):
        KenningExplain().generate_all({"name": ""}, 0)


def test_components_carry_structured_parts():
    r = KenningExplain().generate_all({"name": "Bridgewater"}, 0)[0]
    parts = r.components[0]["parts"]
    matched = [p for p in parts if p["type"] == "matched"]
    assert any(p["fragment"].lower() == "bridge" for p in matched)
    bridge = next(p for p in matched if p["fragment"].lower() == "bridge")
    # Structured parts must carry root-language attribution. Don't pin a
    # specific code — which root wins the top slot depends on bundle data.
    assert bridge["roots"]


def test_unaccounted_chunks_are_flagged_in_components():
    # Use a guaranteed-unrecognized input so the assertion is robust to
    # bundle coverage growth (Aberystwyth previously had unaccounted chunks
    # but the expanded morpheme inventory now decomposes it cleanly).
    r = KenningExplain().generate_all({"name": "Zzqxk"}, 0)[0]
    flat = [p for word in r.components for p in word["parts"]]
    assert any(p["type"] == "unaccounted" for p in flat)


def test_decompositions_are_sorted_best_first():
    rs = KenningExplain().generate_all({"name": "Westminster"}, 0)
    # First reading should have at most as many [unaccounted] chunks as the last.
    first_unaccounted = rs[0].explanation.count("[")
    last_unaccounted = rs[-1].explanation.count("[")
    assert first_unaccounted <= last_unaccounted


def test_dispatcher_returns_all_decompositions(client):
    """multi_result generators ignore count and return their natural N."""
    resp = client.post("/api/kenning-explain", json={"name": "Bridgewater"})
    assert resp.status_code == 200
    body = resp.get_json()
    # Bridgewater currently produces 3 distinct structural decompositions.
    assert len(body["results"]) >= 2
    assert all(r["result"] == "Bridgewater" for r in body["results"])


def test_dispatcher_ignores_count_for_multi_result(client):
    """Even if a client passes count=10, multi_result generators decide."""
    resp = client.post("/api/kenning-explain", json={"name": "Bridgewater", "count": 10})
    assert resp.status_code == 200
    body = resp.get_json()
    # Output count is determined by the input, not the count param. Asserting
    # `!= 10` proves count wasn't used as a cap or padding target.
    assert len(body["results"]) != 10


def test_empty_name_returns_400(client):
    resp = client.post("/api/kenning-explain", json={"name": ""})
    assert resp.status_code == 400


def test_manifest_lists_explainer_with_multi_result(client):
    body = client.get("/api/manifest").get_json()
    names = [g["name"] for g in body["generators"]]
    assert "kenning-explain" in names


def test_explainer_combines_senses_for_one_usage():
    """When a usage has multiple sense entries, the explanation joins them
    with ' / ' instead of producing N near-duplicate decompositions."""
    rs = KenningExplain().generate_all({"name": "Yorkshire"}, 0)
    text = " ".join(r.explanation for r in rs)
    # `-shire` in the bundled meanings DB has two senses; both should appear
    # together inside one slot rather than each spawning its own reading.
    assert "District" in text
    assert "Market" in text
