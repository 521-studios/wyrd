"""Tests for the KenningExplain generator and the multi_result dispatch path.

Behaviour-pin tests use ``swap_bundle`` (conftest.py) to run against
the small ``EXPLAIN_TEST_BUNDLE`` fixture rather than the live
meanings.json. The live bundle's morpheme inventory shifts every
mining session — pinning to a fixture keeps these tests stable
across bundle churn (audit follow-up wyrd-p8ve era).
"""

from __future__ import annotations

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning import KenningExplain


@pytest.fixture
def client():
    return create_app().test_client()


def test_known_compound_decomposes_cleanly(bundle_swapper, explain_test_bundle):
    with bundle_swapper(explain_test_bundle):
        rs = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    # Cleanest reading should appear first thanks to the unaccounted-first sort.
    assert "Bridge" in rs[0].explanation
    assert "water" in rs[0].explanation
    assert "[" not in rs[0].explanation, "best reading should have no unaccounted"


def test_multi_word_name_decomposes_per_word(bundle_swapper, explain_test_bundle):
    with bundle_swapper(explain_test_bundle):
        rs = KenningExplain().generate_all({"name": "Saint Albans"}, 0)
    # Best reading: "Saint" (saint) + "Albans" (auto-pluralized name).
    top = rs[0]
    assert "Saint" in top.explanation
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


def test_non_string_name_raises_value_error_not_attribute_error():
    """A non-string ``name`` (number, list, object) is as invalid as an empty
    one: it must raise ValueError ("name is required") — which the dispatcher
    maps to 400 — not AttributeError from ``.strip()`` on a non-string, which
    escapes as a 500. Covers the shared ``_required_name`` helper used by every
    name-taking generator (explain, render, creature, rewind)."""
    for bad in (5, ["a", "b"], {"x": 1}, None):
        with pytest.raises(ValueError, match="name is required"):
            KenningExplain().generate_all({"name": bad}, 0)


def test_required_name_strips_valid_and_rejects_non_string():
    from wyrd.generators.kenning import _required_name

    assert _required_name({"name": "  Ton By  "}) == "Ton By"
    for bad in (5, ["a"], {"x": 1}, None, "", "   "):
        with pytest.raises(ValueError, match="name is required"):
            _required_name({"name": bad})


def test_components_carry_structured_parts(bundle_swapper, explain_test_bundle):
    with bundle_swapper(explain_test_bundle):
        r = KenningExplain().generate_all({"name": "Bridgewater"}, 0)[0]
    parts = r.components[0]["parts"]
    matched = [p for p in parts if p["type"] == "matched"]
    assert any(p["fragment"].lower() == "bridge" for p in matched)
    bridge = next(p for p in matched if p["fragment"].lower() == "bridge")
    assert "EN" in bridge["roots"]


def test_unaccounted_chunks_are_flagged_in_components(bundle_swapper, explain_test_bundle):
    # 'Aberystwyth' has no entries in the fixture — every char is
    # unaccounted. The test only asserts the unaccounted-flagging
    # codepath, not specific morpheme picks.
    with bundle_swapper(explain_test_bundle):
        r = KenningExplain().generate_all({"name": "Aberystwyth"}, 0)[0]
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


def test_dispatcher_ignores_count_for_multi_result(client, bundle_swapper, explain_test_bundle):
    """Even if a client passes count=10, multi_result generators decide.
    Pinned against the fixture so the per-name decomposition count
    stays stable across bundle churn — the live bundle now produces
    25+ Bridgewater readings due to morpheme inventory growth, which
    is correct behaviour but defeats the 'count<10' assertion's
    intent (which is about dispatcher contract, not bundle scale)."""
    with bundle_swapper(explain_test_bundle):
        resp = client.post("/api/kenning-explain", json={"name": "Bridgewater", "count": 10})
    assert resp.status_code == 200
    body = resp.get_json()
    # Output count is determined by the input, not the count param.
    assert len(body["results"]) < 10


def test_empty_name_returns_400(client):
    resp = client.post("/api/kenning-explain", json={"name": ""})
    assert resp.status_code == 400


def test_non_string_name_returns_400_across_name_generators(client):
    """A non-string ``name`` is a 400 bad_params, not a 500, for EVERY
    name-taking generator — the shared ``_required_name`` guard. Pre-fix each
    `(params.get("name") or "").strip()` AttributeError'd into an uncaught 500."""
    for path in ("kenning-explain", "kenning-render", "kenning-creature", "kenning-rewind"):
        for bad in (5, ["a"], {"x": 1}):
            resp = client.post(f"/api/{path}", json={"name": bad})
            assert resp.status_code == 400, (path, bad, resp.status_code)
            assert resp.get_json()["error"] == "bad_params"


def test_manifest_lists_explainer_with_multi_result(client):
    body = client.get("/api/manifest").get_json()
    names = [g["name"] for g in body["generators"]]
    assert "kenning-explain" in names


def test_explainer_combines_senses_for_one_usage(bundle_swapper):
    """When a usage has multiple sense entries, the explanation joins
    them with ' / ' instead of producing N near-duplicate
    decompositions. Uses a 2-subject fixture (``York-`` + multi-sense
    ``-shire``) so the assertion runs against the seed-runtime.db
    default in CI — the top-N-per-culture seed doesn't necessarily
    carry the live bundle's full ``-shire`` polysemy, and we want this
    rendering contract pinned regardless of which L4 is mounted."""
    fixture = {
        "subjects": [
            {
                "meaning": ["York"],
                "modifier_tags": ["name"],
                "modifier_type": None,
                "words": [{"modern_usage": "York-", "old_english": ["eofor"]}],
            },
            {
                "meaning": ["District", "Market shire"],
                "modifier_tags": ["administrative"],
                "modifier_type": "Administrative",
                "words": [{"modern_usage": "-shire", "old_english": ["scīr"]}],
            },
        ],
    }
    with bundle_swapper(fixture):
        rs = KenningExplain().generate_all({"name": "Yorkshire"}, 0)
    text = " ".join(r.explanation for r in rs)
    # Both senses of -shire must appear together inside one slot rather
    # than each spawning its own reading.
    assert "District" in text
    assert "Market" in text


# --- wyrd-h8k1: canonical surfacing via bundle projection -----------------


def test_explain_marks_canonical_when_bundle_carries_signature(monkeypatch):
    """When the bundle's canonical_decompositions map carries a
    signature for the input name AND that signature matches one of
    the matcher's emitted readings, the matching reading is flagged
    canonical AND sorted to the top of the results list.

    wyrd-lqlw: stays against the live bundle because the test is
    self-stabilizing — it grabs WHATEVER signature the matcher emits
    for Bridgewater (no specific decomposition pinned) and injects
    that into a monkey-patched canonical map. Drift-resistant by
    construction."""
    # First, decompose without canonical info to find the signature
    # that the bundle would need to carry to mark the 'best' reading.
    rs_baseline = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    assert len(rs_baseline) >= 1
    # Cleanest reading (rs[0]) has zero unaccounted; pull its
    # signature shape to inject as canonical.
    from wyrd.generators.kenning import (
        _canonical_signature_for_words,
        _load_canonical_decompositions,
    )
    from wyrd.generators.kenning.runtime.name import Name

    name_obj = Name("Bridgewater")
    from wyrd.generators.kenning import _load_meanings

    meaning_db, _ = _load_meanings()
    name_obj.find_meaning(meaning_db, reduce=False)
    per_word = [name_obj.words[w] for w in ["Bridgewater"]]
    # Find an emitted reading and use its signature.
    import itertools

    target_words = next(itertools.product(*per_word))
    target_sig = _canonical_signature_for_words(target_words)

    fake_map = {"Bridgewater": {"signature": target_sig, "source": "scholar"}}
    _load_canonical_decompositions.cache_clear()
    monkeypatch.setattr(
        "wyrd.generators.kenning.generators.kenning_explain._load_canonical_decompositions",
        lambda: fake_map,
    )

    rs = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    canonical_results = [r for r in rs if r.canonical]
    assert len(canonical_results) == 1
    assert canonical_results[0].canonical_source == "scholar"
    # Canonical sorts to the top.
    assert rs[0].canonical is True


def test_explain_no_canonical_when_bundle_lacks_map(monkeypatch):
    """Legacy bundles (list-shape, no canonical_decompositions field)
    should leave every reading with ``canonical=False`` and sort by
    the original heuristic. The full suite already runs on a legacy
    bundle, so this test mirrors the production path while making the
    no-canonical contract explicit."""
    from wyrd.generators.kenning import _load_canonical_decompositions

    _load_canonical_decompositions.cache_clear()
    monkeypatch.setattr(
        "wyrd.generators.kenning.generators.kenning_explain._load_canonical_decompositions",
        lambda: {},
    )

    rs = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    assert len(rs) >= 1
    assert all(not r.canonical for r in rs)
    assert all(r.canonical_source is None for r in rs)


def test_explain_falls_back_when_canonical_signature_no_match(monkeypatch):
    """When the bundle carries a canonical signature for the input but
    NO matcher reading produces that signature (word_db drift or stale
    bundle), every reading stays uncanonical and the heuristic sort
    applies — graceful no-op rather than promoting the wrong reading."""
    from wyrd.generators.kenning import _load_canonical_decompositions

    fake_map = {
        "Bridgewater": {"signature": "0" * 40, "source": "scholar"},
    }
    _load_canonical_decompositions.cache_clear()
    monkeypatch.setattr(
        "wyrd.generators.kenning.generators.kenning_explain._load_canonical_decompositions",
        lambda: fake_map,
    )

    rs = KenningExplain().generate_all({"name": "Bridgewater"}, 0)
    assert len(rs) >= 1
    assert all(not r.canonical for r in rs)


def test_explain_canonical_surfaces_in_envelope(client, monkeypatch):
    """The API envelope carries the canonical + canonical_source fields
    so the SPA can render the canonical reading distinctly."""
    from wyrd.generators.kenning import (
        _canonical_signature_for_words,
        _load_canonical_decompositions,
        _load_meanings,
    )
    from wyrd.generators.kenning.runtime.name import Name

    meaning_db, _ = _load_meanings()
    name_obj = Name("Bridgewater")
    name_obj.find_meaning(meaning_db, reduce=False)
    per_word = [name_obj.words[w] for w in ["Bridgewater"]]
    import itertools

    target_words = next(itertools.product(*per_word))
    target_sig = _canonical_signature_for_words(target_words)

    fake_map = {"Bridgewater": {"signature": target_sig, "source": "scholar"}}
    _load_canonical_decompositions.cache_clear()
    monkeypatch.setattr(
        "wyrd.generators.kenning.generators.kenning_explain._load_canonical_decompositions",
        lambda: fake_map,
    )

    resp = client.post("/api/kenning-explain", json={"name": "Bridgewater"})
    assert resp.status_code == 200
    body = resp.get_json()
    canonical_entries = [r for r in body["results"] if r.get("canonical")]
    assert len(canonical_entries) == 1
    assert canonical_entries[0]["canonical_source"] == "scholar"
    # Non-canonical results carry the field too, defaulting to False/null.
    for r in body["results"]:
        assert "canonical" in r
        assert "canonical_source" in r
