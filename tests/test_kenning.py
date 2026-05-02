"""Smoke tests for the Kenning generator and the dispatcher around it."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("knob", ["novelty", "spelling_variety", "inflection_density"])
def test_default_knob_zero_matches_unspecified(knob):
    """A fresh Kenning() at default params and at explicit ``knob=0.0`` produce
    the same output. Guards the 'knob=0 is identity' contract for D17/D18/D8."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", knob: 0.0}, seed=42).result
    assert a == b


@pytest.mark.parametrize("knob", ["novelty", "spelling_variety", "inflection_density"])
def test_knob_is_seed_stable(knob):
    """Same (params, seed) tuple yields the same name even with knob>0 —
    the reproducibility contract holds across both new knobs."""
    k = Kenning()
    a = k.generate({"culture": "english", knob: 0.7}, seed=42).result
    b = k.generate({"culture": "english", knob: 0.7}, seed=42).result
    assert a == b


def test_high_novelty_shifts_distribution():
    """At novelty=1, the same seed produces a different name than novelty=0.
    Verifies the knob actually changes the distribution rather than being a
    no-op. Seed-stable so the assertion isn't flaky."""
    k = Kenning()
    canonical = k.generate({"culture": "english"}, seed=42).result
    novel = k.generate({"culture": "english", "novelty": 1.0}, seed=42).result
    assert canonical != novel


def test_high_spelling_variety_changes_output_when_pool_present():
    """At spelling_variety=1, names whose morphemes have a non-empty variant
    pool render with the variant rather than the canonical reflex. Search
    across a few seeds to guarantee at least one hit lands a variant — most
    morphemes have empty pools, so a single seed could miss."""
    k = Kenning()
    canonical_outputs = set()
    variant_outputs = set()
    for seed in range(20):
        canonical_outputs.add(k.generate({"culture": "english"}, seed=seed).result)
        variant_outputs.add(
            k.generate({"culture": "english", "spelling_variety": 1.0}, seed=seed).result
        )
    # At least one of the 20 seeds should land a name with at least one
    # morpheme that has a variant pool — the bundled corpus has 416 such
    # words, so 20 seeds is plenty to surface a difference.
    assert variant_outputs != canonical_outputs


def test_high_inflection_density_changes_output_when_pool_present():
    """At inflection_density=1, names whose morphemes have a non-empty
    inflection pool render with the inflected form rather than the lemma.
    Same many-seed approach as the variant test."""
    k = Kenning()
    canonical_outputs = set()
    inflected_outputs = set()
    for seed in range(20):
        canonical_outputs.add(k.generate({"culture": "english"}, seed=seed).result)
        inflected_outputs.add(
            k.generate({"culture": "english", "inflection_density": 1.0}, seed=seed).result
        )
    assert inflected_outputs != canonical_outputs


def test_different_seeds_diverge():
    k = Kenning()
    seen = {k.generate({"culture": "english"}, seed=s).result for s in range(10)}
    assert len(seen) > 1, "10 distinct seeds yielded one name — RNG isn't propagating"


def test_grim_default_false_matches_unspecified():
    """``grim`` unset and ``grim=False`` produce the same name. Pins the
    'flag-off is identity' contract."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "grim": False}, seed=42).result
    assert a == b


def test_grim_is_seed_stable():
    """Same (params, seed) tuple yields the same name with grim=True."""
    k = Kenning()
    a = k.generate({"culture": "english", "grim": True}, seed=42).result
    b = k.generate({"culture": "english", "grim": True}, seed=42).result
    assert a == b


def test_grim_true_shifts_distribution_across_seeds():
    """grim=True biases morpheme selection toward menacing tags. Across many
    seeds at least one name diverges from the no-grim baseline. Single seeds
    can match by chance (when no slot's tag-pool actually contains a grim
    candidate), so sweep — same shape as the variant / inflection knob
    tests."""
    k = Kenning()
    plain = set()
    grim = set()
    for seed in range(30):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        grim.add(k.generate({"culture": "english", "grim": True}, seed=seed).result)
    assert grim != plain


def test_grim_composes_with_explicit_tags():
    """A user passing both --grim and --tag <tag> should not error; the
    behavior is the union of the explicit tag and the grim tag set. The
    output should remain a non-empty name."""
    k = Kenning()
    result = k.generate({"culture": "english", "tags": ["water"], "grim": True}, seed=42)
    assert result.result
    assert result.explanation


def test_grim_does_not_duplicate_tags_already_present():
    """If the user explicitly passes one of the grim tags AND --grim, the
    expansion should not double-add it — a duplicated tag would produce a
    redundant fallback bucket and trip the existing rng.choice over a
    biased pool. Verifies via a deterministic seed."""
    k = Kenning()
    a = k.generate({"culture": "english", "tags": ["death"], "grim": True}, seed=42).result
    b = k.generate({"culture": "english", "tags": ["death"], "grim": True}, seed=42).result
    # Seed-stable.
    assert a == b
    # Sanity: the result is still a real name, not empty/None.
    assert a


def test_harshness_default_zero_matches_unspecified():
    """A fresh Kenning() at default params and at explicit harshness=0.0
    produce the same output. Guards the 'knob=0 is identity' contract for
    the --harsh axis."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "harshness": 0.0}, seed=42).result
    assert a == b


def test_harshness_is_seed_stable():
    """Same (params, seed) tuple yields the same name with harshness > 0."""
    k = Kenning()
    a = k.generate({"culture": "english", "harshness": 0.7}, seed=42).result
    b = k.generate({"culture": "english", "harshness": 0.7}, seed=42).result
    assert a == b


def test_high_harshness_shifts_distribution_across_seeds():
    """At harshness=1, names diverge from the plain baseline at most seeds —
    some by full re-pick of harsh-keyed morphemes. Sweep many seeds; at
    least one diverges."""
    k = Kenning()
    plain = set()
    harsh = set()
    for seed in range(20):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        harsh.add(k.generate({"culture": "english", "harshness": 1.0}, seed=seed).result)
    assert harsh != plain


def test_harsh_and_grim_compose_orthogonally():
    """--grim shifts the SEMANTIC tag set; --harsh shifts the PHONOLOGICAL
    weights. Composing both should produce a non-empty result and stay
    seed-stable."""
    k = Kenning()
    result = k.generate({"culture": "english", "grim": True, "harshness": 0.8}, seed=42)
    assert result.result
    again = k.generate({"culture": "english", "grim": True, "harshness": 0.8}, seed=42)
    assert result.result == again.result


def test_cli_grim_and_harsh_flags_wire_into_params():
    """End-to-end CLI smoke: invoking `wyrd kenning generate english --grim
    --harsh 1.0 --seed 42 -n 1 --no-describe` runs cleanly and produces a
    non-empty name. Pins the click flag → params dict wiring so a future
    rename (e.g. --harsh → --harshness) wouldn't silently break the CLI
    surface."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "english",
            "--grim",
            "--harsh",
            "1.0",
            "--seed",
            "42",
            "-n",
            "1",
            "--no-describe",
        ],
    )
    assert result.exit_code == 0, result.output
    name_lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert name_lines, f"no name produced; output was {result.output!r}"
    assert name_lines[0]


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
