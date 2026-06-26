"""Smoke tests for the Kenning generator and the dispatcher around it."""

from __future__ import annotations

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning import CULTURES, Kenning, available_tags
from wyrd.generators.kenning.errors import EmptyEligiblePool


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


@pytest.mark.parametrize("knob", ["spelling_variety", "inflection_density"])
def test_default_knob_zero_matches_unspecified(knob):
    """A fresh Kenning() at default params and at explicit ``knob=0.0`` produce
    the same output. Guards the 'knob=0 is identity' contract for D18/D8."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", knob: 0.0}, seed=42).result
    assert a == b


@pytest.mark.parametrize("knob", ["spelling_variety", "inflection_density"])
def test_knob_is_seed_stable(knob):
    """Same (params, seed) tuple yields the same name even with knob>0 —
    the reproducibility contract holds across both new knobs."""
    k = Kenning()
    a = k.generate({"culture": "english", knob: 0.7}, seed=42).result
    b = k.generate({"culture": "english", knob: 0.7}, seed=42).result
    assert a == b


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


def test_mood_empty_matches_unspecified():
    """An empty mood list and an unset 'mood' param produce the same name.
    Pins the 'no mood = identity' contract."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "mood": []}, seed=42).result
    assert a == b


def test_mood_is_seed_stable():
    """Same (params, seed) tuple yields the same name with non-empty mood."""
    k = Kenning()
    a = k.generate({"culture": "english", "mood": ["grim", "harsh"]}, seed=42).result
    b = k.generate({"culture": "english", "mood": ["grim", "harsh"]}, seed=42).result
    assert a == b


@pytest.mark.parametrize("mood_name", ["pastoral", "devotional", "mortuary"])
def test_mood_tag_presets_are_seed_stable(mood_name):
    """Each new tag-only mood is reproducible: same (params, seed) tuple
    yields the same name."""
    k = Kenning()
    a = k.generate({"culture": "english", "mood": [mood_name]}, seed=42).result
    b = k.generate({"culture": "english", "mood": [mood_name]}, seed=42).result
    assert a == b


def test_mood_pastoral_composes_with_harsh():
    """A tag-only mood composes with the phonological-skew mood. Pastoral
    morphemes biased through stop-final / cluster-heavy preference is a
    plausible 'rough wilderness' GM ask — should run cleanly and stay
    seed-stable."""
    k = Kenning()
    a = k.generate({"culture": "english", "mood": ["pastoral", "harsh"]}, seed=42).result
    b = k.generate({"culture": "english", "mood": ["pastoral", "harsh"]}, seed=42).result
    assert a == b
    assert a


def test_mood_harsh_shifts_distribution_across_seeds():
    """`mood: ['harsh']` biases sampling toward stop-final / cluster-heavy
    morphemes. At least one seed in the sweep should diverge from the
    no-mood baseline."""
    k = Kenning()
    plain = set()
    harsh = set()
    for seed in range(20):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        harsh.add(k.generate({"culture": "english", "mood": ["harsh"]}, seed=seed).result)
    assert harsh != plain


def test_mood_harsh_with_graduated_value_is_seed_stable():
    """Colon-suffix syntax 'harsh:0.5' graduates the phonological skew.
    Verifies the syntax parses and the result is reproducible."""
    k = Kenning()
    a = k.generate({"culture": "english", "mood": ["harsh:0.5"]}, seed=42).result
    b = k.generate({"culture": "english", "mood": ["harsh:0.5"]}, seed=42).result
    assert a == b


def test_input_schema_mood_description_surfaces_catalog_dynamically():
    """wyrd-3uzp: the Kenning input_schema's 'mood' description must
    list the catalog's effect names at runtime (not a hardcoded set),
    so adding a register effect to the YAML catalog immediately
    shows up in /help / SPA UI / Lambda-input docs without code
    changes. Pins the post-kq7w.3 dynamic-catalog contract."""
    k = Kenning()
    schema = k.input_schema()
    mood_desc = schema["properties"]["mood"]["description"]
    # Each catalog entry should be enumerated in the description string.
    for name in ("grim", "harsh", "pastoral", "noble", "mystical", "ancient"):
        assert name in mood_desc, (
            f"catalog entry {name!r} missing from input_schema mood description; "
            "the description should consume available_register_effects() at "
            "call time so YAML edits propagate without a code change"
        )


def test_mood_grim_and_harsh_compose():
    """Two moods compose orthogonally: grim shifts the SEMANTIC tag set;
    harsh shifts the PHONOLOGICAL weights. Combined output is non-empty
    and seed-stable."""
    k = Kenning()
    result = k.generate({"culture": "english", "mood": ["grim", "harsh"]}, seed=42)
    assert result.result
    again = k.generate({"culture": "english", "mood": ["grim", "harsh"]}, seed=42)
    assert result.result == again.result


def test_mood_grim_composes_with_explicit_tags():
    """Passing both `mood: ['grim']` and `tags: ['water']` should not error;
    the behavior is the union of the explicit tag and the grim tag set."""
    k = Kenning()
    result = k.generate({"culture": "english", "tags": ["water"], "mood": ["grim"]}, seed=42)
    assert result.result
    assert result.explanation


def test_mood_grim_does_not_duplicate_tags_already_present():
    """If the user passes one of the grim tags AND mood:['grim'], the
    expansion should not double-add it — a duplicated tag would produce
    a redundant fallback bucket. Seed-stable across two calls."""
    k = Kenning()
    a = k.generate({"culture": "english", "tags": ["death"], "mood": ["grim"]}, seed=42).result
    b = k.generate({"culture": "english", "tags": ["death"], "mood": ["grim"]}, seed=42).result
    assert a == b
    assert a


def test_mood_unknown_name_raises():
    """An unknown mood name surfaces a ValueError with the available set —
    saves the caller a guessing game when they typo 'gri' or 'harsh-mode'."""
    import pytest

    k = Kenning()
    with pytest.raises(ValueError, match="unknown mood"):
        k.generate({"culture": "english", "mood": ["whimsical"]}, seed=42)


def test_harshness_param_still_works_for_power_users():
    """Direct `harshness: <float>` still works alongside (or in place of)
    moods. The mood resolution uses max(harshness, mood-derived), so an
    explicit harshness floors the result regardless of moods. Seed-stable."""
    k = Kenning()
    a = k.generate({"culture": "english", "harshness": 0.7}, seed=42).result
    b = k.generate({"culture": "english", "harshness": 0.7}, seed=42).result
    assert a == b


def test_harshness_default_zero_matches_unspecified():
    """`harshness=0.0` and unset produce the same name. Guards the
    'knob=0 is identity' contract on the power-user surface."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42).result
    b = k.generate({"culture": "english", "harshness": 0.0}, seed=42).result
    assert a == b


def test_cli_mood_flag_wires_into_params():
    """End-to-end CLI smoke: `wyrd kenning generate english --mood grim
    --mood harsh --seed 42 -n 1 --no-describe` runs cleanly and produces
    a non-empty name. Pins the click multi-flag → params dict wiring."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "english",
            "--mood",
            "grim",
            "--mood",
            "harsh",
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


def test_cli_mood_unknown_value_exits_nonzero():
    """An unknown mood at the CLI surface exits 1 with a list of valid
    choices on stderr. Catches typos before they reach the JSON path."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate", "english", "--mood", "whimsical", "--seed", "42", "-n", "1"],
    )
    assert result.exit_code != 0
    assert "Unknown mood" in result.output
    assert "grim" in result.output
    assert "harsh" in result.output


def test_coerce_bool_handles_spa_string_path():
    """The SPA renders boolean params as form fields whose value crosses
    the wire as a string. Plain bool('false') == True silently inverts a
    default-False gate. _coerce_bool treats common false-tokens as False
    so the wire-string path matches the JSON-bool path. Pinned because
    the SPA's old text-input fallback for booleans (since fixed) was
    exactly the regression that motivated this helper (wyrd-yan)."""
    from wyrd.generators.kenning import _coerce_bool

    # Real bools pass through.
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False

    # SPA / form-encoded strings — false-tokens become False.
    assert _coerce_bool("false") is False
    assert _coerce_bool("False") is False
    assert _coerce_bool("FALSE") is False
    assert _coerce_bool("0") is False
    assert _coerce_bool("") is False
    assert _coerce_bool("no") is False

    # True-tokens become True.
    assert _coerce_bool("true") is True
    assert _coerce_bool("True") is True
    assert _coerce_bool("1") is True
    assert _coerce_bool("yes") is True
    assert _coerce_bool("on") is True


def test_coerce_float_matches_old_idiom_and_rejects_non_numbers():
    """_coerce_float replaces ``float(params.get(k, 0.0) or 0.0)``. It is
    byte-identical for every scalar / numeric-string / blank value, and a
    non-coercible value (list, dict, non-numeric string) raises ValueError —
    which the dispatcher maps to 400 — instead of a bare float() raising
    TypeError on a list/dict, which escaped uncaught as a 500."""
    from wyrd.generators.kenning import _coerce_float

    # Blank / missing → default (the `... or 0.0` arm of the old idiom).
    assert _coerce_float(None) == 0.0
    assert _coerce_float("") == 0.0
    assert _coerce_float(0) == 0.0
    assert _coerce_float(0.0) == 0.0
    # Scalars + numeric strings (the GET query-string path) coerce.
    assert _coerce_float(0.5) == 0.5
    assert _coerce_float(-0.5) == -0.5
    assert _coerce_float("0.5") == 0.5
    assert _coerce_float("1e3") == 1000.0
    assert _coerce_float(None, 6.0) == 6.0  # non-zero default honored
    # A non-coercible knob value is a bad param (ValueError → 400), not a 500.
    for bad in ([1, 2], {"x": 1}, "abc"):
        with pytest.raises(ValueError, match="not a number"):
            _coerce_float(bad)


def test_non_numeric_knob_value_returns_400_not_500():
    """A list/dict value for a float generation knob (spelling_variety, novelty,
    …) is a 400 bad_params, not a 500. Pre-fix `float([1,2])` raised TypeError
    past the dispatcher's ValueError/KeyError catch → uncaught 500."""
    client = create_app().test_client()
    for knob in ("spelling_variety", "inflection_density", "novelty", "harshness", "cohesion"):
        for bad in ([1, 2], {"x": 1}):
            resp = client.post("/api/kenning", json={"culture": "english", "count": 1, knob: bad})
            assert resp.status_code == 400, (knob, bad, resp.status_code)
            assert resp.get_json()["error"] == "bad_params"
    # a valid knob value still rolls
    ok = client.post("/api/kenning", json={"culture": "english", "count": 1, "novelty": 0.8})
    assert ok.status_code == 200


def test_kenning_generate_string_false_does_not_invert_gate():
    """Defense in depth: even if a future client sends include_fiction
    as the string 'false' (the SPA's pre-fix shape), Kenning.generate
    must NOT silently flip into include-fiction mode. Pin so a refactor
    of params parsing can't reintroduce the bool('false') == True trap."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    plain = k.generate({"culture": "english", "seed": 42}, seed=42).result
    string_false = k.generate({"culture": "english", "include_fiction": "false"}, seed=42).result
    assert plain == string_false


def test_cli_era_flag_year_runs_cleanly():
    """CLI smoke: `wyrd kenning generate english --era 1086` runs cleanly.
    Pins the click `--era` flag → params dict → resolve_era_input wiring
    for the year-input branch."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "english",
            "--era",
            "1086",
            "--seed",
            "42",
            "-n",
            "1",
            "--no-describe",
        ],
    )
    assert result.exit_code == 0, result.output
    name_lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert name_lines and name_lines[0]


def test_cli_era_flag_family_label_form_runs_cleanly():
    """CLI smoke: explicit family/label form. Pins the disambiguation
    surface that lets a user pick across-family era cells from the CLI."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "english",
            "--era",
            "english/oe-late",
            "--seed",
            "42",
            "-n",
            "1",
            "--no-describe",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_era_flag_unknown_label_exits_nonzero():
    """An unknown era label at the CLI surface exits non-zero with a
    friendly error message rather than spilling a raw ValueError or
    KeyError. Catches typos before they reach the API path."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate", "english", "--era", "victorian", "--seed", "42", "-n", "1"],
    )
    assert result.exit_code != 0
    # Friendly "Error: ..." line on stderr names the bad input + the
    # culture's era family. Pre-PR a raw ValueError traceback escaped.
    # Click 8.2+ separates stderr by default; .output is stdout only.
    assert "victorian" in result.stderr
    assert "Error:" in result.stderr


@pytest.mark.parametrize(
    "culture,era_label",
    [
        # Each pair uses a label that ONLY resolves under the culture's
        # mapped era family — a regression that hardcoded
        # default_family='english' (or dropped the _CULTURE_TO_ERA_FAMILY
        # lookup entirely) would fail loudly here. Scottish maps to
        # english by design (see comment on _CULTURE_TO_ERA_FAMILY).
        ("english", "oe-late"),
        ("scottish", "oe-late"),
        ("welsh", "old"),
        ("irish", "middle-irish"),
        ("breton", "middle"),
    ],
)
def test_kenning_generate_era_label_routed_through_culture_map(
    culture: str, era_label: str
) -> None:
    """End-to-end coverage of `_CULTURE_TO_ERA_FAMILY`: each culture
    paired with a label that only resolves under its mapped era family.
    A future refactor that drops the per-culture default_family lookup
    would break here because (e.g.) 'middle-irish' isn't an English-
    family label and would raise."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    result = k.generate({"culture": culture, "era": era_label}, seed=42)
    assert result.result


def test_kenning_generate_invalid_era_raises_friendly_value_error():
    """Bad era input must surface as a ValueError (not KeyError, not raw
    propagation from inside resolve_era_input). The message names the
    bad input and the resolved era family so a 4xx response carries
    enough context for the user."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    with pytest.raises(ValueError, match=r"victorian.*english"):
        k.generate({"culture": "english", "era": "victorian"}, seed=42)
    with pytest.raises(ValueError, match=r"klingon/old"):
        k.generate({"culture": "english", "era": "klingon/old"}, seed=42)


def test_kenning_generate_empty_era_string_treated_as_no_filter():
    """The SPA renders absent form fields as empty strings. An era of
    '' must mean 'no filter' (matching the rest of this generator's
    input handling), not surface as an invalid-era error. Bit-stable
    with the no-era path."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    no_era = k.generate({"culture": "english"}, seed=42).result
    empty_era = k.generate({"culture": "english", "era": ""}, seed=42).result
    assert no_era == empty_era


def test_cli_cohesion_flag_runs_cleanly():
    """CLI smoke: `wyrd kenning generate english --cohesion 1.0` runs
    cleanly. Pins the click `--cohesion` flag → params dict wiring."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "english",
            "--cohesion",
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
    assert name_lines and name_lines[0]


def test_cli_cohesion_zero_matches_default():
    """CLI bit-stability: --cohesion 0.0 must produce the same names as
    the no-flag invocation for the same seed."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    base = ["generate", "english", "--seed", "42", "-n", "5", "--no-describe"]
    no_flag = runner.invoke(cli, base)
    explicit_zero = runner.invoke(cli, [*base, "--cohesion", "0.0"])
    assert no_flag.exit_code == 0 and explicit_zero.exit_code == 0
    assert no_flag.output == explicit_zero.output


def test_cli_cohesion_out_of_range_exits_nonzero():
    """Click FloatRange(0, 1) clamps invalid values at the CLI surface
    so a typo like --cohesion 5 fails fast rather than reaching the
    generator with a nonsense value."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate", "english", "--cohesion", "5.0", "--seed", "42", "-n", "1"],
    )
    assert result.exit_code != 0


def test_cli_include_fiction_flag_runs_cleanly_and_is_seed_stable():
    """End-to-end CLI smoke: --include-fiction runs to completion and
    produces a non-empty name; with no fiction-tagged data in the
    bundle today, the same seed produces the same names with and
    without the flag (bit-stable historical behavior — wyrd-yan)."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    runner = CliRunner()
    base_args = ["generate", "english", "--seed", "42", "-n", "3", "--no-describe"]
    without = runner.invoke(cli, base_args)
    with_flag = runner.invoke(cli, [*base_args, "--include-fiction"])

    assert without.exit_code == 0, without.output
    assert with_flag.exit_code == 0, with_flag.output
    # Strip stderr-only lines (e.g. the seed echo).
    names_without = [ln for ln in without.output.splitlines() if ln and "(seed:" not in ln]
    names_with = [ln for ln in with_flag.output.splitlines() if ln and "(seed:" not in ln]
    assert names_without == names_with, (
        "fiction-flag default behavior should be bit-stable today (no fiction "
        f"data in bundle); without={names_without!r} with={names_with!r}"
    )


def test_components_are_structured():
    result = Kenning().generate({"culture": "english"}, seed=7)
    assert isinstance(result.components, list)
    assert all("usage" in c and "location" in c for c in result.components)


def test_available_tags_excludes_internal():
    tags = available_tags()
    assert "male name" not in tags
    assert "female name" not in tags
    assert "saint" not in tags


def test_tags_options_by_culture_excludes_culture_empty_tags():
    """wyrd-ah53: a tag carried by no morpheme in a culture's pool is omitted for
    that culture (it could never be satisfied — D47), even though it's in the
    culture-agnostic union. 'monster' is a fantasy/pfsrd2 tag with zero English
    morphemes."""
    from wyrd.generators.kenning import _tags_options_by_culture

    by_culture = _tags_options_by_culture()
    assert set(by_culture) == set(CULTURES)
    english = by_culture["english"]
    assert "monster" not in english  # no English monster morpheme
    assert "water" in english  # a real English tag survives
    # every per-culture set is a subset of the global union, and non-empty
    allt = set(available_tags())
    for culture, tags in by_culture.items():
        assert set(tags) <= allt, culture
        assert tags, culture


def test_manifest_tags_carry_per_culture_options():
    """wyrd-ah53: the kenning schema's tags property exposes x-options-by-culture
    (the SPA filters the composer to the chosen culture's satisfiable tags), while
    the items.enum stays the culture-agnostic union for CLI/API validation."""
    app = create_app()
    body = app.test_client().get("/api/manifest").get_json()
    kenning = next(g for g in body["generators"] if g["name"] == "kenning")
    tags_prop = kenning["input_schema"]["properties"]["tags"]
    by_culture = tags_prop["x-options-by-culture"]
    assert "monster" not in by_culture["english"]
    assert "monster" in tags_prop["items"]["enum"]  # union still permissive


def test_culture_empty_tag_raises_no_eligible_name():
    """wyrd-ah53 (the motivating behavior): a tag with zero morphemes in the
    culture's pool — i.e. one OMITTED from x-options-by-culture — actually raises
    'no eligible name' when forced via the (permissive) CLI/API. This is WHY the
    composer must not offer it; closes the loop that the omission tracks the real
    runtime failure, not just an internal map."""
    from wyrd.generators.kenning import _tags_options_by_culture

    assert "monster" not in _tags_options_by_culture()["english"]
    # wyrd-hh2m: now EmptyEligiblePool (a KenningError, not a ValueError) so the
    # web layer can map this valid-but-unsatisfiable case to a 422.
    with pytest.raises(EmptyEligiblePool, match="no eligible name"):
        Kenning().generate({"culture": "english", "tags": ["monster"]}, seed=0)


def test_offered_english_tags_are_actually_placeable():
    """wyrd-ah53 (offering↔placer drift guard): every tag the english composer
    offers yields a successful roll for SOME seed — so the offering map can't
    silently diverge from what select_via_vector_scoring can place. A few seeds
    per tag suffices (the guarantee is 'placeable at all', not 'every seed')."""
    from wyrd.generators.kenning import _tags_options_by_culture

    k = Kenning()
    unplaceable = []
    for tag in _tags_options_by_culture()["english"]:
        if not any(_rolls_ok(k, {"culture": "english", "tags": [tag]}, seed) for seed in range(6)):
            unplaceable.append(tag)
    assert not unplaceable, f"offered english tags that never placed: {unplaceable}"


def _rolls_ok(k, params, seed) -> bool:
    try:
        return bool(k.generate(params, seed=seed).result)
    except (ValueError, EmptyEligiblePool):
        # wyrd-hh2m: an empty pool now raises EmptyEligiblePool, not ValueError —
        # treat it (like a bad-param ValueError) as a failed roll.
        return False


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


def test_envelope_forwards_morphemes_by_word_field():
    """wyrd-pr9g: the API envelope must surface morphemes_by_word so
    the SPA can plumb the same render → age continuity flow the CLI
    exposes via 'kenning generate --json | kenning rewind --from-json'.
    Pre-fix envelope() silently dropped the field; the per-word morpheme
    breakdown was only visible to CLI consumers."""
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/kenning?culture=english&seed=42&count=1")
    assert resp.status_code == 200
    body = resp.get_json()
    result = body["results"][0]
    # Field must be present (None or list — never absent).
    assert "morphemes_by_word" in result, (
        "envelope did not forward morphemes_by_word — SPA continuity flow broken; see wyrd-pr9g."
    )
    # Kenning DOES surface morpheme structure, so the field is non-None.
    assert result["morphemes_by_word"] is not None
    assert isinstance(result["morphemes_by_word"], list)
    # Each word has at least one morpheme with a usage key (mirrors
    # the 'kenning generate --json' shape pinned by wyrd-cp2d tests).
    assert len(result["morphemes_by_word"]) >= 1
    for word in result["morphemes_by_word"]:
        assert isinstance(word, list)
        for morph in word:
            assert "usage" in morph


def test_envelope_morphemes_by_word_is_none_for_non_morphemic_generators():
    """wyrd-pr9g: generators that don't surface morpheme structure
    (e.g. KenningExplain — analytical, not a name builder) leave the
    field at its default None. Envelope shape must stay uniform across
    generators so SPA-side consumers can branch on truthiness without
    a KeyError."""
    from wyrd.envelope import envelope
    from wyrd.registry import GenerationResult

    body = envelope(
        generator="dummy",
        parameters={},
        seed=0,
        results=[GenerationResult(result="x", explanation="")],
    )
    assert body["results"][0]["morphemes_by_word"] is None


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


# wyrd-ywm9 + wyrd-emlb: _rank_siblings filter + rank for the
# morpheme inspector. Drops modern_english-only siblings when a
# historical etymon exists, then orders by stratum + signal density.


def _make_meaning(usage, sources, meanings=None, tags=None):
    """Make a Meaning for tests. `sources` accepts a set of lang
    names (legacy callers; lang → no specific forms) OR a dict of
    lang → forms-list (needed for tests that exercise the wyrd-ubbc
    surface-similarity tiebreaker, which inspects source forms)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    if isinstance(sources, dict):
        sources_dict = sources
    else:
        sources_dict = {lang: [] for lang in sources}
    return Meaning(
        usage=usage,
        tags=tags or [],
        meanings=meanings or [],
        sources=sources_dict,
    )


def test_rank_siblings_empty_and_single():
    from wyrd.generators.kenning import _rank_siblings

    assert _rank_siblings([]) == []
    one = _make_meaning("Green-", {"old_english"}, ["Green"])
    out = _rank_siblings([one])
    assert out == [one]


def test_rank_siblings_drops_modern_english_when_historical_present():
    """The Green- case: modern_english 'gree' (step/rung) shouldn't
    surface when OE 'grēne' (Green / Village green) exists."""
    from wyrd.generators.kenning import _rank_siblings

    modern = _make_meaning("Green-", {"modern_english"}, ["step", "to agree"], tags=[])
    oe = _make_meaning(
        "Green-",
        {"old_english"},
        ["Green", "Village green"],
        tags=["color", "descriptive"],
    )
    out = _rank_siblings([modern, oe])
    assert out == [oe]  # modern dropped, OE survives


def test_rank_siblings_keeps_modern_when_no_historical():
    """When NO sibling has historical roots (genuinely modern morpheme),
    keep the modern_english entry rather than collapsing to []."""
    from wyrd.generators.kenning import _rank_siblings

    modern_a = _make_meaning("Foo", {"modern_english"}, ["a"])
    modern_b = _make_meaning("Foo", {"modern_english"}, ["b"])
    out = _rank_siblings([modern_a, modern_b])
    assert sorted(m.meanings[0] for m in out) == ["a", "b"]


def test_rank_siblings_orders_by_stratum_then_signal():
    """OE > ON > OF > Celtic order — and within same stratum, richer
    meaning + tag counts win."""
    from wyrd.generators.kenning import _rank_siblings

    celtic = _make_meaning(
        "Green-", {"celtic_mix"}, ["Bright", "Shining", "Sunny"], tags=["descriptive"]
    )
    oe_rich = _make_meaning(
        "Green-",
        {"old_english"},
        ["Green", "Village green", "green"],
        tags=["color", "descriptive", "plant", "topography"],
    )
    out = _rank_siblings([celtic, oe_rich])
    assert out[0] is oe_rich
    assert out[1] is celtic


def test_is_pure_proper_noun():
    """wyrd-5z5j: _is_pure_proper_noun is True only when a sibling is
    name/saint-tagged AND carries no common-noun tag."""
    from wyrd.generators.kenning import _is_pure_proper_noun

    # Pure personal name (only a name tag) → pure proper noun.
    assert _is_pure_proper_noun(
        _make_meaning("-bourne", {"old_english"}, ["Bourne"], tags=["male name"])
    )
    # Place element merely co-tagged with a name → NOT pure (keeps common-noun rank).
    assert not _is_pure_proper_noun(
        _make_meaning("-stone", {"old_english"}, ["stone"], tags=["geology", "male name"])
    )
    # Not name/saint-tagged at all → not a proper noun.
    assert not _is_pure_proper_noun(
        _make_meaning("-ton", {"old_english"}, ["estate"], tags=["topography"])
    )


def test_rank_siblings_demotes_pure_proper_noun_below_co_tagged_place_element():
    """wyrd-5z5j: within the same stratum, a PURE proper-noun sense is
    de-prioritized below a common-noun place element — even when the proper
    noun surface-matches the usage better. The headline -bourne case: the
    male name 'Bourne' (surface exactly 'bourne') must NOT outrank OE 'burna'
    (the brook sense). A place element merely co-tagged with a name is NOT
    demoted (covered by test_is_pure_proper_noun)."""
    from wyrd.generators.kenning import _rank_siblings

    surname = _make_meaning(
        "-bourne", {"old_english": ["Bourne", "bourne"]}, ["a surname"], tags=["male name"]
    )
    brook = _make_meaning(
        "-bourne", {"old_english": ["burna"]}, ["brook", "stream"], tags=["topography", "water"]
    )
    # surname surface-matches '-bourne' exactly (1.0) and would win the
    # wyrd-ubbc surface tiebreaker; the pure-proper-noun demotion overrides it.
    out = _rank_siblings([surname, brook])
    assert out[0] is brook, "common-noun brook sense must outrank the pure surname sense"


# wyrd-o53o + wyrd-aizb: derivative-gloss classifier. Suppresses
# morphological pointers ('alternative form of X', 'plural of X',
# 'soft mutation of X') when semantic glosses exist alongside;
# surfaces them only when they're alone (so the card isn't blank).


def test_is_derivative_gloss_matches_alternative_form():
    from wyrd.generators.kenning import _is_derivative_gloss

    assert _is_derivative_gloss("alternative form of homes")
    assert _is_derivative_gloss("archaic form of má")
    assert _is_derivative_gloss("obsolete spelling of moing")


def test_is_derivative_gloss_matches_inflection_chains():
    """Compound grammatical chains like 'singular imperative of X' and
    'third-person singular masculine/neuter accusative of X' both
    classify as derivative."""
    from wyrd.generators.kenning import _is_derivative_gloss

    assert _is_derivative_gloss("singular imperative of sendan")
    assert _is_derivative_gloss("plural of bord")
    assert _is_derivative_gloss("inflection of ingān:")
    assert _is_derivative_gloss("third-person singular masculine/neuter accusative of for")
    assert _is_derivative_gloss("accusative singular indefinite of eykr")
    assert _is_derivative_gloss("past participle of būan")


def test_is_derivative_gloss_matches_welsh_mutations():
    from wyrd.generators.kenning import _is_derivative_gloss

    assert _is_derivative_gloss("soft mutation of llys")
    assert _is_derivative_gloss("nasal mutation of pen")


def test_is_derivative_gloss_rejects_semantic_with_of():
    """Semantic glosses that contain 'of' (descriptions, possessives,
    genealogy) must NOT classify as derivative."""
    from wyrd.generators.kenning import _is_derivative_gloss

    assert not _is_derivative_gloss("Green")
    assert not _is_derivative_gloss("Village green")
    assert not _is_derivative_gloss("a stream of running water")
    assert not _is_derivative_gloss("wife of a thegn")
    assert not _is_derivative_gloss("son of Tegid")
    assert not _is_derivative_gloss("piece of land")
    # 'Form of X' alone is semantic ('Form of an enclosure') — only
    # 'X form of Y' with a leading grammatical primary classifies.
    assert not _is_derivative_gloss("Form of an enclosure")
    # Past as a tense word can't lead alone without 'of' anchor —
    # 'past mistakes' is a fragment, not a pointer.
    assert not _is_derivative_gloss("past mistakes")


def test_split_senses_for_display_prefers_semantic():
    """When both semantic + derivative glosses exist, primary returns
    semantic; derivative returns the suppressed pointers."""
    from wyrd.generators.kenning import _split_senses_for_display

    primary, deriv = _split_senses_for_display(
        ["Green", "Village green", "alternative form of greene"]
    )
    assert primary == ["Green", "Village green"]
    assert deriv == ["alternative form of greene"]


def test_split_senses_for_display_falls_back_to_derivative_when_alone():
    """Single-sibling derivative-only ('-sing- → singular imperative of
    singan'): primary = the derivative so the card isn't blank;
    derivative = [] to avoid duplicating the primary list."""
    from wyrd.generators.kenning import _split_senses_for_display

    primary, deriv = _split_senses_for_display(["singular imperative of singan"])
    assert primary == ["singular imperative of singan"]
    assert deriv == []


def test_meaning_groups_dedups_within_and_groups_by_sibling():
    """wyrd-0y3k: per-sibling sense groups, each deduped article/case/punct-
    insensitively, keeping the shortest representative — so the 'denu-' style
    flat run becomes separated, compact groups."""
    from wyrd.generators.kenning import _meaning_groups
    from wyrd.generators.kenning.runtime.meaning import Meaning

    valley = Meaning(
        "denu",
        tags=[],
        meanings=["valley", "A valley.", "Valley", "Dale"],
        sources={"old_english": ["denu"]},
    )
    hill = Meaning(
        "denu",
        tags=[],
        meanings=["A hill.", "Hill", "hill", "down"],
        sources={"old_english": ["dun"]},
    )
    groups = _meaning_groups([valley, hill])
    assert groups == [["valley", "Dale"], ["Hill", "down"]]


def test_meaning_groups_drops_pure_derivative_siblings():
    """wyrd-0y3k: a sibling whose glosses are ALL morphological pointers
    contributes no sense group."""
    from wyrd.generators.kenning import _meaning_groups
    from wyrd.generators.kenning.runtime.meaning import Meaning

    real = Meaning("x", tags=[], meanings=["Hill"], sources={"old_english": ["x"]})
    deriv = Meaning(
        "x",
        tags=[],
        meanings=["singular imperative of singan"],
        sources={"old_english": ["y"]},
    )
    assert _meaning_groups([real, deriv]) == [["Hill"]]


# wyrd-ubbc: surface-form tiebreaker. When stratum-tied siblings
# are also tied on filter passes, prefer the etymon whose source
# lemma surface-matches the modern_usage best (difflib ratio). Fixes
# 'Hill' resolving to OE 'cyln' (kiln) or 'holt' (forest) instead of
# the obvious 'hyll'.


def test_rank_siblings_prefers_surface_match_within_stratum():
    """The Hill case: OE 'hyll' beats OE 'cyln' (kiln) on surface
    similarity to the '-hill' usage even though both are old_english
    and tied on stratum."""
    from wyrd.generators.kenning import _rank_siblings

    cyln = _make_meaning("-hill", {"old_english": ["cyln", "cylnum"]}, ["Kiln", "kiln"])
    hyll = _make_meaning("-hill", {"old_english": ["hyll", "hyllum"]}, ["Hill", "hill"])
    out = _rank_siblings([cyln, hyll])
    assert out[0] is hyll
    assert out[1] is cyln


def test_rank_siblings_surface_match_handles_macrons():
    """OE lemmas with macrons (brād, dēor) should normalize-strip
    diacritics before similarity scoring. 'brād' should score same
    as 'brad' when compared to 'broad', not lower."""
    from difflib import SequenceMatcher

    from wyrd.generators.kenning import _max_form_similarity, _normalize_for_similarity

    norm_broad = _normalize_for_similarity("broad")
    macron_brad = _make_meaning("broad", {"old_english": ["brād"]}, ["Broad"])
    plain_brad = _make_meaning("broad", {"old_english": ["brad"]}, ["Broad"])
    matcher = SequenceMatcher(autojunk=False, a=norm_broad)
    # NFD-strip: 'brād' → 'brad'. Both score identically vs 'broad'.
    assert _max_form_similarity(matcher, norm_broad, macron_brad) == _max_form_similarity(
        matcher, norm_broad, plain_brad
    )


def test_rank_siblings_surface_match_higher_for_closer_form():
    """Bare lexical check: 'hyll' is more similar to 'hill' than
    'holt' is, so the similarity score is higher for hyll."""
    from difflib import SequenceMatcher

    from wyrd.generators.kenning import _max_form_similarity, _normalize_for_similarity

    norm_hill = _normalize_for_similarity("-hill")
    matcher = SequenceMatcher(autojunk=False, a=norm_hill)
    hyll_m = _make_meaning("-hill", {"old_english": ["hyll"]}, ["Hill"])
    holt_m = _make_meaning("-hill", {"old_english": ["holt"]}, ["Forest"])
    cyln_m = _make_meaning("-hill", {"old_english": ["cyln"]}, ["Kiln"])
    s_hyll = _max_form_similarity(matcher, norm_hill, hyll_m)
    s_holt = _max_form_similarity(matcher, norm_hill, holt_m)
    s_cyln = _max_form_similarity(matcher, norm_hill, cyln_m)
    assert s_hyll > s_holt > s_cyln


def test_rank_siblings_surface_match_does_not_override_stratum():
    """Stratum still wins over surface similarity: an OE etymon with
    sim=0.5 beats a Celtic etymon with sim=1.0 for the same bucket.
    Surface match is a within-stratum tiebreaker, not a primary
    signal (otherwise modern_english homographs would win again)."""
    from wyrd.generators.kenning import _rank_siblings

    celtic_exact = _make_meaning("-hill", {"celtic_mix": ["hill"]}, ["coincidence form"])
    oe_low_sim = _make_meaning("-hill", {"old_english": ["holt"]}, ["Forest", "Grove", "Wood"])
    out = _rank_siblings([celtic_exact, oe_low_sim])
    assert out[0] is oe_low_sim
    assert out[1] is celtic_exact


def test_rank_siblings_returns_new_list():
    """Caller mutations on the result must not leak back to the input."""
    from wyrd.generators.kenning import _rank_siblings

    a = _make_meaning("X-", {"old_english"}, ["a"])
    b = _make_meaning("X-", {"old_scandinavian"}, ["b"])
    inp = [a, b]
    out = _rank_siblings(inp)
    out.clear()
    assert len(inp) == 2  # original untouched


def test_harshness_shifts_vector_distribution_end_to_end():
    """wyrd-rt2m: harshness IS wired into the (now default) vector path — it
    biases the phonological register — so it shifts the generated distribution
    end-to-end. Replaces the end-to-end harshness/mood coverage the deleted
    proportions-path tests provided (those pinned proportions-specific
    semantics that vector composes differently)."""
    k = Kenning()
    plain: set[str] = set()
    harsh: set[str] = set()
    for seed in range(30):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        harsh.add(k.generate({"culture": "english", "harshness": 1.0}, seed=seed).result)
    assert plain != harsh


def test_novelty_shifts_vector_distribution_end_to_end():
    """wyrd-fcub: novelty blends the per-slot score distribution toward uniform
    over the eligible pool, so novelty=1 (every eligible morpheme equally
    likely) produces a different distribution than novelty=0 (score-weighted)."""
    k = Kenning()
    plain: set[str] = set()
    novel: set[str] = set()
    for seed in range(30):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        novel.add(k.generate({"culture": "english", "novelty": 1.0}, seed=seed).result)
    assert plain != novel
    # novelty=1 must still produce VALID names, not empty/None — the uniform
    # blend reweights the pool, it must not empty it.
    assert all(novel), f"novelty=1 produced empty results: {novel}"


def test_novelty_zero_is_noop_byte_identical():
    """novelty=0 must be byte-identical to no-novelty for every seed — the blend
    is score-normalize-only at 0 and weighted_choice is scaling-invariant, so the
    fast-path skip must not be the only thing preserving bit-stability."""
    k = Kenning()
    for seed in range(30):
        base = k.generate({"culture": "english"}, seed=seed).result
        zero = k.generate({"culture": "english", "novelty": 0.0}, seed=seed).result
        assert base == zero, f"seed={seed}: novelty=0 ({zero!r}) != no-novelty ({base!r})"


def test_novelty_is_seed_stable():
    """Same seed + same novelty → identical result (the seed-stability contract
    novelty must preserve)."""
    k = Kenning()
    a = k.generate({"culture": "english", "novelty": 0.7}, seed=42).result
    b = k.generate({"culture": "english", "novelty": 0.7}, seed=42).result
    assert a == b
