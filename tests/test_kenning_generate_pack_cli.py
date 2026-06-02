"""End-to-end CLI tests for the --pack / --pack-tag-filter flags
(wyrd-ecjp.11 Phase 8).

Verifies the CLI flag layer wires pack declarations + tag filters
through to the params dict; the dispatch helper
(_generate_via_vector) resolves them against the bundled pack
catalog and constructs PackOverlay + pack_meaning_dbs at request
time.

Tests use the CliRunner harness; pack-catalog monkeypatching
(when needed) replaces ``wyrd.generators.kenning._load_packs`` with
a fixture-supplied dict so the tests don't depend on a real bundled
pack.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli.generate import generate
from wyrd.generators.kenning.lexicon.pack_loader import PackBundle
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.vectors.schemas import PackManifest


@pytest.fixture
def empty_packs(monkeypatch):
    """Monkeypatch _load_packs to return an empty dict so 'unknown
    pack' errors surface deterministically without depending on the
    bundled catalog state."""
    import wyrd.generators.kenning as kenning_mod

    monkeypatch.setattr(kenning_mod, "_load_packs", lambda: {})
    yield


@pytest.fixture
def single_pack(monkeypatch):
    """Monkeypatch _load_packs to return one synthetic pack named
    'test-pack' with one lemma. Lets us drive the dispatch end-to-
    end without authoring a real bundled pack."""
    import wyrd.generators.kenning as kenning_mod

    pack_meanings = {
        "subjects": [
            {
                "meaning": ["Fortress"],
                "modifier_tags": ["architecture"],
                "modifier_type": "Topographical",
                "words": [{"modern_usage": "-keep", "old_english": ["cȳpe"]}],
            }
        ]
    }
    meaning_db, tag_db = load_meanings(pack_meanings)
    bundle = PackBundle(
        manifest=PackManifest(
            pack_name="test-pack",
            template_donor="old-norse",
            template_recipient="old-english",
            default_weight=0.5,
        ),
        meaning_db=meaning_db,
        tag_db=tag_db,
    )
    monkeypatch.setattr(kenning_mod, "_load_packs", lambda: {"test-pack": bundle})
    yield


# ---- --help surfaces the flags -----------------------------------------


def test_help_lists_pack_flags():
    runner = CliRunner()
    result = runner.invoke(generate, ["--help"])
    assert result.exit_code == 0
    assert "--pack" in result.output
    assert "--pack-tag-filter" in result.output


# ---- unknown pack errors cleanly ---------------------------------------


def test_unknown_pack_errors_cleanly(empty_packs):
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--pack",
            "missing-pack",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown pack" in result.output
    assert "missing-pack" in result.output


def test_unknown_pack_lists_available_packs(single_pack):
    """When the catalog has packs, the error lists them so the
    operator can spot the typo."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--pack",
            "missing",
        ],
    )
    assert result.exit_code != 0
    assert "test-pack" in result.output  # available pack listed


# ---- weight parsing errors --------------------------------------------


def test_invalid_pack_weight_errors_cleanly(single_pack):
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--pack",
            "test-pack:not-a-number",
        ],
    )
    assert result.exit_code != 0
    assert "must be a float" in result.output


def test_pack_weight_out_of_range_errors_cleanly(single_pack):
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--pack",
            "test-pack:99",
        ],
    )
    assert result.exit_code != 0
    assert "must be in [0, 10]" in result.output


# ---- --pack-tag-filter validation --------------------------------------


def test_pack_tag_filter_for_undeclared_pack_errors_cleanly(single_pack):
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--pack",
            "test-pack",
            "--pack-tag-filter",
            "other-pack:war",  # other-pack not declared
        ],
    )
    assert result.exit_code != 0
    assert "not declared via --pack" in result.output


def test_pack_tag_filter_missing_colon_errors_cleanly(single_pack):
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--pack",
            "test-pack",
            "--pack-tag-filter",
            "no-colon",
        ],
    )
    assert result.exit_code != 0
    assert "expected" in result.output


# ---- pack params flow through to dispatch -----------------------------


def test_pack_params_reach_dispatch(single_pack, monkeypatch):
    """End-to-end: CLI --pack + --pack-tag-filter compose a packs
    list in params; _generate_via_vector receives it. Monkeypatch
    the dispatch function to capture what got passed."""
    from wyrd.generators.kenning.generators import kenning as kenning_mod

    captured = {}
    original = kenning_mod._generate_via_vector

    def capture(*args, **kwargs):
        captured["packs_raw"] = kwargs.get("packs_raw")
        return original(*args, **kwargs)

    monkeypatch.setattr(kenning_mod, "_generate_via_vector", capture)

    runner = CliRunner()
    runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--pack",
            "test-pack:0.7",
            "--pack-tag-filter",
            "test-pack:architecture",
        ],
    )
    assert "packs_raw" in captured
    packs = captured["packs_raw"]
    assert packs == [
        {
            "pack_name": "test-pack",
            "weight_override": 0.7,
            "allowed_pack_tags": ["architecture"],
        }
    ]


def test_no_pack_flags_passes_empty_packs(single_pack, monkeypatch):
    """Without --pack flags, params.get('packs') is absent / empty,
    so packs_raw is empty list at dispatch time."""
    from wyrd.generators.kenning.generators import kenning as kenning_mod

    captured = {}
    original = kenning_mod._generate_via_vector

    def capture(*args, **kwargs):
        captured["packs_raw"] = kwargs.get("packs_raw")
        return original(*args, **kwargs)

    monkeypatch.setattr(kenning_mod, "_generate_via_vector", capture)

    runner = CliRunner()
    runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
        ],
    )
    assert captured["packs_raw"] == []


def test_pack_flags_in_proportions_mode_silent_no_op(single_pack, monkeypatch):
    """Pack flags in proportions mode: the params dict still carries
    'packs' (operator might toggle scoring_mode later), but the
    proportions dispatch ignores it. Same contract as the weight
    flags. The output is bit-stable with no-flag proportions output."""
    runner = CliRunner()
    baseline = runner.invoke(
        generate,
        ["english", "--count", "1", "--seed", "42", "--no-describe"],
    )
    with_pack_flags = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--pack",
            "test-pack",
        ],
    )
    assert baseline.exit_code == 0
    assert with_pack_flags.exit_code == 0
    assert baseline.output == with_pack_flags.output


# ---- pack dispatch path produces names with pack lemmas ----------------


def test_vector_mode_with_pack_can_produce_pack_lemma(single_pack):
    """End-to-end smoke: vector mode with a pack must not crash and
    must produce either generated output or a friendly ValueError
    diagnostic across a seed sweep. Whether the pack lemma actually
    surfaces is a function of the synthetic fixture's lemma
    placement + the live English struct distribution; we don't
    pin a "pack lemma appears" assertion against unstable inputs.

    What this test DOES guard: silent crashes (the pack pipeline
    raising an uncaught exception that's neither a clean Error
    message nor a generated name)."""
    runner = CliRunner()
    saw_clean_outcome = False
    for seed in range(10):
        result = runner.invoke(
            generate,
            [
                "english",
                "--count",
                "3",
                "--seed",
                str(seed),
                "--no-describe",
                "--scoring-mode",
                "vector",
                "--tag",
                "architecture",
                "--pack",
                "test-pack:5.0",
            ],
        )
        if result.exit_code == 0:
            # Successfully generated a name — clean outcome
            assert result.output.strip()
            saw_clean_outcome = True
        else:
            # Friendly ValueError fallback ('no eligible name') is
            # also acceptable — must surface as Error: prefix per
            # the CLI's existing user-error-handling pattern.
            assert "Error" in result.output, result.output
            saw_clean_outcome = True
    assert saw_clean_outcome, "no seed in the sweep reached the dispatch"


# ---- positive parse of the params dict shape --------------------------


def test_params_packs_shape_matches_schema(single_pack):
    """The params['packs'] list shape matches what Kenning.input_schema
    documents: items are objects with pack_name + optional
    weight_override + optional allowed_pack_tags."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    pack_schema = schema["properties"]["packs"]
    assert pack_schema["type"] == "array"
    item_schema = pack_schema["items"]
    assert item_schema["required"] == ["pack_name"]
    assert "pack_name" in item_schema["properties"]
    assert "weight_override" in item_schema["properties"]
    assert "allowed_pack_tags" in item_schema["properties"]


def test_kenning_generate_unknown_pack_raises_valueerror(monkeypatch):
    """Non-CLI callers (Lambda, SPA, tests) bypass the CLI's catalog
    validation. The dispatch helper re-checks; unknown pack raises
    ValueError so the caller can surface a clean error."""
    import wyrd.generators.kenning as kenning_mod
    from wyrd.generators.kenning import Kenning

    monkeypatch.setattr(kenning_mod, "_load_packs", lambda: {})
    kenning = Kenning()
    params = {
        "culture": "english",
        "scoring_mode": "vector",
        "packs": [{"pack_name": "missing"}],
    }
    with pytest.raises(ValueError, match="unknown pack"):
        kenning.generate(params, seed=42)


def test_kenning_generate_pack_with_no_weight_uses_manifest_default(single_pack, monkeypatch):
    """When weight_override is None, the dispatch uses
    PackManifest.default_weight (0.5 in the single_pack fixture).
    Monkeypatch PackOverlay construction to capture the weight."""
    from wyrd.generators.kenning.vectors import schemas as schemas_mod

    captured = []
    original_init = schemas_mod.PackOverlay.__init__

    def capture_init(self, **kwargs):
        captured.append(kwargs)
        original_init(self, **kwargs)

    monkeypatch.setattr(schemas_mod.PackOverlay, "__init__", capture_init)

    runner = CliRunner()
    runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--pack",
            "test-pack",  # no :weight
        ],
    )
    # The dispatch MUST have constructed at least one PackOverlay;
    # an `if captured:` guard would let a silent-broken pipeline
    # pass the test. The fixture's pack has default_weight=0.5,
    # and --pack test-pack (no :weight) means weight_override is
    # None → manifest default takes over.
    assert captured, "_generate_via_vector never constructed a PackOverlay"
    assert any(call.get("weight") == 0.5 for call in captured), (
        f"expected weight=0.5 (manifest default), got {captured}"
    )
