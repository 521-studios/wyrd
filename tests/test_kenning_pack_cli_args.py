"""Unit tests for the pack CLI flag parser (wyrd-ecjp.11 Phase 8).

Tests the standalone parsing/validation helpers in cli/_pack_args.py.
End-to-end CLI integration tests live in test_kenning_generate_pack_cli.py.
"""

from __future__ import annotations

import click
import pytest

from wyrd.generators.kenning.cli._pack_args import (
    ParsedPackSpec,
    parse_pack_specs,
    parse_pack_tag_filter_specs,
    validate_packs_against_catalog,
)

# ---- parse_pack_specs ---------------------------------------------------


def test_parse_pack_specs_name_only():
    """Bare name → weight_override is None (runtime falls back to
    PackManifest.default_weight)."""
    result = parse_pack_specs(("neo-khuzdul",))
    assert result == [ParsedPackSpec(pack_name="neo-khuzdul", weight_override=None)]


def test_parse_pack_specs_name_with_weight():
    result = parse_pack_specs(("tatar-9c:0.5",))
    assert result == [ParsedPackSpec(pack_name="tatar-9c", weight_override=0.5)]


def test_parse_pack_specs_multiple():
    """Multi-pack declaration preserves order."""
    result = parse_pack_specs(("pack-a", "pack-b:0.3", "pack-c"))
    assert [s.pack_name for s in result] == ["pack-a", "pack-b", "pack-c"]
    assert result[1].weight_override == 0.3


def test_parse_pack_specs_weight_zero_allowed():
    """Weight = 0 is valid (pack declared but contributes nothing —
    useful for A/B comparison without changing the rest of the CLI)."""
    result = parse_pack_specs(("pack-a:0",))
    assert result[0].weight_override == 0.0


def test_parse_pack_specs_weight_ten_allowed():
    """Weight = 10 is the upper bound (matches the ScoringWeights
    range cap)."""
    result = parse_pack_specs(("pack-a:10",))
    assert result[0].weight_override == 10.0


def test_parse_pack_specs_weight_out_of_range_rejected():
    with pytest.raises(click.BadParameter, match=r"weight must be in \[0, 10\]"):
        parse_pack_specs(("pack-a:11",))
    with pytest.raises(click.BadParameter, match=r"weight must be in \[0, 10\]"):
        parse_pack_specs(("pack-a:-1",))


def test_parse_pack_specs_non_float_weight_rejected():
    with pytest.raises(click.BadParameter, match="weight must be a float"):
        parse_pack_specs(("pack-a:abc",))


def test_parse_pack_specs_empty_name_before_colon_rejected():
    with pytest.raises(click.BadParameter, match="missing pack name"):
        parse_pack_specs((":0.5",))


def test_parse_pack_specs_empty_weight_after_colon_rejected():
    with pytest.raises(click.BadParameter, match="missing weight"):
        parse_pack_specs(("pack-a:",))


def test_parse_pack_specs_empty_name_rejected():
    with pytest.raises(click.BadParameter, match="empty pack name"):
        parse_pack_specs(("",))


def test_parse_pack_specs_duplicate_pack_name_rejected():
    """Two --pack flags with the same name is an operator typo —
    ambiguous which weight wins, so we reject."""
    with pytest.raises(click.BadParameter, match="declared more than once"):
        parse_pack_specs(("pack-a", "pack-a:0.5"))


# ---- parse_pack_tag_filter_specs ----------------------------------------


def test_parse_pack_tag_filter_single_tag():
    result = parse_pack_tag_filter_specs(("tatar-9c:war",), {"tatar-9c"})
    assert result == {"tatar-9c": frozenset({"war"})}


def test_parse_pack_tag_filter_multiple_tags():
    result = parse_pack_tag_filter_specs(("tatar-9c:war,trade,settlement",), {"tatar-9c"})
    assert result == {"tatar-9c": frozenset({"war", "trade", "settlement"})}


def test_parse_pack_tag_filter_strips_whitespace():
    """Operator-friendly: tolerates spaces around commas
    (e.g. shell-quoted 'tatar-9c: war, trade')."""
    result = parse_pack_tag_filter_specs(("tatar-9c: war, trade",), {"tatar-9c"})
    assert result == {"tatar-9c": frozenset({"war", "trade"})}


def test_parse_pack_tag_filter_missing_colon_rejected():
    with pytest.raises(click.BadParameter, match=r"expected '<pack>:<tag"):
        parse_pack_tag_filter_specs(("tatar-9c war",), {"tatar-9c"})


def test_parse_pack_tag_filter_undeclared_pack_rejected():
    """The --pack-tag-filter must reference a previously-declared
    --pack (caught at parse time so the operator gets a friendly
    error rather than the filter silently no-opping)."""
    with pytest.raises(click.BadParameter, match="not declared via --pack"):
        parse_pack_tag_filter_specs(("missing-pack:war",), {"tatar-9c"})


def test_parse_pack_tag_filter_lists_declared_in_error():
    """The error message shows the declared packs so the operator
    can spot a typo without re-reading their command."""
    with pytest.raises(click.BadParameter, match=r"Declared packs: pack-a, pack-b"):
        parse_pack_tag_filter_specs(("typo:war",), {"pack-a", "pack-b"})


def test_parse_pack_tag_filter_empty_tags_rejected():
    with pytest.raises(click.BadParameter, match="missing tag list"):
        parse_pack_tag_filter_specs(("tatar-9c:",), {"tatar-9c"})


def test_parse_pack_tag_filter_empty_pack_rejected():
    with pytest.raises(click.BadParameter, match="missing pack name"):
        parse_pack_tag_filter_specs((":war",), set())


def test_parse_pack_tag_filter_duplicate_pack_rejected():
    """Two --pack-tag-filter flags for the same pack is ambiguous."""
    with pytest.raises(click.BadParameter, match="declared more than once"):
        parse_pack_tag_filter_specs(
            ("tatar-9c:war", "tatar-9c:trade"),
            {"tatar-9c"},
        )


def test_parse_pack_tag_filter_all_empty_tags_rejected():
    """Tag list like ',,,' parses to an empty set after stripping —
    that's not a useful filter, raise."""
    with pytest.raises(click.BadParameter, match="at least one non-empty tag"):
        parse_pack_tag_filter_specs(("tatar-9c: , , ",), {"tatar-9c"})


# ---- validate_packs_against_catalog ------------------------------------


def test_validate_packs_against_catalog_all_known_ok():
    """Empty error path is a no-op (no raise)."""
    validate_packs_against_catalog(
        [ParsedPackSpec(pack_name="pack-a", weight_override=None)],
        {"pack-a", "pack-b"},
    )


def test_validate_packs_against_catalog_unknown_pack_rejected():
    with pytest.raises(click.BadParameter, match="Unknown pack"):
        validate_packs_against_catalog(
            [ParsedPackSpec(pack_name="missing", weight_override=None)],
            {"pack-a", "pack-b"},
        )


def test_validate_packs_against_catalog_lists_available_in_error():
    with pytest.raises(click.BadParameter, match=r"Available: pack-a, pack-b"):
        validate_packs_against_catalog(
            [ParsedPackSpec(pack_name="missing", weight_override=None)],
            {"pack-b", "pack-a"},
        )


def test_validate_packs_against_catalog_no_packs_bundled_error_message():
    """Catalog with no packs surfaces a friendly "(no packs bundled)"
    hint instead of an empty list."""
    with pytest.raises(click.BadParameter, match=r"\(no packs bundled\)"):
        validate_packs_against_catalog(
            [ParsedPackSpec(pack_name="anything", weight_override=None)],
            set(),
        )
