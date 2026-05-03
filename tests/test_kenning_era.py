"""Tests for wyrd.generators.kenning.era — era-cell mapping (wyrd-38d).

Pin-style: each test asserts a single (language, year) → cell decision
or a single (family, label) → range decision. Cell boundaries are
spec'd in the ticket and DECISIONS.md D5-2; if a future refinement
changes them, the tests are the place to record the new contract.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning import era

# --- era_cell dispatch -----------------------------------------------------


@pytest.mark.parametrize(
    "language,year,expected",
    [
        # English family — boundary year lands in the LATER cell
        # (half-open intervals).
        ("old-english", 700, "oe-early"),
        ("old-english", 799, "oe-early"),
        ("old-english", 800, "oe-late"),
        ("old-english", 1099, "oe-late"),
        ("old-english", 1100, "me"),
        ("middle-english", 1300, "me"),
        ("middle-english", 1499, "me"),
        ("middle-english", 1500, "early-modern"),
        ("modern-english", 1699, "early-modern"),
        ("modern-english", 1700, "modern"),
        ("modern-english", 2025, "modern"),
        # Norse family
        ("old-norse", 1000, "on-classical"),
        ("old-norse", 1099, "on-classical"),
        ("old-norse", 1100, "on-late"),
        ("icelandic", 1400, "middle-scandinavian"),
        ("danish", 1700, "modern"),
        # Brythonic
        ("welsh", 800, "old"),
        ("middle-welsh", 1300, "middle"),
        ("breton", 1800, "modern"),
        # Goidelic
        ("old-irish", 800, "old-irish"),
        ("middle-irish", 1000, "middle-irish"),
        ("irish", 1500, "early-modern"),
        ("scottish-gaelic", 2000, "modern"),
        # Latin
        ("latin", 100, "classical"),
        ("latin", 199, "classical"),
        ("latin", 200, "late-vulgar"),
        ("latin", 700, "medieval"),
        ("vulgar-latin", 800, "medieval"),
        ("latin", 1600, "renaissance"),
        # Norman-French
        ("norman-french", 1000, "old-norman"),
        ("norman-french", 1066, "anglo-norman"),
        ("norman-french", 1499, "anglo-norman"),
        ("norman-french", 1500, "modern"),
        ("old-french", 1200, "anglo-norman"),
    ],
)
def test_era_cell_resolves_known_language_year_pairs(
    language: str, year: int, expected: str
) -> None:
    """Spot-check the cell boundary for each family. Half-open
    intervals: a year exactly on a boundary lands in the LATER cell."""
    assert era.era_cell(language, year) == expected


@pytest.mark.parametrize(
    "language",
    [
        "proto-germanic",
        "proto-celtic",
        "proto-indo-european",
        "ancient-greek",
        "modern-greek",
        "hebrew",
        "nahuatl",  # untracked
    ],
)
def test_era_cell_returns_none_for_languages_without_era_family(
    language: str,
) -> None:
    """Proto-languages and untracked classical languages don't have
    era cells — generator interprets None as 'always include'."""
    assert era.era_cell(language, 1000) is None


def test_era_cell_returns_none_for_none_year() -> None:
    """An etymon without an attested_year (NULL in DB) maps to None
    — the generator's --era filter must let it pass through."""
    assert era.era_cell("old-english", None) is None


def test_era_cell_admits_year_below_lowest_open_endpoint() -> None:
    """The lowest cell with start=None is unbounded on the low side,
    so a Republican-era Latin year lands in 'classical' rather than
    falling out. Counter to what the test name might suggest, this
    is the cell-INCLUSIVE behaviour of an open low endpoint."""
    assert era.era_cell("latin", -100) == "classical"


def test_era_cell_returns_none_for_year_above_highest_cell() -> None:
    """Latin's renaissance ends at 1800; a year of 2000 has no cell."""
    assert era.era_cell("latin", 2000) is None


# --- era_year_range inverse lookup -----------------------------------------


def test_era_year_range_returns_inclusive_lower_exclusive_upper() -> None:
    """Half-open intervals: range[0] is inclusive, range[1] exclusive."""
    assert era.era_year_range("english", "oe-late") == (800, 1100)
    assert era.era_year_range("english", "me") == (1100, 1500)


def test_era_year_range_returns_none_for_open_endpoints() -> None:
    """Open-ended cells (oe-early on the low side, modern on the high
    side) report None for the unbounded endpoint."""
    assert era.era_year_range("english", "oe-early") == (None, 800)
    assert era.era_year_range("english", "modern") == (1700, None)


def test_era_year_range_raises_for_unknown_family() -> None:
    """Defensive: a typo in CLI input surfaces as a KeyError naming
    the unknown family rather than silently returning a wrong range."""
    with pytest.raises(KeyError, match="unknown era family"):
        era.era_year_range("klingon", "old")


def test_era_year_range_raises_for_unknown_label_in_known_family() -> None:
    """Defensive: typo on the LABEL surfaces too, with valid labels
    listed in the error message."""
    with pytest.raises(KeyError, match="unknown era cell .* for family"):
        era.era_year_range("english", "victorian")


# --- era_cells_for_family / all_families -----------------------------------


def test_era_cells_for_family_returns_ordered_labels() -> None:
    """Order in the returned tuple matches the chronological order
    of the cells (oldest first), matching ERA_CELLS declaration order.
    Tests that depend on ordered iteration can rely on this."""
    cells = era.era_cells_for_family("english")
    assert cells == ("oe-early", "oe-late", "me", "early-modern", "modern")


def test_all_families_returns_alphabetically_sorted_tuple() -> None:
    """Sorted output makes CLI listings deterministic across hosts."""
    families = era.all_families()
    assert families == tuple(sorted(families))
    assert "english" in families
    assert "norse" in families


# --- language_family -------------------------------------------------------


def test_language_family_returns_correct_family_for_descendants() -> None:
    """A language's family follows the LANGUAGE_TO_FAMILY map. Pin
    the dominant cases so a refactor doesn't accidentally drop one."""
    assert era.language_family("old-english") == "english"
    assert era.language_family("modern-english") == "english"
    assert era.language_family("old-norse") == "norse"
    assert era.language_family("icelandic") == "norse"
    assert era.language_family("welsh") == "brythonic"
    assert era.language_family("scottish-gaelic") == "goidelic"
    assert era.language_family("latin") == "latin"
    assert era.language_family("old-french") == "norman-french"


def test_language_family_returns_none_for_unknown_or_proto_languages() -> None:
    """Unknown / untracked languages get None so the generator
    passes them through the --era filter (era cells don't apply)."""
    assert era.language_family("proto-germanic") is None
    assert era.language_family("proto-indo-european") is None
    assert era.language_family("ancient-greek") is None
    assert era.language_family("klingon") is None
