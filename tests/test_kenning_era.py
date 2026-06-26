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
        # English family — every internal boundary pinned (year just
        # below + year on boundary). Half-open: boundary year lands
        # in the LATER cell.
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
        # Norse — every internal boundary pinned
        ("old-norse", 1000, "on-classical"),
        ("old-norse", 1099, "on-classical"),
        ("old-norse", 1100, "on-late"),
        ("old-norse", 1299, "on-late"),
        ("icelandic", 1300, "middle-scandinavian"),
        ("icelandic", 1400, "middle-scandinavian"),
        ("icelandic", 1499, "middle-scandinavian"),
        ("icelandic", 1500, "modern"),
        ("danish", 1700, "modern"),
        # Brythonic — every internal boundary pinned
        ("welsh", 800, "old"),
        ("welsh", 1099, "old"),
        ("welsh", 1100, "middle"),
        ("middle-welsh", 1300, "middle"),
        ("middle-welsh", 1499, "middle"),
        ("middle-welsh", 1500, "modern"),
        ("breton", 1800, "modern"),
        # Goidelic — every internal boundary pinned (highest
        # regression risk per agent finding).
        ("old-irish", 800, "old-irish"),
        ("old-irish", 899, "old-irish"),
        ("old-irish", 900, "middle-irish"),
        ("middle-irish", 1000, "middle-irish"),
        ("middle-irish", 1199, "middle-irish"),
        ("middle-irish", 1200, "early-modern"),
        ("irish", 1500, "early-modern"),
        ("irish", 1699, "early-modern"),
        ("irish", 1700, "modern"),
        ("scottish-gaelic", 2000, "modern"),
        # Latin — every internal boundary pinned
        ("latin", 100, "classical"),
        ("latin", 199, "classical"),
        ("latin", 200, "late-vulgar"),
        ("latin", 699, "late-vulgar"),
        ("latin", 700, "medieval"),
        ("vulgar-latin", 800, "medieval"),
        ("latin", 1499, "medieval"),
        ("latin", 1500, "renaissance"),
        ("latin", 1600, "renaissance"),
        # Norman-French — every internal boundary pinned
        ("norman-french", 1000, "old-norman"),
        ("norman-french", 1065, "old-norman"),
        ("norman-french", 1066, "anglo-norman"),
        ("norman-french", 1499, "anglo-norman"),
        ("norman-french", 1500, "modern"),
        ("old-french", 1200, "anglo-norman"),
    ],
)
def test_era_cell_resolves_known_language_year_pairs(
    language: str, year: int, expected: str
) -> None:
    """Cell-boundary smoke for every family, both INTERNAL boundaries
    (year-on-boundary + year-just-below). Half-open intervals: the
    boundary year lands in the LATER cell."""
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


@pytest.mark.parametrize(
    "family,expected",
    [
        ("english", ("oe-early", "oe-late", "me", "early-modern", "modern")),
        ("norse", ("on-classical", "on-late", "middle-scandinavian", "modern")),
        ("brythonic", ("old", "middle", "modern")),
        ("goidelic", ("old-irish", "middle-irish", "early-modern", "modern")),
        ("latin", ("classical", "late-vulgar", "medieval", "renaissance")),
        ("norman-french", ("old-norman", "anglo-norman", "modern")),
    ],
)
def test_era_cells_for_family_returns_ordered_labels(
    family: str, expected: tuple[str, ...]
) -> None:
    """Order in the returned tuple matches the chronological order
    of the cells (oldest first), matching ERA_CELLS declaration order.
    Tests that depend on ordered iteration can rely on this. Pins
    every family so a labels-renamed regression surfaces immediately."""
    assert era.era_cells_for_family(family) == expected


def test_era_cells_for_family_raises_for_unknown_family() -> None:
    """Defensive: a typo in CLI input (e.g. 'klingon') surfaces as a
    KeyError naming the unknown family. Pin the error path so a
    silent-default regression is caught."""
    with pytest.raises(KeyError, match="unknown era family"):
        era.era_cells_for_family("klingon")


def test_all_families_returns_full_set_alphabetically_sorted() -> None:
    """Sorted output makes CLI listings deterministic across hosts.
    Equality (rather than 'in') so accidentally dropping a family
    from ERA_CELLS surfaces here."""
    assert era.all_families() == (
        "brythonic",
        "english",
        "goidelic",
        "latin",
        "norman-french",
        "norse",
    )


# --- language_family -------------------------------------------------------


@pytest.mark.parametrize(
    "language,expected_family",
    [
        # English family
        ("old-english", "english"),
        ("middle-english", "english"),
        ("modern-english", "english"),
        ("english", "english"),
        # Norse family — every member of LANGUAGE_TO_FAMILY exercised
        ("old-norse", "norse"),
        ("icelandic", "norse"),
        ("faroese", "norse"),
        ("norwegian", "norse"),
        ("norwegian-bokmal", "norse"),
        ("norwegian-nynorsk", "norse"),
        ("danish", "norse"),
        ("swedish", "norse"),
        # Brythonic family
        ("welsh", "brythonic"),
        ("old-welsh", "brythonic"),
        ("middle-welsh", "brythonic"),
        ("modern-welsh", "brythonic"),
        ("cornish", "brythonic"),
        ("breton", "brythonic"),
        ("old-breton", "brythonic"),
        ("middle-breton", "brythonic"),
        # Goidelic family
        ("irish", "goidelic"),
        ("old-irish", "goidelic"),
        ("middle-irish", "goidelic"),
        ("scottish-gaelic", "goidelic"),
        ("manx", "goidelic"),
        # Latin family
        ("latin", "latin"),
        ("vulgar-latin", "latin"),
        # Norman-French
        ("norman-french", "norman-french"),
        ("old-french", "norman-french"),
        ("middle-french", "norman-french"),
        ("french", "norman-french"),
        # Pseudo-language used in runtime bundle
        ("celtic", "brythonic"),
        ("celtic_mix", "brythonic"),
        # Celtic branch-level languages (wyrd-xdam.3)
        ("brittonic", "brythonic"),
        ("goidelic", "goidelic"),
    ],
)
def test_language_family_covers_every_entry_in_map(language: str, expected_family: str) -> None:
    """Exercise every key in LANGUAGE_TO_FAMILY so a typo or accidental
    drop surfaces here. Parametrised across all 35 mapped languages."""
    assert era.language_family(language) == expected_family


def test_language_family_returns_none_for_unknown_or_proto_languages() -> None:
    """Unknown / untracked languages get None so the generator
    passes them through the --era filter (era cells don't apply)."""
    assert era.language_family("proto-germanic") is None
    assert era.language_family("proto-indo-european") is None
    assert era.language_family("ancient-greek") is None
    assert era.language_family("klingon") is None


# --- era_cell_for_family (disambiguates None reasons) ----------------------


def test_era_cell_for_family_resolves_when_year_in_range() -> None:
    """Direct family→cell dispatch works the same as era_cell when
    the year is in range — same boundary semantics."""
    assert era.era_cell_for_family("english", 950) == "oe-late"
    assert era.era_cell_for_family("latin", 200) == "late-vulgar"


def test_era_cell_for_family_returns_none_only_for_year_out_of_range() -> None:
    """Critical contract for the runtime generator: when the family is
    given directly, None unambiguously means 'year falls outside the
    family's defined cells' — NOT 'no family for this language'.
    Caller can EXCLUDE these from the inventory under --era."""
    # Latin's renaissance ends at 1800; year 2025 is out-of-range.
    assert era.era_cell_for_family("latin", 2025) is None


def test_era_cell_for_family_raises_for_unknown_family() -> None:
    """Defensive: unknown family bubbles up as KeyError, not silent
    None. Otherwise a typo in the family name would look like
    'every year is out-of-range'."""
    with pytest.raises(KeyError, match="unknown era family"):
        era.era_cell_for_family("klingon", 1000)


# --- resolve_era_input (D5-3 CLI/API entry point) --------------------------


def test_resolve_era_input_none_passthrough() -> None:
    """None input means 'no era' — caller gets None back (no render
    range; D44: era only selects the rendered reflex)."""
    assert era.resolve_era_input(None) is None


@pytest.mark.parametrize(
    "year,expected_range",
    [
        (1086, (800, 1100)),  # Domesday — oe-late
        (700, (None, 800)),  # below all internal boundaries → oe-early
        (1500, (1500, 1700)),  # boundary year lands in early-modern
        (1700, (1700, None)),  # modern open-on-high
    ],
)
def test_resolve_era_input_year_resolves_to_cell_range(
    year: int, expected_range: tuple[int | None, int | None]
) -> None:
    """An int year resolves to the half-open range of the cell
    containing it in default_family='english'."""
    assert era.resolve_era_input(year) == expected_range


def test_resolve_era_input_numeric_string_treated_as_year() -> None:
    """CLI passes args as strings; a numeric string takes the year
    branch rather than the label branch."""
    assert era.resolve_era_input("1086") == (800, 1100)


def test_resolve_era_input_float_year_still_resolves() -> None:
    """A float year (a JSON number from the API) coerces through int() like an
    int/numeric-string — guards that the non-string rejection below doesn't
    over-reach and break a valid numeric era."""
    assert era.resolve_era_input(1086.0) == (800, 1100)


def test_resolve_era_input_non_string_non_number_raises_value_error() -> None:
    """A list/dict era (a malformed request) is an UNKNOWN era input — a
    ValueError the dispatcher maps to 400 bad_params — not a bare int() TypeError
    that escapes uncaught as a 500. The validator already raised ValueError for an
    unknown string; this closes the non-string/non-number type hole."""
    for bad in ([1, 2], {"x": 1}, [], ("oe-late",)):
        with pytest.raises(ValueError, match="unknown era input"):
            era.resolve_era_input(bad)


def test_resolve_era_input_year_out_of_range_raises_value_error() -> None:
    """A year outside the family's defined cells should fail loudly,
    not silently return None — that would let a bad --era 9999 sneak
    through as 'no filter'."""
    with pytest.raises(ValueError, match="outside the defined cells"):
        era.resolve_era_input(9999, default_family="latin")


def test_resolve_era_input_bare_label_uses_default_family() -> None:
    """A bare label resolves against default_family."""
    assert era.resolve_era_input("oe-late") == (800, 1100)
    assert era.resolve_era_input("me") == (1100, 1500)


def test_resolve_era_input_bare_label_outside_default_family_raises_with_pointer() -> None:
    """A bare label not defined in default_family raises ValueError —
    NOT a silent cross-family fallback. The message names the families
    that DO define the label so the user can switch to family/label.
    Prevents 'wrong culture' typos from silently routing to a different
    family's range."""
    with pytest.raises(ValueError, match=r"is not defined in family 'english'"):
        era.resolve_era_input("middle-irish")
    # Error mentions the family that does define it.
    with pytest.raises(ValueError, match=r"goidelic/middle-irish"):
        era.resolve_era_input("middle-irish")


def test_resolve_era_input_explicit_family_label_works_for_cross_family() -> None:
    """The explicit family/label form is the way to use a label outside
    default_family. Pinned alongside the strict bare-label test so the
    user-facing migration path is obvious."""
    assert era.resolve_era_input("goidelic/middle-irish") == (900, 1200)


def test_resolve_era_input_family_label_form_disambiguates() -> None:
    """Several families share the label 'modern' — the family/label
    form pins the family explicitly."""
    assert era.resolve_era_input("english/modern") == (1700, None)
    assert era.resolve_era_input("latin/renaissance") == (1500, 1800)
    assert era.resolve_era_input("norman-french/anglo-norman") == (1066, 1500)


def test_resolve_era_input_family_label_form_raises_on_unknown() -> None:
    """A typo in the family/label form surfaces as a KeyError so the
    user sees 'unknown era family' rather than a silent miss."""
    with pytest.raises(KeyError, match="unknown era family"):
        era.resolve_era_input("klingon/old")


def test_resolve_era_input_unknown_label_raises_value_error() -> None:
    """A bare label that no family defines should fail with a
    ValueError pointing the user at the accepted shapes."""
    with pytest.raises(ValueError, match="unknown era input"):
        era.resolve_era_input("victorian")


def test_resolve_era_input_default_family_picks_correct_range_for_shared_label() -> None:
    """A label shared across families resolves to the default_family's
    range. 'modern' exists in every family — passing
    default_family='brythonic' returns brythonic's 'modern' (1500, None);
    with 'english' it returns (1700, None). No fallback; the label is
    defined in default_family so resolution is unambiguous."""
    assert era.resolve_era_input("modern", default_family="brythonic") == (1500, None)
    assert era.resolve_era_input("modern", default_family="english") == (1700, None)


# --- wyrd-rogd.2: compressed STAGE labels in the era resolvers --------------
def test_cells_for_stage_collapses_canonical_language() -> None:
    assert era.cells_for_stage("english", "old-english") == ("oe-early", "oe-late")
    assert era.cells_for_stage("english", "middle-english") == ("me",)
    assert era.cells_for_stage("english", "modern-english") == ("early-modern", "modern")
    assert era.cells_for_stage("english", "not-a-stage") == ()


def test_stage_year_range_unions_constituent_cells() -> None:
    # old-english = oe-early (open-low) + oe-late (..1100) → (None, 1100)
    assert era.stage_year_range("english", "old-english") == (None, 1100)
    assert era.stage_year_range("english", "middle-english") == (1100, 1500)
    # modern-english = early-modern (1500..1700) + modern (1700..open) → (1500, None)
    assert era.stage_year_range("english", "modern-english") == (1500, None)
    assert era.stage_year_range("english", "not-a-stage") is None


def test_resolve_era_input_accepts_a_stage_label() -> None:
    assert era.resolve_era_input("old-english", default_family="english") == (None, 1100)
    assert era.resolve_era_input("modern-english", default_family="english") == (1500, None)
    # raw cells + years still resolve (backward compatible)
    assert era.resolve_era_input("oe-late", default_family="english") == (800, 1100)
    assert era.resolve_era_input(1086, default_family="english") == (800, 1100)


def test_era_cell_for_input_maps_stage_to_representative_cell() -> None:
    # a stage resolves to its first cell, whose canonical language IS the stage
    family, cell = era.era_cell_for_input("old-english", default_family="english")
    assert (family, cell) == ("english", "oe-early")
    assert era.canonical_language_for_cell(family, cell) == "old-english"


def test_unknown_era_label_still_raises() -> None:
    with pytest.raises(ValueError):
        era.resolve_era_input("victorian", default_family="english")


def test_stage_resolution_across_families() -> None:
    # goidelic 'irish' is the multi-cell non-english stage (early-modern+modern)
    assert era.cells_for_stage("goidelic", "irish") == ("early-modern", "modern")
    assert era.resolve_era_input("irish", default_family="goidelic") == (1200, None)
    # norse single-cell stage
    assert era.resolve_era_input("old-norse", default_family="norse") == (None, 1100)
    # brythonic
    assert era.resolve_era_input("welsh", default_family="brythonic") == (1500, None)


def test_cell_label_shadows_a_same_named_stage() -> None:
    # goidelic 'old-irish' / 'middle-irish' are BOTH cells and stage tags; the
    # cell path wins (same range), so resolution is identical either way.
    assert "old-irish" in era.era_cells_for_family("goidelic")
    assert era.resolve_era_input("old-irish", default_family="goidelic") == era.era_year_range(
        "goidelic", "old-irish"
    )
    assert era.era_cell_for_input("old-irish", default_family="goidelic") == (
        "goidelic",
        "old-irish",
    )


def test_latin_stage_span_is_documented_non_contiguous() -> None:
    # The lone non-contiguous stage: 'latin' (classical+medieval+renaissance)
    # spans the vulgar-latin 200-700 gap. CLI-only (latin isn't SPA-surfaced).
    assert era.cells_for_stage("latin", "latin") == ("classical", "medieval", "renaissance")
    assert era.stage_year_range("latin", "latin") == (None, 1800)
    assert era.stage_year_range("latin", "vulgar-latin") == (200, 700)


def test_explicit_family_stage_form() -> None:
    # wyrd-rogd.2: family/STAGE resolves like family/cell (symmetry).
    assert era.resolve_era_input("english/old-english", default_family="brythonic") == (None, 1100)
    fam, cell = era.era_cell_for_input("english/old-english", default_family="brythonic")
    assert (fam, cell) == ("english", "oe-early")


def test_cross_family_stage_label_gives_helpful_error() -> None:
    # 'old-welsh' is a brythonic stage; passed with default english it points
    # the user at the right family/stage pair rather than 'unknown era input'.
    with pytest.raises(ValueError, match=r"brythonic/old-welsh"):
        era.resolve_era_input("old-welsh", default_family="english")
    with pytest.raises(ValueError, match=r"brythonic/old-welsh"):
        era.era_cell_for_input("old-welsh", default_family="english")


def test_invalid_family_label_lists_cells_and_stages_consistently() -> None:
    # wyrd-rogd.2: both resolvers give the same helpful error (cells + stages)
    # for a bad label under an explicit family.
    for fn in (era.resolve_era_input, era.era_cell_for_input):
        with pytest.raises(
            ValueError, match=r"unknown era cell/stage 'invalid' for family 'english'"
        ):
            fn("english/invalid", default_family="brythonic")
