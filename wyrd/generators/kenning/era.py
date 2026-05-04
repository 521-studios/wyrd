"""Era-cell mapping for the (language × era) generation axis (D5-2).

A morpheme attested in 950 AD ('Hædan-tūn') belongs in a different era
cell than one attested in 1240 ('Heydonton'); even within the same
language family the cells split at meaningful boundaries (Conquest,
Great Vowel Shift, etc.). This module is the canonical mapping from
``(language_code, year)`` to a label like ``"oe-late"`` or
``"me"``, plus the inverse range lookup that callers need to filter
the morpheme inventory before generation.

Design intent:

* Era cells live PER LANGUAGE FAMILY, not per individual language —
  Old English / Middle English / Modern English share the same family
  ('english') and so a year-based cell-label resolution works across
  all three.
* Cells are *half-open* intervals ``[start, end)`` so a year exactly
  on a boundary lands deterministically in the later cell. ``start``
  or ``end`` of ``None`` means "open on that side".
* Pre-attested proto-languages (Proto-Germanic, Proto-Celtic,
  Proto-Indo-European) don't have an era family — ``era_cell`` returns
  ``None`` for them, which the generator interprets as 'always include
  regardless of --era filter'.

The cell boundaries come from `wyrd-38d` and the `D5` design
discussion in DECISIONS.md.
"""

from __future__ import annotations

# Each entry: (label, start_year, end_year). start=None → open on the
# low side ("before recorded history"), end=None → open on the high
# side ("ongoing"). Half-open: start <= year < end.
EraCell = tuple[str, int | None, int | None]

ERA_CELLS: dict[str, tuple[EraCell, ...]] = {
    "english": (
        ("oe-early", None, 800),
        ("oe-late", 800, 1100),
        ("me", 1100, 1500),
        ("early-modern", 1500, 1700),
        ("modern", 1700, None),
    ),
    "norse": (
        ("on-classical", None, 1100),
        ("on-late", 1100, 1300),
        ("middle-scandinavian", 1300, 1500),
        ("modern", 1500, None),
    ),
    "brythonic": (
        ("old", None, 1100),
        ("middle", 1100, 1500),
        ("modern", 1500, None),
    ),
    "goidelic": (
        ("old-irish", None, 900),
        ("middle-irish", 900, 1200),
        ("early-modern", 1200, 1700),
        ("modern", 1700, None),
    ),
    "latin": (
        ("classical", None, 200),
        ("late-vulgar", 200, 700),
        ("medieval", 700, 1500),
        ("renaissance", 1500, 1800),
    ),
    "norman-french": (
        # Anglo-Norman is the cross-Channel period; outside-England
        # 'old-norman' through ~1300, then 'modern' is a wide
        # post-medieval bucket. Norman in English place-name corpora
        # is dominated by the anglo-norman span.
        ("old-norman", None, 1066),
        ("anglo-norman", 1066, 1500),
        ("modern", 1500, None),
    ),
}

# Map every language code (as it appears in `etymon.language`) to its
# era-family bucket. Proto-languages and untracked codes intentionally
# omitted — they resolve to None via ``era_cell``.
LANGUAGE_TO_FAMILY: dict[str, str] = {
    # English family
    "old-english": "english",
    "middle-english": "english",
    "modern-english": "english",
    "english": "english",
    # Norse family — Old Norse and its descendants share the cell
    # split, with the modern bucket folding in the continental
    # Scandinavian languages (post-Old-Norse split they all run on
    # roughly the same chronology).
    "old-norse": "norse",
    "icelandic": "norse",
    "faroese": "norse",
    "norwegian": "norse",
    "norwegian-bokmal": "norse",
    "norwegian-nynorsk": "norse",
    "danish": "norse",
    "swedish": "norse",
    # Brythonic family
    "welsh": "brythonic",
    "old-welsh": "brythonic",
    "middle-welsh": "brythonic",
    "cornish": "brythonic",
    "breton": "brythonic",
    "old-breton": "brythonic",
    "middle-breton": "brythonic",
    # Goidelic family
    "irish": "goidelic",
    "old-irish": "goidelic",
    "middle-irish": "goidelic",
    "scottish-gaelic": "goidelic",
    "manx": "goidelic",
    # Latin family (incl. vulgar-latin variant tags)
    "latin": "latin",
    "vulgar-latin": "latin",
    # Norman / Old French — treat all French as norman-french for
    # English-corpus purposes; pure modern-French wouldn't typically
    # surface in a place-name lexicon.
    "norman-french": "norman-french",
    "old-french": "norman-french",
    "middle-french": "norman-french",
    "french": "norman-french",
    # 'celtic_mix' is a pseudo-language used in the runtime bundle
    # for general Celtic compositions; map to brythonic by convention
    # since most Celtic British place names are Brythonic-derived.
    "celtic": "brythonic",
    "celtic_mix": "brythonic",
    # Greek + Hebrew + general non-British classical languages don't
    # have era cells defined yet; let them resolve to None.
}


def language_family(language: str) -> str | None:
    """Return the era-family for ``language``, or None if the language
    has no defined era progression (proto-languages, modern-Greek,
    Hebrew, etc.)."""
    return LANGUAGE_TO_FAMILY.get(language)


def era_cell(language: str, year: int | None) -> str | None:
    """Resolve a ``(language, year)`` pair to an era-cell label.

    Returns None for THREE distinct reasons that callers may want to
    distinguish:

    * ``language`` has no defined era family (proto-languages, untracked
      classical languages). The runtime generator interprets this as
      "always include regardless of --era filter".
    * ``year`` is None (no attestation found yet). Same generator
      interpretation: pass through.
    * ``year`` falls outside the defined range for the family (e.g. a
      Latin year of 2000 — outside renaissance's 1500-1800). Generator
      should EXCLUDE these — the era filter found a valid family but
      the year doesn't match any cell.

    Callers that need to distinguish "no-family pass-through" from
    "year-out-of-range exclude" should call ``language_family`` first
    and then ``era_cell_for_family`` — see the lexicon ``era-cell``
    CLI command for the canonical pattern.

    The half-open ``[start, end)`` convention means a year exactly on
    a boundary lands in the later cell.
    """
    if year is None:
        return None
    family = LANGUAGE_TO_FAMILY.get(language)
    if family is None:
        return None
    return era_cell_for_family(family, year)


def era_cell_for_family(family: str, year: int) -> str | None:
    """Resolve a ``(family, year)`` pair to a cell label, or None when
    the year is outside the family's defined range.

    Distinct from ``era_cell`` in that the family is given directly —
    so a None return here unambiguously means 'year out of range', not
    'no family for this language'. Lets the runtime generator treat
    the two cases differently:

    * "no family" (resolved before calling this) → include in inventory
      regardless of --era.
    * "out of range" (this returns None) → exclude.

    Raises KeyError on an unknown family — fail loudly rather than
    silently treating an out-of-range year as in-range.
    """
    cells = ERA_CELLS.get(family)
    if cells is None:
        raise KeyError(f"unknown era family: {family!r}")
    for label, start, end in cells:
        if start is not None and year < start:
            continue
        if end is not None and year >= end:
            continue
        return label
    return None


def era_year_range(family: str, label: str) -> tuple[int | None, int | None]:
    """Return the ``(start, end)`` half-open year range for a given
    ``(family, label)`` pair. Raises ``KeyError`` for unknown family
    or label so a typo in CLI input surfaces immediately rather than
    silently returning everything."""
    cells = ERA_CELLS.get(family)
    if cells is None:
        raise KeyError(f"unknown era family: {family!r}")
    for cell_label, start, end in cells:
        if cell_label == label:
            return (start, end)
    valid = [c[0] for c in cells]
    raise KeyError(f"unknown era cell {label!r} for family {family!r}; valid: {valid}")


def era_cells_for_family(family: str) -> tuple[str, ...]:
    """Return the ordered tuple of cell labels defined for ``family``.
    Useful for CLI auto-completion and validation."""
    cells = ERA_CELLS.get(family)
    if cells is None:
        raise KeyError(f"unknown era family: {family!r}")
    return tuple(label for label, _, _ in cells)


def all_families() -> tuple[str, ...]:
    """Return the tuple of all defined era families. Stable order
    (sorted alphabetically) so CLI output is deterministic."""
    return tuple(sorted(ERA_CELLS))


def resolve_era_input(
    era: str | int | None,
    *,
    default_family: str = "english",
) -> tuple[int | None, int | None] | None:
    """Convert a CLI/API ``--era`` value to a half-open ``(start, end)``
    year range, or None to mean 'no filter applies'.

    Empty-string ``era`` is NOT silently treated as None — that's an
    upstream concern (``Kenning.generate`` guards before calling). A
    direct caller passing ``""`` will hit the 'unknown era input' error
    path; if you mean 'no filter', pass ``None``.

    Accepted shapes:

    * ``None`` → no filter. Caller leaves the inventory untouched.
    * ``int`` (or numeric str like ``"1086"``) → year-based. Resolves to
      the cell containing that year in ``default_family`` and returns
      that cell's range. ValueError if the year falls outside any
      defined cell for the family.
    * ``str`` cell label (``"oe-late"``) → resolves against
      ``default_family`` only. A label not defined in ``default_family``
      raises ValueError, even if another family defines the same label
      (e.g. ``"middle-irish"`` only exists in ``goidelic`` — passing it
      with ``default_family="english"`` is a typo or wrong-culture
      request, and silently using the goidelic range would surprise the
      caller). The error message names the families where the label IS
      defined so the user can switch to the explicit ``family/label``
      form.
    * ``str`` ``family/label`` form (``"english/oe-late"``) → explicit
      family choice; returns the cell's range from the named family.
      Use this when ``default_family`` doesn't define the label.

    The runtime D5-2 filter (``Meaning.attested_in_era_range``) uses
    the returned range as a half-open interval — a year exactly on
    ``start`` is included, a year exactly on ``end`` is not.
    """
    if era is None:
        return None
    if isinstance(era, int):
        cell = era_cell_for_family(default_family, era)
        if cell is None:
            raise ValueError(
                f"year {era} is outside the defined cells for family {default_family!r}"
            )
        return era_year_range(default_family, cell)
    try:
        year = int(era)
    except ValueError:
        year = None
    if year is not None:
        return resolve_era_input(year, default_family=default_family)
    if "/" in era:
        family, _, label = era.partition("/")
        return era_year_range(family, label)
    # Bare label — must exist in default_family. Cross-family fallback
    # would silently route a typo or wrong-culture label into the wrong
    # range; instead, point the user at the families that DO define the
    # label so they can switch to the explicit ``family/label`` form.
    try:
        return era_year_range(default_family, era)
    except KeyError:
        defined_in = [f for f in sorted(ERA_CELLS) if era in era_cells_for_family(f)]
        if defined_in:
            choices = ", ".join(f"{f}/{era}" for f in defined_in)
            raise ValueError(
                f"era cell {era!r} is not defined in family "
                f"{default_family!r}; use one of: {choices}"
            ) from None
        raise ValueError(
            f"unknown era input {era!r}; pass a year (e.g. 1086), a cell "
            f"label (e.g. 'oe-late'), or a 'family/label' pair"
        ) from None
