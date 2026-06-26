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
    "modern-welsh": "brythonic",
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
    # Celtic BRANCH-level languages (wyrd-xdam.3) — resolve to their own era
    # family. brittonic→the brythonic cells (old/middle/modern → Welsh lineage,
    # the dominant Brythonic in the corpus); goidelic→the goidelic cells.
    "brittonic": "brythonic",
    "goidelic": "goidelic",
    # Greek + Hebrew + general non-British classical languages don't
    # have era cells defined yet; let them resolve to None.
}


# wyrd-skm Phase 3.0b: per-cell canonical language tag.
#
# The era-reflex picker walks a cognate cluster and asks "for THIS era
# cell, which language tag should I prefer when picking a surface
# form?". This maps each `(family, cell_label)` pair to the canonical
# language tag (as it appears in ``etymon.language``) that
# scholarly sources use for that cell.
#
# Cells MISSING from this dict produce an empty result from the
# picker — there is NO family-level fallback. Norse cells past
# 'on-classical' fold continental Scandinavian languages with no
# single canonical tag; Latin's late cells are ambiguous against the
# flat 'latin' / 'vulgar-latin' Wiktionary tagging. Adding an entry
# expands picker coverage; downstream demos treat empty results as
# 'no era data, fall back to canonical_form'.
#
# Where a single language tag spans multiple cells (Old English covers
# both `oe-early` and `oe-late`), the same tag appears twice.
CANONICAL_LANGUAGE_FOR_CELL: dict[tuple[str, str], str] = {
    # English: OE pre-1100 → 'old-english'; ME 1100-1500 →
    # 'middle-english'; modern post-1500 → 'modern-english' (the
    # tagged variant we have most rows for; bare 'english' rows are a
    # smaller superset).
    ("english", "oe-early"): "old-english",
    ("english", "oe-late"): "old-english",
    ("english", "me"): "middle-english",
    ("english", "early-modern"): "modern-english",
    ("english", "modern"): "modern-english",
    # Norse: ON pre-1100 → 'old-norse'. Later cells fold the
    # continental Scandinavian languages (Icelandic / Norwegian /
    # Danish / Swedish) so no single canonical tag fits — those
    # cells are intentionally OMITTED here and the picker returns
    # ``[]`` for them.
    ("norse", "on-classical"): "old-norse",
    # Brythonic: old/middle/modern map to old-welsh/middle-welsh/welsh
    # (Welsh is the dominant Brythonic in our corpus).
    ("brythonic", "old"): "old-welsh",
    ("brythonic", "middle"): "middle-welsh",
    ("brythonic", "modern"): "welsh",
    # Goidelic: same shape against Irish.
    ("goidelic", "old-irish"): "old-irish",
    ("goidelic", "middle-irish"): "middle-irish",
    ("goidelic", "early-modern"): "irish",
    ("goidelic", "modern"): "irish",
    # Latin: Wiktionary uses flat 'latin' / 'vulgar-latin' codes, so
    # most cells map to 'latin' (classical / medieval / renaissance
    # don't carry distinct tags in the corpus). Late-vulgar pulls
    # 'vulgar-latin' since some Wiktionary entries do tag that way.
    ("latin", "classical"): "latin",
    ("latin", "late-vulgar"): "vulgar-latin",
    ("latin", "medieval"): "latin",
    ("latin", "renaissance"): "latin",
    # Norman / Old French: 'old-norman' is best served by Wiktionary's
    # 'old-french' tag; Anglo-Norman cell prefers 'middle-french' /
    # 'old-french'; modern is bare 'french'.
    ("norman-french", "old-norman"): "old-french",
    ("norman-french", "anglo-norman"): "old-french",
    ("norman-french", "modern"): "french",
}


def canonical_language_for_cell(family: str, cell: str) -> str | None:
    """Return the canonical ``etymon.language`` tag preferred for the
    given ``(family, cell)`` pair, or None when no canonical tag is
    defined.

    Used by the era-reflex picker (wyrd-skm Phase 3.0b) to choose the
    first-choice cluster-mate when rendering an etymon at a target
    era. **None is a hard miss — the picker returns ``[]`` for that
    cell**, NOT a family-level fallback. Cells missing from the dict
    represent eras where scholarly sources don't anchor on a single
    canonical language tag (Norse 'modern' folds the continental
    Scandinavian languages; Latin is ambiguous against the flat
    'latin'/'vulgar-latin' tagging Wiktionary uses). Adding entries
    expands the picker's coverage; downstream demos (wyrd-rni /
    wyrd-381) should treat empty results as "no era data — fall back
    to canonical_form" rather than "feature broken".
    """
    return CANONICAL_LANGUAGE_FOR_CELL.get((family, cell))


def _cell_for_family_label(family: str, label: str) -> tuple[str, str]:
    """The ``family/label`` branch of :func:`era_cell_for_input`: explicit
    family, with a cell label or a compressed stage (→ its first cell)."""
    if label in era_cells_for_family(family):
        return (family, label)
    # wyrd-rogd.2: explicit family/STAGE form too (e.g. 'english/old-english')
    stage_cells = cells_for_stage(family, label)
    if stage_cells:
        return (family, stage_cells[0])
    valid = [*era_cells_for_family(family), *family_stage_order(family)]
    raise ValueError(f"unknown era cell/stage {label!r} for family {family!r}; valid: {valid}")


def _cell_for_bare_label(era: str, default_family: str) -> tuple[str, str]:
    """The bare-label branch of :func:`era_cell_for_input`: a cell or stage of
    ``default_family``, else a cross-family hint, else 'unknown era input'."""
    if era in era_cells_for_family(default_family):
        return (default_family, era)
    # wyrd-rogd.2: a compressed STAGE label maps to a representative cell (its
    # first) so canonical_language_for_cell resolves to the stage's language —
    # all the stage's cells share it, so any one renders the era correctly.
    stage_cells = cells_for_stage(default_family, era)
    if stage_cells:
        return (default_family, stage_cells[0])
    defined_in = [f for f in sorted(ERA_CELLS) if era in era_cells_for_family(f)]
    if defined_in:
        choices = ", ".join(f"{f}/{era}" for f in defined_in)
        raise ValueError(
            f"era cell {era!r} is not defined in family {default_family!r}; use one of: {choices}"
        )
    # wyrd-rogd.2: same cross-family hint for a STAGE label of another family
    # (e.g. 'old-welsh' with default_family='english').
    defined_in_stages = [f for f in sorted(ERA_CELLS) if cells_for_stage(f, era)]
    if defined_in_stages:
        choices = ", ".join(f"{f}/{era}" for f in defined_in_stages)
        raise ValueError(
            f"era stage {era!r} is not defined in family {default_family!r}; use one of: {choices}"
        )
    raise ValueError(
        f"unknown era input {era!r}; pass a year (e.g. 1086), a cell label "
        f"(e.g. 'oe-late'), a stage (e.g. 'old-english'), or a 'family/label' pair"
    )


def era_cell_for_input(
    era: str | int,
    *,
    default_family: str,
) -> tuple[str, str]:
    """Resolve an ``--era`` CLI/API value to its ``(family, cell)``
    pair directly, without round-tripping through year ranges.

    This is the era-reflex picker's bridge: callers want to pick a
    canonical language tag from ``CANONICAL_LANGUAGE_FOR_CELL``, which
    is keyed on ``(family, cell)``, but ``resolve_era_input`` returns
    a year range — and re-deriving the cell from the year range loses
    information when ``start`` is None (open-low cells like
    'oe-early').

    Accepted shapes mirror ``resolve_era_input``:

    * ``int`` (or numeric str) → year-based; resolves to the cell
      containing that year in ``default_family``.
    * ``str`` cell label (``"oe-late"``) → resolves against
      ``default_family`` only.
    * ``str`` ``family/label`` → explicit family choice.

    Raises ``ValueError`` when the input is malformed or out of range,
    matching ``resolve_era_input``'s error contract. ``None`` and
    empty string are intentionally NOT accepted — there's no cell for
    'no filter', and silently treating ``""`` as 'no filter' would
    surprise an era-reflex caller. Pass an explicit cell label or
    year instead.
    """
    if era is None or era == "":
        raise ValueError(
            "era must be a year, cell label, or family/label pair (got None or empty string)"
        )
    if isinstance(era, int):
        cell = era_cell_for_family(default_family, era)
        if cell is None:
            raise ValueError(
                f"year {era} is outside the defined cells for family {default_family!r}"
            )
        return (default_family, cell)
    try:
        year = int(era)
    except (ValueError, TypeError):
        # Mirrors resolve_era_input: ValueError → a non-numeric string (a cell
        # label); TypeError → a non-string/non-int (a list/dict from a malformed
        # request), which is rejected as malformed just below rather than letting
        # int()'s bare TypeError escape uncaught (a 500 instead of the documented
        # ValueError → 400).
        year = None
    if year is not None:
        return era_cell_for_input(year, default_family=default_family)
    if not isinstance(era, str):
        raise ValueError(f"era must be a year, cell label, or family/label pair (got {era!r})")
    if "/" in era:
        family, _, label = era.partition("/")
        return _cell_for_family_label(family, label)
    return _cell_for_bare_label(era, default_family)


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


def all_families() -> tuple[str, ...]:  # noqa: V103 — public re-export (era/__init__.py) accessor pinned by tests
    """Return the tuple of all defined era families. Stable order
    (sorted alphabetically) so CLI output is deterministic."""
    return tuple(sorted(ERA_CELLS))


def family_stage_order(family: str) -> list[str]:
    """wyrd-lftl: the distinct canonical language tags for ``family``, in
    era-cell order — the column set the SPA col-3 reflex grid renders.

    Collapses era cells that share a canonical language tag, because the
    era-reflex picker keys on the language tag, not the cell: english's
    ``oe-early`` + ``oe-late`` both resolve to ``old-english`` and would
    otherwise render two identical columns; ``early-modern`` + ``modern``
    both resolve to ``modern-english``. So english yields three stages
    (old-english / middle-english / modern-english), norse one
    (old-norse — its later cells have no canonical tag), brythonic three
    (old-welsh / middle-welsh / welsh), etc.

    First-appearance order across the family's cells, so the stages read
    oldest → newest. Returns ``[]`` for an unknown family (no cells) so
    callers can treat 'no era axis' uniformly rather than catching KeyError.
    """
    cells = ERA_CELLS.get(family)
    if cells is None:
        return []
    stages: list[str] = []
    for label, _start, _end in cells:
        lang = CANONICAL_LANGUAGE_FOR_CELL.get((family, label))
        if lang and lang not in stages:
            stages.append(lang)
    return stages


def cells_for_stage(family: str, stage: str) -> tuple[str, ...]:
    """wyrd-rogd.2: the era cells a compressed STAGE label collapses — every
    cell in ``family`` whose canonical language is ``stage`` (a
    ``family_stage_order`` tag like ``old-english``), in cell order. Empty when
    ``stage`` isn't a stage of ``family``. The inverse of the
    ``CANONICAL_LANGUAGE_FOR_CELL`` collapse that ``family_stage_order`` exposes,
    so the Configure-column era control can offer stages instead of raw cells."""
    cells = ERA_CELLS.get(family)
    if cells is None:
        return ()
    return tuple(
        label
        for label, _start, _end in cells
        if CANONICAL_LANGUAGE_FOR_CELL.get((family, label)) == stage
    )


def stage_year_range(family: str, stage: str) -> tuple[int | None, int | None] | None:
    """wyrd-rogd.2: the UNION half-open year range of a compressed STAGE label's
    constituent cells (``old-english`` = oe-early + oe-late → ``(None, 1100)``;
    ``modern-english`` = early-modern + modern → ``(1500, None)``). None when
    ``stage`` isn't a stage of ``family``. An open bound on ANY constituent cell
    (``start``/``end`` is None) makes that side of the union open — the stage
    inherits its oldest cell's start and its newest cell's end.

    Returns a single ``(start, end)`` SPAN, so it is exact only for a
    CONTIGUOUS stage — every stage of the SPA-surfaced families
    (``_CULTURE_TO_ERA_FAMILY``: english / brythonic / goidelic) is contiguous.
    The ``latin`` family's ``latin`` stage is the lone non-contiguous case
    (classical + medieval + renaissance, with the ``vulgar-latin`` cell's
    200–700 window between them), so its span ``(None, 1800)`` over-includes that
    gap — reachable only via the CLI ``latin/<stage>`` path, not the SPA."""
    cells = cells_for_stage(family, stage)
    if not cells:
        return None
    ranges = [era_year_range(family, c) for c in cells]
    start = None if any(r[0] is None for r in ranges) else min(r[0] for r in ranges)
    end = None if any(r[1] is None for r in ranges) else max(r[1] for r in ranges)
    return (start, end)


def _range_for_family_label(family: str, label: str) -> tuple[int | None, int | None]:
    """The ``family/label`` branch of :func:`resolve_era_input`: a cell range,
    else a compressed-stage UNION range, else a bare KeyError / ValueError."""
    try:
        return era_year_range(family, label)
    except KeyError:
        # wyrd-rogd.2: explicit family/STAGE form (e.g. 'english/old-english')
        stage_range = stage_year_range(family, label)
        if stage_range is not None:
            return stage_range
        if family not in ERA_CELLS:
            raise  # unknown family — surface the bare KeyError
        valid = [*era_cells_for_family(family), *family_stage_order(family)]
        raise ValueError(
            f"unknown era cell/stage {label!r} for family {family!r}; valid: {valid}"
        ) from None


def _range_for_bare_label(era: str, default_family: str) -> tuple[int | None, int | None]:
    """The bare-label branch of :func:`resolve_era_input`: a cell or
    compressed-stage range of ``default_family``, else a cross-family hint,
    else 'unknown era input'."""
    try:
        return era_year_range(default_family, era)
    except KeyError:
        # wyrd-rogd.2: a compressed STAGE label (e.g. 'old-english') isn't a
        # cell but collapses several — resolve to the UNION of their ranges so
        # the Configure dropdown can offer stages, not raw cells. A label that is
        # BOTH a cell and a stage (goidelic 'old-irish' / 'middle-irish') already
        # resolved via the cell path above — the cell intentionally shadows the
        # stage (same range there, so identical), and this fallback never sees it.
        stage_range = stage_year_range(default_family, era)
        if stage_range is not None:
            return stage_range
        defined_in = [f for f in sorted(ERA_CELLS) if era in era_cells_for_family(f)]
        if defined_in:
            choices = ", ".join(f"{f}/{era}" for f in defined_in)
            raise ValueError(
                f"era cell {era!r} is not defined in family "
                f"{default_family!r}; use one of: {choices}"
            ) from None
        # wyrd-rogd.2: same cross-family hint for a STAGE label of another family.
        defined_in_stages = [f for f in sorted(ERA_CELLS) if cells_for_stage(f, era)]
        if defined_in_stages:
            choices = ", ".join(f"{f}/{era}" for f in defined_in_stages)
            raise ValueError(
                f"era stage {era!r} is not defined in family "
                f"{default_family!r}; use one of: {choices}"
            ) from None
        raise ValueError(
            f"unknown era input {era!r}; pass a year (e.g. 1086), a cell label "
            f"(e.g. 'oe-late'), a stage (e.g. 'old-english'), or a 'family/label' pair"
        ) from None


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

    Ranges are half-open intervals — a year exactly on ``start`` is
    included, a year exactly on ``end`` is not. (The D5-2 inventory
    filter that consumed these ranges was retired by D44 — era now only
    selects the rendered reflex; the resolver survives as the era-input
    VALIDATOR and the render-language anchor.)
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
    except (ValueError, TypeError):
        # ValueError: a non-numeric string (a cell label like "oe-late") — falls
        # through to the label paths below. TypeError: a non-string/non-int value
        # (a list/dict from a malformed request) — int() can't take it; it is
        # rejected as an unknown era input just below rather than escaping
        # uncaught (a bare int(["x"]) TypeError surfaces as a 500, not a 400).
        year = None
    if year is not None:
        return resolve_era_input(year, default_family=default_family)
    if not isinstance(era, str):
        raise ValueError(
            f"unknown era input {era!r}; expected a year, a cell label "
            "(e.g. 'oe-late'), or a 'family/label' string"
        )
    if "/" in era:
        family, _, label = era.partition("/")
        return _range_for_family_label(family, label)
    # Bare label — must exist in default_family. Cross-family fallback
    # would silently route a typo or wrong-culture label into the wrong
    # range; instead, point the user at the families that DO define the
    # label so they can switch to the explicit ``family/label`` form.
    return _range_for_bare_label(era, default_family)
