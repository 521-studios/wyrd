"""Empirical-baseline priors extraction (wyrd-ecjp.2, D36.4 / D36.7).

The vector-driven generator's empirical-baseline scoring axis reads
per-(culture x position x tag x era) frequency priors. This module
extracts those priors from the lexicon DB (toponym_etymology +
toponym_etymology_element + etymon + etymon_tag) and persists them as

  * Two L3 tables in the lexicon DB
    (``empirical_priors_native``, ``empirical_priors_loan``) — primary
    storage, re-generated on demand.
  * A committed JSON sidecar (produced via ``lexicon
    dump-empirical-priors``) — the diff against the previous commit
    is the operator-visible "what did the corpus do this rebuild" view.

LLM-free, deterministic, idempotent. Re-running on an unchanged DB
state produces byte-identical SQL state + byte-identical JSON. D36.9's
re-runnable-derived-artifact contract.

Scope notes (v1):

* Native baseline currently extracts the ``culture=english`` slice only
  (toponym.country='England'). Other cultures (welsh, irish, scottish,
  breton) ship runtime-side via the legacy ``<culture>_proportions.json``
  artifacts; future mining + a country-to-culture mapping expand
  coverage. Non-English toponyms surface as ``skipped_country_unmapped``
  in the extraction summary so the gap stays visible.
* The smoothing rule (hierarchical fallback per D36.7) lives at
  *lookup* time, not at *bake* time — this module persists raw counts.
* Tag aggregation rule lives at lookup time too (per D36.2). A lemma
  carrying tags [plant, tree] contributes its occurrence ONCE per tag
  to the priors tables (one row per cell); at scoring time the request
  vector's per-tag weight selects the aggregation
  ``baseline(lemma) = sum_over_tags_T(request.sem.tags[T] * priors[T])``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wyrd.generators.kenning.era import ERA_CELLS, LANGUAGE_TO_FAMILY, era_cell, era_year_range
from wyrd.generators.kenning.lexicon import LexiconDB

# Culture inference from toponym.country. v1 covers the English slice
# only (where the lexicon DB actually has toponym coverage). Future
# coverage of the other CULTURES happens by adding mining-source mappings.
_COUNTRY_TO_CULTURE: dict[str, str] = {
    "England": "english",
    # 'Ireland' has 674 rows but isn't validated against the irish
    # culture's runtime expectations yet — leave it as a deliberate gap
    # so the priors-vs-legacy proportions diff doesn't false-positive.
    # Toponyms with a country not in this map count into
    # ``skipped_country_unmapped`` so the gap stays visible.
}

# Culture name -> era-family-bucket key (per ``era.ERA_CELLS``). The
# era_midpoint is resolved in the RECIPIENT culture's family so that
# every element of a single toponym lands in the same era bucket
# regardless of its donor language. Otherwise an Old Norse element
# in an Old English toponym would silently land in a different era
# cell than its sibling OE element (norse's on-classical = 1000;
# english's oe-late = 950) — a per-toponym semantic incoherence.
#
# This map MUST cover every value in ``_COUNTRY_TO_CULTURE.values()``.
# ``era_midpoint_for_culture`` raises if it doesn't, so config drift
# fires loudly rather than silently bucketing every row of the affected
# culture into ``skipped_no_era``.
_CULTURE_TO_FAMILY: dict[str, str] = {
    "english": "english",
    "welsh": "brythonic",
    "breton": "brythonic",
    "irish": "goidelic",
    "scottish": "goidelic",
}

# Era-midpoint convention for open-ended cells. The era schema in
# vector_schemas requires an int; cells with only one bound use a
# fixed-offset midpoint relative to the defined bound.
#
# Closed cell (start, end): (start + end) // 2.
# Open-low (None, end):     end - _OPEN_BOUND_OFFSET.
# Open-high (start, None):  start + _OPEN_BOUND_OFFSET.
_OPEN_BOUND_OFFSET = 100

PositionLabel = Literal["pre", "inner", "post"]


def _era_midpoint(start: int | None, end: int | None) -> int:
    """Compute the integer-year midpoint of an era cell.

    Open-bound cells use the ``_OPEN_BOUND_OFFSET`` convention so every
    cell has a deterministic single-int identity.
    """
    if start is not None and end is not None:
        return (start + end) // 2
    if start is None and end is not None:
        return end - _OPEN_BOUND_OFFSET
    if start is not None and end is None:
        return start + _OPEN_BOUND_OFFSET
    # Both None: undefined cell. Era cells in ERA_CELLS always have at
    # least one bound, so this case shouldn't trigger; raise loudly
    # rather than silently return 0.
    raise ValueError("era cell has both start and end as None")


def _era_midpoint_for_language(language: str, year: int) -> int | None:
    """Resolve (language, attested_year) -> integer-year midpoint of
    the era cell that year falls into, in the LANGUAGE's family.

    Private — the priors extractor uses ``era_midpoint_for_culture``
    so elements of a single toponym share an era bucket regardless of
    their donor language. This helper is retained for callers that
    genuinely need per-language era resolution (e.g. ad-hoc tooling).

    Returns None when no era cell matches (proto-languages, untracked
    language codes, year outside every cell's range).
    """
    cell_label = era_cell(language, year)
    if cell_label is None:
        return None
    family = LANGUAGE_TO_FAMILY.get(language)
    if family is None:
        return None
    start, end = era_year_range(family, cell_label)
    return _era_midpoint(start, end)


def era_midpoint_for_culture(culture: str, year: int) -> int | None:
    """Resolve (culture, attested_year) -> integer-year midpoint of
    the era cell that year falls into, in the RECIPIENT CULTURE's
    family.

    Raises ``ValueError`` when ``culture`` is in
    ``_COUNTRY_TO_CULTURE.values()`` (i.e. a culture the extractor
    routes rows to) but missing from ``_CULTURE_TO_FAMILY`` or with
    no entry in ``ERA_CELLS`` — these are configuration drift between
    the two maps and would otherwise silently bucket every row of the
    affected culture into ``skipped_no_era``.

    Returns ``None`` when ``culture`` is entirely unknown (caller's
    error; raise also makes sense but we leave that to the caller)
    or when the year falls outside every cell's range in the family.
    """
    if culture not in _CULTURE_TO_FAMILY:
        return None
    family = _CULTURE_TO_FAMILY[culture]
    cells = ERA_CELLS.get(family)
    if cells is None:
        raise ValueError(
            f"culture {culture!r} maps to family {family!r}, "
            f"but {family!r} is not in era.ERA_CELLS — "
            "fix _CULTURE_TO_FAMILY or add the family's cells"
        )
    for _label, start, end in cells:
        # Half-open semantics matching era.era_cell: start <= year < end.
        in_low = start is None or year >= start
        in_high = end is None or year < end
        if in_low and in_high:
            return _era_midpoint(start, end)
    return None


def _position_label(ordinal: int, element_count: int) -> PositionLabel:
    """Derive a position label from (ordinal, element_count).

    Single-element 'words' use ``'pre'`` — the solo morpheme is
    treated as the head. Multi-element compounds use ``'pre'`` /
    ``'inner'`` / ``'post'`` by ordinal (first / middle / last).
    The choice for single-element 'pre' is this module's convention,
    not inherited from any legacy proportions data.
    """
    if element_count == 1:
        return "pre"
    if ordinal == 0:
        return "pre"
    if ordinal == element_count - 1:
        return "post"
    return "inner"


@dataclass(frozen=True)
class _NativeKey:
    culture: str
    position: PositionLabel
    tag: str
    era_midpoint: int
    lemma_ref: str


@dataclass(frozen=True)
class _LoanKey:
    donor: str
    recipient: str
    position: PositionLabel
    tag: str
    era_midpoint: int
    lemma_ref: str


@dataclass(frozen=True)
class ExtractionResult:
    """Per-run summary used by the CLI orchestrator + tests.

    Partition invariant (enforced by ``__post_init__``):
        rows_scanned == rows_emitted
                      + skipped_country_unknown
                      + skipped_country_unmapped
                      + skipped_no_era
                      + skipped_no_tag

    Each SQL row contributes to exactly one bucket. Skip-reason fields
    are mutually exclusive — the extractor's gate order is country
    (unknown then unmapped) → era → tag, and the first failing gate
    wins.
    """

    rows_scanned: int
    rows_emitted: int
    native_cells_written: int
    loan_cells_written: int
    skipped_country_unknown: int
    skipped_country_unmapped: int
    skipped_no_era: int
    skipped_no_tag: int

    def __post_init__(self) -> None:
        skipped = (
            self.skipped_country_unknown
            + self.skipped_country_unmapped
            + self.skipped_no_era
            + self.skipped_no_tag
        )
        if self.rows_emitted + skipped != self.rows_scanned:
            raise ValueError(
                "ExtractionResult partition invariant violated: "
                f"rows_scanned={self.rows_scanned} != "
                f"rows_emitted={self.rows_emitted} + "
                f"skipped_country_unknown={self.skipped_country_unknown} + "
                f"skipped_country_unmapped={self.skipped_country_unmapped} + "
                f"skipped_no_era={self.skipped_no_era} + "
                f"skipped_no_tag={self.skipped_no_tag}"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_scanned": self.rows_scanned,
            "rows_emitted": self.rows_emitted,
            "native_cells_written": self.native_cells_written,
            "loan_cells_written": self.loan_cells_written,
            "skipped_country_unknown": self.skipped_country_unknown,
            "skipped_country_unmapped": self.skipped_country_unmapped,
            "skipped_no_era": self.skipped_no_era,
            "skipped_no_tag": self.skipped_no_tag,
        }


# SQL: one row per (toponym_etymology_element x tag). Etymon_tag is
# LEFT-joined so untagged etymons come through with tag=NULL and get
# counted into ``skipped_no_tag`` rather than silently dropped at the
# SQL layer. The ``cnt`` derived-table subquery counts elements per
# etymology so we can derive position via (ordinal, element_count).
_EXTRACT_SQL = """
SELECT
    tee.ordinal                AS ordinal,
    cnt.element_count          AS element_count,
    t.country                  AS country,
    t.modern_name              AS modern_name,
    COALESCE(t.region, '')     AS region,
    te.attested_year           AS attested_year,
    te.source_id               AS source_id,
    te.page                    AS page,
    e.canonical_form           AS canonical_form,
    e.language                 AS language,
    tag.tag                    AS tag
FROM toponym_etymology_element tee
JOIN (
    SELECT toponym_etymology_id, COUNT(*) AS element_count
    FROM toponym_etymology_element
    GROUP BY toponym_etymology_id
) cnt ON cnt.toponym_etymology_id = tee.toponym_etymology_id
JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
JOIN toponym t              ON t.id = te.toponym_id
JOIN etymon e               ON e.id = tee.etymon_id
LEFT JOIN etymon_tag tag    ON tag.etymon_id = e.id
"""

# Stable iteration order — same DB state, same row order, same
# extraction. Sort by content-addressable fields (toponym identity +
# attested year + source + page + ordinal + tag + donor language +
# canonical form) so two DBs independently rebuilt from the same L2
# JSONL produce the same row order regardless of insert-time rowids.
_EXTRACT_SQL_ORDERED = (
    _EXTRACT_SQL + " ORDER BY t.country, t.modern_name, "
    "COALESCE(t.region, ''), te.attested_year, te.source_id, "
    "COALESCE(te.page, ''), tee.ordinal, "
    "COALESCE(tag.tag, ''), e.language, e.canonical_form"
)


def extract_priors(
    db: LexiconDB,
    *,
    progress_every: int = 0,
) -> tuple[dict[_NativeKey, int], dict[_LoanKey, int], ExtractionResult]:
    """Walk the lexicon DB and tally (cell -> count) for both prior tables.

    Returns ``(native_counts, loan_counts, summary)``. The two count
    dicts are dense — only observed cells are present. Empty input
    (no etymology elements) returns empty dicts and a zero-counts
    summary.

    ``progress_every``: emit a stderr progress line every N rows
    scanned (0 = silent). Convention from CLAUDE.md: long-running
    mining/ingestion CLIs MUST emit periodic progress so operators
    aren't watching a silent process for minutes.

    ``LexiconDB`` always sets ``row_factory = sqlite3.Row``
    (lexicon.py LexiconDB.__init__), so rows support named attribute
    access without a defensive fallback.
    """
    native: dict[_NativeKey, int] = defaultdict(int)
    loan: dict[_LoanKey, int] = defaultdict(int)
    rows_scanned = 0
    rows_emitted = 0
    skipped_country_unknown = 0
    skipped_country_unmapped = 0
    skipped_no_era = 0
    skipped_no_tag = 0

    for row in db.conn.execute(_EXTRACT_SQL_ORDERED):
        rows_scanned += 1
        if progress_every and rows_scanned % progress_every == 0:
            print(
                f"  [{rows_scanned}] scanned  emitted={rows_emitted} "
                f"skip_country_unknown={skipped_country_unknown} "
                f"skip_country_unmapped={skipped_country_unmapped} "
                f"skip_no_era={skipped_no_era} "
                f"skip_no_tag={skipped_no_tag}",
                file=sys.stderr,
                flush=True,
            )

        country = row["country"]
        attested = row["attested_year"]
        tag = row["tag"]

        # Gate 1: country.
        if not country:
            skipped_country_unknown += 1
            continue
        culture = _COUNTRY_TO_CULTURE.get(country)
        if culture is None:
            skipped_country_unmapped += 1
            continue

        # Gate 2: era. era_midpoint_for_culture raises on config drift
        # (missing _CULTURE_TO_FAMILY entry for a culture we route to)
        # so only out-of-range year and missing attested_year reach
        # this skip counter.
        midpoint = era_midpoint_for_culture(culture, attested) if attested is not None else None
        if midpoint is None:
            skipped_no_era += 1
            continue

        # Gate 3: tag. LEFT JOIN keeps untagged etymons in the result
        # set so the gap surfaces here rather than at the SQL layer.
        if tag is None:
            skipped_no_tag += 1
            continue

        position = _position_label(row["ordinal"], row["element_count"])
        lemma_ref = f"{row['language']}:{row['canonical_form']}"

        native[
            _NativeKey(
                culture=culture,
                position=position,
                tag=tag,
                era_midpoint=midpoint,
                lemma_ref=lemma_ref,
            )
        ] += 1
        loan[
            _LoanKey(
                donor=row["language"],
                recipient=culture,
                position=position,
                tag=tag,
                era_midpoint=midpoint,
                lemma_ref=lemma_ref,
            )
        ] += 1
        rows_emitted += 1

    summary = ExtractionResult(
        rows_scanned=rows_scanned,
        rows_emitted=rows_emitted,
        native_cells_written=len(native),
        loan_cells_written=len(loan),
        skipped_country_unknown=skipped_country_unknown,
        skipped_country_unmapped=skipped_country_unmapped,
        skipped_no_era=skipped_no_era,
        skipped_no_tag=skipped_no_tag,
    )
    return dict(native), dict(loan), summary


def _replace_priors(
    db: LexiconDB,
    native: dict[_NativeKey, int],
    loan: dict[_LoanKey, int],
) -> None:
    """Clear and repopulate the two priors tables atomically.

    Replace-not-merge: every successful run starts from an empty
    state. A stale row that no longer corresponds to a current
    observation would survive a merge but is meaningless data.

    Wrapped in ``with db.conn:`` so the DELETE + INSERT pair is one
    transaction — a mid-way failure rolls back rather than leaving
    both tables truncated.
    """
    with db.conn:
        db.conn.execute("DELETE FROM empirical_priors_native")
        db.conn.execute("DELETE FROM empirical_priors_loan")
        db.conn.executemany(
            "INSERT INTO empirical_priors_native "
            "(culture, position, tag, era_midpoint, lemma_ref, count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                (k.culture, k.position, k.tag, k.era_midpoint, k.lemma_ref, n)
                for k, n in native.items()
            ),
        )
        db.conn.executemany(
            "INSERT INTO empirical_priors_loan "
            "(donor, recipient, position, tag, era_midpoint, lemma_ref, count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (k.donor, k.recipient, k.position, k.tag, k.era_midpoint, k.lemma_ref, n)
                for k, n in loan.items()
            ),
        )


def mine_empirical_baselines(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 0,
) -> dict[str, int]:
    """Extract empirical baselines from the lexicon DB into the
    ``empirical_priors_native`` + ``empirical_priors_loan`` tables.

    ``apply=False`` is a dry-run that returns the would-write summary
    without modifying the DB.
    ``apply=True`` atomically replaces existing rows in both tables.

    Idempotent: re-running on the same DB state produces identical
    table contents (cleared + repopulated each apply run).

    ``progress_every``: pass-through to ``extract_priors`` — emit a
    stderr progress line every N rows. 0 (default) is silent.
    """
    native, loan, summary = extract_priors(db, progress_every=progress_every)
    if apply:
        _replace_priors(db, native, loan)
    return summary.as_dict()


# JSON dump format. The lemma-keyed dict per cell is the natural shape
# for the runtime EmpiricalPriors loader.
#
# {
#   "version": "<content-hash>",
#   "native": [
#     {
#       "culture": "english",
#       "position": "pre",
#       "tag": "topography",
#       "era_midpoint": 950,
#       "lemmas": {"old-english:hyll": 7, "old-english:dūn": 3, ...}
#     },
#     ...
#   ],
#   "loan": [ { "donor": "old-norse", "recipient": "english", ... } ]
# }
def _cell_records(
    rows: Iterable[tuple],
    cell_fields: tuple[str, ...],
) -> list[dict]:
    grouped: dict[tuple, dict[str, int]] = {}
    for *cell, lemma_ref, count in rows:
        cell_key = tuple(cell)
        grouped.setdefault(cell_key, {})[lemma_ref] = int(count)
    records = []
    for cell_key in sorted(grouped.keys()):
        record = dict(zip(cell_fields, cell_key, strict=True))
        record["lemmas"] = dict(sorted(grouped[cell_key].items()))
        records.append(record)
    return records


_NATIVE_CELL_FIELDS = ("culture", "position", "tag", "era_midpoint")
_LOAN_CELL_FIELDS = ("donor", "recipient", "position", "tag", "era_midpoint")


def dump_empirical_priors_to_json(
    db: LexiconDB,
    output_path: Path,
    *,
    version: str = "unversioned",
) -> dict[str, int]:
    """Read both priors tables and emit a sorted, byte-stable JSON
    artifact to ``output_path``.

    Sort order: cells in lexicographic order of their field tuple;
    per-cell lemmas in lexicographic order of lemma_ref. ORDER BY in
    the SELECT clauses is belt-and-suspenders — ``_cell_records``
    re-sorts everything, but the explicit ORDER BY documents intent
    and protects any future caller that bypasses ``_cell_records``.

    ``version`` is written verbatim into the JSON. Callers may pass a
    content hash, a sequential integer, or a date string per their
    versioning policy.
    """
    native_rows = db.conn.execute(
        "SELECT culture, position, tag, era_midpoint, lemma_ref, count "
        "FROM empirical_priors_native "
        "ORDER BY culture, position, tag, era_midpoint, lemma_ref"
    )
    loan_rows = db.conn.execute(
        "SELECT donor, recipient, position, tag, era_midpoint, lemma_ref, count "
        "FROM empirical_priors_loan "
        "ORDER BY donor, recipient, position, tag, era_midpoint, lemma_ref"
    )
    artifact = {
        "version": version,
        "native": _cell_records(native_rows, _NATIVE_CELL_FIELDS),
        "loan": _cell_records(loan_rows, _LOAN_CELL_FIELDS),
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return {
        "native_cells": len(artifact["native"]),
        "loan_cells": len(artifact["loan"]),
    }


# Convenience for tests + external callers that want to confirm the
# era-midpoint convention is the one this module advertises.
def known_era_midpoints() -> dict[tuple[str, str], int]:
    """Return ``{(family, cell_label): midpoint_year}`` for every era
    cell defined in ``era.ERA_CELLS``. Useful for tests + tooling that
    needs to project rows into priors-space.
    """
    out: dict[tuple[str, str], int] = {}
    for family, cells in ERA_CELLS.items():
        for label, start, end in cells:
            out[(family, label)] = _era_midpoint(start, end)
    return out
