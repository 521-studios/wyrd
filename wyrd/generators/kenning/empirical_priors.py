"""Empirical-baseline priors extraction (wyrd-ecjp.2, D36.4 / D36.7).

The vector-driven generator's empirical-baseline scoring axis reads
per-(culture x position x tag x era) frequency priors. This module
extracts those priors from the lexicon DB (toponym_etymology +
toponym_etymology_element + etymon + etymon_tag) and persists them as

  * Two L3 tables in the lexicon DB
    (``empirical_priors_native``, ``empirical_priors_loan``) — primary
    storage, re-generated on demand.
  * A committed JSON sidecar at
    ``wyrd/generators/kenning/data/empirical_priors.json`` — produced
    via ``--dump-json``; the diff against the previous commit is the
    operator-visible "what did the corpus do this rebuild" view.

LLM-free, deterministic, idempotent. Re-running on an unchanged DB
state produces byte-identical SQL state + byte-identical JSON. D36.9's
re-runnable-derived-artifact contract.

Scope notes (v1):

* Native baseline currently extracts the ``culture=english`` slice only
  (toponym.country='England'). Other cultures (welsh, irish, scottish,
  breton) ship runtime-side via the legacy ``<culture>_proportions.json``
  artifacts; future mining + a country-to-culture mapping expand
  coverage.
* The smoothing rule (hierarchical fallback per D36.7) lives at
  *lookup* time, not at *bake* time — this module persists raw counts.
* Tag aggregation rule (weighted-by-request-tag-weight per D36.2) also
  lives at lookup time. Lemmas carrying multiple tags contribute their
  occurrence once per tag.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click

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
}

# Culture name -> era-family-bucket key (per ``era.ERA_CELLS``). The
# era_midpoint is resolved in the RECIPIENT culture's family so that
# every element of a single toponym lands in the same era bucket
# regardless of its donor language. Otherwise an Old Norse element
# in an Old English toponym would silently land in a different era
# cell than its sibling OE element (norse's on-classical = 1000;
# english's oe-late = 950) — a per-toponym semantic incoherence.
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


def era_midpoint_for(language: str, year: int) -> int | None:
    """Resolve (language, attested_year) -> integer-year midpoint of
    the era cell that year falls into, in the LANGUAGE's family.

    Returns None when no era cell matches (proto-languages, untracked
    language codes, year outside every cell's range). Callers skip
    rows with None midpoint rather than bucket them into a sentinel.

    For the priors extractor, prefer ``era_midpoint_for_culture`` so
    elements of a single toponym share an era bucket regardless of
    their donor language.
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

    The priors extractor uses this (not ``era_midpoint_for``) so every
    element of a single toponym shares one era bucket. An Old Norse
    loan in an English toponym attested 950 is reported as English/
    oe-late (midpoint 950), the same bucket as a sibling OE element
    in the same toponym.
    """
    family = _CULTURE_TO_FAMILY.get(culture)
    if family is None:
        return None
    cells = ERA_CELLS.get(family)
    if cells is None:
        return None
    for _label, start, end in cells:
        # Half-open semantics matching era.era_cell: start <= year < end.
        in_low = start is None or year >= start
        in_high = end is None or year < end
        if in_low and in_high:
            return _era_midpoint(start, end)
    return None


def _position_label(ordinal: int, element_count: int) -> str:
    """Derive a position label from (ordinal, element_count).

    Mirrors the legacy ``english_proportions.structures`` convention:
    a single-element 'word' uses ``'pre'`` (the head/solo position);
    multi-element compounds use ``'pre'`` / ``'inner'`` / ``'post'``
    by ordinal.
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
    position: str
    tag: str
    era_midpoint: int
    lemma_ref: str


@dataclass(frozen=True)
class _LoanKey:
    donor: str
    recipient: str
    position: str
    tag: str
    era_midpoint: int
    lemma_ref: str


@dataclass(frozen=True)
class ExtractionResult:
    """Per-run summary used by the CLI orchestrator + tests."""

    rows_scanned: int
    rows_with_era: int
    rows_with_tag: int
    native_cells_written: int
    loan_cells_written: int
    skipped_no_country: int
    skipped_no_era: int
    skipped_no_tag: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_scanned": self.rows_scanned,
            "rows_with_era": self.rows_with_era,
            "rows_with_tag": self.rows_with_tag,
            "native_cells_written": self.native_cells_written,
            "loan_cells_written": self.loan_cells_written,
            "skipped_no_country": self.skipped_no_country,
            "skipped_no_era": self.skipped_no_era,
            "skipped_no_tag": self.skipped_no_tag,
        }


# SQL: one row per (toponym_etymology_element x tag). Etymon_tag is
# joined inner so rows without tags get filtered at the SQL layer; the
# element_count window function gives us position derivation.
_EXTRACT_SQL = """
SELECT
    te.id              AS te_id,
    tee.ordinal        AS ordinal,
    cnt.element_count  AS element_count,
    t.country          AS country,
    e.canonical_form   AS canonical_form,
    e.language         AS language,
    tag.tag            AS tag,
    te.attested_year   AS attested_year
FROM toponym_etymology_element tee
JOIN (
    SELECT toponym_etymology_id, COUNT(*) AS element_count
    FROM toponym_etymology_element
    GROUP BY toponym_etymology_id
) cnt ON cnt.toponym_etymology_id = tee.toponym_etymology_id
JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
JOIN toponym t              ON t.id = te.toponym_id
JOIN etymon e               ON e.id = tee.etymon_id
JOIN etymon_tag tag         ON tag.etymon_id = e.id
"""

# Stable iteration order — same DB state, same row order, same
# extraction. Sort native and loan keys identically so the JSON dump
# is byte-stable on idempotent re-runs.
_EXTRACT_SQL_ORDERED = (
    _EXTRACT_SQL + " ORDER BY t.country, te.id, tee.ordinal, tag.tag, e.language, e.canonical_form"
)


def extract_priors(
    db: LexiconDB,
) -> tuple[dict[_NativeKey, int], dict[_LoanKey, int], ExtractionResult]:
    """Walk the lexicon DB and tally (cell -> count) for both prior tables.

    Returns ``(native_counts, loan_counts, summary)``. The two count
    dicts are dense — only observed cells are present. Empty input
    (no dated etymologies) returns empty dicts and a zero-counts
    summary.
    """
    native: dict[_NativeKey, int] = defaultdict(int)
    loan: dict[_LoanKey, int] = defaultdict(int)
    rows_scanned = 0
    rows_with_era = 0
    rows_with_tag = 0
    skipped_no_country = 0
    skipped_no_era = 0

    for row in db.conn.execute(_EXTRACT_SQL_ORDERED):
        rows_scanned += 1
        rows_with_tag += 1  # The SQL inner-joins on etymon_tag.
        country = row["country"] if isinstance(row, sqlite3.Row) else row[3]
        # Tolerate raw-tuple Row factory: lock down indices.
        if not isinstance(row, sqlite3.Row):
            te_id, ordinal, element_count, country, canonical, lang, tag, attested = row
        else:
            ordinal = row["ordinal"]
            element_count = row["element_count"]
            country = row["country"]
            canonical = row["canonical_form"]
            lang = row["language"]
            tag = row["tag"]
            attested = row["attested_year"]

        culture = _COUNTRY_TO_CULTURE.get(country or "")
        if culture is None:
            skipped_no_country += 1
            continue

        midpoint = era_midpoint_for_culture(culture, attested) if attested is not None else None
        if midpoint is None:
            skipped_no_era += 1
            continue
        rows_with_era += 1

        position = _position_label(ordinal, element_count)
        lemma_ref = f"{lang}:{canonical}"

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
                donor=lang,
                recipient=culture,
                position=position,
                tag=tag,
                era_midpoint=midpoint,
                lemma_ref=lemma_ref,
            )
        ] += 1

    summary = ExtractionResult(
        rows_scanned=rows_scanned,
        rows_with_era=rows_with_era,
        rows_with_tag=rows_with_tag,
        native_cells_written=len(native),
        loan_cells_written=len(loan),
        skipped_no_country=skipped_no_country,
        skipped_no_era=skipped_no_era,
        skipped_no_tag=0,  # SQL inner-joins on tag; no skipped-no-tag.
    )
    return dict(native), dict(loan), summary


def _replace_priors(
    db: LexiconDB,
    native: dict[_NativeKey, int],
    loan: dict[_LoanKey, int],
) -> None:
    """Clear and repopulate the two priors tables.

    Replace-not-merge: every successful run starts from an empty
    state. This is the right shape for a derived artifact — a stale
    row that no longer corresponds to a current observation would
    survive a merge but is meaningless data.
    """
    db.conn.execute("DELETE FROM empirical_priors_native")
    db.conn.execute("DELETE FROM empirical_priors_loan")
    db.conn.executemany(
        "INSERT INTO empirical_priors_native "
        "(culture, position, tag, era_midpoint, lemma_ref, count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(k.culture, k.position, k.tag, k.era_midpoint, k.lemma_ref, n) for k, n in native.items()],
    )
    db.conn.executemany(
        "INSERT INTO empirical_priors_loan "
        "(donor, recipient, position, tag, era_midpoint, lemma_ref, count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (k.donor, k.recipient, k.position, k.tag, k.era_midpoint, k.lemma_ref, n)
            for k, n in loan.items()
        ],
    )


def mine_empirical_baselines(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 500,  # noqa: ARG001 — reserved for future progress reporting
) -> dict[str, int]:
    """Extract empirical baselines from the lexicon DB into the
    ``empirical_priors_native`` + ``empirical_priors_loan`` tables.

    ``apply=False`` is a dry-run that returns the would-write summary
    without modifying the DB.
    ``apply=True`` replaces existing rows in both tables and commits.

    Idempotent: re-running on the same DB state produces identical
    table contents (cleared + repopulated each apply run).
    """
    native, loan, summary = extract_priors(db)
    if apply:
        _replace_priors(db, native, loan)
        db.commit()
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
    per-cell lemmas in lexicographic order of lemma_ref. Stable on
    re-runs against the same DB state — the resulting file diffs
    cleanly under git when priors evolve.

    ``version`` is written verbatim into the JSON. Callers may pass a
    content hash, a sequential integer, or a date string per their
    versioning policy.
    """
    native_rows = list(
        db.conn.execute(
            "SELECT culture, position, tag, era_midpoint, lemma_ref, count "
            "FROM empirical_priors_native"
        )
    )
    loan_rows = list(
        db.conn.execute(
            "SELECT donor, recipient, position, tag, era_midpoint, lemma_ref, count "
            "FROM empirical_priors_loan"
        )
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


# --- CLI entry points (registered in cli.py) ----------------------------


def cli_mine_empirical_baselines(
    db_path: Path,
    apply_changes: bool,
) -> None:
    """Thin wrapper used by the ``lexicon mine-empirical-baselines`` CLI
    command. Kept separate so tests can drive the extraction without
    Click's invocation contract.
    """
    with LexiconDB(db_path) as db:
        result = mine_empirical_baselines(db, apply=apply_changes)
    click.echo("mine-empirical-baselines:", err=True)
    click.echo(
        "  scanned={scanned:>5}  with_era={era:>5}  "
        "native_cells={nc:>5}  loan_cells={lc:>5}  "
        "skip_country={sc:>5}  skip_era={se:>5}".format(
            scanned=result["rows_scanned"],
            era=result["rows_with_era"],
            nc=result["native_cells_written"],
            lc=result["loan_cells_written"],
            sc=result["skipped_no_country"],
            se=result["skipped_no_era"],
        ),
        err=True,
    )
    if not apply_changes:
        click.echo(
            "(dry-run; pass --apply to write empirical_priors_* rows)",
            err=True,
        )


def cli_dump_empirical_priors(
    db_path: Path,
    output_path: Path,
    version: str,
) -> None:
    """Thin wrapper for the ``lexicon dump-empirical-priors`` CLI."""
    with LexiconDB(db_path) as db:
        result = dump_empirical_priors_to_json(db, output_path, version=version)
    click.echo("dump-empirical-priors:", err=True)
    click.echo(
        f"  wrote {result['native_cells']} native cells + "
        f"{result['loan_cells']} loan cells to {output_path}",
        err=True,
    )
