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
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from wyrd.generators.kenning.era.cells import ERA_CELLS
from wyrd.generators.kenning.lexicon import _LANG_CODE_TO_JSON_FIELD, LexiconDB
from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

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
# culture into ``skipped_year_*``.
_CULTURE_TO_FAMILY: dict[str, str] = {
    "english": "english",
    "welsh": "brythonic",
    "breton": "brythonic",
    "irish": "goidelic",
    "scottish": "goidelic",
}

# Era-midpoint convention for open-ended cells. The era schema in
# vectors.schemas requires an int; cells with only one bound use a
# fixed-offset midpoint relative to the defined bound.
#
# Closed cell (start, end): (start + end) // 2.
# Open-low (None, end):     end - _OPEN_BOUND_OFFSET.
# Open-high (start, None):  start + _OPEN_BOUND_OFFSET.
_OPEN_BOUND_OFFSET = 100

PositionLabel = Literal["bare", "pre", "inner", "post"]


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


def era_midpoint_for_culture(culture: str, year: int) -> int | None:
    """Resolve (culture, attested_year) -> integer-year midpoint of
    the era cell that year falls into, in the RECIPIENT CULTURE's
    family.

    Raises ``ValueError`` when ``culture`` is missing from
    ``_CULTURE_TO_FAMILY`` or when its family is missing from
    ``ERA_CELLS`` — these are configuration drift between the maps
    and would otherwise silently bucket every row of the affected
    culture into the ``skipped_year_*`` counters. Production callers
    always reach this function via a culture from
    ``_COUNTRY_TO_CULTURE.values()``, which by contract is also in
    ``_CULTURE_TO_FAMILY``; an unknown culture string is a programming
    error, not a recoverable miss.

    Returns ``None`` only when the year falls outside every cell's
    range in the family — that's the legitimate data-quality gap
    the ``skipped_year_out_of_range`` counter surfaces.
    """
    if culture not in _CULTURE_TO_FAMILY:
        raise ValueError(
            f"culture {culture!r} is missing from _CULTURE_TO_FAMILY — "
            "an unknown culture is a config or caller error, not a "
            "recoverable miss. Add the entry or fix the caller."
        )
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

    Single-element 'words' use ``'bare'`` — the sole morpheme, matching
    D40's derived-position vocabulary (sole piece → ``bare``, first →
    ``pre``, interior → ``inner``, last → ``post``). Multi-element
    compounds use ``'pre'`` / ``'inner'`` / ``'post'`` by ordinal.

    wyrd-nkiq: this MUST agree with the runtime lookup label
    (``runtime.vector_name_select._slot_position_label``), which resolves
    a no-dash (single-morpheme) slot to ``'bare'``. Baking single-element
    words at ``'pre'`` here (the old convention) stranded them in a cell
    the ``'bare'`` lookup never queried — the baseline axis silently
    fell through to the coarse position-agnostic fallback for every
    single-word generation.
    """
    if element_count == 1:
        return "bare"
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
                      + skipped_year_unknown
                      + skipped_year_out_of_range
                      + skipped_no_tag

    Each SQL row contributes to exactly one bucket. Skip-reason fields
    are mutually exclusive — the extractor's gate order is country
    (unknown then unmapped) → year (unknown then out-of-range) → tag,
    and the first failing gate wins.

    Year skips are split into two reasons because they need different
    remediation: ``_unknown`` means ``toponym_etymology.attested_year``
    is NULL (operator: backfill years from source citations);
    ``_out_of_range`` means the year is present but falls outside
    every era cell's range in the recipient culture's family
    (operator: extend ``ERA_CELLS`` or accept the proto-period gap).
    """

    rows_scanned: int
    rows_emitted: int
    native_cells_written: int
    loan_cells_written: int
    skipped_country_unknown: int
    skipped_country_unmapped: int
    skipped_year_unknown: int
    skipped_year_out_of_range: int
    skipped_no_tag: int

    def __post_init__(self) -> None:
        skipped = (
            self.skipped_country_unknown
            + self.skipped_country_unmapped
            + self.skipped_year_unknown
            + self.skipped_year_out_of_range
            + self.skipped_no_tag
        )
        if self.rows_emitted + skipped != self.rows_scanned:
            raise ValueError(
                "ExtractionResult partition invariant violated: "
                f"rows_scanned={self.rows_scanned} != "
                f"rows_emitted={self.rows_emitted} + "
                f"skipped_country_unknown={self.skipped_country_unknown} + "
                f"skipped_country_unmapped={self.skipped_country_unmapped} + "
                f"skipped_year_unknown={self.skipped_year_unknown} + "
                f"skipped_year_out_of_range={self.skipped_year_out_of_range} + "
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
            "skipped_year_unknown": self.skipped_year_unknown,
            "skipped_year_out_of_range": self.skipped_year_out_of_range,
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
    te.attested_year           AS attested_year,
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

# Within-DB-stable iteration order — same DB state, same row order.
# Sort by (toponym_etymology_id, ordinal, tag) which uniquely orders
# rows within one DB without a full content-key sort.
#
# Note: this ORDER BY is for DEBUGGABLE ITERATION (deterministic log
# / trace output if extract_priors is observed mid-run). It is NOT
# what guarantees byte-stable JSON dump output — that derives from
# commutative dict-accumulation, content-PRIMARY-KEYs on the priors
# tables, the dump SELECTs' own ORDER BY clauses, and _cell_records
# re-sorting. A wider content-key ORDER BY would also give cross-DB
# consistent iteration order, but no downstream contract depends on
# that, and the 10-column sort cost showed up as a perf concern on
# corpus-scale extracts (Gemini round-4 finding). See
# test_dump_json_byte_stable_across_independent_dbs for the end-to-
# end byte-stability pin.
_EXTRACT_SQL_ORDERED = (
    _EXTRACT_SQL + " ORDER BY tee.toponym_etymology_id, tee.ordinal, COALESCE(tag.tag, '')"
)


# wyrd-3x9e: every celtic-family L3 language collapses into the single
# ``celtic_mix`` bundle field, and at runtime ``_lemma_ref_for`` maps that field
# back to the lone representative ``celtic`` (``_BUNDLE_FIELD_TO_L3_LANG``). So a
# native prior keyed on any OTHER celtic language — ``brittonic:``, ``welsh:``,
# ``old-irish:``, … — is unreachable: the runtime lookup only ever synthesizes
# ``celtic:<form>``. Collapse the celtic family to that representative when
# building the ``lemma_ref`` key so the emitted key matches the runtime lookup.
#
# The member set is DERIVED from the emit map (so emit-map growth can't silently
# reintroduce an unreachable key) and unioned with the two coarse family tags the
# emit map doesn't carry but the L3 ``etymon.language`` column does. This is a
# CELTIC-SCOPED fix (wyrd-3x9e owner decision) — the analogous full
# representative normalization for the other bundle fields is deliberately out of
# scope.
_CELTIC_LEMMA_LANGUAGES: frozenset[str] = frozenset(
    lang for lang, field in _LANG_CODE_TO_JSON_FIELD.items() if field == "celtic_mix"
) | {"brittonic", "goidelic"}
# The runtime representative for ``celtic_mix`` — MUST stay in sync with
# ``_BUNDLE_FIELD_TO_L3_LANG["celtic_mix"]`` (runtime/vector_name_select.py); the
# end-to-end reachability test pins the two together.
_CELTIC_LEMMA_REPRESENTATIVE: str = "celtic"


def _classify_prior_row(row) -> tuple[str, _NativeKey | None, _LoanKey | None]:
    """Apply the country / culture / year / era / tag gates to one scanned row.

    Returns ``(status, native_key, loan_key)``. ``status`` is ``"ok"`` (both keys
    set) or one of the skip-reason names (both keys ``None``) — the names match
    the ``ExtractionResult`` skip fields, so the caller can tally them directly.
    """
    # Gate 1: country.
    country = row["country"]
    if not country:
        return "skipped_country_unknown", None, None
    culture = _COUNTRY_TO_CULTURE.get(country)
    if culture is None:
        return "skipped_country_unmapped", None, None

    # Gate 2a: attested year present. era_midpoint_for_culture raises on config
    # drift (missing _CULTURE_TO_FAMILY entry for a culture we route to), so NULL
    # year is the only "missing year" path here.
    attested = row["attested_year"]
    if attested is None:
        return "skipped_year_unknown", None, None
    # Gate 2b: year falls into an era cell. None means the year is outside every
    # cell's range — operator: extend ERA_CELLS or accept the proto-period gap.
    midpoint = era_midpoint_for_culture(culture, attested)
    if midpoint is None:
        return "skipped_year_out_of_range", None, None

    # Gate 3: tag. LEFT JOIN keeps untagged etymons in the result set so the gap
    # surfaces here rather than at the SQL layer.
    tag = row["tag"]
    if tag is None:
        return "skipped_no_tag", None, None

    position = _position_label(row["ordinal"], row["element_count"])
    # wyrd-3x9e: normalize the celtic family to its runtime representative for the
    # lemma_ref key ONLY. loan_key.donor below keeps the raw language — donor
    # identity is a separate axis, outside this celtic-scoped fix.
    language = row["language"]
    lemma_language = (
        _CELTIC_LEMMA_REPRESENTATIVE if language in _CELTIC_LEMMA_LANGUAGES else language
    )
    lemma_ref = f"{lemma_language}:{row['canonical_form']}"
    native_key = _NativeKey(
        culture=culture,
        position=position,
        tag=tag,
        era_midpoint=midpoint,
        lemma_ref=lemma_ref,
    )
    loan_key = _LoanKey(
        donor=row["language"],
        recipient=culture,
        position=position,
        tag=tag,
        era_midpoint=midpoint,
        lemma_ref=lemma_ref,
    )
    return "ok", native_key, loan_key


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
    skips: Counter[str] = Counter()

    def _emit_progress() -> None:
        print(
            f"  [{rows_scanned}] scanned  emitted={rows_emitted} "
            f"skip_country_unknown={skips['skipped_country_unknown']} "
            f"skip_country_unmapped={skips['skipped_country_unmapped']} "
            f"skip_year_unknown={skips['skipped_year_unknown']} "
            f"skip_year_out_of_range={skips['skipped_year_out_of_range']} "
            f"skip_no_tag={skips['skipped_no_tag']}",
            file=sys.stderr,
            flush=True,
        )

    for row in db.conn.execute(_EXTRACT_SQL_ORDERED):
        rows_scanned += 1
        if progress_every and rows_scanned % progress_every == 0:
            _emit_progress()

        status, native_key, loan_key = _classify_prior_row(row)
        if status != "ok":
            skips[status] += 1
            continue
        native[native_key] += 1
        loan[loan_key] += 1
        rows_emitted += 1

    # Final progress line — CLAUDE.md requires "Always echo a final
    # line at completion so the last partial chunk shows up", even
    # when total isn't an exact multiple of progress_every.
    if progress_every and rows_scanned % progress_every != 0:
        _emit_progress()

    summary = ExtractionResult(
        rows_scanned=rows_scanned,
        rows_emitted=rows_emitted,
        native_cells_written=len(native),
        loan_cells_written=len(loan),
        skipped_country_unknown=skips["skipped_country_unknown"],
        skipped_country_unmapped=skips["skipped_country_unmapped"],
        skipped_year_unknown=skips["skipped_year_unknown"],
        skipped_year_out_of_range=skips["skipped_year_out_of_range"],
        skipped_no_tag=skips["skipped_no_tag"],
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


def load_empirical_priors_from_json(input_path: Path) -> EmpiricalPriors:
    """Counterpart to :func:`dump_empirical_priors_to_json`. Reads the
    JSON sidecar that the lexicon emits + returns a populated
    :class:`EmpiricalPriors` instance.

    Used by the runtime path (wyrd-ecjp.5 NameGenerator rewrite — the
    vector-scoring branch reads priors at Lambda startup once, then
    every per-request scoring call hits the in-memory mapping). Bundle
    builders may also embed the JSON directly into ``meanings.json``
    for single-file deploys (wyrd-ecjp.8).

    JSON shape (per :func:`dump_empirical_priors_to_json`):
        {
          "version": str,
          "native": [
            {"culture": str, "position": str, "tag": str,
             "era_midpoint": int, "lemmas": {lemma_ref: count}}, ...
          ],
          "loan": [
            {"donor": str, "recipient": str, "position": str,
             "tag": str, "era_midpoint": int,
             "lemmas": {lemma_ref: count}}, ...
          ]
        }

    Tolerant of missing keys: an artifact emitted by an older
    revision (pre-loan-priors, say) loads cleanly with the missing
    side defaulting to an empty dict — the scoring runtime degrades
    gracefully when a pack lookup misses.

    Deterministic: same JSON input -> same EmpiricalPriors instance
    contents (Python dict iteration order doesn't matter for the
    scoring callers; the JSON's sorted shape preserves bit-stability
    of the underlying lemma->count mappings).
    """
    with input_path.open(encoding="utf-8") as f:
        artifact = json.load(f)
    return load_empirical_priors_from_payload(artifact)


def load_empirical_priors_from_payload(artifact: dict) -> EmpiricalPriors:
    """wyrd-ecjp.10b: same parsing as :func:`load_empirical_priors_from_json`
    but takes an already-parsed JSON payload. Used by the bundled-priors
    loader (``kenning._load_empirical_priors``) which reads the JSON
    from an ``importlib.resources`` Traversable and can't depend on a
    filesystem Path.
    """
    version = artifact.get("version", "unversioned")
    native: dict[tuple, dict[str, float]] = {}
    loan: dict[tuple, dict[str, float]] = {}

    for cell in artifact.get("native", []):
        key = (
            cell["culture"],
            cell["position"],
            cell["tag"],
            int(cell["era_midpoint"]),
        )
        native[key] = {ref: float(c) for ref, c in cell["lemmas"].items()}

    for cell in artifact.get("loan", []):
        key = (
            cell["donor"],
            cell["recipient"],
            cell["position"],
            cell["tag"],
            int(cell["era_midpoint"]),
        )
        loan[key] = {ref: float(c) for ref, c in cell["lemmas"].items()}

    return EmpiricalPriors(
        native=native,
        loan_relationship=loan,
        version=version,
    )


def collect_empirical_priors(
    db: LexiconDB,
    *,
    version: str = "unversioned",
) -> dict[str, Any] | None:
    """Project the L3 ``empirical_priors_native`` + ``empirical_priors_loan``
    tables into the same payload shape ``dump_empirical_priors_to_json``
    writes to disk — used by the L4 emit to fold the priors into the
    runtime DB rather than relying on a separate ``priors.json`` sidecar.

    Returns ``None`` when neither table has any rows (the L3 hasn't run
    ``mine-empirical-baselines`` yet) OR when the tables don't exist
    at all (pre-empirical-priors L3 schema — the alembic migration
    hasn't run yet). Callers treat ``None`` as "no priors; runtime
    falls back to empty ``EmpiricalPriors``"."""
    try:
        native_rows = list(
            db.conn.execute(
                "SELECT culture, position, tag, era_midpoint, lemma_ref, count "
                "FROM empirical_priors_native "
                "ORDER BY culture, position, tag, era_midpoint, lemma_ref"
            )
        )
        loan_rows = list(
            db.conn.execute(
                "SELECT donor, recipient, position, tag, era_midpoint, lemma_ref, count "
                "FROM empirical_priors_loan "
                "ORDER BY donor, recipient, position, tag, era_midpoint, lemma_ref"
            )
        )
    except sqlite3.OperationalError:
        # Pre-priors L3 (alembic migration hasn't run); treat as "no
        # priors data" rather than crashing the L4 emit. Loud-fail
        # here would block ``export-runtime-db`` against any L3
        # snapshotted before the priors tables existed.
        return None
    if not native_rows and not loan_rows:
        return None
    return {
        "version": version,
        "native": _cell_records(iter(native_rows), _NATIVE_CELL_FIELDS),
        "loan": _cell_records(iter(loan_rows), _LOAN_CELL_FIELDS),
    }


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
    # Stream to the file handle rather than materializing the full
    # JSON string in memory. At current corpus scale (~6k cells) the
    # difference is negligible, but as the priors artifact grows the
    # streaming form keeps peak memory bounded.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "native_cells": len(artifact["native"]),
        "loan_cells": len(artifact["loan"]),
    }


# Convenience for tests + external callers that want to confirm the
# era-midpoint convention is the one this module advertises.
def known_era_midpoints() -> dict[tuple[str, str], int]:  # noqa: V103 — public era-midpoint accessor pinned by tests
    """Return ``{(family, cell_label): midpoint_year}`` for every era
    cell defined in ``era.ERA_CELLS``. Useful for tests + tooling that
    needs to project rows into priors-space.
    """
    out: dict[tuple[str, str], int] = {}
    for family, cells in ERA_CELLS.items():
        for label, start, end in cells:
            out[(family, label)] = _era_midpoint(start, end)
    return out
