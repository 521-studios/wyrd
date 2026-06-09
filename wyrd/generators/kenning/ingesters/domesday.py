"""Domesday Book (1086) ingester — wyrd-el93.

Pulls Hull University's open Domesday dataset (Williams-Erskine / J.J.N.
Palmer team, CC-BY-NC-SA) into the lexicon DB as toponym rows +
attestations dated 1086. Two operator wins:

1. **Toponym corpus expansion** — ~14,767 confirmed Domesday settlements
   with modern names, OS coords, and Phillimore-volume citations.
   Backfills the toponym table beyond the LLM-mined scholar gazetteers.
2. **rando-port validation signal** — when a modern Vill name decomposes
   into rando-port morphemes (-tun, -ham, -aecer, etc.), those morphemes
   are validated as historically used in 1086. Surfaces in the
   dashboard via the live-etymon set (matcher pass attributes morphemes
   per ticket wyrd-XXXX, follow-up).

**License**: Hull's data is CC-BY-NC-SA 4.0 (J.J.N. Palmer team).
Attribution is recorded in the ``source`` row's notes field.
Computational derivatives (toponym presence, etymon attribution
signals) flow into our DB + dashboard; raw Hull spellings are NOT
shipped in the runtime bundle (per wyrd-license-posture memory —
keeps the bundle commerce-friendly).

**Data shape** (Hull Places database — .mdb file via access_parser):
- ``Places`` table: 14,767 settlements, columns =
  ``PlacesIdx``, ``County``, ``Phillimore``, ``Hundred``, ``Vill``,
  ``Area``, ``XRefs``, ``OSrefs``, ``OScodes``.
- ``PlaceForm`` table: 20,458 form-variant rows. ``PlaceFormSub > 1``
  rows are modern-name alternates (e.g. Great/Little Abington split
  from a single Domesday vill). NOT historical Latin spellings.

The ingester writes one ``toponym`` row per distinct (Vill, county)
and one ``toponym_attestation`` row per Domesday entry (date_year=1086,
source_doc=Phillimore citation + Hundred + OS ref).

OScode-based filtering: rows tagged ``'speculative'`` / ``'unknown'``
are skipped by default (uncertain modern identifications, low value
for morpheme validation). Pass ``include_uncertain=True`` to include.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Standard Domesday 3-letter county codes → modern county / shire names.
# Sourced from Hull's General metadata + cross-referenced with
# Williams-Erskine. Used as the ``country='England'`` /
# ``region=<full county name>`` mapping at toponym-insert time so the
# DB carries human-readable values rather than abbreviation codes.
COUNTY_CODE_TO_NAME: dict[str, str] = {
    "BDF": "Bedfordshire",
    "BRK": "Berkshire",
    "BKM": "Buckinghamshire",
    "CAM": "Cambridgeshire",
    "CHS": "Cheshire",
    "CON": "Cornwall",
    "DBY": "Derbyshire",
    "DEV": "Devon",
    "DOR": "Dorset",
    "ESS": "Essex",
    "GLS": "Gloucestershire",
    "HAM": "Hampshire",
    "HEF": "Herefordshire",
    "HRT": "Hertfordshire",
    "HUN": "Huntingdonshire",
    "KEN": "Kent",
    "LAN": "Lancashire",
    "LEI": "Leicestershire",
    "LIN": "Lincolnshire",
    "MDX": "Middlesex",
    "NFK": "Norfolk",
    "NTH": "Northamptonshire",
    "NTT": "Nottinghamshire",
    "OXF": "Oxfordshire",
    "RUT": "Rutland",
    "SHR": "Shropshire",
    "SOM": "Somerset",
    "STS": "Staffordshire",
    "SFK": "Suffolk",
    "SRY": "Surrey",
    "SSX": "Sussex",
    "WAR": "Warwickshire",
    "WIL": "Wiltshire",
    "WOR": "Worcestershire",
    "YKS": "Yorkshire",
    # Cumberland / Westmorland: not formally surveyed in 1086 but a few
    # marginal entries; preserve code passthrough for any stray rows.
    "CUL": "Cumberland",
    "WES": "Westmorland",
    # Wales / Scotland-fringe codes used inconsistently by Hull team.
    "MON": "Monmouthshire",
    "FLN": "Flintshire",
}

# Country attribution per county code. Most Domesday counties are in
# England; the marginal Welsh entries (Monmouthshire was a debated
# border region for centuries — sometimes treated as Welsh, sometimes
# English) get country='Wales'. Hull's dataset doesn't formally label
# country, but Domesday-historians treat MON / FLN as the Welsh
# marches. Codes not in this set default to ``'England'``.
COUNTY_CODE_TO_COUNTRY: dict[str, str] = {
    "MON": "Wales",
    "FLN": "Wales",
}

DOMESDAY_SOURCE_ID = "open_domesday_hull"
DOMESDAY_SOURCE_TITLE = "Open Domesday (Hull dataset, J.J.N. Palmer team)"
DOMESDAY_SOURCE_NOTES = (
    "Domesday Book (1086) settlements + modern identifications. "
    "Data: J.J.N. Palmer + team, University of Hull (CC-BY-NC-SA 4.0). "
    "Presentation + folio images: Anna Powell-Smith, opendomesday.org "
    "(CC-BY-SA 3.0). Ingested per wyrd-el93. Raw Hull spellings are NOT "
    "redistributed in the runtime bundle to keep meanings.json "
    "commerce-friendly; only computational derivatives (toponym "
    "presence, etymon attribution) flow downstream."
)


def upsert_domesday_source(conn: sqlite3.Connection) -> None:
    """Ensure the ``open_domesday_hull`` source row exists. Idempotent."""
    conn.execute(
        """
        INSERT OR REPLACE INTO source
            (id, author, title, year, region, language_focus, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            DOMESDAY_SOURCE_ID,
            "J.J.N. Palmer and team",
            DOMESDAY_SOURCE_TITLE,
            1086,
            "England",
            # The Domesday text itself is Latin / OE / Norman-French, but
            # the ``form`` values we write are MODERN English names
            # (Hull's identification, not the Latin original). Tag the
            # source as modern-english so downstream phonology /
            # period-form consumers don't treat the forms as
            # historical Latin spellings.
            "modern-english (modern identifications of 1086 Latin entries)",
            DOMESDAY_SOURCE_NOTES,
        ),
    )


def ingest_domesday(
    conn: sqlite3.Connection,
    mdb_path: Path,
    *,
    apply: bool = False,
    include_uncertain: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Parse the Hull Places MDB and write toponym + attestation rows.

    Thin wrapper that parses the MDB via ``access_parser`` and routes
    the resulting table dicts into ``_ingest_from_tables``. Split so
    tests can construct fixture table dicts directly without needing a
    real MDB on disk.
    """
    # Deferred import — access_parser is an optional ingest-only
    # dependency (lives in [project.optional-dependencies].ingest,
    # not base deps; see pyproject.toml). Operators running this
    # command install via ``pip install -e '.[ingest]'``. The Lambda
    # bundle build skips it because its sdist-only / non-aarch64
    # wheels break the build pipeline.
    try:
        from access_parser import AccessParser
    except ImportError as exc:
        raise ImportError(
            "access_parser is required for ingest-domesday. Install with: "
            "pip install -e '.[ingest]'"
        ) from exc

    apdb = AccessParser(str(mdb_path))
    places = apdb.parse_table("Places")
    placeforms = apdb.parse_table("PlaceForm")
    return _ingest_from_tables(
        conn,
        places,
        placeforms,
        apply=apply,
        include_uncertain=include_uncertain,
        progress_callback=progress_callback,
    )


@dataclass
class _DomesdayIngestState:
    """Mutable walk state threaded through the per-placeform ingest helpers:
    the telemetry counts, the two dedup indexes (toponym (modern_name, region)
    → id; attestation (toponym_id, form, year, source_doc) presence set), the
    apply/dry-run + include-uncertain flags, the in-run seen-key set, and the
    dry-run negative-id counter (decremented per would-be-inserted toponym so
    each gets a unique id for the attestation dedup)."""

    counts: dict[str, int]
    toponym_id_cache: dict[tuple[str, str], int]
    attestation_index: set[tuple[int, str, int, str]]
    apply: bool
    include_uncertain: bool
    seen_this_run: set[tuple[str, str]] = field(default_factory=set)
    dry_run_id_counter: int = 0


def _build_phillimore_lookup(places: dict[str, list[Any]]) -> dict[int, str | None]:
    """PlacesIdx → Phillimore-citation lookup, so each attestation can be
    stamped with the right page reference."""
    lookup: dict[int, str | None] = {}
    n_places = len(next(iter(places.values()))) if places else 0
    for i in range(n_places):
        lookup[places["PlacesIdx"][i]] = places["Phillimore"][i]
    return lookup


def _build_dedup_indexes(
    conn: sqlite3.Connection,
) -> tuple[dict[tuple[str, str], int], set[tuple[int, str, int, str]]]:
    """Pre-populate the (modern_name, region) → toponym_id cache and the
    1086-attestation existence index from the current DB in two queries, so
    the walk avoids ~20K per-row SELECTs AND dry-run reports accurate
    inserted-vs-existing counts against real DB state (a faithful preview of
    what apply would do). NULL source_doc falls back to ``DOMESDAY_SOURCE_ID``
    so the index matches what the loop computes for citation-less rows."""
    toponym_id_cache: dict[tuple[str, str], int] = {}
    for r in conn.execute("SELECT modern_name, region, id FROM toponym"):
        if r[0] is not None and r[1] is not None:
            toponym_id_cache[(r[0], r[1])] = r[2]

    attestation_index: set[tuple[int, str, int, str]] = set()
    for r in conn.execute(
        "SELECT toponym_id, form, date_year, source_doc "
        "FROM toponym_attestation WHERE date_year = 1086"
    ):
        attestation_index.add((r[0], r[1], r[2] or 0, r[3] or DOMESDAY_SOURCE_ID))
    return toponym_id_cache, attestation_index


def _compute_source_doc(philli: str | None, hundred: Any, os_ref: Any) -> str:
    """Assemble the attestation ``source_doc`` from the Phillimore citation +
    Hundred + OS-ref parts, falling back to ``DOMESDAY_SOURCE_ID`` when none
    are present (so citation-less rows still dedup against the index)."""
    philli = philli or ""
    parts = [f"Phillimore {philli}"] if philli else []
    if hundred:
        parts.append(f"Hundred={hundred}")
    if os_ref:
        parts.append(f"OS={os_ref}")
    return "; ".join(parts) or DOMESDAY_SOURCE_ID


def _resolve_domesday_toponym(
    conn: sqlite3.Connection,
    state: _DomesdayIngestState,
    modern_name: str,
    country: str,
    region: str,
) -> int:
    """Resolve (modern_name, region) → toponym_id via the cache. Absent →
    insert (apply) or assign a unique negative id (dry-run), counting
    ``toponym_inserted``; present and first-seen this run → count
    ``toponym_existing`` (subsequent same-key rows are silent in-run
    dedupes)."""
    cache_key = (modern_name, region)
    toponym_id = state.toponym_id_cache.get(cache_key)
    if toponym_id is None:
        state.counts["toponym_inserted"] += 1
        if state.apply:
            cur = conn.execute(
                "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
                (modern_name, country, region),
            )
            toponym_id = cur.lastrowid
        else:
            state.dry_run_id_counter -= 1
            toponym_id = state.dry_run_id_counter
        state.toponym_id_cache[cache_key] = toponym_id
    elif cache_key not in state.seen_this_run:
        state.counts["toponym_existing"] += 1
    state.seen_this_run.add(cache_key)
    return toponym_id


def _record_domesday_attestation(
    conn: sqlite3.Connection,
    state: _DomesdayIngestState,
    toponym_id: int,
    modern_name: str,
    source_doc: str,
) -> None:
    """Record (or dedup-skip) the 1086 attestation for this toponym/form,
    keyed by (toponym_id, form, year, source_doc) against the pre-populated
    index so both apply + dry-run report accurate inserted-vs-existing."""
    att_key = (toponym_id, modern_name, 1086, source_doc)
    if att_key in state.attestation_index:
        state.counts["attestation_existing"] += 1
        return
    state.attestation_index.add(att_key)
    state.counts["attestation_inserted"] += 1
    if state.apply:
        conn.execute(
            """
            INSERT INTO toponym_attestation
                (toponym_id, form, date_year, source_doc)
            VALUES (?, ?, ?, ?)
            """,
            (toponym_id, modern_name, 1086, source_doc),
        )


def _ingest_one_placeform(
    conn: sqlite3.Connection,
    placeforms: dict[str, list[Any]],
    i: int,
    phillimore_by_idx: dict[int, str | None],
    state: _DomesdayIngestState,
) -> None:
    """Process one PlaceForm row: apply the uncertain / no-vill / no-county
    filters (counting each skip), resolve the toponym, and record its 1086
    attestation. All writes are gated on ``state.apply``."""
    idx = placeforms["PlacesIdx"][i]
    modern_name = placeforms["Vill"][i]
    county_code = placeforms["County"][i]
    os_ref = placeforms["OSref"][i]
    os_codes = placeforms["OScodes"][i]
    hundred = placeforms["Hundred"][i]

    if not state.include_uncertain and os_codes in ("speculative", "unknown"):
        state.counts["skipped_uncertain"] += 1
        return
    if not modern_name:
        state.counts["skipped_no_vill"] += 1
        return
    if not county_code:
        state.counts["skipped_no_county"] += 1
        return

    region = COUNTY_CODE_TO_NAME.get(county_code, county_code)
    country = COUNTY_CODE_TO_COUNTRY.get(county_code, "England")
    source_doc = _compute_source_doc(phillimore_by_idx.get(idx), hundred, os_ref)
    toponym_id = _resolve_domesday_toponym(conn, state, modern_name, country, region)
    _record_domesday_attestation(conn, state, toponym_id, modern_name, source_doc)


def _ingest_from_tables(
    conn: sqlite3.Connection,
    places: dict[str, list[Any]],
    placeforms: dict[str, list[Any]],
    *,
    apply: bool = False,
    include_uncertain: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Write toponym + attestation rows from already-parsed Hull tables.

    Without ``apply=True`` the call is a dry-count walk: tallies row
    counts but never writes. Idempotent on re-run (existing
    (modern_name, region) toponyms are reused; existing
    (toponym_id, form, date_year, source_doc) attestations are
    detected and skipped). Returns a dict of counts for caller logging.

    ``include_uncertain=False`` (default) skips PlaceForm rows tagged
    ``OScodes='speculative'`` or ``'unknown'`` (~600 rows, lost vills
    or low-confidence modern identifications). Pass True to include
    them in the ingest.

    ``progress_callback(done, total)`` is called every ~5% of the walk
    for CLAUDE.md-shape stderr output. Defaults to silent.
    """
    phillimore_by_idx = _build_phillimore_lookup(places)

    counts: dict[str, int] = {
        "placeform_rows_walked": 0,
        "skipped_uncertain": 0,
        "skipped_no_county": 0,
        "skipped_no_vill": 0,
        "toponym_inserted": 0,
        "toponym_existing": 0,
        "attestation_inserted": 0,
        "attestation_existing": 0,
    }

    if apply:
        upsert_domesday_source(conn)

    toponym_id_cache, attestation_index = _build_dedup_indexes(conn)
    state = _DomesdayIngestState(
        counts=counts,
        toponym_id_cache=toponym_id_cache,
        attestation_index=attestation_index,
        apply=apply,
        include_uncertain=include_uncertain,
    )

    n_pf = len(next(iter(placeforms.values()))) if placeforms else 0
    progress_step = max(1, n_pf // 20)
    for i in range(n_pf):
        counts["placeform_rows_walked"] += 1
        if progress_callback is not None and (i + 1) % progress_step == 0:
            progress_callback(i + 1, n_pf)
        _ingest_one_placeform(conn, placeforms, i, phillimore_by_idx, state)

    if apply:
        conn.commit()

    if progress_callback is not None:
        progress_callback(n_pf, n_pf)

    return counts
