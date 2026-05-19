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
    # Build PlacesIdx → Phillimore-citation lookup so we can stamp the
    # right page reference on each attestation.
    phillimore_by_idx: dict[int, str | None] = {}
    n_places = len(next(iter(places.values()))) if places else 0
    for i in range(n_places):
        idx = places["PlacesIdx"][i]
        phil = places["Phillimore"][i]
        phillimore_by_idx[idx] = phil

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

    # Pre-populate the (modern_name, region) → toponym_id cache from
    # the existing toponym table in one query. Avoids ~20K per-row
    # SELECTs during the walk; also lets dry-run report accurate
    # toponym_inserted vs toponym_existing counts against the current
    # DB state (i.e. dry-run is a faithful preview of what apply
    # would do, not just what the parse would produce).
    toponym_id_cache: dict[tuple[str, str], int] = {}
    for r in conn.execute("SELECT modern_name, region, id FROM toponym"):
        if r[0] is not None and r[1] is not None:
            toponym_id_cache[(r[0], r[1])] = r[2]

    # Pre-populate the attestation existence index so dry-run can
    # also report accurate attestation_inserted vs _existing counts.
    # Keyed by (toponym_id, form, date_year, source_doc) — same shape
    # as the apply path's existence query, but cached once. NULL
    # source_doc values fall back to DOMESDAY_SOURCE_ID so the index
    # matches what the loop computes for citation-less rows
    # (otherwise NULL-source_doc DB rows would dedup-miss and produce
    # duplicate inserts).
    attestation_index: set[tuple[int, str, int, str]] = set()
    for r in conn.execute(
        "SELECT toponym_id, form, date_year, source_doc "
        "FROM toponym_attestation WHERE date_year = 1086"
    ):
        attestation_index.add((r[0], r[1], r[2] or 0, r[3] or DOMESDAY_SOURCE_ID))

    n_pf = len(next(iter(placeforms.values()))) if placeforms else 0
    progress_step = max(1, n_pf // 20)
    # Counter for dry-run would-be-insert IDs. Decrements per new
    # toponym so each gets a UNIQUE negative ID — keeps the
    # attestation_index dedupe semantic correct even when two
    # would-be-inserted toponyms share the same modern_name +
    # source_doc (rare but possible across counties).
    dry_run_id_counter = 0
    # Tracks (modern_name, region) keys we've already counted in this
    # run so the toponym_existing tally doesn't multi-count
    # PlaceFormSub > 1 alternates that map to the same pre-existing
    # toponym row.
    seen_this_run: set[tuple[str, str]] = set()

    for i in range(n_pf):
        counts["placeform_rows_walked"] += 1
        if progress_callback is not None and (i + 1) % progress_step == 0:
            progress_callback(i + 1, n_pf)

        idx = placeforms["PlacesIdx"][i]
        modern_name = placeforms["Vill"][i]
        county_code = placeforms["County"][i]
        os_ref = placeforms["OSref"][i]
        os_codes = placeforms["OScodes"][i]
        hundred = placeforms["Hundred"][i]

        # Filter uncertain identifications unless explicitly included.
        if not include_uncertain and os_codes in ("speculative", "unknown"):
            counts["skipped_uncertain"] += 1
            continue
        if not modern_name:
            counts["skipped_no_vill"] += 1
            continue
        if not county_code:
            counts["skipped_no_county"] += 1
            continue

        region = COUNTY_CODE_TO_NAME.get(county_code, county_code)
        country = COUNTY_CODE_TO_COUNTRY.get(county_code, "England")
        cache_key = (modern_name, region)

        # Compute the source_doc here so both apply and dry-run share
        # the attestation-dedup logic below.
        philli = phillimore_by_idx.get(idx) or ""
        source_doc_parts = [f"Phillimore {philli}"] if philli else []
        if hundred:
            source_doc_parts.append(f"Hundred={hundred}")
        if os_ref:
            source_doc_parts.append(f"OS={os_ref}")
        source_doc = "; ".join(source_doc_parts) or DOMESDAY_SOURCE_ID

        # Resolve toponym via cache. Three outcomes:
        #   * absent: truly new — insert (apply) / assign unique
        #     negative ID (dry-run)
        #   * present + real id: pre-existing in DB — count as existing
        #   * present + negative id: dry-run repeat of a key already
        #     counted; reuse the assigned negative ID
        toponym_id = toponym_id_cache.get(cache_key)
        if toponym_id is None:
            counts["toponym_inserted"] += 1
            if apply:
                cur = conn.execute(
                    "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
                    (modern_name, country, region),
                )
                toponym_id = cur.lastrowid
            else:
                dry_run_id_counter -= 1
                toponym_id = dry_run_id_counter
            toponym_id_cache[cache_key] = toponym_id
        elif cache_key not in seen_this_run:
            # First encounter in THIS run of a pre-existing
            # (modern_name, region). Count once — subsequent rows
            # for the same cache_key are silent in-run dedupes.
            counts["toponym_existing"] += 1
        seen_this_run.add(cache_key)

        # Attestation: dedupe against the pre-populated
        # attestation_index (covers DB-resident attestations and
        # within-this-run repeats) so both apply + dry-run report
        # accurate inserted vs existing counts.
        att_key = (toponym_id, modern_name, 1086, source_doc)
        if att_key in attestation_index:
            counts["attestation_existing"] += 1
        else:
            attestation_index.add(att_key)
            counts["attestation_inserted"] += 1
            if apply:
                conn.execute(
                    """
                    INSERT INTO toponym_attestation
                        (toponym_id, form, date_year, source_doc)
                    VALUES (?, ?, ?, ?)
                    """,
                    (toponym_id, modern_name, 1086, source_doc),
                )

    if apply:
        conn.commit()

    if progress_callback is not None:
        progress_callback(n_pf, n_pf)

    return counts
