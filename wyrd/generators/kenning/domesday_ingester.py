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
            "old-english / latin / norman-french (Domesday Latin text)",
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
    # Deferred import — access_parser is a heavy dep used only here.
    from access_parser import AccessParser

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

    # In-process memo: (modern_name, region) → toponym_id.
    # Without it each Domesday vill would re-issue a SELECT against
    # toponym at insertion time on every row.
    toponym_id_cache: dict[tuple[str, str], int] = {}

    n_pf = len(next(iter(placeforms.values()))) if placeforms else 0
    progress_step = max(1, n_pf // 20)

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
        cache_key = (modern_name, region)

        if not apply:
            # Dry-count path: tally would-be-inserts using the same
            # dedupe key the apply path uses so duplicate
            # (modern_name, region) PlaceForm rows (PlaceFormSub > 1
            # alternates) don't over-count. Otherwise dry-run reports
            # a higher toponym count than the apply path would
            # actually produce.
            if cache_key not in toponym_id_cache:
                counts["toponym_inserted"] += 1
                toponym_id_cache[cache_key] = -1  # dry-run sentinel
            counts["attestation_inserted"] += 1
            continue

        toponym_id = toponym_id_cache.get(cache_key)
        if toponym_id is None:
            existing = conn.execute(
                "SELECT id FROM toponym WHERE modern_name = ? AND region = ?",
                (modern_name, region),
            ).fetchone()
            if existing is not None:
                toponym_id = existing[0]
                counts["toponym_existing"] += 1
            else:
                cur = conn.execute(
                    "INSERT INTO toponym (modern_name, country, region) VALUES (?, 'England', ?)",
                    (modern_name, region),
                )
                toponym_id = cur.lastrowid
                counts["toponym_inserted"] += 1
            toponym_id_cache[cache_key] = toponym_id

        # Attestation: the modern Vill name carries the form (Hull's
        # data doesn't include the Latin original spelling — that lives
        # in Phillimore-cited paper volumes the team isn't authorized
        # to redistribute). The ``form`` field holds the modern name;
        # ``source_doc`` carries the Phillimore citation + Hundred + OS
        # ref so downstream readers can trace back to the manor entry.
        philli = phillimore_by_idx.get(idx) or ""
        source_doc_parts = [f"Phillimore {philli}"] if philli else []
        if hundred:
            source_doc_parts.append(f"Hundred={hundred}")
        if os_ref:
            source_doc_parts.append(f"OS={os_ref}")
        source_doc = "; ".join(source_doc_parts) or DOMESDAY_SOURCE_ID

        existing_atst = conn.execute(
            """
            SELECT id FROM toponym_attestation
             WHERE toponym_id = ? AND form = ? AND date_year = ?
               AND source_doc = ?
            """,
            (toponym_id, modern_name, 1086, source_doc),
        ).fetchone()
        if existing_atst is not None:
            counts["attestation_existing"] += 1
        else:
            conn.execute(
                """
                INSERT INTO toponym_attestation
                    (toponym_id, form, date_year, source_doc)
                VALUES (?, ?, ?, ?)
                """,
                (toponym_id, modern_name, 1086, source_doc),
            )
            counts["attestation_inserted"] += 1

    if apply:
        conn.commit()

    if progress_callback is not None:
        progress_callback(n_pf, n_pf)

    return counts
