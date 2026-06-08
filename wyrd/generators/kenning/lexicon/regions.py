"""Region → country derivation.

Single source of truth for "given a region string the ingest pipeline
sees (Cambridgeshire, Scottish Highlands and Islands, Loire-Inférieure
(Brittany), England, ...), what country does it belong to?"

Used at three layers:

1. Mining-side JSONL backfill — when raw mining output (one line per
   place-name mention) carries the source's region but not country, the
   country is derived here so the L2 file is self-contained.
2. Ingest-side — ``_upsert_toponym`` in ``ingest.py`` derives country
   from region when the caller doesn't supply it explicitly, so any
   in-flight ingest path picks up country without per-CLI plumbing.
3. Backfill migration — ``backfill_toponym_country`` updates existing
   ``toponym`` rows whose ``country`` column is NULL but whose ``region``
   maps to a known country.

The map is conservative: regions that span multiple countries
(``British Isles``, ``Europe``, ``England and Wales``) resolve to
``None`` rather than guessing, so the operator can review + fix by hand.
"""

from __future__ import annotations

# region → country. Exhaustive over the regions present in
# ``_KNOWN_SKEAT_BOOKS`` as of 2026-05-25, plus the gazetteer-side
# region values populated by the ``<culture>_place_names.json`` ingest.
_REGION_TO_COUNTRY: dict[str, str] = {
    # England — counties, historical divisions, and catch-alls.
    "England": "England",
    "England (Danelaw)": "England",
    "Bedfordshire": "England",
    "Berkshire": "England",
    "Buckinghamshire": "England",
    "Cambridgeshire": "England",
    "Cumberland and Westmorland": "England",
    "Cumberland": "England",
    "Devon": "England",
    "Dorset": "England",
    "East Riding of Yorkshire": "England",
    "Essex": "England",
    "Gloucestershire": "England",
    "Hampshire": "England",
    "Herefordshire": "England",
    "Hertfordshire": "England",
    "Huntingdonshire": "England",
    "Kent": "England",
    "Lancashire": "England",
    "Leicestershire": "England",
    "Lincolnshire": "England",
    "Middlesex": "England",
    "Norfolk": "England",
    "Northamptonshire": "England",
    "Northumberland and Durham": "England",
    "Northumberland": "England",
    "Nottinghamshire": "England",
    "Oxfordshire": "England",
    "Rutland": "England",
    "Shropshire": "England",
    "Somerset": "England",
    "South-West Yorkshire": "England",
    "Staffordshire": "England",
    "Suffolk": "England",
    "Surrey": "England",
    "Sussex": "England",
    "Warwickshire": "England",
    "West Riding of Yorkshire": "England",
    "Westmorland": "England",
    "Wiltshire": "England",
    "Worcestershire": "England",
    "Yorkshire": "England",
    # Scotland — country itself + counties / regions covered by the
    # bundled Scottish scholarship.
    "Scotland": "Scotland",
    "Argyll": "Scotland",
    "Ross and Cromarty": "Scotland",
    "Scottish Highlands and Islands": "Scotland",
    "Stirlingshire": "Scotland",
    # Wales.
    "Wales": "Wales",
    "Wales and Monmouthshire": "Wales",
    # Ireland.
    "Ireland": "Ireland",
    # Isle of Man — distinct from the constituent countries but covered
    # by the bundled gazetteer.
    "Isle of Man": "Isle of Man",
    # France — covers Brittany (Loire-Inférieure / Loire-Atlantique;
    # Quilgars 1906 covers Breton place-names in this French
    # département) plus general-France sources (Longnon, Arbois,
    # Joret's Normandy patois).
    "France": "France",
    "Loire-Inférieure (Brittany)": "France",
    "Brittany": "France",
    # Deliberately omitted (return None — multi-country / unclear):
    #   "British Isles"
    #   "Europe"
    #   "England and Wales"
}


def country_for_region(region: str | None) -> str | None:
    """Return the country string for ``region`` (or ``None`` when the
    region is unknown, missing, or spans multiple countries — the
    caller leaves ``toponym.country`` NULL in that case)."""
    if region is None:
        return None
    return _REGION_TO_COUNTRY.get(region)
