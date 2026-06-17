"""One-time L2 vocabulary normalization (wyrd-3q6m.4).

Rewrites the ``region`` and ``country`` fields of the committed ``data/mining``
JSONL to canonical form, idempotently and in place (safe because L2 is
git-tracked — the original coding is recoverable from history; region/country are
provenance metadata, not D21 evidence).

Three transforms, applied to canonical-state rows:

1. **Region folds** — ``canonicalize_region(region, country)`` per the Class-A
   model (SUF→Suffolk, East/West Sussex→Sussex, casing, …). Non-England and
   unfoldable values (zones/quarantine) are left unchanged and counted in the
   report (deferred — wyrd-3q6m.4 leave-and-report decision).
2. **Yorkshire→Riding recode** — bare ``Yorkshire`` rows whose place name maps
   to a single Riding in ``yorkshire.csv``'s ``ReferenceTitle`` (Smith's
   per-Riding EPNS volumes) become ``North/East/West Riding``. Names with no /
   multiple Riding citations stay ``Yorkshire`` (a valid county node).
3. **Country normalization** — ``Republic of Ireland``→``Ireland``,
   ``The Isle of Man``→``Isle of Man``, ``Brittany``→``France``; a null country
   is derived from the (canonicalized) region via ``country_for_region`` where
   possible.

A region change also rewrites the ``<name>@<region>`` ``ref`` on toponym rows and
the matching ``toponym_ref`` on etymology rows, so cross-refs stay intact.

Idempotent: canonical→canonical is identity, so a second run is a no-op.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon.region_model import (
    RegionValidationError,
    canonicalize_region,
)
from wyrd.generators.kenning.lexicon.regions import country_for_region

# Country-field canonicalization (wyrd-3q6m.4 decisions).
COUNTRY_ALIASES = {
    "Republic of Ireland": "Ireland",
    "The Isle of Man": "Isle of Man",
    "Brittany": "France",
}

_RIDING_RE = re.compile(r"\b(North|East|West) Riding of Yorkshire\b")


@dataclass
class NormReport:
    """Counts + deferred values for an observability line (D24)."""

    toponyms_seen: int = 0
    region_folded: int = 0
    yorkshire_recoded: int = 0
    # bare "Yorkshire" rows left at the county node (no per-place Riding
    # provenance — e.g. Domesday/gazetteer sources, not kepn). Valid, just coarser.
    yorkshire_left_county: int = 0
    country_aliased: int = 0
    country_derived: int = 0
    refs_rewritten: int = 0
    # region value -> count, for England-scoped values left unchanged (zones /
    # quarantine / unknown) and the bare-Yorkshire rows with no resolvable Riding.
    unfoldable: dict[str, int] = field(default_factory=dict)


def build_riding_map(yorkshire_csv: Path) -> dict[str, str]:
    """``PlaceName`` → ``"<North|East|West> Riding"`` parsed from the kepn
    ``yorkshire.csv`` ``ReferenceTitle`` column (Smith's per-Riding volumes).

    A place citing zero or more-than-one distinct Riding is omitted, so it stays
    at the ``Yorkshire`` county node rather than being assigned a guessed Riding.
    """
    out: dict[str, str] = {}
    with open(yorkshire_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ridings = {m.group(1) for m in _RIDING_RE.finditer(row.get("ReferenceTitle") or "")}
            if len(ridings) == 1:
                out[row["PlaceName"]] = f"{next(iter(ridings))} Riding"
    return out


def _split_ref(ref: str) -> tuple[str, str]:
    """Split ``"<name>@<region>"`` into (name, region) on the last ``@``."""
    name, _, region = ref.rpartition("@")
    return name, region


def _new_region(
    name: str,
    region: str | None,
    country: str | None,
    riding_map: dict[str, str],
    report: NormReport,
) -> str | None:
    """Canonical region for (name, region, country). Yorkshire→Riding via the
    provenance map; otherwise the Class-A canonicalizer. Unfoldable England
    values are returned unchanged and counted."""
    if region == "Yorkshire" and name in riding_map:
        return riding_map[name]
    try:
        return canonicalize_region(region, country=country)
    except RegionValidationError:
        report.unfoldable[region] = report.unfoldable.get(region or "", 0) + 1
        return region


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    riding_map: dict[str, str] | None = None,
    report: NormReport,
) -> list[dict[str, Any]]:
    """Return canonical rows with region/country/refs normalized. Pure: callers
    pass the full canonical-state row set for a file (or the whole corpus).

    Two passes: (1) rewrite toponym region/country/ref and record the
    ``(name, old_region) → new_region`` remap; (2) rewrite etymology
    ``toponym_ref`` from that remap so cross-refs follow the rename.
    """
    riding_map = riding_map or {}
    remap: dict[tuple[str, str | None], str | None] = {}
    for row in rows:
        # canonical toponym rows carry modern_name; skip event rows (op:remove)
        # and any non-toponym record.
        if row.get("_type") == "toponym" and "modern_name" in row:
            _normalize_toponym(row, riding_map, remap, report)
    for row in rows:
        _rewrite_ref(row, remap, report)
    return rows


def _normalize_toponym(
    row: dict[str, Any],
    riding_map: dict[str, str],
    remap: dict[tuple[str, str | None], str | None],
    report: NormReport,
) -> None:
    """Rewrite one canonical toponym row's region/country/ref + record the remap."""
    report.toponyms_seen += 1
    name = row["modern_name"]
    old_region = row.get("region")
    new_region = _new_region(name, old_region, row.get("country"), riding_map, report)
    remap[(name, old_region)] = new_region
    if new_region != old_region:
        if old_region == "Yorkshire":
            report.yorkshire_recoded += 1
        else:
            report.region_folded += 1
        row["region"] = new_region
        row["ref"] = f"{name}@{new_region if new_region is not None else '-'}"
        report.refs_rewritten += 1
    elif old_region == "Yorkshire":
        report.yorkshire_left_county += 1
    # country: alias-fold, then derive from the (now-canonical) region if null.
    country = row.get("country")
    if country in COUNTRY_ALIASES:
        row["country"] = COUNTRY_ALIASES[country]
        report.country_aliased += 1
    elif country is None:
        derived = country_for_region(new_region)
        if derived is not None:
            row["country"] = derived
            report.country_derived += 1


def _rewrite_ref(
    row: dict[str, Any],
    remap: dict[tuple[str, str | None], str | None],
    report: NormReport,
) -> None:
    """Follow a region rename on any toponym cross-ref: etymology rows
    (``toponym_ref``) and toponym event rows like op:remove (``ref`` with no
    ``modern_name``). Non-toponym refs (etymon ``lang:form``, source ``kepn``)
    don't parse to a remapped key, so they pass through."""
    if "toponym_ref" in row:
        field_name = "toponym_ref"
    elif row.get("_type") == "toponym" and "modern_name" not in row and "ref" in row:
        field_name = "ref"
    else:
        return
    ref = row[field_name]
    name, region = _split_ref(ref)
    key = (name, region if region != "-" else None)
    if key not in remap:
        return
    new_region = remap[key]
    new_ref = f"{name}@{new_region if new_region is not None else '-'}"
    if new_ref != ref:
        row[field_name] = new_ref
        report.refs_rewritten += 1
