"""Per-toponym era timeline + aggregate era-coverage metrics — wyrd-ub76.

Phase 4a of the wyrd-eni4 epic. The Phase-3 ingesters (Domesday →
Hundred Rolls → Speed → Hearth Tax → OS Open Names) deposit
attestations across the full Anglo-Saxon → present-day arc. This
module surfaces that timeline:

- :func:`fetch_era_timeline` returns one toponym's attestations
  grouped by era.
- :func:`fetch_era_coverage` returns aggregate metrics: how many
  toponyms have attestations in N eras, distribution per era, etc.

The era buckets are coarse on purpose. The interesting trend isn't
"1086 vs 1279" — it's "do we have ANY ME-era form for this place?"
That's the gap a Phase-3 ingester closes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from wyrd.generators.kenning.browse import parse_toponym_ref

ERA_ANGLO_SAXON = "Anglo-Saxon"
ERA_MIDDLE_ENGLISH = "Middle English"
ERA_EARLY_MODERN = "Early Modern"
ERA_MODERN = "Modern"
ERA_UNDATED = "Undated"

# Ordered for display; UNDATED last because it has no place in the
# chronological story.
ERA_ORDER: tuple[str, ...] = (
    ERA_ANGLO_SAXON,
    ERA_MIDDLE_ENGLISH,
    ERA_EARLY_MODERN,
    ERA_MODERN,
    ERA_UNDATED,
)


def bucket_year(year: int | None) -> str:
    """Map a year to an era bucket. ``None`` → ``UNDATED``.

    The 1100 / 1500 / 1800 boundaries are the standard scholarly cuts
    (Norman Conquest aftermath / Tudor / Industrial era) and match
    what the Phase-3 ingesters target. Edge dates round up: 1100 is
    Middle English; 1500 is Early Modern; 1800 is Modern.
    """
    if year is None:
        return ERA_UNDATED
    if year < 1100:
        return ERA_ANGLO_SAXON
    if year < 1500:
        return ERA_MIDDLE_ENGLISH
    if year < 1800:
        return ERA_EARLY_MODERN
    return ERA_MODERN


@dataclass(frozen=True)
class EraAttestation:
    """One historical form attested at a specific date, with its
    source provenance."""

    form: str
    date_year: int | None
    source_doc: str | None


@dataclass(frozen=True)
class EraTimeline:
    """All attestations for one toponym, bucketed by era."""

    toponym_id: int
    modern_name: str
    region: str | None
    country: str | None
    attestations_by_era: dict[str, list[EraAttestation]]

    @property
    def era_count(self) -> int:
        """Number of distinct eras with at least one attestation.
        UNDATED doesn't count — coverage means *chronological* spread."""
        return sum(
            1 for era, items in self.attestations_by_era.items() if era != ERA_UNDATED and items
        )

    @property
    def total_attestations(self) -> int:
        return sum(len(v) for v in self.attestations_by_era.values())


def _resolve_toponym_id(
    conn: sqlite3.Connection, modern_name: str, region: str | None
) -> int | None:
    """Look up a toponym row id by (modern_name, region). Region
    sentinel semantics match :func:`parse_toponym_ref`:

    - ``region is None`` — match any region; if multiple, pick the
      first ASCII-sorted region (deterministic fallback for ambiguous
      names; operator can disambiguate via ``@<region>``).
    - ``region == ""`` — match only the row with null region.
    - ``region == "<name>"`` — exact match.

    Returns ``None`` when no row matches.
    """
    if region is None:
        row = conn.execute(
            "SELECT id FROM toponym WHERE modern_name=? ORDER BY region IS NULL, region LIMIT 1",
            (modern_name,),
        ).fetchone()
    elif region == "":
        row = conn.execute(
            "SELECT id FROM toponym WHERE modern_name=? AND region IS NULL LIMIT 1",
            (modern_name,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM toponym WHERE modern_name=? AND region=? LIMIT 1",
            (modern_name, region),
        ).fetchone()
    return row["id"] if row else None


def fetch_era_timeline(conn: sqlite3.Connection, toponym_ref: str) -> EraTimeline | None:
    """Resolve a toponym ref (``modern_name@region``), pull all its
    attestations, bucket by era. Returns ``None`` when the toponym
    doesn't exist."""
    modern_name, region = parse_toponym_ref(toponym_ref)
    toponym_id = _resolve_toponym_id(conn, modern_name, region)
    if toponym_id is None:
        return None

    head = conn.execute(
        "SELECT modern_name, region, country FROM toponym WHERE id=?", (toponym_id,)
    ).fetchone()

    by_era: dict[str, list[EraAttestation]] = {era: [] for era in ERA_ORDER}
    for r in conn.execute(
        """SELECT form, date_year, source_doc
             FROM toponym_attestation
            WHERE toponym_id=?
            ORDER BY COALESCE(date_year, 99999), id""",
        (toponym_id,),
    ):
        bucket = bucket_year(r["date_year"])
        by_era[bucket].append(
            EraAttestation(
                form=r["form"],
                date_year=r["date_year"],
                source_doc=r["source_doc"],
            )
        )

    return EraTimeline(
        toponym_id=toponym_id,
        modern_name=head["modern_name"],
        region=head["region"],
        country=head["country"],
        attestations_by_era=by_era,
    )


@dataclass(frozen=True)
class EraCoverageReport:
    """Aggregate over the entire toponym corpus."""

    total_toponyms: int
    toponyms_with_attestations: int
    per_era_counts: dict[str, int]
    """Count of toponyms with ≥1 attestation in each era."""
    era_count_distribution: dict[int, int]
    """Histogram: how many toponyms have exactly N eras of coverage.
    Key 0 = no dated attestations; 4 = all four chronological eras."""

    @property
    def coverage_3plus_pct(self) -> float:
        """Percent of all toponyms with ≥3 eras of coverage."""
        if self.total_toponyms == 0:
            return 0.0
        n = sum(c for eras, c in self.era_count_distribution.items() if eras >= 3)
        return 100.0 * n / self.total_toponyms


def fetch_era_coverage(conn: sqlite3.Connection) -> EraCoverageReport:
    """Aggregate era-coverage metrics across the whole toponym table.

    Streams ``toponym_attestation`` rows in toponym_id order and folds
    per-toponym era sets in Python. One SQL pass; no GROUP_CONCAT
    string-parsing (which would be fragile for date_year values that
    aren't simple integers).

    Suitable for the ~tens-of-thousands-scale toponym corpus the
    Phase-3 ingesters produce.
    """
    total = conn.execute("SELECT COUNT(*) AS n FROM toponym").fetchone()["n"]

    per_era: dict[str, int] = dict.fromkeys(ERA_ORDER, 0)
    era_count_dist: dict[int, int] = {}
    toponyms_with_attestations = 0

    current_id: int | None = None
    current_eras: set[str] = set()

    def _flush() -> None:
        nonlocal toponyms_with_attestations
        if current_id is None:
            return
        toponyms_with_attestations += 1
        dated_eras = {e for e in current_eras if e != ERA_UNDATED}
        era_count_dist[len(dated_eras)] = era_count_dist.get(len(dated_eras), 0) + 1
        for e in current_eras:
            per_era[e] += 1

    for r in conn.execute(
        """SELECT toponym_id, date_year
             FROM toponym_attestation
            ORDER BY toponym_id"""
    ):
        if r["toponym_id"] != current_id:
            _flush()
            current_id = r["toponym_id"]
            current_eras = set()
        current_eras.add(bucket_year(r["date_year"]))
    _flush()

    # Toponyms with zero attestations are 0-era too — fold them in.
    zero_era_toponyms = total - toponyms_with_attestations
    if zero_era_toponyms:
        era_count_dist[0] = era_count_dist.get(0, 0) + zero_era_toponyms

    return EraCoverageReport(
        total_toponyms=total,
        toponyms_with_attestations=toponyms_with_attestations,
        per_era_counts=per_era,
        era_count_distribution=era_count_dist,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_era_timeline(timeline: EraTimeline) -> str:
    """Markdown rendering of one toponym's era timeline."""
    head_bits = [timeline.modern_name]
    if timeline.region:
        head_bits.append(f"@{timeline.region}")
    if timeline.country:
        head_bits.append(f"({timeline.country})")
    lines: list[str] = [f"# {' '.join(head_bits)}", ""]
    lines.append(
        f"Eras with coverage: {timeline.era_count}/4 "
        f"(total attestations: {timeline.total_attestations})"
    )
    lines.append("")

    any_emitted = False
    for era in ERA_ORDER:
        items = timeline.attestations_by_era[era]
        if not items:
            continue
        any_emitted = True
        lines.append(f"## {era} ({len(items)})")
        for a in items:
            year = a.date_year if a.date_year is not None else "—"
            doc = f" — {a.source_doc}" if a.source_doc else ""
            lines.append(f"- {year}: `{a.form}`{doc}")
        lines.append("")

    if not any_emitted:
        lines.append("_no attestations on this toponym_")

    return "\n".join(lines).rstrip() + "\n"


def format_era_coverage(report: EraCoverageReport) -> str:
    """Markdown rendering of the corpus-wide coverage report."""
    lines: list[str] = ["# Cross-era coverage report", ""]
    lines.append(f"- Total toponyms: {report.total_toponyms:,}")
    lines.append(f"- Toponyms with ≥1 attestation: {report.toponyms_with_attestations:,}")
    lines.append(f"- Toponyms with ≥3 dated eras: {report.coverage_3plus_pct:.1f}%")
    lines.append("")
    lines.append("## Per-era coverage")
    lines.append("")
    lines.append("| Era | Toponyms covered |")
    lines.append("| --- | ---: |")
    for era in ERA_ORDER:
        lines.append(f"| {era} | {report.per_era_counts[era]:,} |")
    lines.append("")
    lines.append("## Dated-era-count distribution")
    lines.append("")
    lines.append("| # eras dated | toponyms |")
    lines.append("| ---: | ---: |")
    for n in sorted(report.era_count_distribution):
        lines.append(f"| {n} | {report.era_count_distribution[n]:,} |")
    return "\n".join(lines) + "\n"


__all__: Sequence[str] = (
    "ERA_ANGLO_SAXON",
    "ERA_EARLY_MODERN",
    "ERA_MIDDLE_ENGLISH",
    "ERA_MODERN",
    "ERA_ORDER",
    "ERA_UNDATED",
    "EraAttestation",
    "EraCoverageReport",
    "EraTimeline",
    "bucket_year",
    "fetch_era_coverage",
    "fetch_era_timeline",
    "format_era_coverage",
    "format_era_timeline",
)
