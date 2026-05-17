"""Wikipedia place-name backfill report (wyrd-4453).

The ``wyrd/generators/kenning/data/*_place_names.json`` files are
Wikipedia-sourced lists that originally seeded the toponym table. The
epic's terminal goal is to retire those files once scholar-source
corpus mining covers enough of the same entries — at which point the
Wikipedia provenance can be dropped without losing data.

This module reports the retirement-readiness gap per region:

* ``total``       — Wikipedia entries the file lists for the region.
* ``in_db``       — entries that resolve to a ``toponym`` row by exact
  ``modern_name`` match.
* ``attested``    — entries whose ``toponym`` row has ≥1
  ``toponym_attestation`` row (scholar attestation is the retirement
  criterion; an in-DB row alone isn't enough — it might trace back to
  the Wikipedia seed itself).
* ``gap``         — ``total - attested``; how many entries still need
  scholar mining before Wikipedia can be retired safely.

KNOWN LIMITATION (documented intentionally): the name lookup is
case-sensitive exact match against ``toponym.modern_name``. Operators
running the report against the live DB will see false-negatives where
the Wikipedia name and the DB form differ in punctuation, diacritics,
or disambiguation suffixes ("Newcastle" vs "Newcastle upon Tyne"). A
fuzzier matcher is a follow-up — start with the strict numbers so the
retirement gate is conservative.

Read-only: this module NEVER writes to the DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from wyrd.generators.kenning.lexicon import LexiconDB

# The 5 Wikipedia-sourced files seeded by the original toponym ingest.
# Order is the printed-report order. The CLI's ``--language`` flag
# matches against the leading culture token (e.g. "english" →
# "english_place_names.json").
WIKIPEDIA_PLACE_NAME_FILES: tuple[str, ...] = (
    "english_place_names.json",
    "scottish_place_names.json",
    "welsh_place_names.json",
    "irish_place_names.json",
    "breton_place_names.json",
)


@dataclass(frozen=True)
class CountyReport:
    """One subregion's backfill numbers (e.g. "Bedfordshire" inside
    England). The atomic unit of the report — every higher-level
    aggregate sums these.
    """

    name: str
    total: int
    in_db: int
    attested: int

    @property
    def in_db_no_attestation(self) -> int:
        """Entries in the DB but without any scholar attestation.

        Surfaced separately because they're the rows that contributed
        to the DB BUT can't be retired yet — they're still
        Wikipedia-only provenance until a scholar source confirms
        them.
        """
        return self.in_db - self.attested

    @property
    def gap(self) -> int:
        """Wikipedia entries still needing scholar attestation.

        Includes both "not in DB" and "in DB but no attestation" —
        both block the Wikipedia file from being retired.
        """
        return self.total - self.attested

    @property
    def pct_attested(self) -> float:
        return self.attested / self.total if self.total else 0.0


@dataclass(frozen=True)
class RegionReport:
    """One country/region inside a file (e.g. "England", "The Isle of
    Man") — aggregates its counties."""

    name: str
    counties: list[CountyReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(c.total for c in self.counties)

    @property
    def in_db(self) -> int:
        return sum(c.in_db for c in self.counties)

    @property
    def attested(self) -> int:
        return sum(c.attested for c in self.counties)

    @property
    def in_db_no_attestation(self) -> int:
        return self.in_db - self.attested

    @property
    def gap(self) -> int:
        return self.total - self.attested

    @property
    def pct_attested(self) -> float:
        return self.attested / self.total if self.total else 0.0


@dataclass(frozen=True)
class FileReport:
    """One Wikipedia *_place_names.json file — aggregates its regions."""

    filename: str
    regions: list[RegionReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(r.total for r in self.regions)

    @property
    def in_db(self) -> int:
        return sum(r.in_db for r in self.regions)

    @property
    def attested(self) -> int:
        return sum(r.attested for r in self.regions)

    @property
    def in_db_no_attestation(self) -> int:
        return self.in_db - self.attested

    @property
    def gap(self) -> int:
        return self.total - self.attested

    @property
    def pct_attested(self) -> float:
        return self.attested / self.total if self.total else 0.0


def _load_place_names(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load and validate one ``*_place_names.json`` file.

    Expected shape::

        {"<country>": {"<subregion>": ["name1", "name2", ...], ...}, ...}

    Returns the parsed structure verbatim. Raises ``ValueError`` on
    shape mismatch — we want to fail loud on a file that drifted from
    the documented shape so the operator notices, not silently report
    zero.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be dict, got {type(raw).__name__}")
    for country, subregions in raw.items():
        if not isinstance(subregions, dict):
            raise ValueError(
                f"{path}: country {country!r} must map to dict, got {type(subregions).__name__}"
            )
        for subregion, names in subregions.items():
            if not isinstance(names, list):
                raise ValueError(
                    f"{path}: {country!r}.{subregion!r} must map to list, "
                    f"got {type(names).__name__}"
                )
    return raw


def _build_toponym_lookups(db: LexiconDB) -> tuple[set[str], set[str]]:
    """Return ``(in_db_names, attested_names)`` from one DB read.

    Both are sets of ``modern_name`` strings. ``in_db_names`` is every
    name that has a ``toponym`` row; ``attested_names`` is the subset
    that ALSO has ≥1 ``toponym_attestation`` row.

    Built as sets to avoid an N-way query loop — the largest input file
    (English) has ~20k entries and the DB has ~10k toponyms; loading
    both into memory is cheap (single-digit MB) and turns each lookup
    into a hash membership check.

    Same ``modern_name`` may appear in multiple toponym rows (different
    country/region disambiguators); the set collapses them — once the
    name is in the DB it's in the lookup, regardless of how many
    rows share it.
    """
    in_db_names: set[str] = set()
    attested_names: set[str] = set()

    for row in db.conn.execute("SELECT modern_name FROM toponym"):
        in_db_names.add(row["modern_name"])

    # LEFT-side filter: only consider toponyms that exist (toponym_id
    # FK guarantees this, but the query plan stays clearer with the
    # explicit join). Same modern_name → attested as long as ANY of
    # its toponym rows has ≥1 attestation.
    for row in db.conn.execute(
        "SELECT DISTINCT t.modern_name "
        "FROM toponym t "
        "JOIN toponym_attestation a ON a.toponym_id = t.id"
    ):
        attested_names.add(row["modern_name"])

    return in_db_names, attested_names


def _compute_county_report(
    name: str,
    place_names: list[str],
    in_db_names: set[str],
    attested_names: set[str],
) -> CountyReport:
    """Tally one county's Wikipedia entries against the DB lookups.

    Names are counted PER-OCCURRENCE — if the same modern_name appears
    twice in the same county list (Wikipedia happily duplicates), it
    counts twice. Each occurrence is a Wikipedia entry, even though
    the underlying toponym table collapses them. See test
    ``test_duplicate_name_counts_each_occurrence``.
    """
    total = len(place_names)
    in_db = sum(1 for n in place_names if n in in_db_names)
    attested = sum(1 for n in place_names if n in attested_names)
    return CountyReport(name=name, total=total, in_db=in_db, attested=attested)


def _filename_for_language(language: str) -> str:
    """Map a CLI ``--language NAME`` token to its place-names filename.

    Case-insensitive; ``"english"`` → ``"english_place_names.json"``.
    Mainly defensive — operators may type "English" or "ENGLISH"
    without thinking, and a silent no-match would be confusing.
    """
    return f"{language.lower()}_place_names.json"


def compute_backfill_report(
    db: LexiconDB,
    data_dir: Path,
    languages: tuple[str, ...] | None = None,
) -> list[FileReport]:
    """Build the per-file backfill report.

    Args:
        db: read-only ``LexiconDB`` handle.
        data_dir: directory containing the ``*_place_names.json`` files.
        languages: optional filter — tuple of culture tokens (e.g.
          ``("english", "scottish")``). When None or empty, all 5
          files are reported. Missing files in the filter are silently
          skipped (e.g. ``--language bogus`` produces an empty report
          rather than crashing — the CLI surfaces the absence).

    Returns: list of ``FileReport`` in ``WIKIPEDIA_PLACE_NAME_FILES``
    order, filtered to the requested languages.
    """
    in_db_names, attested_names = _build_toponym_lookups(db)

    selected = WIKIPEDIA_PLACE_NAME_FILES
    if languages:
        wanted = {_filename_for_language(lang) for lang in languages}
        selected = tuple(f for f in WIKIPEDIA_PLACE_NAME_FILES if f in wanted)

    reports: list[FileReport] = []
    for filename in selected:
        path = data_dir / filename
        if not path.exists():
            # Don't crash — the operator may legitimately not have all
            # 5 files (e.g. a partial checkout). Emit an empty file
            # report so the CLI can surface it.
            reports.append(FileReport(filename=filename, regions=[]))
            continue
        raw = _load_place_names(path)
        regions: list[RegionReport] = []
        for country, subregions in raw.items():
            counties = [
                _compute_county_report(
                    name=subregion,
                    place_names=names,
                    in_db_names=in_db_names,
                    attested_names=attested_names,
                )
                for subregion, names in subregions.items()
            ]
            regions.append(RegionReport(name=country, counties=counties))
        reports.append(FileReport(filename=filename, regions=regions))
    return reports


def report_to_dict(reports: list[FileReport]) -> dict:
    """Serialize the report to the ``--as-json`` shape.

    Shape::

        {
          "files": {
            "<filename>": {
              "total": int, "in_db": int, "attested": int, "gap": int,
              "regions": {
                "<region>": {
                  "total": int, "in_db": int, "attested": int, "gap": int,
                  "counties": {
                    "<county>": {"total": int, "in_db": int,
                                 "attested": int, "gap": int}, ...
                  }
                }, ...
              }
            }, ...
          },
          "totals": {"total": int, "in_db": int, "attested": int, "gap": int}
        }
    """
    out: dict = {"files": {}, "totals": {"total": 0, "in_db": 0, "attested": 0, "gap": 0}}
    for fr in reports:
        file_entry: dict = {
            "total": fr.total,
            "in_db": fr.in_db,
            "attested": fr.attested,
            "gap": fr.gap,
            "regions": {},
        }
        for rr in fr.regions:
            region_entry: dict = {
                "total": rr.total,
                "in_db": rr.in_db,
                "attested": rr.attested,
                "gap": rr.gap,
                "counties": {},
            }
            for cr in rr.counties:
                region_entry["counties"][cr.name] = {
                    "total": cr.total,
                    "in_db": cr.in_db,
                    "attested": cr.attested,
                    "gap": cr.gap,
                }
            file_entry["regions"][rr.name] = region_entry
        out["files"][fr.filename] = file_entry
        out["totals"]["total"] += fr.total
        out["totals"]["in_db"] += fr.in_db
        out["totals"]["attested"] += fr.attested
        out["totals"]["gap"] += fr.gap
    return out


# Column layout for the human-readable table. The widths were tuned
# against the live-DB numbers: English's "20172" is the widest count;
# "The Isle of Man" is the widest region label at 15 chars; "%_attested"
# is the longest header. Stays under 100 columns at the deepest
# indent (verbose per-county).
_LABEL_WIDTH = 28  # leaves room for "    " indent + 24-char county name
_COL_WIDTH = 9
_PCT_WIDTH = 10


def _row(label: str, total: int, in_db: int, attested: int, indent: int = 0) -> str:
    """Format one data row. Width-aligned to the header columns.

    ``indent`` is the number of leading spaces; the label slot is
    shrunk by that much so the numeric columns stay aligned with the
    header regardless of nesting depth.
    """
    gap = total - attested
    pct = (attested / total * 100) if total else 0.0
    label_slot = max(_LABEL_WIDTH - indent, 1)
    return (
        " " * indent
        + f"{label:<{label_slot}}"
        + f"{total:<{_COL_WIDTH}}"
        + f"{in_db:<{_COL_WIDTH}}"
        + f"{attested:<{_COL_WIDTH}}"
        + f"{gap:<{_COL_WIDTH}}"
        + f"{pct:>{_PCT_WIDTH - 1}.0f}%"
    )


def format_report(reports: list[FileReport], *, verbose: bool = False) -> str:
    """Format the report as a multi-line table.

    With ``verbose=True``, each region expands to its counties. Without
    verbose, the report rolls up at country/region granularity.
    """
    lines: list[str] = []
    header = (
        f"{'':<{_LABEL_WIDTH}}"
        f"{'total':<{_COL_WIDTH}}"
        f"{'in_db':<{_COL_WIDTH}}"
        f"{'attested':<{_COL_WIDTH}}"
        f"{'gap':<{_COL_WIDTH}}"
        f"{'%_attested':>{_PCT_WIDTH}}"
    )
    lines.append("Wikipedia backfill report")
    lines.append(header)

    total = in_db = attested = 0
    for fr in reports:
        lines.append(fr.filename)
        for rr in fr.regions:
            lines.append(_row(rr.name, rr.total, rr.in_db, rr.attested, indent=2))
            if verbose:
                for cr in rr.counties:
                    lines.append(_row(cr.name, cr.total, cr.in_db, cr.attested, indent=4))
        total += fr.total
        in_db += fr.in_db
        attested += fr.attested

    lines.append(_row("TOTAL", total, in_db, attested))
    return "\n".join(lines)
