"""Wikipedia place-name backfill report (wyrd-4453).

The ``wyrd/generators/kenning/data/*_place_names.json`` files are
Wikipedia-sourced lists that originally seeded the toponym table. The
epic's terminal goal is to retire those files once scholar-source
corpus mining covers enough of the same entries — at which point the
Wikipedia provenance can be dropped without losing data.

This module reports the retirement-readiness gap per region:

* ``total``       — Wikipedia entries the file lists for the region.
* ``in_db``       — entries that resolve to a ``toponym`` row by exact
  ``modern_name`` match (after NFC normalization on both sides so a
  diacritic-encoding mismatch doesn't falsely under-count).
* ``attested``    — entries whose ``toponym`` row has ≥1
  ``toponym_attestation`` row from a SCHOLAR source (rando-port and
  any future synthetic-provenance rows are explicitly excluded —
  scholar attestation is the retirement criterion; a row attested
  only by the legacy seed itself doesn't justify retiring that seed).
* ``gap``         — ``total - attested``; how many entries still need
  scholar mining before Wikipedia can be retired safely.

KNOWN LIMITATIONS (documented intentionally):

1. Name matching is case-sensitive exact match against
   ``toponym.modern_name`` (after NFC normalization). Operators
   running the report against the live DB will see false-negatives
   where the Wikipedia name and the DB form differ in punctuation or
   disambiguation suffixes ("Newcastle" vs "Newcastle upon Tyne").
   A fuzzier matcher is a follow-up — start with the strict numbers
   so the retirement gate is conservative.

2. Cross-region permissive matching: a ``Newton`` attested anywhere
   in the DB counts as attested for every Wikipedia ``Newton`` across
   all 5 culture files. Conceptually this OVER-counts retirement
   readiness (different "Newton" places shouldn't transitively cover
   for each other). The current report is intentionally permissive
   — the operator-review step that follows the report is where
   per-region attestation gets vetted. A region-scoped variant is a
   follow-up (see bd ticket pointer in CLAUDE.md if it's filed).

Read-only: this module NEVER writes to the DB.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from wyrd.generators.kenning.lexicon import _NON_SCHOLAR_SOURCES, LexiconDB

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

# Culture tokens accepted by ``--language``. Derived from
# WIKIPEDIA_PLACE_NAME_FILES so the CLI can validate the operator's
# spelling and refuse silently-empty runs (R1 silent-failure-hunter
# MEDIUM). Order matches the file ordering for predictable error
# messages.
SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(
    f.removesuffix("_place_names.json") for f in WIKIPEDIA_PLACE_NAME_FILES
)


def _nfc(text: str) -> str:
    """Apply Unicode NFC normalization. The toponym table and the
    Wikipedia JSONs may store the SAME character sequence with
    different combining-form decomposition (e.g. precomposed
    ``é`` U+00E9 vs decomposed ``e`` U+0065 + ``́``). Match
    by NFC on both sides so a diacritic-encoding mismatch doesn't
    silently under-count attestations. R1 silent-failure-hunter
    MEDIUM."""
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class CountyReport:
    """One subregion's backfill numbers (e.g. "Bedfordshire" inside
    England). The atomic unit of the report — every higher-level
    aggregate sums these.

    Invariants (validated in __post_init__):
        0 <= attested <= in_db <= total
    """

    name: str
    total: int
    in_db: int
    attested: int

    def __post_init__(self) -> None:
        # Validate the count invariant so a producer bug surfaces
        # immediately instead of as a -ve gap deep in a report.
        # R1 type-design MEDIUM.
        if not (0 <= self.attested <= self.in_db <= self.total):
            raise ValueError(
                f"CountyReport({self.name!r}) violates 0 <= attested <= in_db <= total: "
                f"total={self.total} in_db={self.in_db} attested={self.attested}"
            )

    @property
    def in_db_no_attestation(self) -> int:
        """Entries in the DB but without any scholar attestation.

        Surfaced separately because they're the rows that contributed
        to the DB BUT can't be retired yet — they're still Wikipedia-
        only provenance until a scholar source confirms them.
        """
        return self.in_db - self.attested

    @property
    def gap(self) -> int:
        """Wikipedia entries still needing scholar attestation.

        Equals ``(total - in_db) + (in_db - attested)``: both the
        not-in-DB rows and the in-DB-but-unattested rows still block
        Wikipedia retirement.
        """
        return self.total - self.attested

    @property
    def pct_attested(self) -> float:
        return self.attested / self.total if self.total else 0.0


@dataclass(frozen=True)
class RegionReport:
    """One country/region inside a file (e.g. "England", "The Isle of
    Man") — aggregates its counties.

    ``counties`` is a tuple (not a list) so the frozen=True guarantee
    actually holds — a list field on a frozen dataclass blocks
    rebinding but not in-place mutation. R1 type-design MEDIUM.
    """

    name: str
    counties: tuple[CountyReport, ...] = field(default_factory=tuple)

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
    """One Wikipedia *_place_names.json file — aggregates its regions.

    ``regions`` is a tuple (see RegionReport for why)."""

    filename: str
    regions: tuple[RegionReport, ...] = field(default_factory=tuple)

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
    shape mismatch or malformed JSON — we want to fail loud on a file
    that drifted from the documented shape so the operator notices,
    not silently report zero.
    """
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
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
    """Return ``(in_db_names, attested_names)`` from two DB reads.

    Both are sets of NFC-normalized ``modern_name`` strings.
    ``in_db_names`` is every name that has a ``toponym`` row;
    ``attested_names`` is the subset that ALSO has ≥1
    ``toponym_attestation`` row FROM A SCHOLAR SOURCE (rando-port
    and any other entries in ``_NON_SCHOLAR_SOURCES`` are excluded —
    a name attested only by the Wikipedia/seed sources is not
    evidence of scholar coverage, so it doesn't satisfy the
    retirement gate). R1 silent-failure-hunter HIGH.

    Built as sets to avoid an N-way query loop — the largest input
    file (English) has ~20k entries; loading both into memory is
    cheap (single-digit MB) and turns each lookup into a hash
    membership check.

    Same ``modern_name`` may appear in multiple toponym rows
    (different country/region disambiguators); the set collapses them
    — once the name is in the DB it's in the lookup, regardless of
    how many rows share it. See module docstring KNOWN LIMITATIONS
    #2 for the cross-region permissive-match implication.
    """
    in_db_names: set[str] = set()
    # SELECT DISTINCT collapses duplicates at the DB level so the
    # Python loop only sees unique names. Same modern_name commonly
    # appears across multiple toponym rows (different country/region
    # disambiguators); pre-filtering avoids redundant set insertions
    # and reduces row-fetch bandwidth on large corpora.
    for row in db.conn.execute("SELECT DISTINCT modern_name FROM toponym"):
        in_db_names.add(_nfc(row["modern_name"]))

    # Exclude non-scholar source_docs (rando-port today; any future
    # synthetic-provenance entries) — a name attested ONLY by the
    # seed sources is not evidence of scholar coverage. The CTE
    # form lets the planner pre-filter attestations before joining.
    attested_names: set[str] = set()
    placeholders = ",".join("?" * len(_NON_SCHOLAR_SOURCES))
    sql = (
        "SELECT DISTINCT t.modern_name "
        "FROM toponym t "
        "JOIN toponym_attestation a ON a.toponym_id = t.id "
        f"WHERE a.source_doc IS NULL OR a.source_doc NOT IN ({placeholders})"
    )
    for row in db.conn.execute(sql, tuple(_NON_SCHOLAR_SOURCES)):
        attested_names.add(_nfc(row["modern_name"]))

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
    ``test_duplicate_name_counts_each_occurrence``. Names are NFC-
    normalized to match the same normalization applied at DB-read
    time in ``_build_toponym_lookups``.
    """
    total = len(place_names)
    normalized = [_nfc(n) for n in place_names]
    in_db = sum(1 for n in normalized if n in in_db_names)
    attested = sum(1 for n in normalized if n in attested_names)
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
          files are reported. Unknown tokens raise ValueError (the
          CLI converts to a BadParameter so the operator sees the
          full set of valid options). R1 silent-failure-hunter
          MEDIUM: a typo'd ``--language welch`` used to silently
          produce an empty report.

    Returns: list of ``FileReport`` in ``WIKIPEDIA_PLACE_NAME_FILES``
    order, filtered to the requested languages.
    """
    if languages:
        bad = [lang for lang in languages if lang.lower() not in SUPPORTED_LANGUAGES]
        if bad:
            raise ValueError(
                f"unknown --language token(s): {sorted(bad)!r}; "
                f"valid: {list(SUPPORTED_LANGUAGES)!r}"
            )

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
            reports.append(FileReport(filename=filename, regions=()))
            continue
        raw = _load_place_names(path)
        regions = tuple(
            RegionReport(
                name=country,
                counties=tuple(
                    _compute_county_report(
                        name=subregion,
                        place_names=names,
                        in_db_names=in_db_names,
                        attested_names=attested_names,
                    )
                    for subregion, names in subregions.items()
                ),
            )
            for country, subregions in raw.items()
        )
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


# Column layout for the human-readable table. ``_COL_WIDTH`` is wide
# enough for 6-digit counts (corpus-scale: English currently ~20k,
# Irish ~41k). Bump to 10 if any single bucket ever exceeds 999k.
# ``_LABEL_WIDTH=40`` leaves room for the 4-space indent + 36-char
# county/place names — long English names like
# "Sutton-under-Whitestonecliffe" (29 chars) sit comfortably. Total
# table width stays under 100 columns at the deepest indent.
_LABEL_WIDTH = 40
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
