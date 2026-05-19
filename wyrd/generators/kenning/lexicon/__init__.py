"""Lexicon data store: SQLite-backed authoring DB for place-name etymology.

Authoritative for authoring (mining sources, tracking citations, recording
disagreement). The runtime keeps reading meanings.json, which is exported from
this DB by `wyrd kenning lexicon export-meanings`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wyrd.generators.kenning.era import canonical_language_for_cell, era_year_range
from wyrd.generators.kenning.phonology_rules import rule_form as phonology_rule_form

if TYPE_CHECKING:
    from wyrd.generators.kenning.skeat_parser import ParsedEntry

# wyrd-67fv: every authoring concern lives in a dedicated submodule
# (``lexicon.constants``, ``lexicon.db``, ``lexicon.schema``,
# ``lexicon.citations``, ``lexicon.seed``, ``lexicon.empirical_priors``,
# ``lexicon.sql``). The blocks below re-export the PUBLIC surface — plus
# the small set of underscore-prefixed names tests actually import — so
# external callers (cli.py, rewind.py, disambiguator.py, the test suite)
# keep their existing ``from wyrd.generators.kenning.lexicon import …``
# imports unchanged. Underscore helpers without an external caller are
# NOT re-exported; in-package callers reach them via the submodule path
# directly (``from wyrd.generators.kenning.lexicon.schema import …``).
from wyrd.generators.kenning.lexicon.citations import (  # noqa: E402, F401
    _normalize_for_quote_match,  # tested directly by test_kenning_lexicon
    _quote_body_excerpt,  # tested directly by test_kenning_lexicon
    backfill_citation_pages,
    detect_running_headers,
    page_for_offset,
    parse_running_header_pages,
    parse_skeat_section_header_pages,
)
from wyrd.generators.kenning.lexicon.constants import (  # noqa: E402, F401
    LANGUAGE_FIELDS,
    NON_LANGUAGE_FIELDS,  # re-exported for tests; not used inside this module
    normalize_ocr_form,
    position_from_usage,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB  # noqa: E402, F401
from wyrd.generators.kenning.lexicon.enrichment import (  # noqa: E402, F401
    INFLECTION_RULES,
    MUTATION_RULES,
    annotate_fragments_with_corpus_evidence,
    clear_enrichment,
    cluster_ocr_variants,
    derive_lemma_candidate,
    derive_lemma_candidates,
    derive_mutation_lemma_candidate,
    fuzzy_search_attestations,
    levenshtein,
    link_lemmas,
    reverse_search_attestations,
)
from wyrd.generators.kenning.lexicon.schema import (  # noqa: E402, F401
    _migrate_toponym_etymology_canonical,  # used by test_kenning_toponym_etymology_canonical
    init_schema,
    migrate_schema,
    record_mining_run,
)
from wyrd.generators.kenning.lexicon.seed import seed_from_meanings  # noqa: E402, F401

# Underscore-prefixed alias for the existing internal callers; the
# canonical name is the un-prefixed export from ``lexicon.constants``.
# Kept until the rest of the package's call sites migrate.
_position_from_usage = position_from_usage


# --- meaning_synset (wyrd-7tz Phase 1) -------------------------------------
#
# Manual / seed-data driven for now. Phase 2 will add an LLM-assisted
# classification pass over the full lexicon. Phase 3 will wire the
# query layer into generator transforms (calque, anglicize/foreignize,
# drift-toward-X). Phase 1 ships:
#   - schema (in lexicon.sql + migrate_schema)
#   - seed catalog from data/meaning_synsets.json
#   - manual assign / candidates / list helpers
#   - CLI subcommands for human curation


def _meaning_synsets_seed() -> dict:
    """Read the bundled seed catalog from package data."""
    raw = (
        resources.files("wyrd.generators.kenning.data").joinpath("meaning_synsets.json").read_text()
    )
    return json.loads(raw)


def seed_meaning_synsets(db: LexiconDB) -> dict[str, int]:
    """Idempotently populate meaning_synset from the bundled catalog.

    Inserts canonical_label rows that don't yet exist; for existing
    rows, updates hypernym + notes if they drifted. Returns
    {'inserted': N, 'updated': M, 'unchanged': K} so callers can
    report.

    Hypernyms resolve in a two-pass walk so a child can be declared
    before its parent in the JSON without breaking the FK. Pass one
    inserts/updates rows with hypernym=NULL; pass two writes the
    hypernym links once every label has a known id.
    """
    catalog = _meaning_synsets_seed()
    entries: list[dict] = catalog["synsets"]
    inserted = updated = unchanged = 0
    # Pass 1: rows with no hypernym dependency. Use INSERT OR IGNORE
    # for the canonical_label uniqueness; update notes separately so
    # we can count drift.
    label_to_id: dict[str, int] = {}
    for entry in entries:
        label = entry["label"]
        notes = entry.get("notes")
        cur = db.conn.execute(
            "SELECT id, notes FROM meaning_synset WHERE canonical_label = ?",
            (label,),
        )
        row = cur.fetchone()
        if row is None:
            cur = db.conn.execute(
                "INSERT INTO meaning_synset (canonical_label, notes) VALUES (?, ?)",
                (label, notes),
            )
            label_to_id[label] = cur.lastrowid
            inserted += 1
            continue
        label_to_id[label] = row["id"]
        if row["notes"] != notes:
            db.conn.execute(
                "UPDATE meaning_synset SET notes = ? WHERE id = ?",
                (notes, row["id"]),
            )
            updated += 1
        else:
            unchanged += 1
    # Pass 2: hypernym links. Always run on every entry so a hypernym
    # removed from the JSON file gets its FK NULL'd out (drift-aware
    # like the notes update in pass 1). Missing hypernyms (typo or
    # out-of-catalog) emit a warning so seed-file typos surface, but
    # don't fail the whole seed run — the row keeps its current
    # hypernym_id.
    import warnings as _warnings

    for entry in entries:
        hypernym_label = entry.get("hypernym")
        if hypernym_label is None:
            db.conn.execute(
                "UPDATE meaning_synset SET hypernym_id = NULL WHERE canonical_label = ?",
                (entry["label"],),
            )
            continue
        if hypernym_label not in label_to_id:
            _warnings.warn(
                f"meaning_synsets.json: synset {entry['label']!r} references "
                f"unknown hypernym {hypernym_label!r}; leaving FK unchanged",
                stacklevel=2,
            )
            continue
        db.conn.execute(
            "UPDATE meaning_synset SET hypernym_id = ? WHERE canonical_label = ?",
            (label_to_id[hypernym_label], entry["label"]),
        )
    db.commit()
    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}


def list_meaning_synsets(db: LexiconDB, *, with_member_counts: bool = False) -> list[dict]:
    """Return all meaning_synset rows ordered by canonical_label.

    With ``with_member_counts=True`` includes a 'member_count' integer
    per row — useful for the CLI to surface which synsets are still
    empty after the LLM pass.
    """
    if with_member_counts:
        sql = """
            SELECT s.id, s.canonical_label, s.hypernym_id, s.notes,
                   COUNT(ems.etymon_id) AS member_count
            FROM meaning_synset s
            LEFT JOIN etymon_meaning_synset ems ON ems.meaning_synset_id = s.id
            GROUP BY s.id
            ORDER BY s.canonical_label
        """
    else:
        sql = """
            SELECT id, canonical_label, hypernym_id, notes
            FROM meaning_synset
            ORDER BY canonical_label
        """
    return [dict(row) for row in db.conn.execute(sql)]


def assign_etymon_to_meaning_synset(
    db: LexiconDB,
    etymon_id: int,
    synset_label: str,
    *,
    fit: str = "core",
) -> bool:
    """Add a (etymon, meaning_synset) membership row, or update its fit
    if the row already exists. Returns True on first-time insert,
    False on update / unchanged.

    Raises ValueError on an unknown etymon_id, unknown synset_label, or
    bad fit value — fail loudly so a typo at the CLI surface doesn't
    silently insert nothing.
    """
    if fit not in ("core", "peripheral"):
        raise ValueError(f"fit must be 'core' or 'peripheral', got {fit!r}")
    etymon_row = db.conn.execute("SELECT id FROM etymon WHERE id = ?", (etymon_id,)).fetchone()
    if etymon_row is None:
        raise ValueError(f"unknown etymon_id: {etymon_id}")
    synset_row = db.conn.execute(
        "SELECT id FROM meaning_synset WHERE canonical_label = ?",
        (synset_label,),
    ).fetchone()
    if synset_row is None:
        raise ValueError(f"unknown meaning_synset label: {synset_label!r}")
    existing = db.conn.execute(
        """
        SELECT fit FROM etymon_meaning_synset
        WHERE etymon_id = ? AND meaning_synset_id = ?
        """,
        (etymon_id, synset_row["id"]),
    ).fetchone()
    if existing is None:
        db.conn.execute(
            """
            INSERT INTO etymon_meaning_synset (etymon_id, meaning_synset_id, fit)
            VALUES (?, ?, ?)
            """,
            (etymon_id, synset_row["id"], fit),
        )
        db.commit()
        return True
    if existing["fit"] != fit:
        db.conn.execute(
            """
            UPDATE etymon_meaning_synset SET fit = ?
            WHERE etymon_id = ? AND meaning_synset_id = ?
            """,
            (fit, etymon_id, synset_row["id"]),
        )
        db.commit()
    return False


def get_meaning_synsets_for_etymon(db: LexiconDB, etymon_id: int) -> list[dict]:
    """Return the synset rows an etymon belongs to, ordered by fit
    ('core' first) then canonical_label."""
    return [
        dict(row)
        for row in db.conn.execute(
            """
            SELECT s.id, s.canonical_label, s.hypernym_id, ems.fit
            FROM etymon_meaning_synset ems
            JOIN meaning_synset s ON s.id = ems.meaning_synset_id
            WHERE ems.etymon_id = ?
            ORDER BY CASE ems.fit WHEN 'core' THEN 0 ELSE 1 END, s.canonical_label
            """,
            (etymon_id,),
        )
    ]


def get_meaning_preserving_candidates(
    db: LexiconDB,
    etymon_id: int,
    *,
    target_language: str | None = None,
    fit: str | None = None,
    include_self: bool = False,
    dedupe: bool = True,
) -> list[dict]:
    """Return etymons that share at least one meaning_synset with the
    target etymon — i.e. candidates for a meaning-preserving
    substitution by the upcoming Lab transforms.

    Each result row carries:
      - etymon_id, canonical_form, language
      - synset_label, meaning_synset_id (the shared meaning_synset)
      - target_fit (the source etymon's fit in this synset)
      - candidate_fit (the candidate etymon's fit)

    ``dedupe`` (default True) collapses (target, candidate) pairs that
    share multiple synsets into one row per candidate, picking the
    'best' shared synset by fit ('core' beats 'peripheral'); ties break
    on canonical_label. This is what the Phase 3 transforms want
    (replace-root etc. need ONE candidate per etymon, not one per
    shared synset). Pass ``dedupe=False`` to get the raw cartesian for
    introspection — the audit/debugging case.

    ``target_language`` restricts candidates to one language (e.g. for
    anglicize/foreignize transforms). ``fit`` restricts BOTH the target
    side and the candidate side (e.g. 'core' for high-confidence
    substitutions only). ``include_self`` keeps the target etymon in
    the result set; default is to exclude it.
    """
    if fit is not None and fit not in ("core", "peripheral"):
        raise ValueError(f"fit filter must be 'core' or 'peripheral', got {fit!r}")
    sql = """
        SELECT
          ems_other.etymon_id              AS etymon_id,
          e.canonical_form                 AS canonical_form,
          e.language                       AS language,
          s.canonical_label                AS synset_label,
          s.id                             AS meaning_synset_id,
          ems_self.fit                     AS target_fit,
          ems_other.fit                    AS candidate_fit
        FROM etymon_meaning_synset ems_self
        JOIN meaning_synset s         ON s.id = ems_self.meaning_synset_id
        JOIN etymon_meaning_synset ems_other ON ems_other.meaning_synset_id = s.id
        JOIN etymon e                 ON e.id = ems_other.etymon_id
        WHERE ems_self.etymon_id = ?
    """
    params: list = [etymon_id]
    if not include_self:
        sql += " AND ems_other.etymon_id != ?"
        params.append(etymon_id)
    if target_language is not None:
        sql += " AND e.language = ?"
        params.append(target_language)
    if fit is not None:
        sql += " AND ems_self.fit = ? AND ems_other.fit = ?"
        params.extend([fit, fit])
    sql += " ORDER BY s.canonical_label, e.language, e.canonical_form"
    rows = [dict(row) for row in db.conn.execute(sql, params)]
    if not dedupe:
        return rows
    # Collapse to one row per candidate etymon. Sort the per-candidate
    # rows by (target_fit_priority, candidate_fit_priority,
    # synset_label) so 'core/core' beats 'core/peripheral' beats
    # 'peripheral/peripheral'; ties break on label for determinism.
    fit_rank = {"core": 0, "peripheral": 1}
    by_candidate: dict[int, dict] = {}
    for row in rows:
        eid = row["etymon_id"]
        existing = by_candidate.get(eid)
        if existing is None:
            by_candidate[eid] = row
            continue
        new_key = (
            fit_rank[row["target_fit"]],
            fit_rank[row["candidate_fit"]],
            row["synset_label"],
        )
        old_key = (
            fit_rank[existing["target_fit"]],
            fit_rank[existing["candidate_fit"]],
            existing["synset_label"],
        )
        if new_key < old_key:
            by_candidate[eid] = row
    return sorted(
        by_candidate.values(),
        key=lambda r: (r["language"], r["canonical_form"]),
    )


# wyrd-jott Phase 1: Goidelic initial-mutation rules (séimhiú = lenition,
# urú = eclipsis). UNLIKE the suffix-strip rules in INFLECTION_RULES,
# these strip from the START of the form because Irish / Scottish
# Gaelic / Old Irish mark grammatical mutation by prepending or
# transforming the initial consonant.
#
# Each rule is ``(prefix, replacement, label)``:
#   - ``prefix`` is what to look for at the start of the inflected form
#   - ``replacement`` is the first character of the unmutated lemma
#   - ``label`` describes the mutation
#
# Example: 'mboga' (eclipsis of 'b' → 'mb' prefix) → strip 'mb', prepend
# 'b' → 'boga'.
#
# CONSERVATIVE SUBSET: ch- and th- (lenition of c-/t-) are DELIBERATELY
# OMITTED. Both are also real digraph-initial Irish lemmas (chéile,
# cheap, cheart; thart, thiar, thuas) so the strip rule produces too
# many false positives. The remaining lenition rules cover m-, b-, d-,
# g-, p-, f-, s- which rarely have the lenited digraph as a true
# lemma-initial spelling. Eclipsis rules are unambiguous (the eclipsing
# digraphs are never real word-initial sequences in Goidelic).
#
# h-prefix (vowel-initial mutation) and t-prefix are also OMITTED for
# false-positive risk — many real Irish lemmas start with h- (loans
# like 'hata' 'hat') or t- (typical lemma initials).

# --- D5-1 / wyrd-3ux: attested-year lookup --------------------------------
#
# Post-mining stage that scans source bodies for date-citation patterns
# near each `etymon_text_match.matched_form` and records the EARLIEST
# plausible year into `etymon_text_match.attested_year`. LLM-free,
# idempotent, reversible (clear-enrichment --stage=attested-years).
#
# Year range = [_ATTESTED_YEAR_MIN, _ATTESTED_YEAR_MAX] = [100, 1700]
# (matches llm_extractor's prompt-side capture). Roman-era through pre-
# modern. Filters out scholarly publication years (most are 1800s+),
# page numbers (small integers), and most modern stray digits.
#
# Foundation for D5-2 era-cell sampling.

_ATTESTED_YEAR_MIN_LOOKUP = 100
_ATTESTED_YEAR_MAX_LOOKUP = 1700


# Form-attached year-citation pattern. Three accepted shapes, all
# requiring the year to follow the matched form within a small window of
# punctuation / whitespace:
#
#   1. "c." prefix — explicit "circa" marker (any year in range):
#         Tune c. 950        Tunna, c. 1066        Tune (c. 950)
#
#   2. Parenthesized year — distinctive citation form:
#         Tune (1086)        Tunes (1242)
#
#   3. Bare year ≥ 700 — post-Roman British/OE attestations begin
#      around then, so a 3-4 digit number ≥700 is far more likely to be
#      a real date than a page reference (page numbers in toponym
#      dictionaries cluster in the low hundreds):
#         Tune, 1086 (DB)    Tunes; 1242    Tunna 800
#
# Years <700 only qualify under (1) or (2). This filters the dominant
# false-positive class — common-word reverse-search forms (`with`,
# `wind`, `port`) followed by page references like "form, 134" — while
# admitting bona-fide Roman-era citations when they're explicitly marked
# with "c." or parens.
#
# Built fresh per matched_form because re.escape() makes it form-specific.
def _build_form_year_pattern(form: str) -> re.Pattern[str]:
    return re.compile(
        r"\b" + re.escape(form) + r"\b"
        r"(?:"
        # c./circa prefix — accept any 3-4 digit year (Roman era OK)
        r"[\s,;:.()\[\]]{0,4}c\.?\s*(\d{3,4})\b"
        r"|"
        # parenthesized year — strong citation marker
        r"[\s,;:.]*\((\d{3,4})\b"
        r"|"
        # bare year ≥ 700 — post-Roman / OE attestations. Three-digit
        # range (700-999) plus four-digit range (1000-1700) explicitly
        # listed so a 4-digit 8xxx/9xxx publication ID, ISBN fragment,
        # or page reference can't slip in. The trailing |1700 admits the
        # year-range upper bound that 1[0-6]\d{2} cuts off at 1699.
        r"[\s,;:.]{0,4}(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b"
        r")"
    )


def _extract_attested_year_from_body(text: str, form: str) -> int | None:
    """Return the EARLIEST plausibly-attested year cited DIRECTLY against
    ``form`` in ``text``, or None if no qualifying citation is found.

    ``text`` is expected lowercased + OCR-normalized (the same shape the
    reverse-search snippets are stored in). ``form`` should match.

    A candidate year qualifies when:
      * The matched_form is followed (within a few chars of intervening
        punctuation / space, and optionally a "c." prefix) by a 3-4
        digit year — see _build_form_year_pattern. This is the dominant
        scholarly toponym citation shape.
      * The year parses to an integer in [_ATTESTED_YEAR_MIN_LOOKUP,
        _ATTESTED_YEAR_MAX_LOOKUP].

    "Earliest" is a deliberate v1 choice: scholarly toponym citations
    often run ascending ("Tune, 1086; Tunes, 1242; Tunne, 1340"), and the
    earliest reflects the earliest known attestation of the form. Future
    work (D5-2) may want all years, not just the earliest.
    """
    if not form:
        return None
    # ``text`` is lowercased + OCR-normalized before reaching this fn
    # (see lookup_attested_years). Lowercase the form so case-mismatched
    # matched_form values still produce hits.
    pattern = _build_form_year_pattern(form.lower())
    earliest: int | None = None
    for m in pattern.finditer(text):
        # Three capture groups in the alternation; exactly one will be
        # populated per match.
        captured = m.group(1) or m.group(2) or m.group(3)
        if captured is None:
            continue
        year = int(captured)
        if year < _ATTESTED_YEAR_MIN_LOOKUP or year > _ATTESTED_YEAR_MAX_LOOKUP:
            continue
        if earliest is None or year < earliest:
            earliest = year
    return earliest


_TOPONYM_NOTE_YEAR_PATTERN = re.compile(r"\b(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b")

# A digit run preceded by one of these abbreviations is a page or
# volume reference, not a date. Page numbers in long EPNS volumes
# occasionally exceed 700, so filtering on year-range alone leaks
# them through.
#
# Matched as whole words at the END of the preceding slice — `\b`
# prevents false-skips on words that happen to end in "p." (e.g.
# "Bp." for Bishop) by requiring a word boundary before the marker.
# Substring match would also incorrectly fire "p." inside "chap.",
# but that's a no-op (chap is itself a marker); the real FP class
# was words like "Hp." or stray "p" letters at the end of the slice.
_TOPONYM_NOTE_PAGE_MARKER_RE = re.compile(
    r"\b(?:p|pp|vol|vols|ch|chap|no|nr)\.\s*$",
    re.IGNORECASE,
)


def _earliest_year_in_notes(notes: str | None) -> int | None:
    """Find the earliest plausible year (700-1700) in a
    ``toponym_etymology.notes`` value. Skips digit runs preceded by
    page / volume markers (``"p. 755"``, ``"vol. 1244"``) — those are
    bibliographic references, not dates. Returns None when no
    qualifying year appears."""
    if not notes:
        return None
    earliest: int | None = None
    for m in _TOPONYM_NOTE_YEAR_PATTERN.finditer(notes):
        ystart = m.start()
        # Pass the full prefix and rely on the regex's $ anchor to
        # match only when the marker IMMEDIATELY precedes the year.
        # A fixed-size window would miss "p.   1086" with extra spaces;
        # the linear search cost is trivial since notes is already in
        # memory and finditer call frequency is bounded by year-hit
        # count (a few per row in the worst case).
        if _TOPONYM_NOTE_PAGE_MARKER_RE.search(notes, 0, ystart):
            continue
        year = int(m.group(1))
        if earliest is None or year < earliest:
            earliest = year
    return earliest


def _scan_etymon_text_match_for_years(db: LexiconDB, sources_path: Path, *, apply: bool) -> dict:
    """Stream ``etymon_text_match`` rows ordered by source_id; for each
    row, scan the matching source body for a form-attached year
    citation (PR #47 / wyrd-3ux pattern).

    Memory characteristics:
    * ONE source body in memory at a time (the heavy thing — 100s of MB
      for big OCR'd corpora).
    * ONE row's metadata at a time during iteration.
    * candidate_updates accumulates (year, row_id) tuples for every hit
      and flushes via executemany at the end. At ~3% hit rate even a
      million-row text-match table yields ~30k tuples (~1 MB), which
      is fine; chunking the writes would be premature optimisation.
    """
    available_sources = {f.stem: f for f in sources_path.glob("*.txt")}

    cur = db.conn.execute(
        "SELECT id, source_id, matched_form FROM etymon_text_match "
        "WHERE attested_year IS NULL ORDER BY source_id"
    )

    candidate_updates: list[tuple[int, int]] = []  # (year, row_id)
    sources_missing: set[str] = set()
    rows_scanned = 0
    current_source_id: str | None = None
    text: str | None = None
    for row in cur:
        rows_scanned += 1
        if row["source_id"] != current_source_id:
            current_source_id = row["source_id"]
            source_file = available_sources.get(current_source_id)
            if source_file is None:
                sources_missing.add(current_source_id)
                text = None
                continue
            text = source_file.read_text(errors="replace").lower()
            text = normalize_ocr_form(text)
        if text is None:
            continue
        year = _extract_attested_year_from_body(text, row["matched_form"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
        "sources_missing": len(sources_missing),
    }


def _scan_toponym_etymology_for_years(db: LexiconDB, *, apply: bool) -> dict:
    """wyrd-bag: scan ``toponym_etymology.notes`` for the earliest
    plausible year. Unlike the ``etymon_text_match`` scan, this doesn't
    need source-body files — the LLM-extracted notes are stored inline
    in the DB and are densely populated with scholarly citations
    (``"Tune, 1086 (DB); Tunes, 1242"``).

    Memory characteristics:
    * Full match list (candidate_updates) is held in memory before the
      executemany write. At ~80% density on the production corpus
      (~5,200 rows × 0.8 = ~4,200 tuples = ~130 KB) this is fine.
    * If a future corpus pushes this past tens of millions of rows,
      switching to chunked writes (yield + flush per N hits) would
      cap peak memory; not worth the complexity at current scale.
    """
    cur = db.conn.execute(
        "SELECT id, notes FROM toponym_etymology WHERE attested_year IS NULL AND notes IS NOT NULL"
    )

    candidate_updates: list[tuple[int, int]] = []
    rows_scanned = 0
    for row in cur:
        rows_scanned += 1
        year = _earliest_year_in_notes(row["notes"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE toponym_etymology SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
    }


def lookup_attested_years(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
) -> dict:
    """Populate ``attested_year`` on the two row sources that carry
    date-citation evidence (D5-1):

    * ``etymon_text_match.attested_year`` (PR #47 / wyrd-3ux) — scans
      source bodies via the form-attached pattern. Lower density (~3%)
      because reverse-search rows are mentions, not citations.
    * ``toponym_etymology.attested_year`` (PR #5x / wyrd-bag) — scans
      LLM-extracted notes for the earliest year ≥700. Higher density
      (~80%) because the notes are scholarly date strings.

    LLM-free, idempotent, reversible. Per D21/D22 this is enrichment —
    operates on already-mined data without touching mining evidence.
    Re-runs are no-ops on rows where ``attested_year`` is already set.
    Reverse via ``clear-enrichment --stage=attested-years --apply``.

    Returns a dict with both per-source breakdowns and aggregate keys
    so existing callers (PR #47's CLI output, tests, etc.) continue
    to read ``rows_scanned`` / ``candidates`` / ``rows_written``
    unchanged while new callers can read the per-source detail.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    etm = _scan_etymon_text_match_for_years(db, sources_path, apply=apply)
    te = _scan_toponym_etymology_for_years(db, apply=apply)

    return {
        # Aggregate keys (PR #47 back-compat).
        "rows_scanned": etm["rows_scanned"] + te["rows_scanned"],
        "candidates": etm["candidates"] + te["candidates"],
        "rows_written": etm["rows_written"] + te["rows_written"],
        "sources_missing": etm["sources_missing"],
        "applied": apply,
        # Per-source breakdown for richer CLI output + new tests.
        "etymon_text_match": etm,
        "toponym_etymology": te,
    }


# --- wyrd-skm Phase 3.0a: toponym-attestation mining ----------------------
#
# Scan toponym_etymology.notes for (form, year) pairs and write them to
# toponym_attestation. Per-toponym dated historical spellings are the
# raw input that wyrd-skm Phase 3.0b derives per-etymon period forms
# from. Schema for toponym_attestation pre-existed (lexicon.sql); this
# pass is the populator.
#
# LLM-free, idempotent, reversible (clear-enrichment --stage=attestations).

# Domesday Book (1086) is the load-bearing dated reference in English
# toponym scholarship — almost every Mawer / Skeat / Ekwall entry cites
# its Domesday spelling. We canonicalize it to the year so a single
# regex captures both "Cestretone in Domesday Book" and "Domesday Book
# has Chingestone". 'D.B.' is the same source, abbreviated.
_DOMESDAY_YEAR = 1086

# Form character class — leading capital plus letters / OE specials /
# Welsh + Norman diacritics / hyphens.
#
# The base segment is bounded at 4-30 chars (lead + {3,29} continuation);
# hyphenated suffixes can extend the total to ~43 chars (Hædan-ham at 9,
# Bedingafelda at 12, Llanfaên-y-bryn at 15 are typical). The lower
# bound excludes 3-letter sentence connectives ("The", "But", "And",
# "Of") that would otherwise leak through the year-anchor patterns; the
# upper bound is generous enough that real toponym variants haven't
# tripped it on the production corpus.
#
# The diacritic set covers the production toponym_etymology.notes
# corpus: Welsh ŵâêôûŷ + macron-vowel ē (Welsh-stratum work landed in
# PR #105), Norman çéè (early French/Anglo-Norman charter spellings),
# OE specials æðþœǣ + macron set āīōū. Both cases of each diacritic so
# capitalised lemmas (``Hēafod``, ``Ŷrwyrne``, ``Llanfaên``) match.
#
# The follow-up ``_form_passes_filter`` check additionally requires at
# least one lowercase letter so pure-uppercase source abbreviations
# (LPR, LI, LF, DB) don't slip through.
_FORM_CHARSET = r"A-Za-zÆÐÞŒæðþœǣĒĀĪŌŪēāīōūȳŴÂÊÔÛŶŵâêôûŷÇÉÈçéè"
_FORM_PATTERN = (
    rf"[A-ZÆÐÞŒĒĀĪŌŪŴÂÊÔÛŶÇÉÈ][{_FORM_CHARSET}]{{3,29}}(?:[-][{_FORM_CHARSET}]{{1,12}})*"
)


# wyrd-hcd9: source-attribution chain FP detector.
#
# Mawer / Skeat / Ekwall use the convention `<year> <source>` to
# attribute SAME-FORM attestations to multiple sources, chained by
# COMMAS:
#
#   Chevington 1535 VE, 1539 Wills, 1544 LP
#                       ↑ source name, NOT a place form
#
# The regex would otherwise match ``Wills, 1544`` as form=Wills
# year=1544 because it follows the FORM-comma-YEAR shape. The
# distinguishing structural signal is two-fold:
#   (a) the form is preceded by ``<year> `` (a 3-4 digit year + space)
#   (b) the form is followed by ``, <year>`` (next source-attribution)
#
# COMMA-after-form is the load-bearing distinguisher from
# semicolon-separated multi-form chains
# (``Edreston ; 1242 Cl. ...``) where the form IS a real attestation:
# scholarly convention puts commas between same-form sources and
# semicolons between different-form attestations. Both conditions
# together identify the source-attribution-chain FP without
# false-suppressing real chained attestations.
_SOURCE_CHAIN_PRECEDING_YEAR_RE = re.compile(r"\d{3,4}\s+$")
# Note: no ``^`` anchor — ``re.match(string, pos)`` already anchors
# at ``pos``, and the pattern's ``^`` would only match at string
# start (position 0), not at ``pos``. ``\s*,\s*\d{3,4}\b`` covers
# the post-form `, 1544` lookahead with optional whitespace either
# side of the comma.
_SOURCE_CHAIN_FOLLOWING_YEAR_RE = re.compile(r"\s*,\s*\d{3,4}\b")


def _form_passes_filter(form: str) -> bool:
    """Final form-quality gate beyond the regex character class.

    Place-name attestations always carry at least one lowercase letter
    (``Cestretone``, ``Hædan-ham``, ``Wyntewurthe``). Pure-uppercase
    runs that match the regex are scholarly source abbreviations
    (``LPR``, ``LI``, ``LF``, ``DB``, ``MS``, ``FA``) and never
    legitimate forms — those exist in citation suffixes and sometimes
    pick up year-shaped digits nearby that survive the page-marker
    guard. Requiring a lowercase letter gives us a single check that
    handles the entire abbreviation false-positive class.
    """
    if not any(c.islower() for c in form):
        return False
    return form.lower() not in _ATTEST_FORM_BLACKLIST


# Year pattern — same range as _earliest_year_in_notes (post-Roman
# 700-1700) so we share its filter on page-references / publication
# years.
_YEAR_PATTERN = r"7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700"

# "FORM in YEAR" or "FORM, YEAR" or "FORM, in YEAR" — the canonical
# citation shapes in Mawer / Skeat / Ekwall. Matches "Cestretone in
# 1210", "Iselham, 1302", "Tadelowe, in 1302", "Spelt Knesworthe in
# 1316" (the Spelt prefix doesn't need its own pattern — it leaves the
# form right where this one anchors).
#
# Two branches in the connector group — punctuation-then-optional-"in"
# OR bare-"in" — keeps an explicit anchor between form and year. A
# pure-whitespace separator (``"After 1066"``) would generate too many
# false positives when a sentence happens to have a capitalized word
# followed by a year-shaped digit run.
#
# Trailing ``(?!\s*\(p+\.)`` negative lookahead suppresses
# academic-citation shape "Author, 1086 (p. 59)" — when ``(p.`` /
# ``(pp.`` immediately follows the year-paren, the year is a
# publication date and the preceding capitalised word is a scholar
# name, not a place-name attestation. Real attestations cite the
# SOURCE in the parens (``"Cestretone, 1086 (D.B.)"``,
# ``"Iselham, 1302 (F.A.)"``); page references show up as the LATER
# component of a parenthetical (``"(D.B., p. 102)"``), which the
# lookahead ignores because the ``(`` is followed by a non-``p`` char.
_ATTEST_FORM_YEAR_RE = re.compile(
    rf"\b(?P<form>{_FORM_PATTERN})"
    r"(?:\s*[,;]\s*(?:in\s+)?|\s+in\s+)"
    rf"(?P<year>{_YEAR_PATTERN})\b"
    r"(?!\s*\(p+\.)"
)

# Chain-element shape — the LAST item of a comma/semicolon-separated
# chain often drops the explicit connector between form and year:
# ``"Cestretone in 1210; Cestrede, 1218; Chestreton 1242"``. The
# leading ``;`` (or ``.``) plus optional whitespace anchors this
# pattern to chain positions; ``After 1066 the conquest came`` won't
# match because no ``;`` precedes ``After``. Bare-whitespace connector
# is still safe within this anchor since chain context already implies
# we're inside a citation list.
_ATTEST_CHAIN_FORM_YEAR_RE = re.compile(
    rf"[;]\s*(?P<form>{_FORM_PATTERN})\s+(?P<year>{_YEAR_PATTERN})\b"
)

# Domesday-anchored patterns. Spelled-out (``Domesday Book``) and
# abbreviated (``D.B.``) shapes each cover the form-before-marker and
# marker-before-form orderings; the four-pattern set folds to two
# regex alternations on the marker side.
#
# ``D\.\s*B\.`` carries an explicit trailing ``(?:\b|,|;|\s)`` — the
# ``\b`` alone fails to match between ``.`` and a space (both are
# non-word) so without the alternation a sentence like ``"Cestretone
# D.B. has more"`` would mis-match. Pinned by a regression test.
_ATTEST_FORM_DOMESDAY_RE = re.compile(
    rf"\b(?P<form>{_FORM_PATTERN})\s+in\s+Domesday(?:\s+Book)?\b"
    rf"|\b(?P<form2>{_FORM_PATTERN})\s*,?\s+D\.\s*B\.(?:\b|,|;|\s)"
)

# Marker-before-form variants of the same data.
_ATTEST_DOMESDAY_HAS_FORM_RE = re.compile(
    rf"\bDomesday(?:\s+Book)?\s+has\s+(?P<form>{_FORM_PATTERN})"
    rf"|\bD\.\s*B\.\s+has\s+(?P<form2>{_FORM_PATTERN})"
)

# Domesday-anchored regex set, paired with the year that should be
# stamped onto every match. The driver loop in
# ``_extract_attestation_pairs`` iterates this tuple — adding a new
# Domesday phrasing means adding one regex here, not a fresh code path.
_DOMESDAY_RES: tuple[tuple[re.Pattern[str], int], ...] = (
    (_ATTEST_FORM_DOMESDAY_RE, _DOMESDAY_YEAR),
    (_ATTEST_DOMESDAY_HAS_FORM_RE, _DOMESDAY_YEAR),
)


# Forms that match the regex character class but are never legitimate
# place-name attestations. Most false positives filter via the
# year-anchor (a year must follow IMMEDIATELY) plus
# ``_form_passes_filter``'s lowercase-letter requirement; this list
# handles the leftover mixed-case false positives that survive both
# filters. Lowercased entries; membership lookup is case-insensitive.
_ATTEST_FORM_BLACKLIST = frozenset(
    {
        # 1. Domesday-citation noise — "Domesday Book has Foo" would
        #    otherwise match "Domesday" as a form on its own under the
        #    year-anchored pattern when a year happens to be nearby.
        "domesday",
        "book",
        # 2. Citation-prefix words — Mawer / Skeat lead with these
        #    before naming the form. The regex captures them when the
        #    next year-shaped digit is close enough.
        "spelt",
        "formerly",
        "apparently",
        # 3. Generic English connectives that survive the lowercase-
        #    required filter (mixed-case at sentence start).
        "the",
        "from",
        "where",
        "there",
        "these",
        "this",
        "that",
        "and",
        "but",
        "with",
        "here",
        # 4. Scholar surnames — appear inline in citation prose
        #    ("according to Kemble"). Never the toponym itself.
        "kemble",
        "skeat",
        "ekwall",
        "mawer",
        "joyce",
        "thorpe",
        "kelly",
        # 5. Source / archival names that read proper-noun-ish.
        "pipe",
        "roll",
        "red",
        "inquisitio",
        # 6. Mixed-case source abbreviations that the lowercase-required
        #    filter doesn't catch. Pure-uppercase abbreviations (LPR,
        #    LI, DB) are filtered in `_form_passes_filter` directly.
        "cod",
        "dipl",
        "vol",
        "ipm",
        "ipms",
    }
)


def _extract_attestation_pairs(notes: str | None) -> list[tuple[str, int]]:
    """Extract ``(form, year)`` attestation pairs from a
    ``toponym_etymology.notes`` value.

    Returns deduped tuples ordered by year ascending, with ties broken
    by form (alphabetical). The ordering is deterministic so callers
    can rely on first-row-per-key idempotency under the unique-index
    DB constraint.

    Pattern set (highest precision first):

    1. ``FORM in YEAR`` / ``FORM, YEAR`` / ``FORM; YEAR`` — the
       dominant scholarly shape. Year is filtered to 700-1700 via the
       same range used by ``_earliest_year_in_notes``. The
       ``(?!\\s*\\(p+\\.)`` negative lookahead rejects
       academic-citation shape ``"Author, 1086 (p. 59)"``.
    2. ``;FORM YEAR`` — chain-element bare connector. The LAST item
       of a citation chain often drops the explicit comma/in
       connector; the leading semicolon anchors this to chain
       positions so sentence-flow false positives don't leak.
    3. ``FORM in Domesday[ Book]`` — Domesday-anchored citation;
       year=1086.
    4. ``Domesday[ Book] has FORM`` — inverted shape for the same.
    5. ``FORM, D.B.`` / ``FORM D.B.`` — the abbreviated form.

    Page-reference false positives are guarded TWO ways: first the
    same ``_earliest_year_in_notes`` shared filter (a digit run
    preceded IMMEDIATELY by ``p.`` / ``pp.`` / ``vol.`` is rejected;
    catches ``"Bedinga feld, p. 59"`` shape), and second the
    parenthetical-page lookahead in pattern 1 above. The two combined
    cover both pre-year (``p. 1086``) and post-year (``1086 (p. 59)``)
    page-marker placements.

    Forms that match the regex syntactically but represent scholarly
    metadata (``Domesday``, ``Spelt``, source-author surnames) are
    filtered via ``_ATTEST_FORM_BLACKLIST``.
    """

    def _matched_form(m: re.Match[str]) -> str | None:
        """Resolve which named group fired in a Domesday alternation.

        Each compiled regex carries either a ``form`` group (single
        branch) or both ``form`` and ``form2`` (two branches sharing
        one regex). Returning the first non-None capture, stripped of
        trailing punctuation, gives one accessor for both shapes.
        """
        for key in ("form", "form2"):
            captured = m.groupdict().get(key)
            if captured is not None:
                return captured.rstrip(",.;:")
        return None

    if not notes:
        return []
    pairs: set[tuple[str, int]] = set()

    def _admit_year_match(m: re.Match[str]) -> None:
        """Validate and absorb a ``(form, year)`` match from the year-
        anchored or chain-anchored regex.

        Shared guards:
        * form-quality filter (lowercase-required, blacklist),
        * year-range bounds,
        * immediate-predecessor page-marker check (rejects
          ``"Bedinga feld, p. 1086"`` shape where the year is actually
          a page reference; probe scope is narrow ``0:year_start``
          since pages cited AFTER the year are caught by the
          ``(?!\\s*\\(p+\\.)`` lookahead on ``_ATTEST_FORM_YEAR_RE``),
        * source-attribution-chain check (rejects ``"1539 Wills,
          1544 LP"`` shape where the form is a SOURCE name in a
          multi-source chain — see _SOURCE_CHAIN_*_RE comments).
        """
        form = m.group("form").rstrip(",.;:")
        if not _form_passes_filter(form):
            return
        year = int(m.group("year"))
        if year < _ATTESTED_YEAR_MIN_LOOKUP or year > _ATTESTED_YEAR_MAX_LOOKUP:
            return
        if _TOPONYM_NOTE_PAGE_MARKER_RE.search(notes, 0, m.start("year")):
            return
        # Source-attribution-chain check: form preceded by `<year> `
        # AND followed by `, <year>` is the source-name FP shape
        # (`Wills, 1544 LP`). Both conditions required so real
        # attestation chains (`Edreston ; 1242 ...`) aren't
        # suppressed.
        form_start = m.start("form")
        form_end = m.end("form")
        if _SOURCE_CHAIN_PRECEDING_YEAR_RE.search(
            notes, 0, form_start
        ) and _SOURCE_CHAIN_FOLLOWING_YEAR_RE.match(notes, form_end):
            return
        pairs.add((form, year))

    # Year-anchored pattern: explicit connector between form and year.
    for m in _ATTEST_FORM_YEAR_RE.finditer(notes):
        _admit_year_match(m)

    # Chain-element pattern: ``;FORM YEAR`` — the last item of a
    # comma/semicolon-separated citation chain often drops the
    # explicit connector. The leading ``;`` anchors this to chain
    # positions so sentence flow ("After 1066") can't slip through.
    for m in _ATTEST_CHAIN_FORM_YEAR_RE.finditer(notes):
        _admit_year_match(m)

    # Domesday-anchored patterns (year fixed to 1086 per
    # _DOMESDAY_RES). The driver tuple folds the four shape variants
    # (form-before / marker-before / spelled / abbreviated) into two
    # regexes; adding a new Domesday phrasing means appending one
    # regex to the tuple.
    for pattern, year in _DOMESDAY_RES:
        for m in pattern.finditer(notes):
            form = _matched_form(m)
            if form is None or not _form_passes_filter(form):
                continue
            pairs.add((form, year))

    return sorted(pairs, key=lambda p: (p[1], p[0]))


def mine_toponym_attestations(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 500,
) -> dict:
    """Populate ``toponym_attestation`` from ``toponym_etymology.notes``
    (wyrd-skm Phase 3.0a). For every etymology row carrying inline
    scholarly date citations like ``"Cestretone in 1210; Hadenham in
    Domesday Book"`` we extract ``(form, year)`` pairs and INSERT them
    into ``toponym_attestation`` keyed on the etymology's
    ``toponym_id``. Source attribution rides on the original
    ``toponym_etymology.source_id`` so a Skeat-derived Cestretone(1210)
    is distinguishable from a Mawer-derived one.

    LLM-free, idempotent, reversible (clear-enrichment
    --stage=attestations). Re-runs are no-ops thanks to the unique
    index ``idx_attestation_unique`` on
    ``(toponym_id, form, date_year, source_doc)``.

    Progress lines emit to stderr every ``progress_every`` rows and
    once at completion (CLAUDE.md mining-progress shape).
    ``progress_every`` is clamped to ≥1 so a caller passing 0 doesn't
    trip a modulo-zero (or, on the rate computation, divide-by-zero).

    Returns a result dict with row-counts so callers can report.
    """
    import sys
    import time

    def _emit_progress(scanned: int, total: int | None = None) -> None:
        """Print one CLAUDE.md-shape progress line to stderr.

        ``total=None`` is the mid-loop case (we don't know the final
        count yet); ``total=scanned`` is the post-loop closer that
        guarantees the final partial chunk surfaces. The rate clamps
        to a small floor so a sub-tick scan doesn't blow up.
        """
        elapsed = max(time.monotonic() - started, 1e-6)
        denom = scanned if scanned else 1
        head = f"[{scanned}/{total}]" if total is not None else f"[{scanned}]"
        print(
            f"  {head}  rows_with_pairs={rows_with_pairs} "
            f"candidates={len(candidate_inserts)} "
            f"rows_written={rows_written} "
            f"({elapsed / denom:.4f}s/entry)",
            file=sys.stderr,
            flush=True,
        )

    progress_every = max(progress_every, 1)
    cur = db.conn.execute(
        "SELECT toponym_id, source_id, notes FROM toponym_etymology "
        "WHERE notes IS NOT NULL AND notes != ''"
    )

    rows_scanned = 0
    rows_written = 0
    candidate_inserts: list[tuple[int, str, int, str]] = []
    rows_with_pairs = 0
    started = time.monotonic()
    for row in cur:
        rows_scanned += 1
        pairs = _extract_attestation_pairs(row["notes"])
        if pairs:
            rows_with_pairs += 1
            for form, year in pairs:
                candidate_inserts.append((row["toponym_id"], form, year, row["source_id"]))
        if rows_scanned % progress_every == 0:
            _emit_progress(rows_scanned)

    if apply and candidate_inserts:
        # INSERT OR IGNORE relies on the unique index added by
        # _create_toponym_attestation_unique_index. Without it, re-runs
        # would duplicate every row.
        result = db.conn.executemany(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            candidate_inserts,
        )
        rows_written = result.rowcount
        db.commit()

    _emit_progress(rows_scanned, total=rows_scanned)

    return {
        "rows_scanned": rows_scanned,
        "rows_with_pairs": rows_with_pairs,
        "candidates": len(candidate_inserts),
        "rows_written": rows_written,
        "applied": apply,
    }


# --- wyrd-skm Phase 3.0b: cognate-cluster era-reflex picker ----------------
#
# Given an etymon (e.g. OE ``ceaster``) and a target era cell, walk
# the cognate cluster (D27/D28 — etymons sharing the same root via
# inheritance/borrowing edges) and return cluster mates whose
# language tag is the canonical pick for that era cell.
#
# Example: ``ceaster`` (OE, cluster_id=3617) at era=me → ME variants
# (Chestre, Chester, Cestre, Chestir, ...) tagged middle-english in
# the same cluster. The downstream wyrd-rni / wyrd-381 demos consume
# this primitive to render compound names at user-chosen eras.
#
# Coverage: 24% of OE toponym etymons currently have cognate_id
# (cluster_cognates pass output); the remainder return an empty list
# and consumers fall back to the original etymon's canonical_form.
# This is a known coverage floor that improves as the descent graph
# grows.


@dataclass
class EraReflex:
    """A single cluster-mate of an etymon at a target era.

    ``etymon_id`` is the cluster mate's row id; ``form`` is its
    ``canonical_form``; ``language`` is its ``etymon.language`` tag.
    Multiple reflexes may surface for the same era when scholarly
    sources record multiple period-specific spellings (the four
    Middle-English variants of OE ``ceaster``: Chestre / Chester /
    Cestre / Chestir / etc.).

    ``source`` distinguishes how the reflex was derived. Consumers
    that surface era progressions (KenningRewind, KenningEraMap) can
    mark phonology-rule-derived forms differently so the user knows
    the form is an inferred derivation rather than an attested
    cluster mate. Values: ``"cluster"`` (Tier 1, cognate cluster),
    ``"descent"`` (Tier 2, descent edge), ``"period-form"`` (Tier 3,
    projected from toponym_attestation), ``"phonology-rule:v1"``
    (Tier 4, derived via phonology_rules.apply_rules). Default
    ``"cluster"`` so existing callers stay unchanged.
    """

    etymon_id: int
    form: str
    language: str
    source: str = "cluster"


def etymon_era_reflexes(
    db: LexiconDB,
    etymon_id: int,
    *,
    target_language: str | None = None,
    target_family_cell: tuple[str, str] | None = None,
) -> list[EraReflex]:
    """Return cluster-mates of ``etymon_id`` matching a target era.

    Two callable shapes:

    * ``target_language='middle-english'`` — direct language pick;
      every cluster mate with that exact language tag is returned.
    * ``target_family_cell=('english', 'me')`` — resolves to the
      canonical language tag via
      ``era.canonical_language_for_cell``, then proceeds as the
      direct-language path.

    Returns ``[]`` when the target cell has no canonical language
    tag (``Norse/modern`` etc.) or when neither lookup path produces
    a match. Output is sorted by ``form`` for deterministic output
    across PYTHONHASHSEED — callers that depend on first-row
    stability get the alphabetically-first reflex.

    **Four lookup tiers** in order of precedence — each is its own
    helper (``_tier1_cluster_reflexes`` etc.) that this dispatcher
    composes. Tiers 1 + 2 are ALTERNATES (cluster wins when
    ``cognate_id`` is set; otherwise descent fires). Tiers 3 + 4
    are sequential FALLBACKS (each runs only if every prior tier
    returned empty).

    1. **Cognate cluster** (D27/D28) — cluster mates of the target
       language. ~24% of OE toponym etymons. High-quality path.
    2. **Direct descent edges** — immediate inheritance / borrowing
       children of the target language. ~4% of OE toponym etymons.
    3. **Period-form projection** (wyrd-unuo Phase 3.3) — projected
       forms from ``etymon_period_form`` whose ``date_year`` falls
       in the target cell's year range. Requires
       ``target_family_cell``. Closes the ~72% coverage gap for
       isolated OE etymons.
    4. **Phonology rule** (wyrd-98cs) — derives the target-era form
       via phonology_rules.apply_rules forward / inverse cell walks.
       Lower precision (no mining evidence); reflexes carry
       ``source='phonology-rule:v1'`` so consumers can mark inferred
       forms differently from attested ones.

    Reflexes are filtered to ``merged_into_id IS NULL`` so OCR-
    cluster losers (D22) don't surface; the merge target is the
    canonical reflex for that surface.
    """
    # Defensive guard: exactly ONE target must be provided. An
    # empty-string ``target_language`` would silently slip the
    # ``is None`` check and resolve to zero rows (the SQL ``language
    # = ''`` predicate is technically valid); catch that explicitly.
    # Passing both targets at once is also a caller bug — silently
    # ignoring one would mask a bad merge.
    if (target_language is None) == (target_family_cell is None):
        raise ValueError("must pass exactly one of target_language or target_family_cell")
    if target_language is not None and not target_language:
        raise ValueError("target_language must not be an empty string")
    if target_language is None:
        assert target_family_cell is not None  # narrowed by the guard above
        family, cell = target_family_cell
        target_language = canonical_language_for_cell(family, cell)
        if target_language is None:
            return []

    # Fetch every etymon column we'll need across all four tiers in
    # one round-trip — Tier 4 also reads canonical_form + language, so
    # a second SELECT later would be wasteful.
    row = db.conn.execute(
        "SELECT cognate_id, canonical_form, language FROM etymon WHERE id = ?",
        (etymon_id,),
    ).fetchone()
    if row is None:
        return []

    # Tier 1 + Tier 2 are alternates — cognate cluster wins when the
    # etymon has cognate_id; otherwise descent edges. The "first
    # non-empty wins" pattern doesn't apply here; we pick exactly one.
    if row["cognate_id"] is not None:
        results = _tier1_cluster_reflexes(db, row["cognate_id"], target_language)
    else:
        results = _tier2_descent_reflexes(db, etymon_id, target_language)

    # Tier 3 + Tier 4 are sequential fallbacks: each runs only if the
    # prior tier produced nothing. Both surface lower-confidence data
    # so we'd rather return cluster/descent reflexes alone when they
    # exist.
    if not results and target_family_cell is not None:
        results = _tier3_period_form_reflexes(db, etymon_id, target_language, target_family_cell)
    if not results:
        results = _tier4_phonology_reflexes(
            etymon_id, row["canonical_form"], row["language"], target_language
        )

    return results


def _tier1_cluster_reflexes(
    db: LexiconDB, cognate_id: int, target_language: str
) -> list[EraReflex]:
    """Tier 1: cluster-mates query. Selects every etymon sharing
    ``cognate_id`` whose language matches the target. Highest-quality
    reflex source — backed by the D27/D28 cluster_cognates output."""
    cur = db.conn.execute(
        "SELECT id, canonical_form, language FROM etymon "
        "WHERE cognate_id = ? AND language = ? AND merged_into_id IS NULL "
        "ORDER BY canonical_form",
        (cognate_id, target_language),
    )
    return [
        EraReflex(
            etymon_id=r["id"],
            form=r["canonical_form"],
            language=r["language"],
            source="cluster",
        )
        for r in cur
    ]


def _tier2_descent_reflexes(db: LexiconDB, etymon_id: int, target_language: str) -> list[EraReflex]:
    """Tier 2: direct descent fallback. Only walks immediate children
    via inheritance / borrowing edges (peer 'cognate' edges excluded
    — too loose for v1 era-rendering). Deeper traversal would re-
    implement cluster_cognates."""
    cur = db.conn.execute(
        "SELECT DISTINCT child.id, child.canonical_form, child.language "
        "FROM etymon_descent ed "
        "JOIN etymon child ON ed.child_id = child.id "
        "WHERE ed.parent_id = ? "
        "  AND ed.edge_type IN ('inheritance', 'borrowing') "
        "  AND child.language = ? "
        "  AND child.merged_into_id IS NULL "
        "ORDER BY child.canonical_form",
        (etymon_id, target_language),
    )
    return [
        EraReflex(
            etymon_id=r["id"],
            form=r["canonical_form"],
            language=r["language"],
            source="descent",
        )
        for r in cur
    ]


def _tier3_period_form_reflexes(
    db: LexiconDB,
    etymon_id: int,
    target_language: str,
    target_family_cell: tuple[str, str],
) -> list[EraReflex]:
    """Tier 3: period-form projection (wyrd-unuo Phase 3.3). Queries
    ``etymon_period_form`` for projected period forms whose
    ``date_year`` falls in the cell's year range. Returns empty when
    the cell has no registered range. Joins against ``etymon`` to
    filter ``merged_into_id IS NULL`` — OCR-cluster losers' projected
    period forms don't surface; the merge winner is the canonical
    voice for that surface."""
    family, cell = target_family_cell
    try:
        start, end = era_year_range(family, cell)
    except KeyError:
        return []
    # Build year-range filter; treat None bounds as "open on that
    # side" (no constraint).
    clauses = ["pf.etymon_id = ?", "e.merged_into_id IS NULL"]
    params: list[int] = [etymon_id]
    if start is not None:
        clauses.append("pf.date_year >= ?")
        params.append(start)
    if end is not None:
        clauses.append("pf.date_year < ?")
        params.append(end)
    cur = db.conn.execute(
        f"""
        SELECT DISTINCT pf.form, MIN(pf.date_year) AS year
        FROM etymon_period_form pf
        JOIN etymon e ON e.id = pf.etymon_id
        WHERE {" AND ".join(clauses)}
        GROUP BY pf.form
        ORDER BY pf.form
        """,
        params,
    )
    return [
        EraReflex(
            etymon_id=etymon_id,
            form=r["form"],
            language=target_language,
            source="period-form",
        )
        for r in cur
    ]


def _tier4_phonology_reflexes(
    etymon_id: int,
    canonical_form: str,
    source_language: str,
    target_language: str,
) -> list[EraReflex]:
    """Tier 4: phonology-rule fallback (wyrd-98cs). Walks the
    registered sound-change cells (phonology_rules.py: OE→ME,
    ME→EModE, EModE→ModE, OW→ModW) forward or inverse to derive a
    target-era form from ``canonical_form``. Lower precision than
    Tier 1-3 (no mining evidence behind it), so the resulting
    EraReflex carries ``source='phonology-rule:v1'`` for callers
    that want to mark inferred forms differently. Returns empty when
    no rule fires (the form passes through unchanged)."""
    phon_form = phonology_rule_form(canonical_form, source_language, target_language)
    if phon_form is None:
        return []
    return [
        EraReflex(
            etymon_id=etymon_id,
            form=phon_form,
            language=target_language,
            source="phonology-rule:v1",
        )
    ]


# --- wyrd-unuo Phase 3.3: per-etymon period-form projection ---------------
#
# Project per-etymon period-keyed surface forms from
# toponym_attestation rows. For each binary toponym breakdown, find
# the longest suffix of the attested historical form that matches a
# known reflex of the LAST morpheme (canonical_form, cognate-cluster
# mates, etymon_variant rows). The remaining prefix is projected
# onto the FIRST morpheme.
#
# Bradford(1377) "Bradeford" → split as "Brade" + "ford":
#     - "ford" matches OE 'ford' canonical
#     - "Brade" is the projected first-morpheme period form for OE 'brad'
#
# Chesterton(1210) "Cestretone" → split as "Cestre" + "tone":
#     - "tone" matches gmw-msc 'tone' (in OE 'tūn' cluster)
#     - "Cestre" is the projected first-morpheme period form for OE 'ceaster'
#
# v1 limitations:
# - Binary breakdowns only. Ternary (Hadenham → "had + en + ham") is
#   skipped; the alignment ambiguity for middle morphemes makes
#   precision unreliable without phonetic alignment.
# - Suffix-anchoring only. First-morpheme prefix-anchoring would
#   double the matchable population but introduces FPs when the
#   modern compound starts with a common scholarly word (e.g.
#   'New' / 'Old' / 'Saint' prefixes).
# - No phonological-distance alignment. Forms that drifted
#   orthographically from any cognate-cluster mate (rare in our
#   corpus but real) are skipped.


def _strip_diacritics(text: str) -> str:
    """NFKD-decompose ``text`` and drop combining marks. Used for
    suffix matching where 'tūn' (with macron) should match against
    a historical 'tun' surface."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _suffix_candidates_for_etymon(
    db: LexiconDB,
    etymon_id: int,
    canonical_form: str,
    cognate_id: int | None,
) -> set[str]:
    """Build the set of forms a historical-form suffix may match
    against for THIS etymon: canonical_form + cognate-cluster mates
    (any language; gmw-msc / Middle Scots forms like 'tone' are real
    historical reflexes that survive into the broader cognate set) +
    etymon_variant rows.

    All forms returned lowercased. Both diacritic-bearing and
    diacritic-stripped variants are included so 'tūn' and 'tun' both
    match against historical 'tun'-suffix forms.
    """
    candidates: set[str] = set()

    def _add(form: str) -> None:
        if not form or len(form) < 2:
            return
        candidates.add(form.lower())
        candidates.add(_strip_diacritics(form.lower()))

    _add(canonical_form)
    if cognate_id is not None:
        cur = db.conn.execute(
            "SELECT canonical_form FROM etymon WHERE cognate_id = ? AND merged_into_id IS NULL",
            (cognate_id,),
        )
        for row in cur:
            _add(row["canonical_form"])
    cur = db.conn.execute(
        "SELECT form FROM etymon_variant WHERE etymon_id = ?",
        (etymon_id,),
    )
    for row in cur:
        _add(row["form"])
    return candidates


def _find_longest_suffix_match(attested_form: str, candidates: set[str]) -> str | None:
    """Return the longest suffix of ``attested_form`` (preserving
    casing) that matches any entry in ``candidates`` (case-insensitive
    + diacritic-insensitive). Returns None when no candidate is a
    suffix of the attested form.

    Matches are minimum-length 2 to avoid spurious 1-char hits
    (a trailing 's' / 'e' is too noisy to project as a morpheme
    surface).
    """
    af_lower = attested_form.lower()
    af_stripped = _strip_diacritics(af_lower)
    best_len = 0
    for cand in candidates:
        if len(cand) < 2:
            continue
        if (af_lower.endswith(cand) or af_stripped.endswith(cand)) and len(cand) > best_len:
            best_len = len(cand)
    if best_len == 0:
        return None
    return attested_form[-best_len:]


def project_period_forms(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 200,
) -> dict:
    """Project per-etymon period-keyed surface forms from
    toponym_attestation rows (wyrd-unuo Phase 3.3). For each binary
    toponym breakdown, segment the attested form's suffix against the
    last morpheme's known reflexes; the remaining prefix is the first
    morpheme's projected period form.

    v1 limits (mirroring the module-level comment block above for
    callers that read help() / IDE hovers):

    * **Binary breakdowns only** — ternary+ skipped because middle-
      morpheme alignment isn't reliable without phonetic distance.
    * **Suffix-anchoring only** — last-morpheme matched against
      cluster mates / variants; first-morpheme is whatever's left
      after the suffix split.
    * **Skip merged-into etymons** — toponyms whose breakdown
      points at OCR-cluster losers (merged_into_id IS NOT NULL)
      are skipped so loser etymons don't accumulate stale period-
      form rows that then surface via the era-reflex Tier 3.
    * **Reasonability gates** — both projected segments must be
      ≥2 chars; sub-2-char prefixes / suffixes are noise.

    Idempotent via the unique index ``idx_period_form_unique``;
    re-runs are no-ops on already-projected (etymon, form, year,
    source_doc) tuples.

    Progress lines emit to stderr every ``progress_every`` rows
    (CLAUDE.md mining-progress shape). ``progress_every`` clamps to
    ≥1.

    Returns a dict with row-counts so callers can report.
    """
    import sys
    import time

    progress_every = max(progress_every, 1)
    started = time.monotonic()

    cur = db.conn.execute(
        "SELECT id, toponym_id, form, date_year, source_doc "
        "FROM toponym_attestation "
        "WHERE date_year IS NOT NULL "
        "ORDER BY toponym_id"
    )

    rows_scanned = 0
    rows_projected = 0
    candidate_inserts: list[tuple[int, str, int, str | None, int]] = []
    # Eager-load all binary breakdowns in one query (instead of one
    # per toponym_id during the scan loop). At ~700 unique toponyms
    # the per-row query was ~500ms cumulative; the eager-load is
    # ~50ms total. Scales linearly to 100k+ toponyms without
    # hitting the per-row cliff.
    cached_breakdowns: dict[int, list[list[dict]]] = _preload_binary_breakdowns(db)
    cached_candidates: dict[int, set[str]] = {}

    for ta in cur:
        rows_scanned += 1
        toponym_id = ta["toponym_id"]
        breakdowns = cached_breakdowns.get(toponym_id, [])
        for breakdown in breakdowns:
            first, last = breakdown[0], breakdown[1]
            if last["etymon_id"] not in cached_candidates:
                cached_candidates[last["etymon_id"]] = _suffix_candidates_for_etymon(
                    db,
                    last["etymon_id"],
                    last["canonical_form"],
                    last["cognate_id"],
                )
            candidates = cached_candidates[last["etymon_id"]]
            match = _find_longest_suffix_match(ta["form"], candidates)
            if match is None:
                continue
            last_form = match
            first_form = ta["form"][: len(ta["form"]) - len(match)]
            # Reasonability gates: both projected segments must be
            # ≥2 chars (single-letter prefixes / suffixes are noise).
            if len(first_form) < 2 or len(last_form) < 2:
                continue
            candidate_inserts.append(
                (first["etymon_id"], first_form, ta["date_year"], ta["source_doc"], ta["id"])
            )
            candidate_inserts.append(
                (last["etymon_id"], last_form, ta["date_year"], ta["source_doc"], ta["id"])
            )
            rows_projected += 1
        if rows_scanned % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"  [{rows_scanned}]  rows_projected={rows_projected} "
                f"candidates={len(candidate_inserts)} "
                f"({elapsed / rows_scanned:.4f}s/entry)",
                file=sys.stderr,
                flush=True,
            )

    rows_written = 0
    if apply and candidate_inserts:
        result = db.conn.executemany(
            "INSERT OR IGNORE INTO etymon_period_form "
            "(etymon_id, form, date_year, source_doc, attestation_id) "
            "VALUES (?, ?, ?, ?, ?)",
            candidate_inserts,
        )
        rows_written = result.rowcount
        db.commit()

    elapsed = max(time.monotonic() - started, 1e-6)
    print(
        f"  [{rows_scanned}/{rows_scanned}]  rows_projected={rows_projected} "
        f"candidates={len(candidate_inserts)} rows_written={rows_written} "
        f"({elapsed / max(rows_scanned, 1):.4f}s/entry)",
        file=sys.stderr,
        flush=True,
    )

    return {
        "rows_scanned": rows_scanned,
        "rows_projected": rows_projected,
        "candidates": len(candidate_inserts),
        "rows_written": rows_written,
        "applied": apply,
    }


def _preload_binary_breakdowns(db: LexiconDB) -> dict[int, list[list[dict]]]:
    """Eager-load every binary breakdown for toponyms that have at
    least one toponym_attestation row. Returns ``{toponym_id:
    [breakdown, ...]}`` where each breakdown is a list of 2 element
    dicts ordered by ordinal.

    Single SQL query (vs. ``_get_binary_breakdowns`` called per
    toponym during the scan loop) — avoids the N+1 pattern when
    project_period_forms iterates the attestation table. Filters
    out breakdowns whose components are merged_into-id tombstones,
    same rule as the per-toponym helper.
    """
    cur = db.conn.execute(
        """
        SELECT te.toponym_id, te.id AS te_id, tee.ordinal, tee.etymon_id,
               e.canonical_form, e.language, e.cognate_id, e.merged_into_id
        FROM toponym_etymology te
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON tee.etymon_id = e.id
        WHERE te.toponym_id IN (
            SELECT DISTINCT toponym_id FROM toponym_attestation
            WHERE date_year IS NOT NULL
        )
        ORDER BY te.toponym_id, te.id, tee.ordinal
        """
    )
    grouped: dict[tuple[int, int], list[dict]] = {}
    skip_te_keys: set[tuple[int, int]] = set()
    for row in cur:
        key = (row["toponym_id"], row["te_id"])
        if row["merged_into_id"] is not None:
            skip_te_keys.add(key)
            continue
        grouped.setdefault(key, []).append(
            {
                "ordinal": row["ordinal"],
                "etymon_id": row["etymon_id"],
                "canonical_form": row["canonical_form"],
                "language": row["language"],
                "cognate_id": row["cognate_id"],
            }
        )
    out: dict[int, list[list[dict]]] = {}
    for (topo_id, te_id), elements in grouped.items():
        if len(elements) == 2 and (topo_id, te_id) not in skip_te_keys:
            out.setdefault(topo_id, []).append(elements)
    return out


def _get_binary_breakdowns(db: LexiconDB, toponym_id: int) -> list[list[dict]]:
    """Return all toponym_etymology breakdowns for ``toponym_id``
    that have exactly 2 elements (binary breakdowns), ordered by
    ordinal. Each breakdown is a list of dicts.

    Skips ternary+ breakdowns since v1's suffix-anchoring algorithm
    only handles the binary case reliably. Multiple breakdowns per
    toponym are common (different scholarly proposals); each is
    projected independently.

    Filters out breakdowns where ANY element points at a
    merged_into_id-tagged tombstone — projecting a period form to
    a loser etymon would surface stale rows via Tier 3 of the
    era-reflex picker. The OCR-cluster winner is the canonical
    voice; if a downstream pass needs the loser's projections,
    it should re-resolve via the merge chain.
    """
    cur = db.conn.execute(
        """
        SELECT te.id AS te_id, tee.ordinal, tee.etymon_id,
               e.canonical_form, e.language, e.cognate_id,
               e.merged_into_id
        FROM toponym_etymology te
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON tee.etymon_id = e.id
        WHERE te.toponym_id = ?
        ORDER BY te.id, tee.ordinal
        """,
        (toponym_id,),
    )
    grouped: dict[int, list[dict]] = {}
    skip_te_ids: set[int] = set()
    for row in cur:
        if row["merged_into_id"] is not None:
            skip_te_ids.add(row["te_id"])
            continue
        grouped.setdefault(row["te_id"], []).append(
            {
                "ordinal": row["ordinal"],
                "etymon_id": row["etymon_id"],
                "canonical_form": row["canonical_form"],
                "language": row["language"],
                "cognate_id": row["cognate_id"],
            }
        )
    return [bd for te_id, bd in grouped.items() if len(bd) == 2 and te_id not in skip_te_ids]


_COGNATE_BRIDGING_EDGES = ("inheritance", "borrowing")
_CLUSTER_COGNATES_METHOD = "cluster-cognates-v1"


def cluster_cognates(db: LexiconDB, *, apply: bool = False) -> dict:
    """Walk etymon_descent inheritance + borrowing edges from each root
    and assign cognate_id to every reachable descendant (D27 / wyrd-81n).

    A "root" is an etymon that participates in inheritance/borrowing
    descent edges as a parent but never as a child — i.e. the most-
    ancestral known form in its cognate chain (typically a Proto-* form).
    Every descendant reachable via inheritance or borrowing edges from
    that root gets cognate_id = root.id, so cross-language cognates
    cluster behind a single canonical pointer.

    Edge type semantics (D27):
      inheritance — direct lineage. Bridges cognate cluster.
      borrowing   — borrowed across language lines. Bridges cognate cluster (a
                    borrowed word IS part of the borrowing language's
                    cognate set in practice).
      cognate     — peer relation, NOT a chain. Does not bridge — would
                    over-unify, since Wiktionary's cognate sections
                    sometimes cross probable-but-unproven boundaries.
      derivation, calque, compound, unknown — context-specific; treat
                    as non-bridging for v1, refine if mining surfaces
                    a clear case.

    Determinism: when an etymon is reachable from multiple roots (rare;
    happens when scholars disagree on the chain), the smallest root id
    wins. Iteration order is sorted by root id so the assignment is
    bit-stable across runs.

    Operates in CANONICAL space (per D22 / wyrd-223): edges'
    parent_id and child_id are resolved through merged_into_id before
    use, so a descent edge that points at an OCR-merge tombstone is
    treated as if it pointed at the canonical winner. cognate_id is
    written on canonical etymons only; tombstones stay NULL (and are
    rolled up at query time via merged_into_id chain). This bridges
    cross-source canonical-form mismatches like wiktextract's
    `tun` vs the place-name dictionaries' `tūn`.

    With apply=False (default) reports candidate counts without writing.
    With apply=True writes cognate_id + cognate_method='cluster-cognates-v1'
    only on rows whose current (cognate_id, cognate_method) doesn't already
    match the target — re-runs against unchanged data become real no-ops
    instead of redundant UPDATEs. Reverse with
    `clear-enrichment --stage=cognates --apply`.

    Returns a dict of:
      - roots: number of canonical root etymons walked
      - candidates: total canonical etymons that received (or would
        receive) a cognate_id assignment
      - applied: whether writes happened
      - rows_written: count of UPDATE statements that actually changed
        a row (always 0 in dry-run; ≤ candidates when applied)
      - cycle_orphans: count of canonical etymons that participate in
        bridging edges but couldn't be assigned because they sit in a
        cycle with no external root. Healthy data should report 0 here;
        non-zero means the descent graph has at least one closed loop
        with no anchor — worth flagging in operator output.
    """
    # f-string interpolation of `placeholders` is safe here:
    # _COGNATE_BRIDGING_EDGES is a module-level tuple of code-controlled
    # strings, never user input, so no SQL injection risk. Edge values
    # themselves are bound via parameters.
    placeholders = ", ".join(["?"] * len(_COGNATE_BRIDGING_EDGES))

    # Build the canonical edge set: every descent edge with both
    # endpoints resolved through merged_into_id. Returns
    # (parent_canon_id, child_canon_id) tuples. Self-loops introduced
    # by the resolution (parent and child both merge into the same
    # canonical) are filtered out — they'd waste a BFS node and offer
    # zero clustering signal.
    canonical_edges = [
        (row["parent_canon"], row["child_canon"])
        for row in db.conn.execute(
            f"""
            SELECT
              COALESCE(p.merged_into_id, d.parent_id) AS parent_canon,
              COALESCE(c.merged_into_id, d.child_id) AS child_canon
            FROM etymon_descent d
            JOIN etymon p ON p.id = d.parent_id
            JOIN etymon c ON c.id = d.child_id
            WHERE d.edge_type IN ({placeholders})
            """,
            _COGNATE_BRIDGING_EDGES,
        ).fetchall()
        if row["parent_canon"] != row["child_canon"]
    ]

    # Build child-of-parent index for the BFS. Parent → set of children.
    children_by_parent: dict[int, set[int]] = {}
    bridging_participants: set[int] = set()
    parent_set: set[int] = set()
    child_set: set[int] = set()
    for parent_id, child_id in canonical_edges:
        children_by_parent.setdefault(parent_id, set()).add(child_id)
        bridging_participants.add(parent_id)
        bridging_participants.add(child_id)
        parent_set.add(parent_id)
        child_set.add(child_id)

    # Roots: canonical etymons that appear as parent but never as
    # child in the canonical edge set.
    roots = sorted(parent_set - child_set)

    assignments: dict[int, int] = {}
    for root_id in roots:
        if root_id in assignments:
            # Already claimed by an earlier (smaller-id) root via cross-edges.
            continue
        assignments[root_id] = root_id
        frontier: list[int] = [root_id]
        while frontier:
            next_frontier: list[int] = []
            for node_id in frontier:
                for child_id in children_by_parent.get(node_id, ()):
                    if child_id not in assignments:
                        assignments[child_id] = root_id
                        next_frontier.append(child_id)
            frontier = next_frontier

    # Cycle-orphans: canonical etymons that participate in bridging
    # edges but never reached a root. A pure cycle has no anchor.
    cycle_orphans = bridging_participants - set(assignments.keys())

    rows_written = 0
    if apply:
        for etymon_id, cognate_id in assignments.items():
            cur = db.conn.execute(
                "UPDATE etymon SET cognate_id = ?, cognate_method = ? "
                "WHERE id = ? "
                "  AND (cognate_id IS NOT ? OR cognate_method IS NOT ?)",
                (
                    cognate_id,
                    _CLUSTER_COGNATES_METHOD,
                    etymon_id,
                    cognate_id,
                    _CLUSTER_COGNATES_METHOD,
                ),
            )
            rows_written += cur.rowcount
        db.commit()

    return {
        "roots": len(roots),
        "candidates": len(assignments),
        "applied": apply,
        "rows_written": rows_written,
        "cycle_orphans": len(cycle_orphans),
    }


# --- generic-language bridging --------------------------------------------


def bridge_generic_language(
    db: LexiconDB,
    *,
    generic_lang: str,
    candidate_langs: tuple[str, ...],
    apply: bool = False,
) -> dict:
    """Bridge a generic-language etymon (e.g. 'celtic') to the matching
    specific-language entry (e.g. 'irish' / 'welsh' / 'old-irish') by
    setting merged_into_id (D22 OCR-cluster style — non-destructive).

    Place-name dictionaries often write generic language tags like
    'celtic' for morphemes whose specific Celtic-family origin isn't
    pinned (or doesn't matter to the place-name analysis). Wiktextract
    entries are language-specific. This pass bridges the lookup
    mismatch by canonicalizing each generic-language etymon onto a
    specific-language match with the same canonical_form.

    For each generic-language etymon (canonical, not already merged):
      - Look up specific-language etymons matching `LOWER(canonical_form)`
        across `candidate_langs`.
      - If matches exist, pick the highest-priority one per
        `candidate_langs` order (typically Proto-* > Old-* > Middle-* >
        modern). Within a language, pick the smallest etymon id for
        determinism.
      - Set the generic-language etymon's `merged_into_id` to the picked
        canonical winner.

    The cluster-cognates pass is already redirect-aware, so any descent
    edges that previously pointed at the specific-language etymon now
    also cluster the merged generic etymon via the merged_into_id rollup
    chain.

    With apply=False (default) reports candidate counts without writing.
    Returns:
      - generic_etymons: total canonical generic-language etymons examined
      - bridged: count that found a specific-language match (would
        merge / did merge)
      - unmatched: count with no specific-language candidate (will
        remain as standalone canonical entries)
      - rows_written: actual UPDATE row count when apply=True (always 0
        in dry-run)
    """
    if not candidate_langs:
        raise ValueError("candidate_langs must be non-empty")

    # Build a (lower(canonical_form), language) → smallest_id lookup over
    # canonical specific-language etymons. One pass over the candidate
    # set keeps the bridging O(N) on the generic-language list.
    placeholders = ", ".join(["?"] * len(candidate_langs))
    candidate_index: dict[tuple[str, str], int] = {}
    for row in db.conn.execute(
        f"""
        SELECT id, canonical_form, language
        FROM etymon
        WHERE merged_into_id IS NULL
          AND language IN ({placeholders})
        ORDER BY id
        """,
        candidate_langs,
    ).fetchall():
        key = (row["canonical_form"].lower(), row["language"])
        # First-seen wins (smallest id) per (form, lang) — within a
        # language the lower id is the older entry, deterministic.
        if key not in candidate_index:
            candidate_index[key] = row["id"]

    # Walk generic-language canonical etymons; pick the first matching
    # candidate language per the priority order in `candidate_langs`.
    generic_rows = db.conn.execute(
        "SELECT id, canonical_form FROM etymon "
        "WHERE language = ? AND merged_into_id IS NULL ORDER BY id",
        (generic_lang,),
    ).fetchall()

    bridges: list[tuple[int, int]] = []  # (generic_id, target_id)
    for gen_row in generic_rows:
        form_lower = gen_row["canonical_form"].lower()
        for cand_lang in candidate_langs:
            target_id = candidate_index.get((form_lower, cand_lang))
            if target_id is not None:
                bridges.append((gen_row["id"], target_id))
                break

    rows_written = 0
    if apply and bridges:
        # Mirror cluster_ocr_variants's chain-flatten (D22): set
        # merged_into_id on the generic row AND re-route any
        # pre-existing redirect that pointed AT the generic row.
        # Without the OR-clause a 2-deep chain X → generic → target
        # would form, and the single-level COALESCE rollup in
        # etymon_consensus / etymon_canonical would split witnesses
        # across two GROUP BY buckets.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, gid, gid, tid) for gid, tid in bridges],
        )
        rows_written = cur.rowcount
        # Re-parent any inflected children the generic row was acting
        # as a lemma for. Mining evidence stays on the original etymon.
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, gid) for gid, tid in bridges],
        )
        db.commit()

    return {
        "generic_etymons": len(generic_rows),
        "bridged": len(bridges),
        "unmatched": len(generic_rows) - len(bridges),
        "rows_written": rows_written,
        "applied": apply,
    }


# --- inflected/Anglicized 'celtic' bridging -------------------------------


# Hand-curated mapping from "celtic place-name form as scholar dictionaries
# write it" → "Wiktionary canonical lemma form (any Goidelic/Brythonic
# language)".
#
# Place-name etymologies tagged 'celtic' surface in three shapes:
#   1. Goidelic genitive / dative / vocative inflections (choill = gen of
#      coill 'wood'; tairbh = gen of tarbh 'bull'; tulaigh = gen/dat of
#      tulach 'hill').
#   2. Eclipsis or lenition consonant mutations (gcorr = eclipsed corr
#      'crane'; mhic = lenited mac 'son'; chluana = lenited gen of cluain).
#   3. Anglicized spellings (drum from druim 'ridge'; lough from loch;
#      kin from cinn — gen of ceann 'head'; cashel from caiseal).
#
# bridge_generic_language only handles same-form lookups, so it
# misses all three. This table maps each high-witness inflected/Anglicized
# form to the dictionary lemma that Wiktionary keys on. The lookup is then
# searched across the celtic candidate languages, preferring the candidate
# whose target is in a cognate cluster.
#
# Convention: keys are lowercase celtic place-name forms; values are the
# lemma form (typically Modern Irish / Welsh / Scottish-Gaelic — the modern
# Wiktionary headword orthography).
_CELTIC_FORM_BRIDGES: dict[str, str] = {
    # Goidelic genitive of common landscape nouns
    "choill": "coill",  # gen of coill 'wood'
    "coille": "coill",
    "coillte": "coill",
    "cluana": "cluain",  # gen of cluain 'meadow'
    "chluana": "cluain",  # lenited gen
    "achaidh": "achadh",  # gen of achadh 'field'
    "fearna": "fearn",  # gen of fearn 'alder'
    "easa": "eas",  # gen of eas 'waterfall'
    "gart": "gort",  # gen of gort 'tilled field'
    "inse": "inis",  # gen of inis 'island'
    "innis": "inis",
    "luachra": "luachair",  # gen of luachair 'rushes'
    "tairbh": "tarbh",  # gen of tarbh 'bull'
    "thairbh": "tarbh",  # lenited
    "tulaigh": "tulach",  # gen/dat of tulach 'hill'
    "cnuic": "cnoc",  # gen of cnoc 'hill'
    "capaill": "capall",  # gen of capall 'horse'
    "caorach": "caora",  # gen pl of caora 'sheep'
    "cille": "cill",  # gen of cill 'church'
    "cloiche": "cloch",  # gen of cloch 'stone'
    "coraidh": "cora",  # gen of cora 'weir'
    "croiche": "croch",  # gen of croch 'cross/gallows'
    "croise": "cros",  # gen of cros 'cross'
    "curra": "currach",  # gen of currach 'marsh'
    "daingin": "daingean",  # gen of daingean 'fortress'
    "draoighin": "draighean",  # blackthorn (gen)
    "brighe": "brí",  # gen of brí 'hill'
    "craoibhe": "craobh",  # gen of craobh 'branch/tree'
    "cairn": "carn",  # gen pl of carn 'cairn'
    "cinn": "ceann",  # gen of ceann 'head'
    "aodha": "aodh",  # gen of Aodh (personal name root)
    # Eclipsis (n-/m-/g-/d-/b- prefixed by negation/genitive triggers)
    "gcorr": "corr",  # eclipsed corr 'crane'
    "mban": "bean",  # eclipsed gen pl of bean 'woman'
    "mhic": "mac",  # lenited gen of mac 'son'
    "dhoire": "doire",  # lenited doire 'oak grove'
    # Anglicized place-name spellings
    "drum": "druim",  # Anglicized of druim 'ridge'
    "kin": "ceann",  # Anglicized of cinn (gen of ceann)
    "lough": "loch",  # Anglicized of loch
    "more": "mór",  # Anglicized of mór 'great'
    "boher": "bóthar",  # Anglicized of bóthar 'road'
    "caher": "cathair",  # Anglicized of cathair 'fort'
    "cashel": "caiseal",  # Anglicized of caiseal 'stone fort'
    "carrig": "carraig",  # variant of carraig 'rock'
    "craig": "carraig",
    "creag": "carraig",
    "derry": "doire",  # Anglicized of doire
    "baun": "bán",  # Anglicized of bán 'white'
    "buidhe": "buí",  # older spelling of buí 'yellow'
    "ruadh": "rua",  # older spelling of rua 'red'
    "magh": "mag",  # plain — older spelling
    "maigh": "mag",  # gen/dat of magh
    "din": "dún",  # variant of dún 'fort'
    "clon": "cluain",  # Anglicized of cluain
    "cloon": "cluain",
    "cnocan": "cnocán",  # diminutive of cnoc
    # Other landscape forms (lemma matches a different Wiktionary spelling)
    "ath": "áth",  # ford
    "beal": "béal",  # mouth/opening
    "beinn": "beann",  # peak/mountain
    "clar": "clár",  # plain/board
    "cu": "cú",  # hound
    "eudan": "éadan",  # face/forehead
    "leamhan": "leamhán",  # elm
    "suidhe": "suí",  # seat
    "airgeat": "airgead",  # silver
    "aluin": "álainn",  # beautiful
    "brean": "bréan",  # foul-smelling
}


def bridge_celtic_forms(
    db: LexiconDB,
    *,
    apply: bool = False,
    table: dict[str, str] | None = None,
    candidate_langs: tuple[str, ...] = (
        "irish",
        "scottish-gaelic",
        "manx",
        "old-irish",
        "middle-irish",
        "welsh",
        "old-welsh",
        "middle-welsh",
        "breton",
        "old-breton",
        "middle-breton",
        "cornish",
        "proto-celtic",
    ),
) -> dict:
    """Bridge celtic place-name etymons to Wiktionary lemmas via a
    hand-curated form→lemma table.

    Differs from `bridge_generic_language` in three ways:
      1. Uses a curated form→lemma table (so Goidelic inflections,
         eclipsis/lenition, and Anglicized spellings can be mapped to
         their Wiktionary headwords).
      2. Iterates ALL celtic rows (including pre-existing tombstones)
         so the chain-flatten can re-route an existing stub-bridge
         (celtic→old-irish that has no cognate cluster) to a clustered alternative
         (celtic→irish with a cognate cluster).
      3. Among multiple matching candidate languages, prefers the one
         whose target is in a cognate cluster (cognate_id IS NOT NULL),
         falling back to the priority order in `candidate_langs`.

    Default `candidate_langs` is biased toward MODERN reflexes (Irish,
    Scottish-Gaelic, Welsh) ahead of Old-* / Proto- because Wiktionary's
    Celtic etymology coverage is denser at the modern end — those entries
    are the ones with descent edges that cluster-cognates can walk.

    Returns:
      - examined: total celtic etymons examined (canonical + tombstones)
      - bridged: count that found a (preferred) target
      - unmatched: count with form not in the bridge table
      - missing_target: count where the table named a lemma that doesn't
        exist as a canonical etymon in any candidate language
      - rows_written: actual UPDATE row count when apply=True
    """
    table = _CELTIC_FORM_BRIDGES if table is None else table

    # Build (lower(canonical_form), language) → (live_canonical_id, cognate_id)
    # over canonical candidate-language etymons. We resolve through any
    # merged_into_id chain so a target form named in the table that has
    # itself been OCR-merged into a canonical still routes correctly.
    placeholders = ", ".join(["?"] * len(candidate_langs))
    candidate_index: dict[tuple[str, str], tuple[int, int | None]] = {}
    for row in db.conn.execute(
        f"""
        SELECT id, canonical_form, language, merged_into_id, cognate_id
        FROM etymon
        WHERE language IN ({placeholders})
        ORDER BY id
        """,
        candidate_langs,
    ).fetchall():
        # Resolve through redirect (one hop is enough for our data shape;
        # OCR-cluster passes don't produce multi-step chains within a
        # single language).
        live_id = row["id"]
        live_synset = row["cognate_id"]
        if row["merged_into_id"] is not None:
            live = db.conn.execute(
                "SELECT id, cognate_id FROM etymon WHERE id = ?",
                (row["merged_into_id"],),
            ).fetchone()
            if live is not None:
                live_id = live["id"]
                live_synset = live["cognate_id"]
        key = (row["canonical_form"].lower(), row["language"])
        # First-seen wins per (form, lang). The ORDER BY id keeps it
        # deterministic (older entry wins).
        if key not in candidate_index:
            candidate_index[key] = (live_id, live_synset)

    # Walk ALL celtic rows (canonical + tombstones). Tombstones whose
    # current target is unclustered get re-routed by the chain-flatten
    # OR-clause to the clustered candidate this pass picks.
    celtic_rows = db.conn.execute(
        "SELECT id, canonical_form, merged_into_id "
        "FROM etymon WHERE language = 'celtic' ORDER BY id"
    ).fetchall()

    bridges: list[tuple[int, int]] = []  # (celtic_id, target_id)
    missing_target = 0
    unmatched = 0
    for row in celtic_rows:
        form = row["canonical_form"].lower()
        lemma = table.get(form)
        if lemma is None:
            unmatched += 1
            continue

        # Find the best candidate: prefer clustered targets (cognate_id IS
        # NOT NULL) in priority order, fall back to first-found unclustered.
        best_clustered: int | None = None
        best_unclustered: int | None = None
        for cand_lang in candidate_langs:
            entry = candidate_index.get((lemma.lower(), cand_lang))
            if entry is None:
                continue
            tid, syn = entry
            if syn is not None and best_clustered is None:
                best_clustered = tid
                break  # first clustered wins
            if best_unclustered is None:
                best_unclustered = tid
        target_id = best_clustered if best_clustered is not None else best_unclustered

        if target_id is None:
            missing_target += 1
            continue
        if target_id == row["id"]:
            continue  # bridging to self is a no-op
        bridges.append((row["id"], target_id))

    rows_written = 0
    if apply and bridges:
        # Chain-flatten + lemma-reparent in batch. The OR-clause re-routes
        # any pre-existing redirect that pointed AT this celtic etymon
        # (e.g. an existing stub-bridge) onto the freshly-picked
        # clustered target, preventing a 2-deep chain that would split
        # witnesses in the single-level COALESCE rollup.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, sid, sid, tid) for sid, tid in bridges],
        )
        rows_written = cur.rowcount
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, sid) for sid, tid in bridges],
        )
        db.commit()

    return {
        "examined": len(celtic_rows),
        "bridged": len(bridges),
        "unmatched": unmatched,
        "missing_target": missing_target,
        "rows_written": rows_written,
        "applied": apply,
    }


# --- phonological bridging for OE place-name forms ------------------------


# Hand-curated mapping from "scholarly OE form as place-name dictionaries
# write it" → "Wiktionary canonical OE form". Wiktionary uses formal
# scholarly orthography (macrons, æ, weak-final consonants); place-name
# dictionaries write modernized / Norman-Anglicized spellings inherited
# from medieval scribal practice. The mapping captures the most common
# pairs that surface in the high-witness end of our place-name corpus.
# Extend the table when new high-witness mismatches surface.
#
# Convention: keys are lowercase scholarly forms; values are the
# Wiktionary canonical (with macrons, æ, etc.). The bridge uses
# merged_into_id (D22 non-destructive) — no mining evidence is
# destroyed.
_OE_PHONOLOGICAL_BRIDGES: dict[str, str] = {
    # -tūn / settlement family
    "ton": "tūn",
    "tun": "tūn",
    "tone": "tūn",
    # -lēah / clearing family
    "lea": "lēah",
    "leah": "lēah",
    "leak": "lēah",  # OCR / spelling variant
    "ley": "lēah",
    "ly": "lēah",
    # -cot / -cote (cottage)
    "cote": "cot",
    "cotum": "cot",  # dative-pl
    # -burh / fortified-place family
    "burgh": "burh",
    "bury": "burh",
    "byrig": "burh",  # dative-sg of burh
    # -dæl / valley
    "dale": "dæl",
    # -heall / hall
    "hall": "heall",
    # -ieg / island. Wiktionary's headword is `īeg` but the macron-stripped
    # `ieg` is what the OCR-normalize pass treats as canonical; `īeg`
    # itself is not in the corpus. Snap value to the live form.
    "ey": "ieg",
    "eg": "ieg",
    # -burna / stream
    "burn": "burna",
    "burne": "burna",
    # -healh / nook
    "hale": "healh",
    "halh": "healh",
    # -nīwe / new
    "new": "nīwe",
    # -ōra / shore, edge
    "ore": "ōra",
    # -wudu / wood
    "wood": "wudu",
    "wode": "wudu",
    # -brād / broad
    "brade": "brād",
    # -brycg / bridge
    "bridge": "brycg",
    # -stān / stone
    "stone": "stān",
    # -hyll / hill
    "hill": "hyll",
    # -hyrst / wooded hill
    "hurst": "hyrst",
    # -hlāw / mound
    "low": "hlāw",
    # -mos / moss/marsh
    "moss": "mos",
    # -pōl / pool — bridge value snapped to 'pol' (the live canonical;
    # 'pōl' is itself a tombstone merged into 'pol' by normalize-ocr).
    "pool": "pol",
    # -sealh / willow
    "salh": "sealh",
    # -stede / place
    "stead": "stede",
    # -wella / spring
    "well": "wella",
    # -ing / -ingas (people-of suffix; place-name dicts often drop hyphen)
    "ing": "-ing",
    "ingas": "-ingas",
    # æcer / acre
    "acre": "æcer",
    # æsc / ash
    "ash": "æsc",
    # -hām / homestead
    "ham": "hām",
    # -wella / spring (additional inflected/spelling variants beyond 'well')
    "wella": "welle",
    # -ing-family additions: scholar-spelled variants of the people-of suffix
    "inga": "ing",
    # -cipp / log
    "cippa": "cipp",
    # -hār / grey, hoary
    "hāran": "hār",
    # -clopp / lump, hill
    "cloppa": "clopp",
    # -ticcen / kid (young goat)
    "ticce": "ticcen",
    # -cirice / church (kirk is the Northumbrian / Norse-influenced form)
    "kirk": "cirice",
    # -healh / nook (hala is a common scholarly variant)
    "hala": "healh",
    # -scylfe / shelf, ledge
    "scelf": "scylfe",
    # -ieg / island (alternative scholarly spelling)
    "ēg": "ieg",
    # -hara / hare
    "hare": "hara",
    # -hæsel / hazel (relies on redirect-follow to canonical haesel)
    "hasel": "hæsel",
    # -hlīep / leap (Hartlepool's "harts-leap" element)
    "hlyp": "hlīep",
    # -lacu / stream (lech is also occasionally read as 'lece' = bog;
    # the place-name reading is dominantly the water sense, hence lacu)
    "lech": "lacu",
}


def _bridge_same_language_phonological(
    db: LexiconDB, *, language: str, table: dict[str, str], apply: bool
) -> dict:
    """Shared engine for same-language phonological bridges.

    OE and ON place-name etymologies both suffer from the same structural
    mismatch against Wiktionary: scholar place-name dictionaries write
    modernized / Anglicized spellings (ton, lea, by, holm) while
    Wiktionary uses scholarly orthography with macrons, æ, þ/ð, etc.
    (tūn, lēah, býr, hólmr). Each language gets its own hand-curated
    bridge table; this function applies one against the canonical rows
    of `language`, walking merged_into_id chains so a bridge value that
    names a tombstone still resolves to the live canonical.

    Returns the same shape as the public bridge_phonological_* wrappers.
    """
    all_rows = db.conn.execute(
        "SELECT id, canonical_form, merged_into_id FROM etymon WHERE language = ? ORDER BY id",
        (language,),
    ).fetchall()
    chain: dict[int, int | None] = {r["id"]: r["merged_into_id"] for r in all_rows}

    def _resolve_canonical(start_id: int) -> int:
        cid = start_id
        visited: set[int] = set()
        while (next_id := chain.get(cid)) is not None and cid not in visited:
            visited.add(cid)
            cid = next_id
        return cid

    target_index: dict[str, int] = {}
    for row in all_rows:
        key = row["canonical_form"].lower()
        if key not in target_index:
            target_index[key] = _resolve_canonical(row["id"])

    examined_rows = [r for r in all_rows if r["merged_into_id"] is None]

    bridges: list[tuple[int, int]] = []
    missing_target = 0
    for row in examined_rows:
        form = row["canonical_form"].lower()
        wiktionary_form = table.get(form)
        if wiktionary_form is None:
            continue
        target_id = target_index.get(wiktionary_form.lower())
        if target_id is None:
            missing_target += 1
            continue
        if target_id == row["id"]:
            continue
        bridges.append((row["id"], target_id))

    rows_written = 0
    if apply and bridges:
        # Chain-flatten + lemma-reparent in batch (mirrors
        # cluster_ocr_variants's D22 pattern). The OR-clause re-routes
        # any pre-existing redirect that pointed AT this place-name
        # etymon onto the canonical target, preventing a 2-deep chain
        # that would split witnesses in the single-level COALESCE
        # rollup used by etymon_consensus / etymon_canonical.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, sid, sid, tid) for sid, tid in bridges],
        )
        rows_written = cur.rowcount
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, sid) for sid, tid in bridges],
        )
        db.commit()

    return {
        "examined": len(examined_rows),
        "bridged": len(bridges),
        "unmatched": len(examined_rows) - len(bridges) - missing_target,
        "missing_target": missing_target,
        "rows_written": rows_written,
        "applied": apply,
    }


def bridge_phonological_oe(db: LexiconDB, *, apply: bool = False) -> dict:
    """Bridge OE place-name etymons to their Wiktionary canonical
    equivalents via a hand-curated mapping table.

    Place-name dictionaries write modernized OE forms (`ton`, `lea`,
    `burgh`, `dale`); Wiktionary uses scholarly orthography (`tūn`,
    `lēah`, `burh`, `dæl`). normalize-ocr handles macron-strip OCR
    variants but not vowel-weakening / silent-e / gh-spelling shifts.
    This pass uses `_OE_PHONOLOGICAL_BRIDGES` to merge known pairs
    via merged_into_id (D22 non-destructive shape) so the redirect-aware
    cluster-cognates pass rolls the place-name etymons up into the
    Wiktionary cognate clusters.

    Returns:
      - examined: total canonical OE etymons examined
      - bridged: count that found a phonological-bridge target
      - unmatched: count with no entry in the phonological table
      - missing_target: count where the table named a target but no
        OE etymon exists with that canonical form (operator should
        add the target via mining or extend the table)
      - rows_written: actual UPDATE row count when apply=True
    """
    return _bridge_same_language_phonological(
        db, language="old-english", table=_OE_PHONOLOGICAL_BRIDGES, apply=apply
    )


# --- phonological bridging for ON place-name forms ------------------------


# Hand-curated mapping from "scholarly ON form as place-name dictionaries
# write it" → "Wiktionary canonical ON form". Wiktionary uses formal
# scholarly orthography (acutes, þ/ð, -r endings, ǫ); place-name
# dictionaries write Anglicized / modernized spellings inherited from
# medieval English / Norse-influenced scribal practice. The mapping
# captures the most common pairs that surface in the high-witness end
# of our place-name corpus.
# Extend the table when new high-witness mismatches surface.
#
# Convention: keys are lowercase scholarly forms; values are the
# Wiktionary canonical (with acutes, þ/ð, ǫ, -r endings, etc.).
# The bridge uses merged_into_id (D22 non-destructive) — no mining
# evidence is destroyed.
_ON_PHONOLOGICAL_BRIDGES: dict[str, str] = {
    # -býr / settlement, farm
    "by": "býr",
    "byr": "býr",
    # -hólmr / island, holm — bridge value is the macron-stripped 'holmr'
    # since hólmr itself is not in the corpus as a canonical form.
    "holm": "holmr",
    # -kirkja / church (Norse loanword underlying kirk)
    "kirk": "kirkja",
    # -dalr / valley
    "dal": "dalr",
    "dale": "dalr",
    # -garðr / yard, enclosure
    "gardr": "garðr",
    "garth": "garðr",
    # -gríss / pig
    "griss": "gríss",
    # -skógr / wood, forest
    "skogr": "skógr",
    "skégr": "skógr",  # OCR variant
    # -þorp / village (relies on redirect-follow: þorp tombstoned to thorp)
    "thorpe": "þorp",
    "torp": "þorp",
    # -þveit / clearing (relies on redirect-follow: þveit tombstoned to thveit)
    "thwaite": "þveit",
    "thwait": "þveit",
    # -tún / enclosure (Norse cognate of OE tūn)
    "tun": "tún",
    # -vík / bay, inlet
    "vik": "vík",
    # -dýr / animal, deer
    "dyr": "dýr",
    # -hestr / horse
    "hest": "hestr",
    # -kjarr / brushwood, copse
    "kiarr": "kjarr",
    # -krókr / hook, bend
    "krokr": "krókr",
    # -mór / moor
    "mor": "mór",
    # -mýrr / bog
    "myrr": "mýrr",
    # -norðr / north
    "nord": "norðr",
    # -rauðr / red
    "rauthr": "rauðr",
    "raudr": "rauðr",
    # -sauðr / sheep
    "saudr": "sauðr",
    # -vágr / wave, sea-creek
    "vagr": "vágr",
    # -vagn / wagon
    "vogn": "vagn",
    # -vǫllr / field
    "vollr": "vǫllr",
    # -blár / blue
    "bla": "blár",
    # -hǫgg / cut, blow
    "hogg": "hǫgg",
    # -buskr / bush
    "buski": "buskr",
    # -veiðr / hunting (also -veiði)
    "veidi": "veiðr",
    # -flatr / flat
    "flad": "flatr",
    # -hagi / pasture, enclosed grazing
    "hain": "hagi",
}


def bridge_phonological_on(db: LexiconDB, *, apply: bool = False) -> dict:
    """Bridge ON place-name etymons to their Wiktionary canonical
    equivalents via a hand-curated mapping table.

    Place-name dictionaries write Anglicized / modernized ON forms
    (`by`, `holm`, `dale`, `thwaite`, `kirk`, `gardr`); Wiktionary
    uses scholarly orthography with acutes, þ/ð, -r endings, ǫ
    (`býr`, `hólmr`, `dalr`, `þveit`, `kirkja`, `garðr`). normalize-ocr
    handles macron-strip OCR variants but not the Anglicization of
    -r endings, þ → th, ð → d, ǫ → o, etc. This pass uses
    `_ON_PHONOLOGICAL_BRIDGES` to merge known pairs via merged_into_id
    (D22 non-destructive shape) so the redirect-aware cluster-cognates
    pass rolls the place-name etymons up into the Wiktionary cognate
    clusters.

    Returns:
      - examined: total canonical ON etymons examined
      - bridged: count that found a phonological-bridge target
      - unmatched: count with no entry in the phonological table
      - missing_target: count where the table named a target but no
        ON etymon exists with that canonical form (operator should
        add the target via mining or extend the table)
      - rows_written: actual UPDATE row count when apply=True
    """
    return _bridge_same_language_phonological(
        db, language="old-norse", table=_ON_PHONOLOGICAL_BRIDGES, apply=apply
    )


# --- ingest from parsed corpus entries -------------------------------------


def _upsert_toponym(db: LexiconDB, modern_name: str, region: str | None) -> int:
    cur = db.conn.execute(
        """
        SELECT id FROM toponym
        WHERE modern_name = ?
          AND COALESCE(country, '') = ''
          AND COALESCE(region, '') = COALESCE(?, '')
        """,
        (modern_name, region),
    )
    row = cur.fetchone()
    if row is not None:
        return row["id"]
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
        (modern_name, region),
    )
    return cur.lastrowid


def ingest_parsed_entries(
    db: LexiconDB,
    parsed_entries: list[ParsedEntry],
    source_id: str,
    *,
    region: str | None = None,
) -> dict[str, int]:
    """Persist ParsedEntry rows from a Skeat-style parser into the DB.

    For each entry:
      - Upsert toponym row (region carries the source's coverage area).
      - For HIGH/MEDIUM confidence: insert a toponym_etymology row pointing
        at <source_id>, plus one toponym_etymology_element per parsed
        element. Each element's etymon is upserted, glosses/tags attached,
        and a citation row added.
      - For LOW confidence: only the toponym row is written, so we still
        record that the source covers this place even when we couldn't
        recover a breakdown.

    Returns a counts dict for sanity-checking.
    """
    counts = {
        "toponyms": 0,
        "etymologies": 0,
        "elements": 0,
        "etymons_touched": 0,
    }
    for entry in parsed_entries:
        toponym_id = _upsert_toponym(db, entry.toponym, region)
        counts["toponyms"] += 1

        if entry.confidence == "low" or not entry.elements:
            continue

        confidence = entry.confidence if entry.confidence in ("high", "medium", "low") else "low"
        cur = db.conn.execute(
            """
            INSERT INTO toponym_etymology
                (toponym_id, source_id, historical_form, confidence, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                toponym_id,
                source_id,
                entry.historical_form,
                confidence,
                entry.source_quote or None,
            ),
        )
        etymology_id = cur.lastrowid
        counts["etymologies"] += 1

        for ordinal, elem in enumerate(entry.elements):
            etymon_id = db.upsert_etymon(elem.form, elem.language)
            counts["etymons_touched"] += 1
            if elem.gloss:
                db.add_gloss(etymon_id, elem.gloss)
            db.add_citation(etymon_id, source_id, short_quote=entry.source_quote or None)
            db.conn.execute(
                """
                INSERT INTO toponym_etymology_element
                    (toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern)
                VALUES (?, ?, ?, ?, ?)
                """,
                (etymology_id, ordinal, etymon_id, elem.inflection, None),
            )
            counts["elements"] += 1

    db.commit()
    return counts


# --- meanings.json export --------------------------------------------------
#
# Inverse of `seed_from_meanings`: walk the lexicon DB, apply the D4 promotion
# rule, and emit a meanings.json document the runtime can load. The runtime
# shape is unchanged (D1) — generation still consumes meanings.json — so the
# export is the bridge that lets new mining reach users.
#
# Inflections (D8) and OCR-cluster losers (D22) ride along in the lemma's
# language form list rather than getting their own subjects, so the consensus
# rollup the runtime sees matches the rollup the lexicon enforces.

# Reverse of LANGUAGE_FIELDS: lexicon language code → meanings.json field name.
# Only the canonical key is emitted; the misspelled aliases ('old _english',
# 'old_scandanavian') are accepted on ingestion but not regenerated on export.
_LANG_CODE_TO_JSON_FIELD = {
    "old-english": "old_english",
    "old-norse": "old_scandinavian",
    "norman-french": "old_french",
    "celtic": "celtic_mix",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "modern-english": "modern_english",
    "biblical": "biblical",
    # wyrd-4hx7: Wiktionary-canonical language codes (per
    # wiktextract_ingester._canonical_language) routed into the same
    # 9 bundle fields the historical scholarly corpus used. The
    # bundle field structure stays at 9 buckets; the lexicon keeps
    # finer-grained languages on the etymon row for future per-language
    # queries. Adding new bundle field names later (e.g. "welsh",
    # "irish") would require runtime + culture-mapping work; routing
    # into existing buckets is the no-runtime-change path.
    "welsh": "celtic_mix",
    "old-welsh": "celtic_mix",
    "middle-welsh": "celtic_mix",
    "irish": "celtic_mix",
    "old-irish": "celtic_mix",
    "middle-irish": "celtic_mix",
    "scottish-gaelic": "celtic_mix",
    "manx": "celtic_mix",
    "cornish": "celtic_mix",
    "breton": "celtic_mix",
    "old-breton": "celtic_mix",
    "middle-breton": "celtic_mix",
    "proto-celtic": "celtic_mix",
    "proto-brythonic": "celtic_mix",
    "old-french": "old_french",
    "anglo-norman": "old_french",
    "middle-french": "old_french",
    "middle-english": "modern_english",
    "scots": "modern_english",
    "icelandic": "old_scandinavian",
    "faroese": "old_scandinavian",
    "old-high-german": "germanic",
    "middle-high-german": "germanic",
    "gothic": "germanic",
    "old-saxon": "germanic",
    "old-dutch": "germanic",
    "old-frisian": "germanic",
    "proto-germanic": "germanic",
    "proto-west-germanic": "germanic",
    "ancient-greek": "greek",
    "proto-greek": "greek",
    # wyrd-vsrn Phase 2c: wave-2 non-Latin source-language buckets.
    # Each canonical wave-2 language gets its own bundle field; the
    # precursor / postcursor stack codes (per fantasy_pipeline's
    # _PRECURSOR_POSTCURSOR_STACK) all funnel into the canonical
    # bucket for their family — same pattern as celtic_mix bundles
    # welsh / old-welsh / middle-welsh / etc.
    "he": "hebrew",
    "hbo": "hebrew",
    "ar": "arabic",
    "fa": "persian",
    "peo": "persian",
    "fa-cls": "persian",
    "xpr": "persian",
    "pal": "persian",
    "ira-pro": "persian",
    "sa": "sanskrit",
    "iir-pro": "sanskrit",
    "inc-pro": "sanskrit",
    "pra": "sanskrit",
    "pi": "sanskrit",
    "akk": "akkadian",
    "sux": "akkadian",  # Sumerian — Mesopotamian substrate, group with akkadian
    "egy": "egyptian",
    "cop": "egyptian",  # Coptic — late-stage descendant of Ancient Egyptian
    "arc": "aramaic",
    "syc": "aramaic",  # Classical Syriac — late descendant of Aramaic
    "sem-pro": "hebrew",  # Proto-Semitic — group with Hebrew (closest canonical)
    "sem-wes-pro": "hebrew",
    "afa-pro": "hebrew",  # Proto-Afroasiatic — same bucket
    "axm": "armenian",  # Old / Classical Armenian — own bucket
}

# Per-language witness thresholds calibrated against corpus availability and
# spot-checked quality at w=2 (analysis 2026-05-02). Languages absent from
# this map fall back to the global ``min_witnesses``. Rationale per language:
#   old-english : 3 — well-mined (32 sources); strict gate keeps Tier-1
#                     prose-extraction noise out (~20% noise at w=2).
#   celtic      : 2 — Qwen weak on Celtic per D13; corpus structurally
#                     thinner per yield. Quality at w=2 ~90% clean.
#   old-norse   : 2 — only one ON-focused dictionary in the corpus; w=3
#                     filters out 93% of ON purely on corpus thinness.
#   modern-english,
#   norman-french,
#   latin,
#   biblical    : 2 — small populations at w≥2 (≤20 each); spot-check 100%
#                     clean. The cost of admitting them is negligible and
#                     the gain (modern English placename elements like mill,
#                     stone, head; NF castle, monte; Latin ecclesia,
#                     ceaster) is meaningful for generation breadth.
#   germanic, greek: nothing reaches w≥2 in the current corpus, so the
#                     threshold is moot — they ride the rando-port path only.
RECOMMENDED_LANG_THRESHOLDS: dict[str, int] = {
    "old-english": 3,
    "celtic": 2,
    "old-norse": 2,
    "modern-english": 2,
    "norman-french": 2,
    "latin": 2,
    "biblical": 2,
}


def _load_norman_manorial_family_tokens() -> frozenset[str]:
    """Return the surname-only tokens of Anglo-Norman manorial families
    (e.g. 'Cary', 'Lacy', 'Mandeville', 'Zouche'). Loaded from the
    canonical JSON at ``data/norman_manorial_families.json``.

    Used by :func:`collect_canonical_decompositions` to skip canonical
    picks for toponyms whose final whitespace-split token matches a
    known manorial family — those names get a runtime-synthesized
    decomposition by ``_norman_manorial_subjects`` in ``__init__.py``
    that the build-time tiebreaker shouldn't pre-empt.

    The token is the last whitespace-split word (matches the
    surname-only matching policy that ``_norman_manorial_subjects``
    documents).
    """
    data = resources.files("wyrd.generators.kenning.data").joinpath("norman_manorial_families.json")
    families = json.loads(data.read_text())
    return frozenset(family.split()[-1] for family in families)


def collect_canonical_decompositions(db: LexiconDB) -> dict[str, dict[str, str]]:
    """Project the lexicon's canonical decomposition picks into a
    bundle-shaped lookup keyed by toponym ``modern_name``.

    Used by ``lexicon export-meanings`` to emit a ``canonical_decompositions``
    field that ``KenningExplain`` (Lambda / SPA — no DB access) can read
    at runtime to mark and front-load the canonical reading among the
    matcher's alternatives (wyrd-h8k1).

    Output shape:
    ``{modern_name: {"signature": sha1_hex, "source": canonical_source}}``.

    Multiple toponym rows sharing a ``modern_name`` (one per region)
    collapse to the lex-first ``(region, id)`` ordered row's canonical.
    The Lambda has no region context at decomposition time so this
    matches ``load_names_with_regions``'s dedup policy: deterministic,
    same-name entries get the same canonical pick across re-runs.

    Toponyms whose final whitespace-split token matches a known
    Anglo-Norman manorial family ('Castle Cary', 'Stoke Mandeville',
    'Newton Lacy') are SKIPPED — the runtime manorial-affix detector
    in ``_norman_manorial_subjects`` synthesizes a more specific
    decomposition (e.g. ``castle + Cary (Norman manorial family)``)
    than the build-time tiebreaker can produce against DB-only
    morphemes. Without this guard, an enrichment pass that adds
    short morphemes (``ca``, ``ry``) lets the matcher perfectly
    decompose these names at export time, the canonical pick wins
    the rank-0 sort at runtime, and the manorial UX silently
    regresses. wyrd-j43l deploy-gate caught this on 'Castle Cary'.

    Returns an empty dict when the ``toponym_decomposition`` table
    doesn't exist yet — older DBs predate the wyrd-08m Phase 1 migration
    and shouldn't crash the bundle export. Run ``lexicon migrate`` to
    create the table, then ``lexicon decompose --apply`` to populate
    canonical picks.
    """
    table_exists = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='toponym_decomposition'"
    ).fetchone()
    if not table_exists:
        return {}
    manorial_tokens = _load_norman_manorial_family_tokens()
    rows = db.conn.execute(
        """
        SELECT t.modern_name,
               td.decomposition_signature,
               td.canonical_source
          FROM toponym t
          JOIN toponym_decomposition td ON td.toponym_id = t.id
         WHERE td.is_canonical = 1
         ORDER BY t.modern_name,
                  COALESCE(t.region, ''),
                  t.id,
                  td.id
        """
    ).fetchall()
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["modern_name"]
        if name in out:
            # Same-name multi-region collisions collapse to the
            # lex-first row's canonical; later rows skipped.
            continue
        tokens = name.split()
        if len(tokens) >= 2 and tokens[-1] in manorial_tokens:
            # Skip — runtime manorial-affix detector wins for these.
            # Requires ≥2 tokens because the manorial-affix shape is
            # ``<base> <family>``; a solo "Lacy" toponym would still
            # legitimately need a build-time canonical pick.
            continue
        out[name] = {
            "signature": row["decomposition_signature"],
            "source": row["canonical_source"] or "",
        }
    return out


def collect_fantasy_morphemes(db: LexiconDB) -> dict[str, dict[str, Any]]:
    """wyrd-vz7f: project usable ``fantasy_morpheme`` rows into a
    bundle-shaped lookup keyed by ``input_name`` so the runtime
    generator can surface creature etymology without a DB.

    Walks ``fantasy_morpheme`` rows where ``usable=1`` (etymon
    successfully resolved) and ``etymon.merged_into_id IS NULL`` (the
    linked etymon isn't an OCR-cluster tombstone). For each, joins
    canonical_form / language / english_shaped from ``etymon`` and
    pulls glosses from ``etymon_gloss``. Era reflexes are computed
    via ``etymon_era_reflexes`` for every target language in the
    creature's family chain — same precision as the toponym path,
    so a creature whose linked etymon is in old-english (Angel →
    OE 'Engel') surfaces ME / EModE / ModE forms when they exist.

    Returns ``{}`` when the ``fantasy_morpheme`` table is missing
    (older DBs predate wyrd-ami) — defensive against the same shape
    of failure that wyrd-c1vq fixed for ``toponym_decomposition``.

    Output shape:
    ``{input_name: {input_name, etymon_id, language, canonical_form,
                    english_shaped, glosses, citation, era_reflexes}}``.
    ``era_reflexes`` follows the wyrd-jbcu source-aware schema
    (``{lang: [{form, source}, ...]}``).
    """
    table_exists = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fantasy_morpheme'"
    ).fetchone()
    if not table_exists:
        return {}

    from wyrd.generators.kenning.era import (
        CANONICAL_LANGUAGE_FOR_CELL,
        language_family,
    )

    # Follow the merged_into chain so OCR-cluster losers route to their
    # winner's canonical_form. wyrd-ami linked at a moment in time;
    # post-link OCR clustering may have moved the canonical voice to
    # the merge winner, but the user's "Harpy" → ancient-greek ἅρπυια
    # lookup is still semantically valid. The COALESCE pattern matches
    # the etymon_consensus view's two-step rollup.
    #
    # ``english_shaped`` semantics: when the winner's column is NULL
    # but the loser's was populated, the COALESCE returns NULL —
    # winner-authoritative per D22's sacred-evidence rule. The loser's
    # english_shaped was a property of the loser's specific form, not
    # the cluster's; it stays attached to the loser row but isn't
    # promoted to the winner's voice.
    rows = db.conn.execute(
        """
        SELECT fm.input_name,
               COALESCE(target.id, e.id) AS etymon_id,
               COALESCE(target.canonical_form, e.canonical_form) AS canonical_form,
               COALESCE(target.language, e.language) AS language,
               COALESCE(target.english_shaped, e.english_shaped) AS english_shaped,
               fm.citation
          FROM fantasy_morpheme fm
          JOIN etymon e ON e.id = fm.etymon_id
          LEFT JOIN etymon target ON target.id = e.merged_into_id
         WHERE fm.usable = 1
         ORDER BY fm.input_name
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        etymon_id = row["etymon_id"]
        glosses = [
            r["gloss"]
            for r in db.conn.execute(
                "SELECT DISTINCT gloss FROM etymon_gloss WHERE etymon_id = ? ORDER BY gloss",
                (etymon_id,),
            )
        ]
        # Per-target-language era reflexes: same precision as the
        # toponym path's _fetch_root_era_reflexes (cluster + descent +
        # period-form + phonology-rule via etymon_era_reflexes).
        family = language_family(row["language"])
        era_reflexes: dict[str, list[dict[str, str]]] = {}
        if family is not None:
            target_languages = sorted(
                {
                    lang
                    for (fam, _cell), lang in CANONICAL_LANGUAGE_FOR_CELL.items()
                    if fam == family
                }
            )
            for target_language in target_languages:
                refs = etymon_era_reflexes(db, etymon_id, target_language=target_language)
                if not refs:
                    continue
                # Same dedupe-by-form-keep-best-source pattern as
                # _fetch_root_era_reflexes (wyrd-jbcu).
                best: dict[str, str] = {}
                for r in refs:
                    existing = best.get(r.form)
                    if existing is None or _better_era_reflex_source(r.source, existing):
                        best[r.form] = r.source
                era_reflexes[target_language] = [
                    {"form": form, "source": best[form]} for form in sorted(best)
                ]
        out[row["input_name"]] = {
            "input_name": row["input_name"],
            "etymon_id": etymon_id,
            "language": row["language"],
            "canonical_form": row["canonical_form"],
            "english_shaped": row["english_shaped"] or "",
            "glosses": glosses,
            "citation": row["citation"] or "",
            "era_reflexes": era_reflexes,
        }
    return out


def export_meanings(
    db: LexiconDB,
    *,
    min_witnesses: int = 3,
    lang_thresholds: dict[str, int] | None = None,
    include_rando: bool = True,
    include_wiktionary_empirical: bool = True,
) -> list[dict[str, Any]]:
    """Walk the lexicon and emit a meanings.json structure.

    Promotion rule (D4): a family root is included if any of:
    (a) any etymon in the family is cited by 'rando-port' AND ``include_rando``
        is true (legacy seed kept until corroborated), OR
    (b) the family's witness count (``etymon_consensus.witnesses``) is at
        least the threshold for that family's language, OR
    (c) any etymon in the family is cited by 'wiktionary-empirical' AND
        ``include_wiktionary_empirical`` is true (empirical class — wyrd-4hx7
        corpus-mining headwords matched against unaccounted-fragment misses;
        treated like rando-port: bypass the scholar-witness gate).

    The threshold per language is taken from ``lang_thresholds`` (defaults to
    ``RECOMMENDED_LANG_THRESHOLDS``); languages absent from the map use
    ``min_witnesses``. Pass ``lang_thresholds={}`` to apply a uniform
    ``min_witnesses`` threshold across all languages.

    Family roots are computed by the same two-step rollup the
    ``etymon_consensus`` view uses (``merged_into_id`` then ``lemma_id``), so
    OCR-cluster losers and inflected variants don't get their own subjects —
    their canonical_forms ride along in the lemma's language form list (D8,
    D18, D22).

    Subjects are reconstructed by grouping family roots that share the same
    (modifier_type, glosses, tags) signature — the natural inverse of how
    seed_from_meanings shaped the original rando-port subjects. The
    reconstruction is lossy where the original meanings.json had two distinct
    subjects with identical signatures, but the runtime ``load_meanings``
    keys by ``modern_usage`` regardless of subject boundary.
    """
    if lang_thresholds is None:
        lang_thresholds = RECOMMENDED_LANG_THRESHOLDS
    families = _collect_families(
        db,
        min_witnesses=min_witnesses,
        lang_thresholds=lang_thresholds,
        include_rando=include_rando,
        include_wiktionary_empirical=include_wiktionary_empirical,
    )
    subjects = _group_families_into_subjects(families)
    if include_rando:
        subjects.extend(_orphan_reflex_subjects(db))
    return subjects


def _build_witness_filter(
    lang_thresholds: dict[str, int],
    min_witnesses: int,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment + bind params that gate etymon_consensus
    rows by per-language thresholds with ``min_witnesses`` as the fallback.

    Returns ``(sql_fragment, params)`` where the fragment is parenthesized
    and references ``language`` / ``witnesses`` columns.
    """
    if not lang_thresholds:
        return "(witnesses >= ?)", [min_witnesses]
    clauses: list[str] = []
    params: list[Any] = []
    sorted_langs = sorted(lang_thresholds)
    for lang in sorted_langs:
        clauses.append("(language = ? AND witnesses >= ?)")
        params.extend([lang, lang_thresholds[lang]])
    placeholders = ",".join("?" * len(sorted_langs))
    clauses.append(f"(language NOT IN ({placeholders}) AND witnesses >= ?)")
    params.extend(sorted_langs)
    params.append(min_witnesses)
    return f"({' OR '.join(clauses)})", params


def _collect_families(
    db: LexiconDB,
    *,
    min_witnesses: int,
    lang_thresholds: dict[str, int],
    include_rando: bool,
    include_wiktionary_empirical: bool = True,
) -> list[dict[str, Any]]:
    """Build per-family-root data: forms-by-language, glosses, tags, reflexes.

    Three admission paths feed ``promoted``:
      * scholar-witness threshold (``etymon_consensus``)
      * rando-port seed admit (legacy Wikipedia-derived bundle)
      * wiktionary-empirical admit (wyrd-4hx7 corpus-mined gap-fills)

    Each empirical-class admit is gated by its own boolean flag so callers
    can A/B by toggling either branch (e.g. ``--no-include-rando`` or
    ``--no-include-wiktionary-empirical``) without disturbing the other.
    """
    members_by_root, root_of = _build_family_rollup(db)
    root_ids = _select_promoted_root_ids(
        db,
        lang_thresholds=lang_thresholds,
        min_witnesses=min_witnesses,
        include_rando=include_rando,
        include_wiktionary_empirical=include_wiktionary_empirical,
        root_of=root_of,
    )
    return _iterate_families_with_progress(db, root_ids, members_by_root)


def _build_family_rollup(
    db: LexiconDB,
) -> tuple[dict[int, list[int]], Callable[[int], int]]:
    """Compute the etymon → root_id rollup in PYTHON rather than via
    a SQL CREATE TEMP TABLE. The relational form needs
    ``LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id,
    e.lemma_id)`` — a function-on-columns join predicate that no index
    can satisfy. On a 731K-row etymon table that's a many-minute scan,
    whether materialised once into a temp table or recomputed per
    query. A flat ``SELECT id, merged_into_id, lemma_id FROM etymon``
    plus a Python dict produces the same rollup in seconds:
    731K rows * ~1µs/row = ~1s vs ~minutes for the SQL JOIN.

    The rollup follows the consensus view's two-step rule (D22 +
    D8 flatten):
      1) target = merged_into_id OR lemma_id OR self
      2) root   = lemma_id of target OR target itself
    so OCR-cluster losers and inflected children both surface their
    ultimate lemma as root_id.

    Returns ``(members_by_root, root_of_callable)``.
    """
    rollup_rows = db.conn.execute("SELECT id, merged_into_id, lemma_id FROM etymon").fetchall()
    lemma_by_id = {r["id"]: r["lemma_id"] for r in rollup_rows}
    target_by_id = {r["id"]: (r["merged_into_id"] or r["lemma_id"] or r["id"]) for r in rollup_rows}

    def root_of(eid: int) -> int:
        target = target_by_id.get(eid, eid)
        return lemma_by_id.get(target) or target

    members_by_root: dict[int, list[int]] = {}
    for eid in target_by_id:
        members_by_root.setdefault(root_of(eid), []).append(eid)
    return members_by_root, root_of


def _select_promoted_root_ids(
    db: LexiconDB,
    *,
    lang_thresholds: dict[str, int],
    min_witnesses: int,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    root_of: Callable[[int], int],
) -> list[int]:
    """Promoted root_ids come from three sources:

    * consensus witness threshold per language (the etymon_consensus view
      already keys on lemma_id, no rollup needed),
    * any etymon cited by 'rando-port' (legacy seed),
    * any etymon cited by 'wiktionary-empirical' (wyrd-4hx7).

    For the two empirical-class branches we SELECT the cited etymon_ids
    flat and roll them up via ``root_of`` — much faster than the JOIN-
    based CTE.
    """
    witness_sql, witness_params = _build_witness_filter(lang_thresholds, min_witnesses)
    promoted: set[int] = set()
    for row in db.conn.execute(
        f"SELECT lemma_id AS root_id FROM etymon_consensus WHERE {witness_sql}",
        witness_params,
    ):
        promoted.add(row["root_id"])
    if include_rando:
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'rando-port'"
        ):
            promoted.add(root_of(row["etymon_id"]))
    if include_wiktionary_empirical:
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'wiktionary-empirical'"
        ):
            promoted.add(root_of(row["etymon_id"]))
    return sorted(promoted)


def _iterate_families_with_progress(
    db: LexiconDB,
    root_ids: list[int],
    members_by_root: dict[int, list[int]],
) -> list[dict[str, Any]]:
    """Walk promoted root_ids, gathering each family's data and
    emitting stderr progress every ~2% of total. ``WYRD_EXPORT_QUIET=1``
    silences the progress lines.

    Without progress reporting the user has no way to estimate how
    far along a multi-minute re-export is. Step chosen to keep
    stderr output to ~50 lines on the typical 5-10K-root corpus
    while still emitting first-rate-of-change signal within the
    first minute.
    """
    import os
    import sys
    import time

    quiet = os.environ.get("WYRD_EXPORT_QUIET") == "1"
    n_total = len(root_ids)
    progress_every = max(1, n_total // 50) if n_total else 1
    started = time.monotonic()
    families: list[dict[str, Any]] = []
    for i, root_id in enumerate(root_ids, 1):
        member_ids = members_by_root.get(root_id, [root_id])
        family = _gather_family(db, root_id, member_ids)
        if family is not None and family["forms_by_lang"]:
            families.append(family)
        if not quiet and (i % progress_every == 0 or i == n_total):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            print(
                f"  collect_families {i}/{n_total} "
                f"({100 * i / n_total:.1f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"({rate:.0f} roots/s)",
                file=sys.stderr,
                flush=True,
            )
    return families


def _gather_family(db: LexiconDB, root_id: int, member_ids: list[int]) -> dict[str, Any] | None:
    """Collect forms / glosses / tags / reflexes for one family root.

    Includes the root itself plus all etymons that roll up to it
    (inflected children, OCR-cluster losers, and combinations thereof).
    ``member_ids`` is the precomputed family-membership list (root +
    children) — caller is responsible for the rollup, computed once in
    ``_collect_families`` to avoid the per-root SQL JOIN that scaled
    badly on the 731K-row etymon table.
    Pure orchestrator — each per-aspect aggregation lives in a focused
    helper below.
    """
    root_row = db.conn.execute(
        "SELECT canonical_form, language, modifier_type, position_pref FROM etymon WHERE id = ?",
        (root_id,),
    ).fetchone()
    if root_row is None:
        return None

    if not member_ids:
        return None
    placeholders = ",".join("?" * len(member_ids))
    member_rows = db.conn.execute(
        f"""
        SELECT id, canonical_form, language, lemma_id, merged_into_id, inflection,
               english_shaped, original_script, transliteration,
               pronunciation_ipa, pronunciation_dialect, stratum
        FROM etymon
        WHERE id IN ({placeholders})
        ORDER BY language, canonical_form
        """,
        member_ids,
    ).fetchall()
    if not member_rows:
        return None

    member_ids = [r["id"] for r in member_rows]
    member_form_by_id = {r["id"]: (r["language"], r["canonical_form"]) for r in member_rows}
    member_inflection_by_id: dict[int, str | None] = {r["id"]: r["inflection"] for r in member_rows}
    # wyrd-vsrn Phase 2c + wyrd-qhs0 Phase 2d: per-member wyrd-ha9q
    # rendering data. NULL when the row's source language is Latin-
    # script or wyrd-ha9q's derive / wiktextract ingest produced no
    # value. Keyed by member_id so absorb-helpers can look up cheaply.
    member_english_shaped_by_id: dict[int, str | None] = {
        r["id"]: r["english_shaped"] for r in member_rows
    }
    member_original_script_by_id: dict[int, str | None] = {
        r["id"]: r["original_script"] for r in member_rows
    }
    member_transliteration_by_id: dict[int, str | None] = {
        r["id"]: r["transliteration"] for r in member_rows
    }
    # pronunciation is paired (IPA + optional dialect tag); keyed by
    # member_id with a tuple so the absorb-helper can drop both
    # together when either side is NULL (matches the upsert semantics
    # that updated them atomically).
    member_pronunciation_by_id: dict[int, tuple[str, str | None] | None] = {
        r["id"]: (r["pronunciation_ipa"], r["pronunciation_dialect"])
        if r["pronunciation_ipa"]
        else None
        for r in member_rows
    }
    # wyrd-lr4 Phase 2: per-member within-language stratum tag
    # (Welsh-only in Phase 1; French / OE / ON follow). NULL for
    # languages without a Phase 1 classifier and for legacy DBs that
    # pre-date the column.
    member_stratum_by_id: dict[int, str | None] = {r["id"]: r["stratum"] for r in member_rows}
    canonical_forms_lower = {f.lower() for _lang, f in member_form_by_id.values()}

    reflex_links = _fetch_member_reflex_links(db, member_ids)

    return {
        "root_id": root_id,
        "root_canonical_form": root_row["canonical_form"],
        "root_language": root_row["language"],
        "modifier_type": root_row["modifier_type"],
        "position_pref": root_row["position_pref"],
        "forms_by_lang": _build_forms_by_lang(root_row, member_rows),
        "member_form_by_id": member_form_by_id,
        "member_descendants": _compute_member_descendants(member_rows),
        "member_variants": _fetch_member_variants(db, member_ids, canonical_forms_lower),
        "member_inflection_by_id": member_inflection_by_id,
        "member_english_shaped_by_id": member_english_shaped_by_id,
        "member_original_script_by_id": member_original_script_by_id,
        "member_transliteration_by_id": member_transliteration_by_id,
        "member_pronunciation_by_id": member_pronunciation_by_id,
        "member_stratum_by_id": member_stratum_by_id,
        "member_citations": _fetch_member_citations(db, member_ids),
        "member_attested_years": _fetch_member_attested_years(db, member_ids),
        "glosses": _fetch_member_glosses(db, member_ids),
        # wyrd-i1s1: union member tags (lemma + inflections + OCR
        # losers) with cognate-cluster-mate tags so semantic signal
        # from non-promoted cluster mates (ME / ModE / NF cognates of
        # a promoted OE root) reaches the bundle subject.
        "tags": sorted(
            set(_fetch_member_tags(db, member_ids)) | set(_fetch_cluster_mate_tags(db, root_id))
        ),
        "reflexes": _fetch_member_reflexes(db, member_ids, reflex_links),
        # wyrd-obpw Phase 3.3: era reflexes for the family root,
        # keyed by target language tag. SPA-side rewinder reads this
        # at runtime from the bundle (Lambda has no lexicon DB).
        "era_reflexes": _fetch_root_era_reflexes(db, root_id, root_row["language"]),
    }


def _fetch_root_era_reflexes(
    db: LexiconDB, root_id: int, root_language: str
) -> dict[str, list[dict[str, str]]]:
    """wyrd-obpw Phase 3.3 + wyrd-jbcu source-aware schema: per-root
    era reflexes for bundle export.

    Returns a dict mapping target language tag → sorted list of
    ``{"form": str, "source": str}`` dicts. The SPA-side rewinder
    consumes this at runtime; ``source`` distinguishes attestation-
    backed reflexes ('cluster' / 'descent' / 'period-form') from
    phonology-rule-derived ones ('phonology-rule:v1') so consumers
    can render inferred forms differently if they want to.

    For each language tag in ``CANONICAL_LANGUAGE_FOR_CELL`` of the
    root's family, calls ``etymon_era_reflexes`` and collects the
    forms. When the same form arrives via multiple tiers (cluster
    mate AND a phonology rule that landed on the same surface), the
    higher-quality source wins per ``_ERA_REFLEX_SOURCE_PRIORITY``.

    Empty languages are omitted (no entry in the returned dict).
    Returns ``{}`` when:

    * the root's language has no era family (proto-languages,
      untracked classical languages),
    * no cluster mates / descent edges / period-form rows / phonology
      rule walks match any target language.

    Computed at bundle-build time only — the runtime caller doesn't
    have DB access and reads from the bundle's ``era_reflexes`` field.
    """
    from wyrd.generators.kenning.era import (
        CANONICAL_LANGUAGE_FOR_CELL,
        language_family,
    )

    family = language_family(root_language)
    if family is None:
        return {}
    target_languages: set[str] = {
        lang for (fam, _cell), lang in CANONICAL_LANGUAGE_FOR_CELL.items() if fam == family
    }
    out: dict[str, list[dict[str, str]]] = {}
    for target_language in sorted(target_languages):
        reflexes = etymon_era_reflexes(db, root_id, target_language=target_language)
        if not reflexes:
            continue
        # Dedupe by form, keeping the highest-quality source on
        # collision. Same form might surface via cluster (high) and
        # phonology-rule (low) — prefer the cluster.
        best: dict[str, str] = {}
        for r in reflexes:
            existing = best.get(r.form)
            if existing is None or _better_era_reflex_source(r.source, existing):
                best[r.form] = r.source
        out[target_language] = [{"form": form, "source": best[form]} for form in sorted(best)]
    return out


# Lower number = higher quality. Used by _fetch_root_era_reflexes when
# the same form surfaces via multiple tiers, and by _emit_era_reflexes
# when multiple linked families contribute the same form. Unknown
# sources fall through to default priority so a future tier doesn't
# silently downgrade everything.
_ERA_REFLEX_SOURCE_PRIORITY: dict[str, int] = {
    "cluster": 0,
    "descent": 1,
    "period-form": 2,
    "phonology-rule:v1": 3,
}
_ERA_REFLEX_SOURCE_DEFAULT_PRIORITY: int = 2


def _better_era_reflex_source(candidate: str, current: str) -> bool:
    """True iff ``candidate`` outranks ``current`` per the era-reflex
    source priority. Used to resolve same-form collisions in both
    ``_fetch_root_era_reflexes`` (per-tier dedupe) and
    ``_emit_era_reflexes`` (cross-family merge). Unknown sources fall
    through to default priority on both sides — the comparison stays
    well-defined and a new tier doesn't silently win or lose against
    everything."""
    return _ERA_REFLEX_SOURCE_PRIORITY.get(
        candidate, _ERA_REFLEX_SOURCE_DEFAULT_PRIORITY
    ) < _ERA_REFLEX_SOURCE_PRIORITY.get(current, _ERA_REFLEX_SOURCE_DEFAULT_PRIORITY)


def _build_forms_by_lang(root_row: Any, member_rows: list[Any]) -> dict[str, list[str]]:
    """Group canonical forms by language, root-first.

    Root form first per language so the lemma's own canonical_form leads
    the list (predictable order for snapshot tests).
    """
    forms_by_lang: dict[str, list[str]] = {}
    forms_by_lang.setdefault(root_row["language"], []).append(root_row["canonical_form"])
    for r in member_rows:
        bucket = forms_by_lang.setdefault(r["language"], [])
        if r["canonical_form"] not in bucket:
            bucket.append(r["canonical_form"])
    return forms_by_lang


_GLOSS_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+or\s+", re.IGNORECASE)


def _fetch_member_glosses(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted glosses across the family's members.

    Drops glosses that are concatenations of already-present sibling
    glosses (mining noise where the LLM emitted a comma/semicolon list as
    a single gloss when the source bullet split it). Concretely, if
    'lake', 'pond' are present and 'lake, pond' also got extracted, drop
    the latter — it adds no information and clutters the explainer.
    Splitter recognizes ',', ';', and ' or '; case-insensitive on the
    'or' connector. Single-token glosses are kept unconditionally.
    """
    placeholders = ",".join("?" * len(member_ids))
    raw = [
        row["gloss"]
        for row in db.conn.execute(
            f"SELECT DISTINCT gloss FROM etymon_gloss "
            f"WHERE etymon_id IN ({placeholders}) ORDER BY gloss",
            member_ids,
        )
    ]
    return _filter_concatenation_glosses(raw)


def _filter_concatenation_glosses(glosses: list[str]) -> list[str]:
    """Drop glosses whose tokens are entirely covered by other singleton
    glosses already in the list. Pure (no DB), so the caller's order
    is preserved for the survivors."""
    singleton_set = {g.strip().lower() for g in glosses if not _GLOSS_SPLIT_RE.search(g)}
    out: list[str] = []
    for g in glosses:
        if g.strip().lower() in singleton_set and not _GLOSS_SPLIT_RE.search(g):
            out.append(g)
            continue
        tokens = [t.strip().lower() for t in _GLOSS_SPLIT_RE.split(g) if t.strip()]
        if len(tokens) > 1 and all(t in singleton_set for t in tokens):
            continue
        out.append(g)
    return out


def _fetch_member_tags(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted tags across the family's members."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        row["tag"]
        for row in db.conn.execute(
            f"SELECT DISTINCT tag FROM etymon_tag WHERE etymon_id IN ({placeholders}) ORDER BY tag",
            member_ids,
        )
    ]


def _fetch_cluster_mate_tags(db: LexiconDB, root_id: int) -> list[str]:
    """wyrd-i1s1: tags from cognate-cluster mates of ``root_id``.

    The family rollup (``_build_family_rollup``) traverses
    ``merged_into_id`` and ``lemma_id`` chains only — cluster mates
    sharing a ``cognate_id`` are SEPARATE family roots. Their tags
    don't surface in the rolled-up family's tag list.

    But cluster mates of a promoted OE root (typically ME / ModE
    cognates that didn't pass the per-language witness threshold on
    their own) carry valuable semantic-tag signal that the bundle
    consumer ought to see attached to the bundle subject the OE root
    grounds. This helper pulls those tags so they can be merged into
    ``family["tags"]`` alongside ``_fetch_member_tags``'s output.

    Excludes the root itself (its tags ride in via ``_fetch_member_tags``)
    and OCR-merge tombstones. Returns an empty list when the root has
    no ``cognate_id`` (no cluster) — most non-promoted etymons.
    """
    return [
        row["tag"]
        for row in db.conn.execute(
            """
            SELECT DISTINCT t.tag
            FROM etymon mate
            JOIN etymon_tag t ON t.etymon_id = mate.id
            WHERE mate.cognate_id = (
                SELECT cognate_id FROM etymon WHERE id = ?
              )
              AND mate.cognate_id IS NOT NULL
              AND mate.id != ?
              AND mate.merged_into_id IS NULL
            ORDER BY t.tag
            """,
            (root_id, root_id),
        )
    ]


def _fetch_member_reflex_links(db: LexiconDB, member_ids: list[int]) -> dict[int, list[int]]:
    """Per-reflex linked etymon ids (within this family).

    Lets us narrow the exported language array per word at group-build
    time — without this, a subject grouping a Celtic family and an OE
    family would emit each reflex's word with both languages even when
    the reflex only links to one of them.
    """
    placeholders = ",".join("?" * len(member_ids))
    reflex_links: dict[int, list[int]] = {}
    for row in db.conn.execute(
        f"SELECT re.reflex_id, re.etymon_id "
        f"FROM reflex_etymon re "
        f"WHERE re.etymon_id IN ({placeholders}) "
        f"ORDER BY re.reflex_id, re.etymon_id",
        member_ids,
    ):
        reflex_links.setdefault(row["reflex_id"], []).append(row["etymon_id"])
    return reflex_links


def _fetch_member_reflexes(
    db: LexiconDB, member_ids: list[int], reflex_links: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Reflex rows (modern surface forms) linked to any family member."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        {
            "id": row["id"],
            "surface_form": row["surface_form"],
            "position": row["position"],
            "linked_member_ids": reflex_links.get(row["id"], []),
        }
        for row in db.conn.execute(
            f"SELECT DISTINCT r.id, r.surface_form, r.position "
            f"FROM reflex r "
            f"JOIN reflex_etymon re ON re.reflex_id = r.id "
            f"WHERE re.etymon_id IN ({placeholders}) "
            f"ORDER BY r.position, r.surface_form",
            member_ids,
        )
    ]


_LOW_CONFIDENCE_METHODS = frozenset({"fuzzy-search-v1", "llm-disambiguated-v1"})


def _fetch_member_variants(
    db: LexiconDB, member_ids: list[int], canonical_forms_lower: set[str]
) -> dict[int, list[tuple[str, int]]]:
    """Spelling variant pool per member_id (D18).

    Pulls from etymon_text_match.matched_form: real attested 19th-c.
    spellings (denu/dene/denū/dená) and post-disambiguator winners. These
    are the surface-form randomization targets for archaic-feel
    generation. Dedupes against canonical_forms_lower so the pool only
    contains FORMS NEW TO THE GENERATOR — emitting "denu" when "denu" is
    also the canonical_form would be redundant.

    Drops "low-confidence singletons": variants whose total match_count
    is 1 AND every contributing row was produced by fuzzy-search or
    llm-disambiguator (vs reverse-search, which scans for canonical-form
    occurrences and is high-precision). A 1-edit-distance fuzzy hit with
    a single attestation in the corpus is the dominant OCR-noise class
    (saearp, drnim, etc.); requiring either ≥2 attestations or any
    high-confidence method removes them without throwing away legitimate
    archaic spellings.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_variants: dict[int, list[tuple[str, int]]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, matched_form, "
        f"  SUM(match_count) AS total_count, "
        f"  GROUP_CONCAT(DISTINCT method) AS methods "
        f"FROM etymon_text_match "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, LOWER(matched_form) "
        f"ORDER BY etymon_id, total_count DESC, matched_form",
        member_ids,
    ):
        if row["matched_form"].lower() in canonical_forms_lower:
            continue
        methods = set((row["methods"] or "").split(","))
        if row["total_count"] <= 1 and methods.issubset(_LOW_CONFIDENCE_METHODS):
            continue
        member_variants.setdefault(row["etymon_id"], []).append(
            (row["matched_form"], row["total_count"])
        )
    return member_variants


_NON_SCHOLAR_SOURCES = frozenset({"rando-port"})


def _fetch_member_citations(db: LexiconDB, member_ids: list[int]) -> dict[int, list[str]]:
    """Distinct sorted scholarly source_ids per member_id from
    etymon_citation. The runtime explainer surfaces this so a GM holding
    a generated name can see which scholars attest each morpheme.
    Sorted alphabetically for deterministic bundle output.

    Filters out non-scholarly seeds (rando-port — the Wikipedia-derived
    legacy bootstrap that ships every legacy etymon but isn't a real
    citation a GM would recognize). Members with no scholarly citations
    drop out of the result entirely so the emitter omits the
    `<lang>_citations` field on rando-only words.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_citations: dict[int, list[str]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, source_id "
        f"FROM etymon_citation "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, source_id "
        f"ORDER BY etymon_id, source_id",
        member_ids,
    ):
        if row["source_id"] in _NON_SCHOLAR_SOURCES:
            continue
        member_citations.setdefault(row["etymon_id"], []).append(row["source_id"])
    return member_citations


def _fetch_member_attested_years(db: LexiconDB, member_ids: list[int]) -> dict[int, int]:
    """Earliest attested year per member_id, drawn from BOTH row sources
    that carry year evidence (D5-1):

    * ``etymon_text_match.attested_year`` (PR #47 / wyrd-3ux) —
      reverse-search rows where the etymon's canonical form was found
      in a source body, paired with a form-attached year citation.
    * ``toponym_etymology.attested_year`` (PR #53 / wyrd-bag) — joined
      via toponym_etymology_element so a year cited against a toponym
      lands on each of its breakdown's element etymons.

    Returns ``{member_id: earliest_year}``; members with no attested
    year on either side are absent (caller emits no ``_attested_years``
    sibling for them — D5-2 generator interprets None as 'no era
    filter applies'). Sorted output is incidental — the dict is built
    by iterating SQL results, then the consumer re-sorts when emitting.
    """
    # Use a CTE to bind member_ids exactly once. A naive UNION ALL with
    # two `IN (?,?,?)` branches would require duplicating the bind list,
    # which is fragile if a third year-source is added in future. The
    # CTE makes the contract explicit: 'these are the etymons we care
    # about; gather years from any source that mentions them'.
    targets_values = ",".join("(?)" for _ in member_ids)
    member_years: dict[int, int] = {}
    cur = db.conn.execute(
        f"""
        WITH targets(etymon_id) AS (VALUES {targets_values})
        SELECT etymon_id, MIN(year) AS earliest_year FROM (
            SELECT etm.etymon_id, etm.attested_year AS year
            FROM etymon_text_match etm
            JOIN targets t ON t.etymon_id = etm.etymon_id
            WHERE etm.attested_year IS NOT NULL
            UNION ALL
            SELECT tee.etymon_id, te.attested_year AS year
            FROM toponym_etymology_element tee
            JOIN targets t ON t.etymon_id = tee.etymon_id
            JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
            WHERE te.attested_year IS NOT NULL
        )
        GROUP BY etymon_id
        """,
        member_ids,
    )
    for row in cur:
        member_years[row["etymon_id"]] = row["earliest_year"]
    return member_years


def _compute_member_descendants(member_rows: list[Any]) -> dict[int, list[int]]:
    """Transitive descendants per member_id (DB-free DFS).

    Lets a reflex linked to a lemma pick up the lemma's inflected children
    + OCR-cluster losers rather than just the directly-linked etymon. The
    graph is shallow (D22 flatten-at-merge-time keeps chains at depth ≤
    2), so a single inversion pass suffices.
    """
    children: dict[int, list[int]] = {}
    for r in member_rows:
        parent_id = r["lemma_id"] or r["merged_into_id"]
        if parent_id is not None:
            children.setdefault(parent_id, []).append(r["id"])
    member_descendants: dict[int, list[int]] = {}
    for r in member_rows:
        member_id = r["id"]
        descendants = [member_id]
        stack = list(children.get(member_id, []))
        while stack:
            d = stack.pop()
            if d in descendants:
                continue
            descendants.append(d)
            stack.extend(children.get(d, []))
        member_descendants[member_id] = descendants
    return member_descendants


def _group_families_into_subjects(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group families by (modifier_type, glosses, tags) into meanings.json subjects.

    Each group becomes one subject. Reflexes linked to any family in the
    group become "words"; families without any reflex contribute a synthesized
    word using the family's canonical_form (for newly-mined etymons that
    haven't been wired into a modern surface form yet).
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for fam in families:
        key = (
            fam["modifier_type"] or "",
            tuple(sorted(fam["glosses"])),
            tuple(sorted(fam["tags"])),
        )
        groups.setdefault(key, []).append(fam)

    subjects: list[dict[str, Any]] = []
    for (mod_type, glosses_tuple, tags_tuple), fams in groups.items():
        words = _build_words_for_group(fams)
        if not words:
            continue
        subject: dict[str, Any] = {
            "meaning": list(glosses_tuple),
            "modifier_tags": list(tags_tuple),
            "modifier_type": mod_type or None,
            "words": words,
        }
        subjects.append(subject)

    # Fully-discriminating sort key: every field that varies across subjects
    # must be in the tuple. Otherwise ties fall back to dict insertion order,
    # which traces back to AUTOINCREMENT root_ids and re-shuffles on rebuild.
    subjects.sort(
        key=lambda s: (
            s.get("modifier_type") or "",
            tuple(s["meaning"]),
            tuple(s["modifier_tags"]),
            tuple(w["modern_usage"] for w in s["words"]),
        )
    )
    return subjects


def _build_words_for_group(fams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble the `words` list for a subject grouping multiple families.

    Sort families deterministically, partition into reflex-linked vs.
    reflex-less, then dispatch each group to a focused word-builder.
    """
    # Sort families by canonical_form/language so the export output is
    # deterministic — root_ids are AUTOINCREMENT and therefore unstable
    # across DB rebuilds, which would otherwise churn the diff in version
    # control even when no semantic content changed.
    fams = sorted(fams, key=lambda f: (f["root_canonical_form"], f["root_language"]))

    reflex_to_links, reflex_meta, families_without_reflex = _partition_families_by_reflex(fams)

    words: list[dict[str, Any]] = []
    for reflex_id in sorted(
        reflex_to_links,
        key=lambda rid: (reflex_meta[rid]["position"], reflex_meta[rid]["surface_form"]),
    ):
        words.append(_word_for_reflex(reflex_meta[reflex_id], reflex_to_links[reflex_id]))

    for fam in families_without_reflex:
        words.append(_synthesize_word_for_family(fam))

    return words


def _partition_families_by_reflex(
    fams: list[dict[str, Any]],
) -> tuple[
    dict[int, list[tuple[dict[str, Any], list[int]]]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Split families into reflex-linked and reflex-less groups.

    Returns ``(reflex_to_links, reflex_meta, families_without_reflex)``.
    ``reflex_to_links[reflex_id]`` is a list of ``(family, linked_member_ids)``
    tuples so each reflex's word entry only includes language forms from
    the etymons it's actually linked to, not from every family in the
    subject. Per the original meanings.json shape (e.g. "Alder tree"),
    reflexes are language-specific: '-farne' carries celtic_mix only,
    'Alder-' carries old_english only. seed_from_meanings preserves this
    in reflex_etymon, and the export must too.
    """
    reflex_to_links: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    reflex_meta: dict[int, dict[str, Any]] = {}
    families_without_reflex: list[dict[str, Any]] = []
    for fam in fams:
        if not fam["reflexes"]:
            families_without_reflex.append(fam)
            continue
        for r in fam["reflexes"]:
            reflex_meta[r["id"]] = r
            reflex_to_links.setdefault(r["id"], []).append((fam, r["linked_member_ids"]))
    return reflex_to_links, reflex_meta, families_without_reflex


@dataclass
class _WordLanguageAccumulators:
    """Per-language accumulators populated during family-walk emission.

    Bundle of the 5 dicts that ``_word_for_reflex`` and
    ``_synthesize_word_for_family`` independently maintain in lockstep
    (same keys, populated by the same absorb_* helpers, drained into
    ``_emit_word_languages`` together). Holding them in one object
    keeps the call signature down to one positional arg per consumer
    and makes 'add a new per-language sibling field' a one-line edit
    (D26 pattern) rather than a 6-touch-site refactor.

    wyrd-k55 (PR-review-loop deferred): consolidates what used to be
    five separate locals declared / passed / absorbed in two parallel
    functions.
    """

    forms_by_lang: dict[str, list[str]] = field(default_factory=dict)
    variants: dict[str, dict[str, int]] = field(default_factory=dict)
    inflections: dict[str, dict[str, str]] = field(default_factory=dict)
    citations: dict[str, set[str]] = field(default_factory=dict)
    attested_years: dict[str, dict[str, int]] = field(default_factory=dict)
    # wyrd-vsrn Phase 2c: per-language english_shaped pool, keyed by
    # (lang, canonical_form) → english_shaped. Sparse — only non-Latin-
    # source-lang rows whose wyrd-ha9q derive_english_shaped produced a
    # non-None value land here. Empty dict for Latin-script langs +
    # rows that lacked sufficient transliteration / IPA input.
    english_shaped: dict[str, dict[str, str]] = field(default_factory=dict)
    # wyrd-qhs0 Phase 2d: the other three wyrd-ha9q rendering columns,
    # all per-(lang, canonical_form). Together with english_shaped these
    # are the four renderings the SPA's etymological-provenance panel
    # surfaces (D31 four-rendering rule).
    original_script: dict[str, dict[str, str]] = field(default_factory=dict)
    transliteration: dict[str, dict[str, str]] = field(default_factory=dict)
    # Pronunciation pairs IPA + dialect tag; the bucket value is a dict
    # {"ipa": str, "dialect": str | None}. NULL pronunciation_dialect
    # surfaces as a None value for "dialect".
    pronunciation: dict[str, dict[str, dict[str, str | None]]] = field(default_factory=dict)
    # wyrd-lr4 Phase 2: per-(lang, canonical_form) within-language
    # stratum tag. Sparse — only languages with a Phase 1 classifier
    # populate this (Welsh-family today). Other languages stay empty.
    stratum: dict[str, dict[str, str]] = field(default_factory=dict)


def _word_for_reflex(
    meta: dict[str, Any], link_pairs: list[tuple[dict[str, Any], list[int]]]
) -> dict[str, Any]:
    """Assemble one word entry for a reflex linked to one or more families.

    Walks each linked etymon's descendants so the reflex picks up the
    lemma's inflected children (D8) and OCR-cluster losers (D22) —
    not just the seeded etymon itself.
    """
    accs = _WordLanguageAccumulators()
    for fam, linked_ids in link_pairs:
        for member_id in linked_ids:
            for descendant_id in fam["member_descendants"][member_id]:
                lang, form = fam["member_form_by_id"][descendant_id]
                bucket = accs.forms_by_lang.setdefault(lang, [])
                if form not in bucket:
                    bucket.append(form)
                _absorb_member_variants(accs, fam, descendant_id, lang)
                _absorb_member_inflection(accs, fam, descendant_id, lang, form)
                _absorb_member_citations(accs, fam, descendant_id, lang)
                _absorb_member_attested_years(accs, fam, descendant_id, lang, form)
                _absorb_member_english_shaped(accs, fam, descendant_id, lang, form)
                _absorb_member_original_script(accs, fam, descendant_id, lang, form)
                _absorb_member_transliteration(accs, fam, descendant_id, lang, form)
                _absorb_member_pronunciation(accs, fam, descendant_id, lang, form)
                _absorb_member_stratum(accs, fam, descendant_id, lang, form)
    word: dict[str, Any] = {"modern_usage": meta["surface_form"]}
    _emit_word_languages(word, accs)
    _emit_era_reflexes(word, link_pairs)
    return word


def _emit_era_reflexes(
    word: dict[str, Any],
    link_pairs: list[tuple[dict[str, Any], list[int]]],
) -> None:
    """wyrd-obpw Phase 3.3 + wyrd-jbcu source-aware schema: stamp the
    family root's era_reflexes onto the word dict. Each linked family
    contributes its root's per-target-language reflex list; multiple
    linked families merge per target language with same-form
    collisions resolved by source quality (higher-quality source wins).

    Bundle field: ``era_reflexes`` is ``{target_language: [{form,
    source}, ...]}`` per word. Empty / absent for words whose linked
    families have no era data (proto-languages, untracked classical
    families, or roots whose cluster has no English-family targets).
    """
    merged: dict[str, dict[str, str]] = {}
    for fam, _linked_ids in link_pairs:
        for target_language, entries in fam.get("era_reflexes", {}).items():
            bucket = merged.setdefault(target_language, {})
            for entry in entries:
                form = entry["form"]
                source = entry["source"]
                existing = bucket.get(form)
                if existing is None or _better_era_reflex_source(source, existing):
                    bucket[form] = source
    if merged:
        word["era_reflexes"] = {
            target_language: [{"form": form, "source": forms[form]} for form in sorted(forms)]
            for target_language, forms in sorted(merged.items())
        }


def _synthesize_word_for_family(fam: dict[str, Any]) -> dict[str, Any]:
    """Assemble a synthesized word for a family that has no linked reflex.

    Uses the family's canonical_forms en bloc (no per-reflex narrowing
    applies). The whole family's variants and inflections fold in by
    matching language.
    """
    word: dict[str, Any] = {"modern_usage": _synthesize_modern_usage(fam)}
    accs = _WordLanguageAccumulators(
        forms_by_lang={lang: list(fam["forms_by_lang"][lang]) for lang in fam["forms_by_lang"]},
    )
    for member_id, (member_lang, member_form) in fam["member_form_by_id"].items():
        _absorb_member_variants(accs, fam, member_id, member_lang)
        _absorb_member_inflection(accs, fam, member_id, member_lang, member_form)
        _absorb_member_citations(accs, fam, member_id, member_lang)
        _absorb_member_attested_years(accs, fam, member_id, member_lang, member_form)
        _absorb_member_english_shaped(accs, fam, member_id, member_lang, member_form)
        _absorb_member_original_script(accs, fam, member_id, member_lang, member_form)
        _absorb_member_transliteration(accs, fam, member_id, member_lang, member_form)
        _absorb_member_pronunciation(accs, fam, member_id, member_lang, member_form)
        _absorb_member_stratum(accs, fam, member_id, member_lang, member_form)
    _emit_word_languages(word, accs)
    # Synthesized word case: link_pairs structure isn't used here, so
    # build a single-element link_pairs from the family directly.
    _emit_era_reflexes(word, [(fam, list(fam["member_form_by_id"].keys()))])
    return word


def _absorb_member_variants(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate D18 spelling variants for one (member, language) into the
    per-language pool, summing weights on collision. Caller guarantees
    `lang` matches the member's language so callers don't accidentally
    cross-pollinate across languages."""
    for variant_form, weight in fam.get("member_variants", {}).get(member_id, []):
        lang_variants = accs.variants.setdefault(lang, {})
        lang_variants[variant_form] = lang_variants.get(variant_form, 0) + weight


def _absorb_member_inflection(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """Record a member's D8 inflection label (if any) keyed by its surface
    form. Lemmas have inflection=None and are skipped — only inflected
    children carry a grammatical-case label worth surfacing."""
    inflection = fam.get("member_inflection_by_id", {}).get(member_id)
    if inflection:
        accs.inflections.setdefault(lang, {})[form] = inflection


def _absorb_member_attested_years(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """Record a member's earliest attested year (D5-1, wyrd-bag) keyed
    by its surface form. Members with no attested year are skipped —
    the runtime generator interprets the absence as 'no era constraint
    applies' (treat the form as always-includable under any --era)."""
    year = fam.get("member_attested_years", {}).get(member_id)
    if year is not None:
        accs.attested_years.setdefault(lang, {})[form] = year


def _absorb_member_english_shaped(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-vsrn Phase 2c: record a member's english_shaped rendering
    keyed by its canonical_form. Skipped when the column is NULL or
    empty: the runtime treats the absence as 'use canonical_form for
    display' for that member. NULL is the documented case (Latin-script
    source langs OR rows that lacked transliteration / IPA inputs);
    empty-string would be an unexpected DB shape but the predicate
    covers both for safety."""
    shaped = fam.get("member_english_shaped_by_id", {}).get(member_id)
    if shaped:
        accs.english_shaped.setdefault(lang, {})[form] = shaped


def _absorb_member_original_script(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's vocalized native-script
    form (Hebrew niqqud, Arabic harakat, Egyptian hieroglyphic markup)
    keyed by canonical_form. NULL is the common case for Latin-script
    rows; skip without absorbing."""
    original = fam.get("member_original_script_by_id", {}).get(member_id)
    if original:
        accs.original_script.setdefault(lang, {})[form] = original


def _absorb_member_transliteration(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's academic Latin-script
    transliteration (with diacritics — ʿifrīt, rakṣasa, kɛ́lɛḇ) keyed
    by canonical_form."""
    translit = fam.get("member_transliteration_by_id", {}).get(member_id)
    if translit:
        accs.transliteration.setdefault(lang, {})[form] = translit


def _absorb_member_pronunciation(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's IPA + dialect pair keyed
    by canonical_form. The pair updates atomically (matches the upsert
    semantics in lexicon.py.upsert_etymon's CASE expression on dialect
    — wyrd-ha9q Phase 2a fix for IPA/dialect decoupling)."""
    pron = fam.get("member_pronunciation_by_id", {}).get(member_id)
    if pron:
        ipa, dialect = pron
        accs.pronunciation.setdefault(lang, {})[form] = {"ipa": ipa, "dialect": dialect}


def _absorb_member_stratum(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-lr4 Phase 2: record a member's within-language stratum tag
    keyed by canonical_form. Skipped when stratum is NULL (the common
    case for languages without a Phase 1 classifier — only Welsh-family
    etymons are populated today). The runtime treats absence as 'no
    stratum filter applies' for the consumer wired up in Phase 3."""
    stratum = fam.get("member_stratum_by_id", {}).get(member_id)
    if stratum:
        accs.stratum.setdefault(lang, {})[form] = stratum


def _absorb_member_citations(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate scholarly source_ids for one member into the per-language
    citation set (wyrd-9kh.1). Set semantics dedupe across the descendant
    walk; emit-time sorts deterministically. Caller guarantees `lang`
    matches the member's language."""
    citations = fam.get("member_citations", {}).get(member_id, [])
    if citations:
        accs.citations.setdefault(lang, set()).update(citations)


def _emit_word_languages(word: dict[str, Any], accs: _WordLanguageAccumulators) -> None:
    """Stamp per-language form arrays + sibling _variants /
    _inflections / _citations / _attested_years / _english_shaped
    metadata onto the word dict. Per D26, the metadata fields are
    sibling keys (``<lang>_variants``, ``<lang>_inflections``,
    ``<lang>_citations``, ``<lang>_attested_years``,
    ``<lang>_english_shaped``) so legacy loaders that ignore unknown
    fields keep working.

    Multiple lexicon codes can route to the SAME bundle bucket via
    `_LANG_CODE_TO_JSON_FIELD` — e.g. welsh + old-welsh + middle-welsh
    all land in `celtic_mix`, and wyrd-vsrn's wave-2 stack collapses
    he + hbo + sem-pro + sem-wes-pro + afa-pro into `hebrew`. We MUST
    union (not overwrite) per bundle bucket: if the inner loop rewrote
    `word[json_field]` on each iteration, only the last-sorted lexicon
    code's forms would survive.

    Aggregation is done per bucket via `_BucketAccumulator` so the
    forms / variants / inflections / citations / attested_years /
    english_shaped fields all union under the same bucket key in one
    pass; the final emit walks buckets in stable order so output is
    deterministic regardless of source-lang sort order.
    """
    buckets: dict[str, _BucketAccumulator] = {}
    for lang in sorted(accs.forms_by_lang):
        json_field = _LANG_CODE_TO_JSON_FIELD.get(lang)
        if not json_field:
            continue
        bucket = buckets.setdefault(json_field, _BucketAccumulator())
        for form in accs.forms_by_lang[lang]:
            if form not in bucket.forms_set:
                bucket.forms.append(form)
                bucket.forms_set.add(form)
        if lang in accs.variants:
            for form, weight in accs.variants[lang].items():
                bucket.variants[form] = bucket.variants.get(form, 0) + weight
        if lang in accs.inflections:
            bucket.inflections.update(accs.inflections[lang])
        if lang in accs.citations:
            bucket.citations.update(accs.citations[lang])
        if lang in accs.attested_years:
            for form, year in accs.attested_years[lang].items():
                # Keep the earliest year on collision (matches the
                # rest-of-pipeline ascending-year sort convention).
                existing = bucket.attested_years.get(form)
                if existing is None or year < existing:
                    bucket.attested_years[form] = year
        if lang in accs.english_shaped:
            bucket.english_shaped.update(accs.english_shaped[lang])
        if lang in accs.original_script:
            bucket.original_script.update(accs.original_script[lang])
        if lang in accs.transliteration:
            bucket.transliteration.update(accs.transliteration[lang])
        if lang in accs.pronunciation:
            bucket.pronunciation.update(accs.pronunciation[lang])
        if lang in accs.stratum:
            bucket.stratum.update(accs.stratum[lang])

    for json_field in sorted(buckets):
        bucket = buckets[json_field]
        word[json_field] = bucket.forms
        if bucket.variants:
            word[f"{json_field}_variants"] = _emit_variant_list(bucket.variants)
        if bucket.inflections:
            word[f"{json_field}_inflections"] = _emit_inflection_list(bucket.inflections)
        if bucket.citations:
            word[f"{json_field}_citations"] = sorted(bucket.citations)
        if bucket.attested_years:
            word[f"{json_field}_attested_years"] = _emit_attested_years_list(bucket.attested_years)
        if bucket.english_shaped:
            word[f"{json_field}_english_shaped"] = _emit_english_shaped_list(bucket.english_shaped)
        if bucket.original_script:
            word[f"{json_field}_original_script"] = _emit_original_script_list(
                bucket.original_script
            )
        if bucket.transliteration:
            word[f"{json_field}_transliteration"] = _emit_transliteration_list(
                bucket.transliteration
            )
        if bucket.pronunciation:
            word[f"{json_field}_pronunciation"] = _emit_pronunciation_list(bucket.pronunciation)
        if bucket.stratum:
            word[f"{json_field}_stratum"] = _emit_stratum_list(bucket.stratum)


@dataclass
class _BucketAccumulator:
    """Per-bundle-bucket aggregation state used by _emit_word_languages
    when multiple lexicon codes (e.g. welsh + old-welsh; he + hbo +
    sem-pro) collapse into one bundle field. Fields mirror the
    per-language pools on `_WordLanguageAccumulators` but keyed by the
    BUNDLE bucket so cross-language values union cleanly."""

    forms: list[str] = field(default_factory=list)
    forms_set: set[str] = field(default_factory=set)
    variants: dict[str, int] = field(default_factory=dict)
    inflections: dict[str, str] = field(default_factory=dict)
    citations: set[str] = field(default_factory=set)
    attested_years: dict[str, int] = field(default_factory=dict)
    english_shaped: dict[str, str] = field(default_factory=dict)
    # wyrd-qhs0 Phase 2d: the other three wyrd-ha9q rendering columns.
    original_script: dict[str, str] = field(default_factory=dict)
    transliteration: dict[str, str] = field(default_factory=dict)
    pronunciation: dict[str, dict[str, str | None]] = field(default_factory=dict)
    # wyrd-lr4 Phase 2: per-canonical_form within-language stratum tag.
    stratum: dict[str, str] = field(default_factory=dict)


def _emit_original_script_list(scripts: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: original_script}
    into the meanings.json original_script entry shape. Each dict has
    ``"form"`` (the canonical_form lookup key) and ``"original_script"``
    (the vocalized native-script form from wiktextract head_templates'
    `wv` / `head` arg). Sorted by form for deterministic output."""
    return [
        {"form": form, "original_script": original_form}
        for form, original_form in sorted(scripts.items())
    ]


def _emit_transliteration_list(translits: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: transliteration}
    into the meanings.json transliteration entry shape. Each dict has
    ``"form"`` (canonical_form) and ``"transliteration"`` (academic
    Latin-script form with diacritics, from wiktextract head_templates
    `tr`). Sorted by form for determinism."""
    return [
        {"form": form, "transliteration": translit_form}
        for form, translit_form in sorted(translits.items())
    ]


def _emit_pronunciation_list(
    pronunciations: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: {ipa, dialect}}
    into the meanings.json pronunciation entry shape. Each dict has
    ``"form"`` (canonical_form), ``"ipa"`` (the IPA string), and
    ``"dialect"`` (the first tag on the chosen sound entry, or None
    when the IPA was untagged-canonical). Sorted by form."""
    return [
        {"form": form, "ipa": data["ipa"], "dialect": data.get("dialect")}
        for form, data in sorted(pronunciations.items())
    ]


def _emit_stratum_list(strata: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-lr4 Phase 2: serialize {canonical_form: stratum} into the
    meanings.json stratum entry shape. Each dict has ``"form"`` (the
    canonical_form lookup key, same as the language form array) and
    ``"stratum"`` (the language-specific register tag — see the Phase
    1 classifier for the current vocabulary; Welsh ships with five:
    latin-loan / english-loan / brittonic-substrate / medieval-welsh /
    native-welsh, with French / OE / ON adding their own as Phase 4
    lands). Sorted by form for determinism."""
    return [{"form": form, "stratum": stratum_tag} for form, stratum_tag in sorted(strata.items())]


def _emit_english_shaped_list(shaped: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-vsrn Phase 2c: serialize {canonical_form: english_shaped} into
    the meanings.json english_shaped entry shape. Sorted by form so the
    output is deterministic regardless of aggregation order.

    Each output dict has two keys:
        ``"form"``           — the canonical_form string (the same key the
                                language form array uses, so the runtime
                                can match a chosen form to its shaping).
        ``"english_shaped"`` — the Latin-script rendering produced by
                                wyrd-ha9q's ``derive_english_shaped`` and
                                stored on ``etymon.english_shaped``.
    """
    return [
        {"form": form, "english_shaped": shaped_form}
        for form, shaped_form in sorted(shaped.items())
    ]


def _emit_attested_years_list(years: dict[str, int]) -> list[dict[str, Any]]:
    """Serialize ``{form: year}`` into the meanings.json attested-year
    entry shape. Sorted by ``(year, form)`` so output is deterministic
    and the chronologically-earliest entries lead — useful when the
    runtime --era filter walks the list looking for the first match."""
    return [
        {"form": form, "year": year}
        for form, year in sorted(years.items(), key=lambda kv: (kv[1], kv[0]))
    ]


def _emit_variant_list(variants: dict[str, int]) -> list[dict[str, Any]]:
    """Serialize {form: weight} into the meanings.json variant entry shape.
    Output is sorted by descending weight (most-attested first), with the
    form string as a stable secondary key — matters because two variants
    can share a weight after aggregation across the family."""
    return [
        {"form": form, "weight": weight}
        for form, weight in sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _emit_inflection_list(inflections: dict[str, str]) -> list[dict[str, Any]]:
    """Serialize {form: inflection_label} into the meanings.json inflection
    entry shape. Sorted by form so the output is deterministic regardless of
    aggregation order."""
    return [{"form": form, "inflection": label} for form, label in sorted(inflections.items())]


def _synthesize_modern_usage(family: dict[str, Any]) -> str:
    """Derive a modern_usage for a family that has no linked reflex.

    Uses ``position_pref`` to choose pre/post/inner dash markers; defaults
    to no-dash (the runtime treats undecorated usage as a post-suffix
    via Meaning._set_location).
    """
    form = family["root_canonical_form"]
    position = family.get("position_pref")
    if position == "pre":
        return f"{form}-"
    if position == "inner":
        return f"-{form}-"
    if position == "post":
        return f"-{form}"
    return form


def _orphan_reflex_subjects(db: LexiconDB) -> list[dict[str, Any]]:
    """Emit one subject per reflex that has no linked etymon.

    These come from the rando-port seed where a `word` entry had only
    `modern_usage` (e.g., 'Adam-' as a saint's name) or had a language slot
    with an empty form list (e.g., {'celtic_mix': [], 'modern_usage': 'Bre-'}).
    The runtime needs them in `meaning_db` because per-culture proportions
    JSONs reference them by usage. Emit each as its own subject with empty
    glosses/tags so the runtime can register the usage; promotion to a richer
    subject can land later via the manual seed-source-aware revisit (D7).
    """
    rows = db.conn.execute(
        """
        SELECT r.surface_form
        FROM reflex r
        WHERE NOT EXISTS (
            SELECT 1 FROM reflex_etymon re WHERE re.reflex_id = r.id
        )
        ORDER BY r.position, r.surface_form
        """
    ).fetchall()
    return [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [{"modern_usage": row["surface_form"]}],
        }
        for row in rows
    ]
