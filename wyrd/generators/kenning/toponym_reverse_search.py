"""Toponym reverse-search via source-body pattern mining — wyrd-x82p Phase 1.

Companion to ``mine-attestations`` (wyrd-skm Phase 3.0a). The original
mine-attestations command extracts (form, year) pairs from
``toponym_etymology.notes`` — but those are only available for
toponyms the LLM extractor already decomposed. Many scholar
gazetteer entries also mention place names IN PASSING, in cross-
references like "compare Suttone (Domesday)" or "as found in
Birmingham, 1086".

This module walks each scholar source's BODY TEXT (not just
extracted notes) for the same (form, year) patterns, then maps each
form back to a known toponym in the DB. New attestations from the
broader text get added without needing fresh LLM extraction.

Form-to-toponym mapping:
  * exact match on ``toponym.modern_name`` (case-insensitive)
  * exact match on any existing ``toponym_attestation.form``
    (case-insensitive) for any toponym in the DB

Unmatched forms — place names that scholar prose mentions but our
toponym table doesn't carry — are tallied so the LLM-extraction
pass (wyrd-x82p Phase 2) can target them for new-toponym discovery.
This module deliberately does NOT create new toponym rows; only
emits attestations for known ones.

Dedup: the existing ``idx_attestation_unique`` on
``(toponym_id, form, date_year, source_doc)`` handles re-runs.
Re-running against the same source body is a no-op (every match
INSERT OR IGNOREs against the existing row).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the well-tested (form, year) extractor from mine-attestations.
# It's pure (no DB), takes arbitrary text, returns deduped (form, year)
# tuples. Underscore prefix is convention only — same-package import
# is appropriate here.
from .lexicon import _extract_attestation_pairs


@dataclass
class ReverseSearchReport:
    """Per-source counters for a reverse-search run.

    Each field counts (form, year) pairs that reached a specific
    branch of :func:`reverse_search_source`:

    * ``pairs_extracted`` — pairs admitted by the
      :func:`_extract_attestation_pairs` filter chain (form-quality,
      year-range, page-marker, source-attribution guards). Pairs
      suppressed by that chain are invisible to this counter; see
      wyrd-9ekl for the body-context over-suppression follow-up.
    * ``matched`` — pairs whose form mapped to a known toponym in
      the form→toponym lookup. Eligible for attestation insert.
    * ``unmatched`` — pairs whose form didn't correspond to any
      toponym in the DB. These are Phase 2 LLM-mining candidates
      for new-toponym discovery.
    * ``inserted`` — under ``apply=True``, the count of NEW rows
      that landed in ``toponym_attestation``. Idempotent re-runs
      produce 0 inserts and the matched count flows to
      ``already_present`` instead.
    * ``already_present`` — matched pairs whose
      ``(toponym_id, form, date_year, source_doc)`` tuple was
      already in the DB (UNIQUE-index dedup). Accurate in BOTH
      dry-run and apply modes — the preflight uniqueness SELECT
      computes this regardless of the apply flag so operators
      can predict --apply's effect from dry-run output.
    """

    source_id: str
    pairs_extracted: int = 0
    matched: int = 0
    unmatched: int = 0
    inserted: int = 0
    already_present: int = 0
    # wyrd-9ekl: count of source-attribution-chain pattern matches in
    # the body. With context_is_body=True the extractor admits these
    # as legitimate attestations rather than suppressing, but the
    # counter still surfaces pattern prevalence so operators can see
    # how often the `<year> FORM, <year>` shape occurs.
    suppressed_by_source_chain: int = 0
    unmatched_samples: list[tuple[str, int]] = field(default_factory=list)


# Surrounding quote/bracket chars stripped from BOTH ends — ASCII plus the
# Unicode typographic quotes scholarly prose / OCR / LLM output actually use:
# curly double “ ” (U+201C/D), curly single ‘ ’ (U+2018/9), guillemets « » ‹ ›
# (U+00AB/BB, U+2039/A), low quotes „ ‚ (U+201E, U+201A).
_MATCH_STRIP_ENDS = "\"'([{<~" + "“”‘’«»‹›„‚"
# Trailing-only punctuation strip (closing punctuation + closing quotes), so a
# closing quote that sits AFTER trailing punctuation (``…”.``) still unwraps.
_MATCH_STRIP_TRAILING = ",.;:!?)'\"]}>~" + "”’»›"


def _normalize_for_match(form: str) -> str:
    """Lowercase + strip surrounding whitespace + drop both LEADING
    and TRAILING punctuation. Mirrors the lightweight normalization
    the matching side of mine-attestations already applies; toponym
    .modern_name values in the DB are mixed-case (Birmingham, Castle
    Cary), so a case-insensitive lookup is what makes the
    form-to-toponym join practical.

    Leading-punctuation strip catches scholar-prose shapes like
    ``"Birmingham`` (opening quote), ``(Birmingham`` (parenthetical
    intro), or ``[Birmingham`` (editor bracket). Without this strip,
    these forms would silently never match a known toponym
    (Gemini PR #212 round-2 finding). The strip sets cover the Unicode
    quote variants too — see :data:`_MATCH_STRIP_ENDS`.
    """
    return form.strip().strip(_MATCH_STRIP_ENDS).rstrip(_MATCH_STRIP_TRAILING).lower()


def _build_form_to_toponym_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build a single in-memory dict mapping normalized form →
    toponym_id, drawn from both ``toponym.modern_name`` and any
    existing ``toponym_attestation.form``.

    **Ambiguity handling**: forms that map to MULTIPLE toponym_ids
    (homonyms — Newton across counties, Hereford the city vs. other
    Herefords, etc.) are EXCLUDED from the returned lookup, so
    :func:`reverse_search_source` counts them as ``unmatched``
    rather than guessing which homonym the scholar prose meant.
    Phase 2's LLM extraction with surrounding-context understanding
    is the right place to disambiguate; Phase 1 should not silently
    pick the wrong one (Gemini PR #212 round-2 finding: the earlier
    "lowest id wins" rule produced wrong assignments — Hereford in
    a Herefordshire source body was mapping to Hereford@Cheshire).

    Modern names AND attestation forms both contribute to the set
    of ids per form. The ambiguity check fires when the union has
    more than one id, regardless of which source layer contributed.
    """
    form_to_ids: dict[str, set[int]] = {}
    for row in conn.execute("SELECT id, modern_name FROM toponym"):
        key = _normalize_for_match(row["modern_name"])
        if key:
            form_to_ids.setdefault(key, set()).add(row["id"])
    for row in conn.execute("SELECT toponym_id, form FROM toponym_attestation"):
        key = _normalize_for_match(row["form"])
        if key:
            form_to_ids.setdefault(key, set()).add(row["toponym_id"])
    # Only include unambiguous matches (exactly one toponym id).
    # Multi-id collisions become "unmatched" downstream — Phase 2's
    # LLM extraction can use surrounding source-body context to
    # disambiguate; Phase 1's pattern-based search cannot, so it
    # defers rather than guess.
    return {key: next(iter(ids)) for key, ids in form_to_ids.items() if len(ids) == 1}


def _load_source_body(sources_dir: Path, source_id: str) -> str | None:
    """Read ``<sources_dir>/<source_id>.txt`` and return the raw
    body text. Returns None when:

    * the file doesn't exist — many sources have JSONL but no
      committed .txt (operator decision per book).
    * the file isn't valid UTF-8 — surface as None rather than
      crashing a batch run.
    * the file is empty / whitespace-only — functionally same as
      missing.

    Strips path components from ``source_id`` via ``Path(...).name``
    to prevent traversal even if a future ingester writes a malformed
    identifier (mirrors the hardening pattern in short_quote_refill).
    """
    path = sources_dir / f"{Path(source_id).name}.txt"
    if not path.exists():
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if not body.strip():
        return None
    return body


# Cap on per-source unmatched-sample list. 10 was the original
# parameter default; the parameter has been removed because no caller
# threaded it through (code-reviewer PR #212 finding) and operators
# don't need finer-grained control than what verbose-mode display
# already provides.
_UNMATCHED_SAMPLE_LIMIT = 10


def reverse_search_source(
    conn: sqlite3.Connection,
    source_id: str,
    sources_dir: Path,
    form_to_toponym: dict[str, int],
    *,
    apply: bool = False,
) -> ReverseSearchReport:
    """Walk one source body, extract (form, year) attestation pairs
    via the mine-attestations regex set, map each form to a known
    toponym, and (with ``apply=True``) INSERT OR IGNORE the
    corresponding ``toponym_attestation`` rows.

    Re-running against the same source is a no-op — the existing
    ``idx_attestation_unique`` UNIQUE constraint silently drops the
    duplicates and they're counted as ``already_present`` rather than
    ``inserted``.

    ``form_to_toponym`` is built once by the caller for performance;
    walking 48 sources without that pre-build would re-execute two
    full-table scans per source (one for ``toponym``, one for
    ``toponym_attestation``).

    Dry-run semantics: with ``apply=False``, the function still
    performs the preflight uniqueness check via SELECT so
    ``inserted`` and ``already_present`` accurately predict what
    ``apply=True`` would do. Without the preflight, dry-run would
    show ``already_present=0`` for every source — operators
    couldn't distinguish "N net-new inserts pending" from "0 net
    inserts (all dupes)". silent-failure-hunter PR #212 finding.

    Transaction control: this function does NOT commit. Callers own
    commit granularity (the CLI commits after walking ALL sources,
    so partial-failure rollback is the natural shape). code-reviewer
    PR #212 finding.
    """
    report = ReverseSearchReport(source_id=source_id)
    body = _load_source_body(sources_dir, source_id)
    if body is None:
        return report
    # wyrd-9ekl: body-text context — disable the source-attribution-
    # chain guard (calibrated for short notes strings; over-suppresses
    # in long body text where `<year> FORM, <year>` is dominantly a
    # legitimate attestation chain). Telemetry dict surfaces any
    # would-be suppressions even though the guard is now off.
    telemetry: dict[str, int] = {}
    pairs = _extract_attestation_pairs(body, context_is_body=True, telemetry=telemetry)
    report.suppressed_by_source_chain = telemetry.get("suppressed_by_source_chain", 0)
    report.pairs_extracted = len(pairs)
    for form, year in pairs:
        key = _normalize_for_match(form)
        toponym_id = form_to_toponym.get(key)
        if toponym_id is None:
            report.unmatched += 1
            if len(report.unmatched_samples) < _UNMATCHED_SAMPLE_LIMIT:
                report.unmatched_samples.append((form, year))
            continue
        report.matched += 1
        # Preflight uniqueness check: would this row be a new insert?
        # SELECT against the UNIQUE-index columns answers without
        # mutating state, so dry-run can predict apply=True's
        # already_present count accurately.
        existing = conn.execute(
            """SELECT 1 FROM toponym_attestation
                WHERE toponym_id = ? AND form = ? AND date_year = ?
                  AND source_doc = ? LIMIT 1""",
            (toponym_id, form, year, source_id),
        ).fetchone()
        if existing is not None:
            report.already_present += 1
            continue
        if apply:
            conn.execute(
                """INSERT OR IGNORE INTO toponym_attestation
                   (toponym_id, form, date_year, source_doc)
                   VALUES (?, ?, ?, ?)""",
                (toponym_id, form, year, source_id),
            )
        report.inserted += 1
    return report
