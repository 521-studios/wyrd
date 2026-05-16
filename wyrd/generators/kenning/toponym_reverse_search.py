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
    * ``already_present`` — under ``apply=True``, matched pairs
      whose ``(toponym_id, form, date_year, source_doc)`` tuple
      was already in the DB (UNIQUE-index dedup). Always 0 in
      dry-run mode — see :func:`reverse_search_source` docstring
      for the rationale and the operator-facing implication.
    """

    source_id: str
    pairs_extracted: int = 0
    matched: int = 0
    unmatched: int = 0
    inserted: int = 0
    already_present: int = 0
    unmatched_samples: list[tuple[str, int]] = field(default_factory=list)


def _normalize_for_match(form: str) -> str:
    """Lowercase + strip surrounding whitespace + drop trailing
    punctuation. Mirrors the lightweight normalization the matching
    side of mine-attestations already applies; toponym.modern_name
    values in the DB are mixed-case (Birmingham, Castle Cary), so a
    case-insensitive lookup is what makes the form-to-toponym join
    practical."""
    return form.strip().rstrip(",.;:").lower()


def _build_form_to_toponym_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build a single in-memory dict mapping normalized form →
    toponym_id, drawn from both ``toponym.modern_name`` and any
    existing ``toponym_attestation.form``.

    Collision precedence (deterministic):

    1. **modern_name always beats attestation_form** — when the same
       normalized form appears as both a ``toponym.modern_name`` and
       a ``toponym_attestation.form`` for a different toponym, the
       modern_name's toponym wins regardless of id. The first loop
       populates the dict before the second loop runs, and the
       second loop uses ``if key not in lookup``.
    2. **Among modern_name collisions, lowest toponym_id wins** —
       ``ORDER BY id`` + first-insert-wins. Multiple "Newton"
       toponyms across counties collapse to the lowest-id Newton.
    3. **Among attestation collisions, lowest attestation.id wins** —
       same ORDER-BY-id-and-first-insert mechanism applied to the
       attestation loop.

    Operator implication: when a passing scholar prose mention
    matches a form that resolves to a Newton, the lookup may map
    to a different Newton than the operator-expected one. The
    insert is still high-precision (it's a real attestation in
    that source's prose) but the toponym_id may not match the
    operator's "main" Newton. Curation pipeline + operator review
    catch this case-by-case.
    """
    lookup: dict[str, int] = {}
    # Modern names first — these are the authoritative forms.
    for row in conn.execute("SELECT id, modern_name FROM toponym ORDER BY id"):
        key = _normalize_for_match(row["modern_name"])
        if key and key not in lookup:
            lookup[key] = row["id"]
    # Then historical forms from existing attestations — these
    # supplement the lookup with already-known variant spellings
    # (Cestretone, Eboracum, etc.). Modern names take precedence
    # when there's a collision because they're the canonical
    # identity.
    for row in conn.execute("SELECT toponym_id, form FROM toponym_attestation ORDER BY id"):
        key = _normalize_for_match(row["form"])
        if key and key not in lookup:
            lookup[key] = row["toponym_id"]
    return lookup


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
    pairs = _extract_attestation_pairs(body)
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
