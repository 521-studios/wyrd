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
    """Per-source outcome.

    Counts emit one row per (form, year) pair that survived the
    full filter chain; ``matched`` is the count that mapped to a
    known toponym (and was therefore eligible for attestation
    insert); ``unmatched`` is the count whose form didn't correspond
    to any toponym in the DB (Phase 2 LLM-mining candidates).
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

    Multi-toponym collisions (the same form name maps to >1 toponym)
    are RESOLVED by keeping the lowest toponym_id — deterministic.
    These collisions are most often legitimate (e.g. multiple
    "Newton" toponyms exist across counties); we accept one canonical
    target per form and trust the operator-driven curation pipeline
    to refine if a specific collision matters.
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
    """Read ``sources/<source_id>.txt`` and return the raw body text.
    Returns None when:

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


def reverse_search_source(
    conn: sqlite3.Connection,
    source_id: str,
    sources_dir: Path,
    form_to_toponym: dict[str, int],
    *,
    apply: bool = False,
    unmatched_sample_limit: int = 10,
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
    walking 48 sources without that pre-build would re-query the DB
    21,969 times per source.
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
            if len(report.unmatched_samples) < unmatched_sample_limit:
                report.unmatched_samples.append((form, year))
            continue
        report.matched += 1
        if not apply:
            continue
        cur = conn.execute(
            """INSERT OR IGNORE INTO toponym_attestation
               (toponym_id, form, date_year, source_doc)
               VALUES (?, ?, ?, ?)""",
            (toponym_id, form, year, source_id),
        )
        if cur.rowcount == 1:
            report.inserted += 1
        else:
            report.already_present += 1
    if apply:
        conn.commit()
    return report
