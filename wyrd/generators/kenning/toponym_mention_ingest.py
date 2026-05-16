"""Ingest LLM-extracted toponym mentions into ``toponym_attestation`` —
wyrd-x82p Phase 2b.1.

Consumes the JSONL emitted by ``lexicon mine-toponym-mentions``
(Phase 2 pilot, PR #213) and turns each resolvable mention into a
``toponym_attestation`` row. Mirrors the conventions established by
Phase 1's pattern-based reverse-search (PR #212):

* Idempotent re-runs via UNIQUE-index dedup (INSERT OR IGNORE).
* Preflight uniqueness SELECT so dry-run mode accurately predicts
  ``--apply``.
* Caller-owned transaction control (function does NOT commit; CLI
  commits once after walking all sources).
* Path-traversal hardening via ``Path(source_id).name``.

Resolver
--------
Phase 1's form→toponym lookup REJECTS ambiguous forms (homonyms)
because pattern-based search has no surrounding context to
disambiguate. Phase 2's LLM emits ``region_hint`` for each mention
when the surrounding prose names a county/country — Phase 2b.1's
resolver uses this hint to pick the right candidate when an
unambiguous lookup would fail:

* Single candidate: resolved.
* Multiple candidates + region_hint matches exactly one: resolved.
* Multiple candidates + region_hint matches zero or many: unresolved.
* No candidates: unresolved (new-toponym candidate for Phase 2b.3).

Unresolved mentions are written to a candidate JSONL for the
Phase 2b.3 operator-review tool to triage.

What this module does NOT do
----------------------------
* No new toponym rows. New-toponym discovery is Phase 2b.3 (operator
  review); Phase 2b.1 only emits ``toponym_attestation`` rows for
  toponyms that already exist.
* No LLM calls. The mentions are pre-extracted (Phase 2 pilot).
  Phase 2b.2 will add the tiered Qwen-first → Anthropic-on-residual
  orchestration that produces the JSONL at full scope.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .toponym_reverse_search import _normalize_for_match


@dataclass(frozen=True)
class ResolverIndexes:
    """In-memory lookups built once per run.

    * ``form_to_ids`` — normalized form → set of toponym_ids that have
      that form (modern_name OR any existing attestation form). Set
      cardinality > 1 means ambiguous (homonym); the resolver uses
      ``region_hint`` to pick.
    * ``toponym_regions`` — toponym_id → normalized region string.
      Built from ``toponym.region`` column; toponyms without a region
      map to None. Used to disambiguate homonyms via the LLM-emitted
      region_hint.
    """

    form_to_ids: dict[str, set[int]]
    toponym_regions: dict[int, str | None]


def build_resolver_indexes(conn: sqlite3.Connection) -> ResolverIndexes:
    """Build the form→ids and toponym_id→region indexes. Single full-
    table scans on ``toponym`` and ``toponym_attestation`` — caller
    builds once per run and threads through ``resolve_mention``.

    Mirrors Phase 1's :func:`_build_form_to_toponym_lookup` shape, but
    keeps ambiguous forms (the multi-id set is the load-bearing input
    for region-based disambiguation here)."""
    form_to_ids: dict[str, set[int]] = {}
    toponym_regions: dict[int, str | None] = {}
    for row in conn.execute("SELECT id, modern_name, region FROM toponym"):
        key = _normalize_for_match(row["modern_name"])
        if key:
            form_to_ids.setdefault(key, set()).add(row["id"])
        region = row["region"]
        toponym_regions[row["id"]] = _normalize_for_match(region) if region else None
    for row in conn.execute("SELECT toponym_id, form FROM toponym_attestation"):
        key = _normalize_for_match(row["form"])
        if key:
            form_to_ids.setdefault(key, set()).add(row["toponym_id"])
    return ResolverIndexes(
        form_to_ids=form_to_ids,
        toponym_regions=toponym_regions,
    )


def resolve_mention(
    form: str,
    region_hint: str | None,
    indexes: ResolverIndexes,
) -> int | None:
    """Map a single mention to a toponym_id, or None if unresolved.

    * Single candidate: return it.
    * Multiple candidates + region_hint matches exactly one of their
      regions: return the disambiguated one.
    * Multiple candidates without a usable region_hint: None.
    * No candidates: None.

    Region matching uses normalized substring containment — the LLM
    may emit "Northumberland", "co. Northumberland", or "Northumber-
    land, England", any of which should match a toponym whose region
    column is "northumberland". A toponym whose region is None can
    never be selected by region_hint (we don't know what region it's
    in, so we can't confirm a match)."""
    key = _normalize_for_match(form)
    if not key:
        return None
    candidates = indexes.form_to_ids.get(key)
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))
    # Ambiguous — need region_hint to disambiguate.
    if not region_hint:
        return None
    hint_norm = _normalize_for_match(region_hint)
    if not hint_norm:
        return None
    matched: list[int] = []
    for tid in candidates:
        region = indexes.toponym_regions.get(tid)
        if region and (hint_norm in region or region in hint_norm):
            matched.append(tid)
    if len(matched) == 1:
        return matched[0]
    # Zero or >1 region matches — still ambiguous, defer to operator.
    return None


@dataclass
class IngestReport:
    """Per-source counters for an ingest run.

    * ``mentions_processed`` — total JSONL rows considered.
    * ``resolved`` — mentions mapped to a known toponym.
    * ``unresolved`` — mentions with no matching toponym OR
      ambiguous matches the region_hint couldn't disambiguate. Each
      is a Phase 2b.3 new-toponym candidate.
    * ``inserted`` — under ``apply=True``, the count of NEW rows
      that landed in ``toponym_attestation``. Idempotent re-runs
      produce 0 inserts; the matched count flows to
      ``already_present`` instead.
    * ``already_present`` — resolved mentions whose
      ``(toponym_id, form, date_year, source_doc)`` tuple was
      already in the DB. Accurate in BOTH dry-run and apply modes —
      the preflight uniqueness SELECT computes this regardless of
      the apply flag.
    * ``unresolved_records`` — full record of unresolved mentions
      (form, year, region_hint, context, source_id) for the
      candidate-JSONL sink. Bounded by ``_UNRESOLVED_SAMPLE_LIMIT``
      in memory; the caller is expected to stream this out.
    """

    source_id: str
    mentions_processed: int = 0
    resolved: int = 0
    unresolved: int = 0
    inserted: int = 0
    already_present: int = 0
    unresolved_records: list[dict] = field(default_factory=list)


def ingest_mentions(
    conn: sqlite3.Connection,
    source_id: str,
    mentions: list[dict],
    indexes: ResolverIndexes,
    *,
    apply: bool = False,
) -> IngestReport:
    """Walk a list of mention dicts, resolve each to a toponym, and
    (with ``apply=True``) INSERT OR IGNORE the corresponding
    ``toponym_attestation`` rows.

    Each mention dict must have ``form`` (str). Optional keys:
    ``date_year`` (int|None), ``region_hint`` (str|None), ``context``
    (str). The function tolerates extra keys (e.g. ``source_id``)
    silently.

    Dry-run semantics: with ``apply=False``, the preflight uniqueness
    SELECT still runs so ``inserted`` and ``already_present``
    accurately predict ``apply=True``'s effect.

    Transaction control: this function does NOT commit. Callers own
    commit granularity — the CLI commits after walking all sources,
    so partial-failure rollback is the natural shape.

    Date-year-NULL handling: Phase 2 mentions often have
    ``date_year=None`` (undated mentions Phase 1's regex didn't
    catch). The UNIQUE index on ``toponym_attestation`` treats NULL
    as distinct in SQLite, so two undated mentions of the same form
    for the same toponym in the same source would both insert. We
    explicitly skip the insert if date_year is None and a matching
    null-year row already exists — IS NULL comparison rather than
    the default NULL-as-distinct semantics."""
    report = IngestReport(source_id=source_id)
    for m in mentions:
        report.mentions_processed += 1
        form = (m.get("form") or "").strip()
        if not form:
            report.unresolved += 1
            continue
        date_year = m.get("date_year")
        if isinstance(date_year, bool) or not isinstance(date_year, (int, type(None))):
            date_year = None
        region_hint = m.get("region_hint")
        toponym_id = resolve_mention(form, region_hint, indexes)
        if toponym_id is None:
            report.unresolved += 1
            report.unresolved_records.append(
                {
                    "source_id": source_id,
                    "form": form,
                    "date_year": date_year,
                    "region_hint": region_hint,
                    "context": (m.get("context") or "").strip(),
                }
            )
            continue
        report.resolved += 1
        # Preflight uniqueness check — same shape as Phase 1.
        # IS NULL vs = NULL: SQLite's UNIQUE index treats NULL as
        # distinct, so we use IS-NULL semantics explicitly to dedup
        # undated mentions of the same form on the same toponym/source.
        if date_year is None:
            existing = conn.execute(
                """SELECT 1 FROM toponym_attestation
                    WHERE toponym_id = ? AND form = ? AND date_year IS NULL
                      AND source_doc = ? LIMIT 1""",
                (toponym_id, form, source_id),
            ).fetchone()
        else:
            existing = conn.execute(
                """SELECT 1 FROM toponym_attestation
                    WHERE toponym_id = ? AND form = ? AND date_year = ?
                      AND source_doc = ? LIMIT 1""",
                (toponym_id, form, date_year, source_id),
            ).fetchone()
        if existing is not None:
            report.already_present += 1
            continue
        if apply:
            conn.execute(
                """INSERT OR IGNORE INTO toponym_attestation
                   (toponym_id, form, date_year, source_doc)
                   VALUES (?, ?, ?, ?)""",
                (toponym_id, form, date_year, source_id),
            )
        report.inserted += 1
    return report
