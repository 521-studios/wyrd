"""Ingest LLM-extracted toponym mentions into ``toponym_attestation`` —
wyrd-x82p Phase 2b.1.

Consumes the JSONL emitted by ``lexicon mine-toponym-mentions``
(Phase 2 pilot) and turns each resolvable mention into a
``toponym_attestation`` row. Mirrors the conventions established by
Phase 1's pattern-based reverse-search:

* Idempotent re-runs via UNIQUE-index dedup (INSERT OR IGNORE).
* Preflight uniqueness SET so dry-run mode accurately predicts
  ``--apply``. The check loads the existing attestations for the
  source once into an in-memory set (rather than per-mention
  SELECT), so a 10K-mention source costs one query, not 10K.
* Caller-owned transaction control (function does NOT commit; CLI
  commits once after walking all sources).

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

Region matching is asymmetric: the toponym's canonical region must
appear inside the LLM's hint after tokenization, NOT the reverse.
``"Sussex"`` does not match ``"East Sussex"`` (LLM was imprecise),
and ``"Sussex"`` does not match ``"Essex"`` (word boundary).
Country is not consulted by the resolver — the wyrd corpus is
British-place-name-focused and almost every toponym has the same
country.

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

import re
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


# Unicode-aware word-character class: [\W_] matches non-word and
# underscore, where `\w` includes Unicode letters by default in
# Python 3 str patterns. Critical for OE/ME data: ``æ`` / ``þ`` /
# ``ð`` are Unicode letters that would otherwise be replaced with
# spaces and break region matching for forms like ``"Æthelney"``
# or ``"Eyþórsstaðir"``.
_NON_WORD_TO_SPACE = re.compile(r"[\W_]+")


def _tokenize_for_region(text: str) -> str:
    """Lower-case + replace any non-word run with a single space,
    with surrounding spaces. Lets the padded-substring check work
    across punctuation: ``"co. Northumberland, England"`` →
    ``" co northumberland england "``, which contains the padded
    region ``" northumberland "``. Unicode-aware: ``"Æthelney"``
    stays intact through tokenization rather than getting split."""
    return " " + _NON_WORD_TO_SPACE.sub(" ", text.lower()).strip() + " "


def _region_matches(hint_norm: str, region_norm: str) -> bool:
    """Asymmetric word-boundary match: the toponym's canonical region
    must appear as a token sequence INSIDE the LLM's hint. The
    reverse direction (hint inside region) is NOT a match — a hint
    of ``"Sussex"`` should not silently match a region of
    ``"East Sussex"`` (the LLM was imprecise; we should not fill in
    detail it didn't provide).

    Examples:
    * hint=``"Northumberland"`` region=``"northumberland"`` → match (equal)
    * hint=``"co. Northumberland, England"`` region=``"northumberland"`` → match
      (region is contained in tokenized hint)
    * hint=``"Sussex"`` region=``"east sussex"`` → no match (LLM vague)
    * hint=``"Sussex"`` region=``"essex"`` → no match (word boundary;
      "essex" is not a token-bounded substring of "sussex" after pad)
    * hint=``"Ham"`` region=``"hampshire"`` → no match (word boundary)
    """
    hint_padded = _tokenize_for_region(hint_norm)
    region_padded = _tokenize_for_region(region_norm)
    return region_padded in hint_padded


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

    Region matching is **asymmetric**: the toponym's canonical
    region must appear inside the LLM's hint (after tokenization on
    non-alphanumerics + space-padding for word boundaries). The
    reverse — hint inside region — is NOT a match: an LLM hint of
    "Sussex" should not silently match a toponym in "East Sussex",
    because the LLM was being imprecise and we should not fill in
    detail it didn't provide.

    The LLM may emit "Northumberland", "co. Northumberland", or
    "Northumberland, England" — all match a toponym whose region
    is "northumberland". But "Sussex" must NOT match "Essex" (word
    boundary), and "Sussex" must NOT match "East Sussex"
    (asymmetric direction).

    A toponym whose region is None can never be selected by
    region_hint — we don't know what region it's in.
    """
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
        if region and _region_matches(hint_norm, region):
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
      already in the DB. Accurate in BOTH dry-run and apply modes
      — the preflight uniqueness set is computed regardless of the
      apply flag.
    * ``unresolved_records`` — full record of unresolved mentions
      (form, year, region_hint, context, source_id). The caller
      (CLI) streams this out to a candidate-JSONL file. Currently
      unbounded; large unresolved sets surface a memory concern at
      multi-million-mention scale (Phase 2b.2 follow-up).
    """

    source_id: str
    mentions_processed: int = 0
    resolved: int = 0
    unresolved: int = 0
    inserted: int = 0
    already_present: int = 0
    unresolved_records: list[dict[str, object]] = field(default_factory=list)


def _coerce_form(raw: object) -> str:
    """Return a stripped form string for any input shape. LLM/JSONL
    sloppiness can yield ``None``, an integer, a list, etc.; we treat
    all non-string values as 'no form' rather than crash on
    ``.strip()``. (silent-failure round-1 finding)."""
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _coerce_date_year(raw: object) -> int | None:
    """Normalize a date_year field from the JSONL to int-or-None.
    Accepts ints, integer-valued floats (``1086.0`` from a sloppy
    JSON encoder), and parseable decimal-digit strings. Rejects bools,
    fractional floats, NaN/inf, and non-numeric strings."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        # Accept integer-valued floats (1086.0) — they're a JSON
        # encoder artifact, not a precision loss. Reject NaN/inf
        # (is_integer is False for those).
        if raw == raw and raw not in (float("inf"), float("-inf")) and raw.is_integer():
            return int(raw)
        return None
    if isinstance(raw, str):
        s = raw.strip()
        # ``isdecimal`` (NOT ``isdigit``): isdigit() is True for Unicode
        # digit-but-not-decimal chars like the superscripts ``²``/``⁵`` (and a
        # footnote-marked year ``"1086²"``), which ``int()`` then REJECTS with
        # ValueError — crashing the ingest on input the contract promises to
        # drop as None. isdecimal() admits exactly the category-Nd digits int()
        # accepts (ASCII, Arabic-Indic, fullwidth), so the guard and the parse
        # agree.
        if s.isdecimal():
            return int(s)
        return None
    return None


def _load_existing_attestations(
    conn: sqlite3.Connection, source_id: str
) -> set[tuple[int, str, int | None]]:
    """Return the set of ``(toponym_id, form, date_year)`` tuples
    already attested for ``source_id``. Single bulk SELECT replaces
    the per-mention preflight SELECT (Gemini round-1 N+1 finding).
    For a source with 10K existing attestations this is one round-
    trip; for an in-memory check it's O(1) per mention."""
    rows = conn.execute(
        "SELECT toponym_id, form, date_year FROM toponym_attestation WHERE source_doc = ?",
        (source_id,),
    ).fetchall()
    return {(r["toponym_id"], r["form"], r["date_year"]) for r in rows}


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

    Each mention dict may have ``form`` (str), ``date_year``
    (int|float|str|None), ``region_hint`` (str|None), ``context``
    (str). The function tolerates extra keys silently and treats
    non-string forms as empty.

    Dry-run semantics: with ``apply=False``, the preflight uniqueness
    set still gets consulted so ``inserted`` and ``already_present``
    accurately predict ``apply=True``'s effect.

    Transaction control: this function does NOT commit. Callers own
    commit granularity — the CLI commits after walking all sources,
    so partial-failure rollback is the natural shape.

    Date-year-NULL handling: Phase 2 mentions often have
    ``date_year=None`` (undated mentions Phase 1's regex didn't
    catch). The UNIQUE index on ``toponym_attestation`` treats NULL
    as distinct in SQLite, so two undated mentions of the same form
    for the same toponym in the same source would both insert. We
    keep the preflight in memory as a set of
    ``(toponym_id, form, date_year)`` tuples (where None stays None)
    so dedup uses set-membership semantics, treating NULL as equal."""
    report = IngestReport(source_id=source_id)
    # Single bulk SELECT into a set — O(1) membership instead of an
    # N+1 SELECT-per-mention. Also gets updated in-loop so two
    # identical mentions in the same batch dedup correctly.
    existing: set[tuple[int, str, int | None]] = _load_existing_attestations(conn, source_id)
    # Bulk inserts via executemany after the loop — one round-trip
    # to SQLite for the whole batch beats per-mention INSERT at
    # scale. Stays empty under dry-run or all-dedup.
    to_insert: list[tuple[int, str, int | None, str]] = []
    for m in mentions:
        if not isinstance(m, dict):
            # JSONL row that's valid JSON but not an object (e.g.
            # ``42``, ``"string"``, ``[1,2,3]``). Count toward
            # mentions_processed so the operator sees the malformed
            # rows; can't resolve, can't write to candidates list
            # (no fields to capture).
            report.mentions_processed += 1
            report.unresolved += 1
            continue
        report.mentions_processed += 1
        form = _coerce_form(m.get("form"))
        if not form:
            report.unresolved += 1
            continue
        date_year = _coerce_date_year(m.get("date_year"))
        region_hint = m.get("region_hint")
        if not isinstance(region_hint, str):
            region_hint = None
        toponym_id = resolve_mention(form, region_hint, indexes)
        if toponym_id is None:
            report.unresolved += 1
            ctx = m.get("context")
            report.unresolved_records.append(
                {
                    "source_id": source_id,
                    "form": form,
                    "date_year": date_year,
                    "region_hint": region_hint,
                    "context": ctx.strip() if isinstance(ctx, str) else "",
                }
            )
            continue
        report.resolved += 1
        key = (toponym_id, form, date_year)
        if key in existing:
            report.already_present += 1
            continue
        if apply:
            to_insert.append((toponym_id, form, date_year, source_id))
        # Add to the in-loop set so a second identical mention in
        # the same batch dedups correctly (idempotency for repeated
        # rows within a single source's JSONL).
        existing.add(key)
        report.inserted += 1
    if apply and to_insert:
        conn.executemany(
            """INSERT OR IGNORE INTO toponym_attestation
               (toponym_id, form, date_year, source_doc)
               VALUES (?, ?, ?, ?)""",
            to_insert,
        )
    return report
