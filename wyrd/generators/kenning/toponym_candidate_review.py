"""Operator-review tool for new-toponym candidates — wyrd-x82p Phase 2b.3.

Two-step bulk-edit workflow that triages the unresolved-mention
JSONL emitted by Phase 2b.1's ``--candidates-out``:

1. ``prepare`` — read raw candidates, fuzzy-match each form against
   existing toponyms (top-N suggestions per record), emit a
   ``triage.jsonl`` with an ``action: "defer"`` placeholder that
   the operator hand-edits.
2. ``commit`` — read the edited triage.jsonl, apply each decision:

   * ``action: "map"`` — operator picked ``toponym_id`` from the
     suggestions; write a ``toponym_attestation`` row pointing at
     the existing toponym.
   * ``action: "create"`` — operator specified ``modern_name`` /
     ``country`` / ``region`` for a NEW toponym row, plus an
     attestation row pointing at it.
   * ``action: "skip"`` — operator decided the candidate is not a
     place name (LLM false positive, OCR garbage); drop silently.
   * ``action: "defer"`` — leave in place for a future pass.

Mirrors Phase 1 / Phase 2b.1 conventions: caller-owned transaction
control, idempotent INSERT OR IGNORE, preflight uniqueness checks,
dry-run mode that accurately predicts ``--apply``.

Fuzzy matching uses ``difflib.SequenceMatcher`` ratio over the
normalized form vs. existing modern_names (top-N by ratio). For
the wyrd corpus (~22K toponyms) this is a single full-table scan
per prepare run; acceptable at pilot scale.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from wyrd.generators.kenning.lexicon.controlled_vocab import (
    CountryValidationError,
    canonicalize_country,
)
from wyrd.generators.kenning.lexicon.region_model import (
    RegionValidationError,
    canonicalize_region,
)
from wyrd.generators.kenning.lexicon.regions import country_for_region

from .toponym_reverse_search import _normalize_for_match

# Top-N fuzzy match suggestions per candidate. Three is a good
# balance: enough to catch the right toponym when normalization
# barely loses it (typographic variants), small enough that the
# operator's eye scans the list at a glance.
_FUZZY_SUGGESTION_LIMIT = 3

# Below this similarity ratio, no suggestion is emitted at all —
# noise filtered out so the operator's triage list isn't padded
# with red herrings. ratio < 0.6 is approximately "shares more
# letters than it doesn't in arbitrary order"; the cutoff is
# empirical but conservative.
_FUZZY_MIN_RATIO = 0.6

# Score bump applied to candidates whose toponym region matches
# the operator's region_hint. Small enough not to override a
# genuinely higher ratio; large enough to break ties toward the
# regionally-correct homonym (Newton@Northumberland over
# Newton@Berkshire when hint is Northumberland).
_REGION_HINT_BOOST = 0.10

# Cap on per-run error_records list. Unbounded growth on a
# malformed multi-million-row triage file would balloon memory.
# 100 captured + accurate `errors` counter is enough to diagnose
# without the operator scrolling forever; the CLI surfaces the
# elided count.
_ERROR_RECORDS_CAP = 100

# Same cap shape for demoted_records: a pathological triage file
# where every CREATE collides could otherwise accumulate thousands
# of records. The accurate count flows through ``demoted_count``;
# only the per-row detail list stops growing.
_DEMOTED_RECORDS_CAP = 100


@dataclass(frozen=True)
class FuzzyToponymSuggestion:
    """One fuzzy-matched existing toponym for the operator's
    consideration. ``fuzzy_score`` is ``SequenceMatcher.ratio()``
    over normalized forms; higher is better (1.0 is exact)."""

    toponym_id: int
    modern_name: str
    region: str | None
    country: str | None
    fuzzy_score: float


# Row shape returned by ``_toponym_rows`` and consumed by
# ``fuzzy_match_toponyms``: (id, modern_name, region, country,
# name_norm, region_norm). Pre-normalizing at scan time means a
# single fuzzy_match_toponyms run over thousands of candidates does
# the normalization work ONCE per toponym, not once per (candidate,
# toponym) pair — at ~22K toponyms × hundreds of candidates the
# saved work is millions of redundant strip+lowercase ops.
ToponymRow = tuple[int, str, str | None, str | None, str, str | None]


def _toponym_rows(conn: sqlite3.Connection) -> list[ToponymRow]:
    """Single full-table scan of ``toponym`` — caller builds once
    per prepare-run and reuses across candidates. Returns rows as
    ``(id, modern_name, region, country, name_norm, region_norm)``
    tuples; the trailing two are pre-normalized via
    ``_normalize_for_match`` so the fuzzy-match hot loop reads
    them directly instead of recomputing per candidate."""
    return [
        (
            row["id"],
            row["modern_name"],
            row["region"],
            row["country"],
            _normalize_for_match(row["modern_name"]),
            _normalize_for_match(row["region"]) if row["region"] else None,
        )
        for row in conn.execute("SELECT id, modern_name, region, country FROM toponym")
    ]


def fuzzy_match_toponyms(
    form: str,
    toponyms: list[ToponymRow],
    *,
    region_hint: str | None = None,
    limit: int = _FUZZY_SUGGESTION_LIMIT,
    min_ratio: float = _FUZZY_MIN_RATIO,
) -> list[FuzzyToponymSuggestion]:
    """Return the top-N fuzzy matches of ``form`` against the
    in-memory toponym list. When ``region_hint`` is provided,
    matches whose region matches the hint are boosted (so an
    operator looking at "Newton" with hint "Northumberland" sees
    the Northumberland Newton first even when the Berkshire one
    has a slightly higher pure-form ratio).

    Below ``min_ratio`` the match is dropped entirely — keeps the
    suggestion list focused, not padded with arbitrary noise."""
    form_norm = _normalize_for_match(form)
    if not form_norm:
        return []
    hint_norm = _normalize_for_match(region_hint) if region_hint else None
    # Hoist SequenceMatcher outside the loop so the b2j cache on
    # ``form_norm`` is built once and reused for every toponym
    # comparison — at production scale (~22K toponyms) this is the
    # dominant cost otherwise.
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(form_norm)
    form_len = len(form_norm)
    scored: list[tuple[float, FuzzyToponymSuggestion]] = []
    for tid, modern_name, region, country, name_norm, region_norm in toponyms:
        if not name_norm:
            continue
        # Fast length-bound gate using the theoretical max ratio:
        # SequenceMatcher.ratio() returns 2 * matched_chars / total_chars,
        # so the upper bound is 2 * min(L1, L2) / (L1 + L2) when every
        # character of the shorter string matches. If that ceiling
        # already falls below min_ratio, no real ratio() call can
        # reach the threshold. Cheaper than quick_ratio() and a
        # tighter bound than the naive (1 - min_ratio) length-delta.
        name_len = len(name_norm)
        total_len = form_len + name_len
        if total_len > 0 and 2.0 * min(form_len, name_len) / total_len < min_ratio:
            continue
        matcher.set_seq1(name_norm)
        # real_quick_ratio + quick_ratio are documented upper bounds
        # on the real ratio. real_quick_ratio() is cheapest (just
        # min-count overlap); quick_ratio() is sharper. Two-stage
        # gate skips the full O(N*M) ratio() for the vast majority
        # of toponyms at production scale.
        if matcher.real_quick_ratio() < min_ratio:
            continue
        if matcher.quick_ratio() < min_ratio:
            continue
        ratio = matcher.ratio()
        if ratio < min_ratio:
            continue
        # Region-hint boost: tiny bump for toponyms whose region
        # contains the hint. Region_norm is pre-computed at scan
        # time so this is a cheap substring-in test.
        effective = ratio
        if hint_norm and region_norm and (hint_norm in region_norm or region_norm in hint_norm):
            effective += _REGION_HINT_BOOST
        scored.append(
            (
                effective,
                FuzzyToponymSuggestion(
                    toponym_id=tid,
                    modern_name=modern_name,
                    region=region,
                    country=country,
                    fuzzy_score=round(ratio, 3),
                ),
            )
        )
    # Sort by effective score descending. Tiebreaker is toponym_id
    # — without it, Python falls back to comparing the second tuple
    # element (the FuzzyToponymSuggestion dataclass), which doesn't
    # define ordering and would raise TypeError on ties. Stable +
    # deterministic across runs.
    scored.sort(key=lambda x: (x[0], x[1].toponym_id), reverse=True)
    return [sugg for _, sugg in scored[:limit]]


@dataclass
class PreparedCandidate:
    """One row in the ``triage.jsonl`` that the operator hand-edits.

    Pre-filled fields:
    * The original candidate fields (``source_id`` / ``form`` /
      ``date_year`` / ``region_hint`` / ``context``)
    * ``suggestions`` — top-N fuzzy matches against existing toponyms

    Operator-edited fields (defaults are placeholders):
    * ``action`` — defaults to ``"defer"``. Operator changes to
      ``"map"``, ``"create"``, or ``"skip"``.
    * ``toponym_id`` — operator sets when action is ``"map"`` (pick
      one of the suggested ids, or any id from the toponym table).
    * ``create_modern_name``, ``create_country``, ``create_region``
      — operator sets when action is ``"create"``.
    """

    source_id: str
    form: str
    date_year: int | None
    region_hint: str | None
    context: str
    suggestions: list[FuzzyToponymSuggestion] = field(default_factory=list)
    action: str = "defer"
    toponym_id: int | None = None
    create_modern_name: str | None = None
    create_country: str | None = None
    create_region: str | None = None


def prepare_candidate(
    raw: dict,
    toponyms: list[ToponymRow],
) -> PreparedCandidate:
    """Convert a raw candidate dict (one row from Phase 2b.1's
    ``--candidates-out`` JSONL) into a ``PreparedCandidate`` with
    fuzzy suggestions attached. Tolerates missing optional fields
    in the raw input (``date_year`` etc. may be absent or null).
    Non-string values for string fields (``form``, ``source_id``,
    ``region_hint``, ``context``) get coerced rather than crashing
    on ``.strip()`` — defensive for hand-edited or upstream-malformed
    JSONL rows."""
    form = _coerce_text(raw.get("form"))
    region_hint_raw = raw.get("region_hint")
    region_hint: str | None
    if isinstance(region_hint_raw, str) and region_hint_raw.strip():
        region_hint = region_hint_raw
    else:
        region_hint = None
    suggestions = fuzzy_match_toponyms(form, toponyms, region_hint=region_hint)
    # Use the same coercion as the commit layer so the triage JSONL
    # carries the same year-value the commit phase would compute —
    # avoids surprise data loss between prepare and commit when the
    # upstream candidates carry e.g. an integer-valued float year.
    date_year = _coerce_date_year(raw.get("date_year"))
    context = _coerce_text(raw.get("context"))
    source_id = _coerce_text(raw.get("source_id"))
    return PreparedCandidate(
        source_id=source_id,
        form=form,
        date_year=date_year,
        region_hint=region_hint,
        context=context,
        suggestions=suggestions,
    )


@dataclass
class CommitReport:
    """Per-run counters for a triage-commit pass.

    * ``processed`` — total triage rows considered.
    * ``mapped`` — `action: map` decisions executed, PLUS any
      `action: create` rows that collided with an existing
      (modern_name, country, region) tuple and were demoted to MAP
      against the colliding id.
    * ``created`` — `action: create` decisions that actually
      inserted a new toponym row (no collision).
    * ``demoted_records`` — per-row capture of CREATE rows that
      collided into a MAP. Surfaces to the operator via --verbose
      so they see WHICH of their CREATEs hit an existing toponym
      (silent demotion was a load-bearing UX gap otherwise).
    * ``skipped`` — `action: skip` decisions (no DB write).
    * ``deferred`` — `action: defer` decisions left for later.
    * ``errors`` — rows whose action was malformed or whose
      operator-supplied fields were missing/invalid. Counted but
      NOT applied; first ``_ERROR_RECORDS_CAP`` collected in
      ``error_records`` for the CLI to surface (counter remains
      accurate beyond the cap).
    """

    processed: int = 0
    mapped: int = 0
    created: int = 0
    skipped: int = 0
    deferred: int = 0
    errors: int = 0
    # demoted_count is the authoritative total of CREATE → MAP
    # collisions; demoted_records is the first
    # ``_DEMOTED_RECORDS_CAP`` entries (operator visibility).
    demoted_count: int = 0
    error_records: list[tuple[int, str]] = field(default_factory=list)
    # CREATE → MAP collisions: (row_index, existing_toponym_id,
    # modern_name) so the operator sees which of their CREATEs hit
    # an existing toponym at apply time. Capped via
    # ``_DEMOTED_RECORDS_CAP``; the counter ``demoted_count``
    # stays accurate beyond the cap.
    demoted_records: list[tuple[int, int, str]] = field(default_factory=list)

    def record_error(self, row_idx: int, reason: str) -> None:
        """Count an errored row; keep the first ``_ERROR_RECORDS_CAP``
        ``(row_idx, reason)`` details (the counter stays accurate past
        the cap, only the detail list stops growing)."""
        self.errors += 1
        if len(self.error_records) < _ERROR_RECORDS_CAP:
            self.error_records.append((row_idx, reason))

    def record_demotion(self, row_idx: int, existing_tid: int, modern_name: str) -> None:
        """Count a CREATE→MAP collision-demotion; keep the first
        ``_DEMOTED_RECORDS_CAP`` ``(row_idx, existing_tid, modern_name)``
        details for operator visibility (counter stays accurate past the
        cap)."""
        self.demoted_count += 1
        if len(self.demoted_records) < _DEMOTED_RECORDS_CAP:
            self.demoted_records.append((row_idx, existing_tid, modern_name))


@dataclass(frozen=True)
class _TriageRow:
    """Per-row context derived once in the commit loop and threaded into
    the action handlers (keeps their arg lists small)."""

    idx: int
    form: str
    source_id: str
    date_year: int | None
    apply: bool


def _coerce_text(raw: object) -> str:
    """Defensive ``str.strip()`` for operator-edited JSONL fields.
    Operators occasionally produce non-string values (a number, a
    null, a list) for what should be a string field. Coercing
    instead of crashing on ``.strip()`` lets the downstream
    validation logic (e.g. "form missing" error) catch the bad
    value with a useful message rather than the process aborting
    on a raw AttributeError.

    Returns an empty string for None, non-string types, and
    whitespace-only inputs — downstream truthiness checks treat
    these as missing."""
    if raw is None or not isinstance(raw, str):
        return ""
    return raw.strip()


def _coerce_date_year(raw: object) -> int | None:
    """Same year-coercion semantics as Phase 2b.1's ingester —
    accepts int, integer-valued float, decimal-digit string; rejects
    bool, NaN/inf, non-numeric strings."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if (
        isinstance(raw, float)
        and raw == raw
        and raw not in (float("inf"), float("-inf"))
        and raw.is_integer()
    ):
        return int(raw)
    # ``isdecimal`` (NOT ``isdigit``): isdigit() admits Unicode digit-but-not-
    # decimal chars (superscripts ``²``/``⁵``, a footnote year ``"1086²"``) that
    # ``int()`` then rejects with ValueError — a crash on input the contract
    # promises to drop. isdecimal() == exactly what int() parses (kept in sync
    # with the ingester copy).
    if isinstance(raw, str) and raw.strip().isdecimal():
        return int(raw.strip())
    return None


def _existing_toponym_id_for_create(
    conn: sqlite3.Connection,
    modern_name: str,
    country: str | None,
    region: str | None,
) -> int | None:
    """Pre-check whether a CREATE would collide with an existing
    toponym row (UNIQUE index on modern_name, country, region).
    Returns the existing id if collision; None if a real insert
    is needed."""
    row = conn.execute(
        "SELECT id FROM toponym "
        "WHERE modern_name = ? "
        "  AND COALESCE(country, '') = COALESCE(?, '') "
        "  AND COALESCE(region, '') = COALESCE(?, '') "
        "LIMIT 1",
        (modern_name, country, region),
    ).fetchone()
    return row["id"] if row else None


def _commit_map_row(
    conn: sqlite3.Connection, row: dict, ctx: _TriageRow, report: CommitReport
) -> None:
    """Apply an ``action: map`` row: validate ``toponym_id`` (must be an
    existing int, not a bool) and write one idempotent
    ``toponym_attestation`` (under ``ctx.apply``). Records an error on a
    missing/invalid or non-existent ``toponym_id``."""
    tid = row.get("toponym_id")
    if not isinstance(tid, int) or isinstance(tid, bool):
        report.record_error(ctx.idx, f"action=map but toponym_id missing/invalid: {tid!r}")
        return
    existing = conn.execute("SELECT 1 FROM toponym WHERE id = ? LIMIT 1", (tid,)).fetchone()
    if existing is None:
        report.record_error(ctx.idx, f"action=map but toponym_id {tid} doesn't exist")
        return
    if ctx.apply:
        conn.execute(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (tid, ctx.form, ctx.date_year, ctx.source_id),
        )
    report.mapped += 1


def _commit_create_row(
    conn: sqlite3.Connection, row: dict, ctx: _TriageRow, report: CommitReport
) -> None:
    """Apply an ``action: create`` row: insert a new ``toponym`` (+ its
    attestation), OR — when ``(modern_name, country, region)`` collides
    with an existing toponym — demote to MAP against the colliding id
    (counting the demotion so ``--verbose`` can surface which CREATEs
    collided). Records an error on a missing ``create_modern_name``."""
    modern_name = _coerce_text(row.get("create_modern_name"))
    if not modern_name:
        report.record_error(ctx.idx, "action=create but create_modern_name missing")
        return
    country = row.get("create_country")
    if isinstance(country, str):
        country = country.strip() or None
    else:
        country = None
    region = row.get("create_region")
    if isinstance(region, str):
        region = region.strip() or None
    else:
        region = None
    # Fail-closed Class-A guards (wyrd-fxxf): canonicalize the operator-supplied
    # region + country, COUNTRY FIRST (wyrd-61p9) so the canonical country scopes
    # the region check (uniform with the other ingest guards). A bad value rejects
    # THIS row (record_error) rather than aborting the whole review commit — the
    # review-tool analogue of the parser ingest's hard raise (wyrd-3q6m.3/.5).
    try:
        country = canonicalize_country(country)
        region = canonicalize_region(region, country=country)
        if country is None and region is not None:
            # derive the country from the canonical region, same as the parser
            # ingest's _upsert_toponym — so an operator CREATE with a region but no
            # country doesn't insert a NULL country (wyrd-61p9; regions.py yields
            # canonical countries, so no re-canonicalization is needed).
            country = country_for_region(region)
    except (RegionValidationError, CountryValidationError) as exc:
        report.record_error(ctx.idx, str(exc))
        return
    # Collision detect — if a toponym with this (name, country, region)
    # already exists, treat as map-to-existing. The UNIQUE index would
    # silently no-op under INSERT OR IGNORE, but we count the demotion
    # AND record (row_idx, existing toponym_id, modern_name) so --verbose
    # surfaces which of the operator's CREATEs hit a collision.
    existing_tid = _existing_toponym_id_for_create(conn, modern_name, country, region)
    if existing_tid is not None:
        if ctx.apply:
            conn.execute(
                "INSERT OR IGNORE INTO toponym_attestation "
                "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
                (existing_tid, ctx.form, ctx.date_year, ctx.source_id),
            )
        report.mapped += 1
        report.record_demotion(ctx.idx, existing_tid, modern_name)
        return
    if ctx.apply:
        cursor = conn.execute(
            "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
            (modern_name, country, region),
        )
        new_tid = cursor.lastrowid
        conn.execute(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            (new_tid, ctx.form, ctx.date_year, ctx.source_id),
        )
    report.created += 1


def commit_triage_decisions(
    conn: sqlite3.Connection,
    rows: list[dict],
    *,
    apply: bool = False,
) -> CommitReport:
    """Apply triage decisions to the DB. Each row is one operator-
    edited triage entry (see :class:`PreparedCandidate` for the
    shape). With ``apply=False`` (dry-run), counters are accurate
    but no DB rows are written.

    Caller-owned transaction control — this function does NOT
    commit. The CLI commits once after walking all rows so a
    bad row aborts the whole pass for atomic rollback.

    Action semantics:
    * ``map``: requires ``toponym_id``; writes one
      ``toponym_attestation`` row (idempotent via UNIQUE index).
    * ``create``: requires ``create_modern_name``; pre-checks the
      UNIQUE index to detect collision (treats a collision as if
      the operator had chosen ``map`` against the colliding id —
      prevents accidental duplicate toponyms).
    * ``skip``: no-op.
    * ``defer``: no-op.

    Any other action value, or missing required fields, is
    counted as an error with the row's index + reason recorded.
    """
    report = CommitReport()
    for idx, row in enumerate(rows):
        report.processed += 1
        if not isinstance(row, dict):
            report.record_error(idx, f"row is not a dict: {type(row).__name__}")
            continue
        action = _coerce_text(row.get("action")).lower()
        form = _coerce_text(row.get("form"))
        source_id = _coerce_text(row.get("source_id"))
        date_year = _coerce_date_year(row.get("date_year"))
        if action == "skip":
            report.skipped += 1
            continue
        if action == "defer":
            report.deferred += 1
            continue
        if not form or not source_id:
            report.record_error(
                idx,
                f"missing form or source_id (form={form!r}, source_id={source_id!r})",
            )
            continue
        ctx = _TriageRow(idx=idx, form=form, source_id=source_id, date_year=date_year, apply=apply)
        if action == "map":
            _commit_map_row(conn, row, ctx, report)
        elif action == "create":
            _commit_create_row(conn, row, ctx, report)
        else:
            report.record_error(idx, f"unknown action: {action!r}")
    return report
