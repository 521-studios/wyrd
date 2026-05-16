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


def _toponym_rows(conn: sqlite3.Connection) -> list[tuple[int, str, str | None, str | None]]:
    """Single full-table scan of ``toponym`` — caller builds once
    per prepare-run and reuses across candidates. Returns rows as
    ``(id, modern_name, region, country)`` tuples (lighter than
    sqlite3.Row for the hot fuzzy-loop)."""
    return [
        (row["id"], row["modern_name"], row["region"], row["country"])
        for row in conn.execute("SELECT id, modern_name, region, country FROM toponym")
    ]


def fuzzy_match_toponyms(
    form: str,
    toponyms: list[tuple[int, str, str | None, str | None]],
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
    scored: list[tuple[float, FuzzyToponymSuggestion]] = []
    for tid, modern_name, region, country in toponyms:
        name_norm = _normalize_for_match(modern_name)
        if not name_norm:
            continue
        ratio = SequenceMatcher(None, form_norm, name_norm).ratio()
        if ratio < min_ratio:
            continue
        # Region-hint boost: small bump (+0.10) for toponyms whose
        # region contains the hint. Keeps regular ratios comparable
        # but breaks ties in favor of the regional match.
        effective = ratio
        if hint_norm and region:
            region_norm = _normalize_for_match(region)
            if region_norm and (hint_norm in region_norm or region_norm in hint_norm):
                effective += 0.10
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
    scored.sort(key=lambda x: x[0], reverse=True)
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
    toponyms: list[tuple[int, str, str | None, str | None]],
) -> PreparedCandidate:
    """Convert a raw candidate dict (one row from Phase 2b.1's
    ``--candidates-out`` JSONL) into a ``PreparedCandidate`` with
    fuzzy suggestions attached. Tolerates missing optional fields
    in the raw input (``date_year`` etc. may be absent or null)."""
    form = (raw.get("form") or "").strip()
    region_hint = raw.get("region_hint")
    if not isinstance(region_hint, str):
        region_hint = None
    suggestions = fuzzy_match_toponyms(form, toponyms, region_hint=region_hint)
    date_year = raw.get("date_year")
    if not isinstance(date_year, int) or isinstance(date_year, bool):
        date_year = None
    context = raw.get("context")
    if not isinstance(context, str):
        context = ""
    source_id = raw.get("source_id") or ""
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
    * ``mapped`` — `action: map` decisions executed.
    * ``created`` — `action: create` decisions executed (new
      toponym + attestation).
    * ``skipped`` — `action: skip` decisions (no DB write).
    * ``deferred`` — `action: defer` decisions left for later.
    * ``errors`` — rows whose action was malformed or whose
      operator-supplied fields were missing/invalid. Counted but
      NOT applied; collected in ``error_records`` for the CLI to
      surface.
    """

    processed: int = 0
    mapped: int = 0
    created: int = 0
    skipped: int = 0
    deferred: int = 0
    errors: int = 0
    error_records: list[tuple[int, str]] = field(default_factory=list)


def _coerce_date_year(raw: object) -> int | None:
    """Same year-coercion semantics as Phase 2b.1's ingester —
    accepts int, integer-valued float, digit-only string; rejects
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
    if isinstance(raw, str) and raw.strip().isdigit():
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
            report.errors += 1
            report.error_records.append((idx, f"row is not a dict: {type(row).__name__}"))
            continue
        action = (row.get("action") or "").strip().lower()
        form = (row.get("form") or "").strip()
        source_id = (row.get("source_id") or "").strip()
        date_year = _coerce_date_year(row.get("date_year"))
        if action == "skip":
            report.skipped += 1
            continue
        if action == "defer":
            report.deferred += 1
            continue
        if not form or not source_id:
            report.errors += 1
            report.error_records.append(
                (idx, f"missing form or source_id (form={form!r}, source_id={source_id!r})")
            )
            continue
        if action == "map":
            tid = row.get("toponym_id")
            if not isinstance(tid, int) or isinstance(tid, bool):
                report.errors += 1
                report.error_records.append(
                    (idx, f"action=map but toponym_id missing/invalid: {tid!r}")
                )
                continue
            existing = conn.execute("SELECT 1 FROM toponym WHERE id = ? LIMIT 1", (tid,)).fetchone()
            if existing is None:
                report.errors += 1
                report.error_records.append((idx, f"action=map but toponym_id {tid} doesn't exist"))
                continue
            if apply:
                conn.execute(
                    "INSERT OR IGNORE INTO toponym_attestation "
                    "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
                    (tid, form, date_year, source_id),
                )
            report.mapped += 1
            continue
        if action == "create":
            modern_name = (row.get("create_modern_name") or "").strip()
            if not modern_name:
                report.errors += 1
                report.error_records.append((idx, "action=create but create_modern_name missing"))
                continue
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
            # Collision detect — if a toponym with this (name, country,
            # region) already exists, treat as map-to-existing (which
            # is what the UNIQUE index would silently do anyway under
            # INSERT OR IGNORE, but we want to update the counter so
            # the operator sees "create" turned into "mapped").
            existing_tid = _existing_toponym_id_for_create(conn, modern_name, country, region)
            if existing_tid is not None:
                if apply:
                    conn.execute(
                        "INSERT OR IGNORE INTO toponym_attestation "
                        "(toponym_id, form, date_year, source_doc) "
                        "VALUES (?, ?, ?, ?)",
                        (existing_tid, form, date_year, source_id),
                    )
                report.mapped += 1
                continue
            if apply:
                cursor = conn.execute(
                    "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
                    (modern_name, country, region),
                )
                new_tid = cursor.lastrowid
                conn.execute(
                    "INSERT OR IGNORE INTO toponym_attestation "
                    "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
                    (new_tid, form, date_year, source_id),
                )
            report.created += 1
            continue
        # Unknown action.
        report.errors += 1
        report.error_records.append((idx, f"unknown action: {action!r}"))
    return report
