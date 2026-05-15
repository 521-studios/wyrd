"""Truncated-short_quote refill — wyrd-1hpc.

Companion to ``short_quote_audit`` (wyrd-bd68). When the audit
flags a citation's ``short_quote`` as truncated, this module pulls
the missing context from the source body in ``sources/<book>.txt``.

The LLM-extraction citations often cut off at a context-window cap
mid-sentence. The audit identifies those; this refills them by
locating the last meaningful chunk in the source body and extending
forward to recover the rest of the scholarly prose.

Pilot results (PR #211 baseline, before refill ran):
  | source          | truncated | locatable | rate  |
  |-----------------|----------:|----------:|------:|
  | mawer_1920      |       283 |       208 | 73.5% |
  | joyce_1898      |       151 |       114 | 75.5% |
  | joyce_1913      |       121 |        98 | 81.0% |
  | ekwall_1922     |       134 |       119 | 88.8% |
  | watson_1926     |       117 |        60 | 51.3% (excluded — mostly hallucinations) |

The non-locatable residual is LLM-hallucinated commentary that was
never in the source body — refill can't fix it. File a separate
LLM re-mine ticket for those (the audit module still flags them;
this module's :class:`RefillReport` ``refilled`` vs ``hallucinated``
counters track the two failure modes explicitly).

Idempotency:
  Refilled short_quotes are post-processed to end at a terminal
  character (``.``, ``!``, ``?``, ``"``, ``'``, ``)``, ``]``, ``}``)
  via :func:`_trim_to_terminal_char` so :func:`short_quote_audit.looks_truncated`
  doesn't re-flag them on subsequent audit runs. A re-run of
  ``lexicon refill-short-quotes`` against an already-refilled DB
  is therefore a no-op (looks_truncated returns False on the
  extended rows, so they're skipped before the anchor search runs).

  Forward chunks that contain no terminal character within the
  configured window are NOT applied — the chunk is too fragmentary
  to safely extend the citation with, and the row is counted as
  ``hallucinated`` so operators can investigate (typically the
  source body's prose runs unusually punctuation-light in that
  region).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .short_quote_audit import looks_truncated

# How much forward context to recover past the matched tail. ±200
# chars gives the SPA citation panel enough scholarly prose without
# bloating the etymon_citation row. Matches the
# _TEXT_MATCH_SNIPPET_RADIUS used elsewhere in lexicon.py.
DEFAULT_REFILL_WINDOW = 200

# How many trailing chars of the truncated short_quote to use as the
# search anchor. Longer = more specific = fewer false-positive
# matches in the source body. 80 chars is enough to be unambiguous
# against typical Mawer/Joyce/Ekwall prose density.
_SEARCH_TAIL_LEN = 80

# Fallback shorter anchor when the 80-char anchor doesn't match
# (LLM may have lightly paraphrased the tail). 40 chars is the
# minimum that's still specific enough to avoid wrong-page false
# positives.
_SEARCH_TAIL_FALLBACK_LEN = 40

# Below this length, the anchor isn't unique enough in typical
# scholar-book prose to trust the .find() result — a 5-10 char tail
# could match in an index, a running header, a cross-reference, or
# any incidental occurrence. Set conservatively at 20 chars; at this
# length, the false-positive rate against Mawer/Joyce/Ekwall-style
# prose is empirically near zero. Gemini PR #211 round-3 finding.
_MIN_ANCHOR_LEN = 20

# Characters that satisfy :func:`short_quote_audit.looks_truncated`'s
# terminal-character check. Forward chunks must end at one of these
# so the audit doesn't re-flag the extended row on subsequent runs
# (= idempotency). Mirrors the same constant in short_quote_audit
# rather than importing it because the audit's set is module-private.
_TERMINAL_CHARS: frozenset[str] = frozenset(".!?\"')]}")

# looks_truncated's length floor — short_quotes below this length
# are guaranteed not flagged, so pushing the filter into SQL strictly
# reduces the row count refill_source walks. Kept in sync with
# short_quote_audit.MIN_TRUNCATION_LENGTH; not imported directly to
# avoid coupling refill code to the audit module's constants.
_TRUNCATION_LENGTH_FLOOR = 200


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces. Used both for
    the source body (so OCR-introduced line breaks don't block
    matching) and the citation short_quote tail."""
    return re.sub(r"\s+", " ", text).strip()


def _unique_find(source_text: str, anchor: str) -> int | None:
    """Return the position of ``anchor`` in ``source_text`` if it
    occurs exactly once, else ``None``.

    Rejecting ambiguous matches is the safer default (Gemini PR #211
    round-4 finding): if a 20-40 char anchor appears in multiple
    places in the source body, ``str.find`` returns the first match,
    which may not be the citation's actual location. A wrong-location
    match writes incorrect forward context to the citation. Refusing
    to refill is the right tradeoff — that row counts as
    hallucinated and surfaces for operator review rather than being
    silently corrupted.

    For the 80-char primary anchor, ambiguity is rare. For the
    40-char fallback, it's plausible in repetitive academic prose
    (Mawer/Joyce reuse formulaic phrases). Either way, uniqueness is
    cheap to verify (one extra ``find`` call) and bounds the false-
    positive rate to zero.
    """
    first = source_text.find(anchor)
    if first == -1:
        return None
    second = source_text.find(anchor, first + 1)
    if second != -1:
        return None
    return first


def _trim_to_terminal_char(chunk: str) -> str | None:
    """Trim ``chunk`` so its last character is in :data:`_TERMINAL_CHARS`.
    Returns the trimmed prefix, or ``None`` if no terminal character
    exists anywhere in the chunk.

    Used to ensure refilled short_quotes don't trip
    :func:`short_quote_audit.looks_truncated`'s terminal-char rule on
    re-audit (= idempotency). Scholar place-name prose has frequent
    terminal characters (citation periods like ``Ipm.``, closing
    quotes/brackets), so most forward-windows have a usable trim
    point well before the arbitrary character cap. (Semicolons are
    NOT in the terminal set — they're attestation separators in this
    corpus, not sentence ends.)
    """
    for i in range(len(chunk) - 1, -1, -1):
        if chunk[i] in _TERMINAL_CHARS:
            return chunk[: i + 1]
    return None


@dataclass(frozen=True)
class RefillResult:
    """Per-citation outcome.

    Status is one of:
      * ``"refilled"`` — tail located, forward chunk had a terminal
        character, short_quote extended.
      * ``"hallucinated"`` — tail not found in source body (LLM
        invented prose) OR forward chunk had no terminal character
        within the window (too fragmentary to safely extend).
    """

    citation_id: int
    etymon_ref: str
    status: str
    old_short_quote: str | None = None
    new_short_quote: str | None = None
    recovered_chars: int = 0


@dataclass
class RefillReport:
    """Aggregate result across one source's citations."""

    source_id: str
    total_truncated: int = 0
    refilled: int = 0
    hallucinated: int = 0
    samples: list[RefillResult] = field(default_factory=list)


def refill_short_quote(
    short_quote: str,
    source_text_norm: str,
    *,
    window: int = DEFAULT_REFILL_WINDOW,
) -> tuple[str | None, int]:
    """Locate the tail of ``short_quote`` in the normalized source
    body, extend forward by ``window`` chars, and return the new
    short_quote (or None if the tail can't be located).

    Returns ``(new_short_quote, recovered_chars)``. ``recovered_chars``
    is 0 when the tail wasn't found.

    The "tail" is the post-pipe segment when ``short_quote`` contains
    a ``|`` (the LLM's commentary/source-context delimiter that the
    audit flags as truncated), otherwise the whole quote. Search uses
    the last :data:`_SEARCH_TAIL_LEN` chars, falling back to
    :data:`_SEARCH_TAIL_FALLBACK_LEN` if the longer anchor doesn't
    match (handles minor LLM paraphrase of the tail).
    """
    norm_quote = _normalize_whitespace(short_quote)
    tail = norm_quote.rsplit("|", 1)[1].strip() if "|" in norm_quote else norm_quote
    if not tail:
        return None, 0
    # Long anchor first.
    anchor = tail[-_SEARCH_TAIL_LEN:].strip() if len(tail) > _SEARCH_TAIL_LEN else tail
    # Reject anchors below the minimum length — they're too short to
    # be unique in typical scholar prose (Gemini PR #211 round-3
    # finding). An anchor of e.g. " 1268" would match dozens of
    # places in a Mawer-style gazetteer.
    if len(anchor) < _MIN_ANCHOR_LEN:
        return None, 0
    idx = _unique_find(source_text_norm, anchor)
    if idx is None and len(tail) > _SEARCH_TAIL_FALLBACK_LEN:
        # Fallback: shorter anchor for LLM-paraphrased early tails.
        # Re-check uniqueness here too — the 40-char fallback is
        # more likely than the 80-char primary anchor to match in
        # multiple places, but ambiguous matches still need to be
        # rejected for correctness.
        #
        # AND re-check the minimum-length guard — `.strip()` can
        # collapse a 40-char trailing window to 1-2 chars if the
        # original tail ended in mostly-whitespace (silent-failure-hunter
        # PR #211 round-5 finding). Without this guard, a stripped
        # 2-char anchor like ", " could match uniquely in the source
        # body and silently write wrong forward context.
        anchor = tail[-_SEARCH_TAIL_FALLBACK_LEN:].strip()
        if len(anchor) < _MIN_ANCHOR_LEN:
            return None, 0
        idx = _unique_find(source_text_norm, anchor)
    if idx is None:
        return None, 0
    end_pos = idx + len(anchor)
    forward = source_text_norm[end_pos : end_pos + window]
    if not forward.strip():
        return None, 0
    # Trim to the last terminal character within the recovered window
    # so the new short_quote ends cleanly and won't re-trip the audit's
    # truncation heuristic on the next run (idempotency).
    trimmed = _trim_to_terminal_char(forward)
    if trimmed is None:
        # Forward chunk has no terminal character anywhere — too
        # fragmentary to safely extend the citation. Caller counts as
        # hallucinated so operators can investigate (rare case;
        # typically source body's prose runs unusually
        # punctuation-light here).
        return None, 0
    new_quote = short_quote.rstrip() + trimmed
    return new_quote, len(trimmed)


def source_body_path(sources_dir: Path, source_id: str) -> Path:
    """Return the expected source-body .txt path for ``source_id`` under
    ``sources_dir``. Centralizes the naming convention so callers (the
    CLI's missing-source warning, _load_source_body, future
    rebuild-from-source tools) all agree on where to look.

    Strips directory components from ``source_id`` via
    ``Path(source_id).name`` so a malformed identifier can't escape
    ``sources_dir`` (Gemini PR #211 round-7 path-traversal hardening)."""
    return sources_dir / f"{Path(source_id).name}.txt"


def _load_source_body(sources_dir: Path, source_id: str) -> str | None:
    """Read ``sources/<source_id>.txt`` and return its normalized
    body. Returns None when:

    * the file doesn't exist — audits that span sources without
      bundled .txt files just skip those.
    * the file isn't valid UTF-8 — some OCR-produced source bodies
      have stray non-UTF-8 bytes; surface as None rather than
      crashing the whole refill run mid-source.
    * the file exists but can't be opened (e.g. PermissionError, or
      a directory-where-a-file-should-be) — same rationale; we
      don't want one corrupt entry to halt a batch run
      (code-reviewer PR #211 round-5 finding).
    * the file is empty or whitespace-only — functionally equivalent
      to missing for refill purposes; treat the same so the CLI's
      missing-source warning triggers (Gemini PR #211 round-4).
    """
    path = source_body_path(sources_dir, source_id)
    if not path.exists():
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    normalized = _normalize_whitespace(body)
    if not normalized:
        return None
    return normalized


def _etymon_ref_for(conn: sqlite3.Connection, etymon_id: int) -> str:
    """Lazy lookup: only called when constructing a sample row, so
    the per-citation hot path doesn't pay for a JOIN it usually
    doesn't need (samples cap at ``sample_limit``, default 3, while
    a source may have hundreds of refilled rows)."""
    row = conn.execute(
        "SELECT language || ':' || canonical_form AS ref FROM etymon WHERE id = ?",
        (etymon_id,),
    ).fetchone()
    return row["ref"] if row else f"<unknown-etymon-id:{etymon_id}>"


def refill_source(
    conn: sqlite3.Connection,
    source_id: str,
    sources_dir: Path,
    *,
    apply: bool = False,
    window: int = DEFAULT_REFILL_WINDOW,
    sample_limit: int = 3,
) -> RefillReport:
    """Walk every truncated citation for one source. With
    ``apply=True``, write the refilled short_quotes back via SQL
    UPDATE. With ``apply=False``, run as a dry-run that only reports
    the counts.

    The DB write is a single UPDATE per refilled row, scoped to
    ``etymon_citation.id``. Re-running is safe — refilled rows end
    at a terminal character (per :func:`_trim_to_terminal_char`), so
    :func:`looks_truncated` returns False for them and the row
    skips the anchor search on subsequent runs.
    """
    report = RefillReport(source_id=source_id)
    # No JOIN to etymon — the etymon_ref is only needed when we add a
    # row to report.samples (sample_limit caps at 3 by default), so
    # defer that lookup to :func:`_etymon_ref_for`. The bulk-path
    # scan over hundreds of citations doesn't pay the JOIN cost.
    # Length filter pushed into SQL (Gemini PR #211 round-3 perf):
    # looks_truncated requires length >= MIN_TRUNCATION_LENGTH (200),
    # so anything below that is guaranteed not flagged. Filtering at
    # the DB cuts the row count refill_source walks on sources with
    # many short / healthy citations. The Python-side
    # ``looks_truncated`` call below still runs as the authoritative
    # check (length is necessary but not sufficient for truncation).
    # Load source body LATE so we can still count truncated rows even
    # when the .txt is missing (Gemini PR #211 round-2): operator sees
    # `truncated=N refilled=0 hallucinated=0` and the CLI's warning
    # tells them the source body is the missing piece. Without this
    # ordering, missing-source returns an empty report and the
    # truncated-count is invisible.
    src_body = _load_source_body(sources_dir, source_id)
    # Stream rows via direct cursor iteration rather than fetchall()
    # to bound memory on sources with very large citation counts
    # (Gemini PR #211 round-5). UPDATE writes are deferred into
    # ``updates`` and applied via executemany() AFTER the SELECT
    # cursor closes — sqlite3 can invalidate the iterator if writes
    # happen on the same connection mid-iteration.
    updates: list[tuple[str, int]] = []
    cursor = conn.execute(
        """
        SELECT id, etymon_id, short_quote
          FROM etymon_citation
         WHERE source_id = ?
           AND short_quote IS NOT NULL
           AND length(short_quote) >= ?
        """,
        (source_id, _TRUNCATION_LENGTH_FLOOR),
    )
    for row in cursor:
        sq = row["short_quote"]
        if not looks_truncated(sq):
            continue
        report.total_truncated += 1
        if src_body is None:
            # No source body → can't refill or classify as
            # hallucination (we don't know if it would have located
            # in the missing source). Counted under total_truncated
            # only so the operator sees the gap.
            continue
        new_quote, recovered = refill_short_quote(sq, src_body, window=window)
        if new_quote is None:
            report.hallucinated += 1
            if len(report.samples) < sample_limit:
                report.samples.append(
                    RefillResult(
                        citation_id=row["id"],
                        etymon_ref=_etymon_ref_for(conn, row["etymon_id"]),
                        status="hallucinated",
                        old_short_quote=sq,
                    )
                )
            continue
        report.refilled += 1
        if apply:
            updates.append((new_quote, row["id"]))
        if len(report.samples) < sample_limit:
            report.samples.append(
                RefillResult(
                    citation_id=row["id"],
                    etymon_ref=_etymon_ref_for(conn, row["etymon_id"]),
                    status="refilled",
                    old_short_quote=sq,
                    new_short_quote=new_quote,
                    recovered_chars=recovered,
                )
            )
    if apply and updates:
        conn.executemany(
            "UPDATE etymon_citation SET short_quote = ? WHERE id = ?",
            updates,
        )
        conn.commit()
    return report
