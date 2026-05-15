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
this module's `refill_count` vs `hallucinated_count` split tracks
the two failure modes explicitly).
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


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces. Used both for
    the source body (so OCR-introduced line breaks don't block
    matching) and the citation short_quote tail."""
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class RefillResult:
    """Per-citation outcome.

    Status is one of:
      * "refilled" — tail located, short_quote extended
      * "hallucinated" — tail not found in source (LLM made it up)
      * "not_truncated" — looks_truncated() returned False; skipped
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
    idx = source_text_norm.find(anchor)
    if idx == -1 and len(tail) > _SEARCH_TAIL_FALLBACK_LEN:
        anchor = tail[-_SEARCH_TAIL_FALLBACK_LEN:].strip()
        idx = source_text_norm.find(anchor)
    if idx == -1:
        return None, 0
    end_pos = idx + len(anchor)
    forward = source_text_norm[end_pos : end_pos + window]
    if not forward.strip():
        return None, 0
    # The new short_quote is the original (preserving any pipe-prefixed
    # commentary) plus the recovered forward context appended. Trim
    # leading whitespace on the recovered chunk since the anchor often
    # ends mid-word.
    new_quote = short_quote.rstrip() + forward
    return new_quote, len(forward)


def _load_source_body(sources_dir: Path, source_id: str) -> str | None:
    """Read ``sources/<source_id>.txt`` and return its normalized
    body. Returns None when the file doesn't exist (audits that span
    sources without bundled .txt files just skip those)."""
    path = sources_dir / f"{source_id}.txt"
    if not path.exists():
        return None
    return _normalize_whitespace(path.read_text(encoding="utf-8"))


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
    ``etymon_citation.id``. Re-running is safe — the audit's
    :func:`looks_truncated` returns False on already-refilled rows
    (they now end in terminal punctuation or are too long), so
    refilled rows skip on subsequent runs.
    """
    report = RefillReport(source_id=source_id)
    src_body = _load_source_body(sources_dir, source_id)
    if src_body is None:
        return report
    rows = conn.execute(
        """
        SELECT ec.id, ec.short_quote,
               e.language || ':' || e.canonical_form AS etymon_ref
          FROM etymon_citation ec
          JOIN etymon e ON e.id = ec.etymon_id
         WHERE ec.source_id = ?
           AND ec.short_quote IS NOT NULL
        """,
        (source_id,),
    ).fetchall()
    for row in rows:
        sq = row["short_quote"]
        if not looks_truncated(sq):
            continue
        report.total_truncated += 1
        new_quote, recovered = refill_short_quote(sq, src_body, window=window)
        if new_quote is None:
            report.hallucinated += 1
            if len(report.samples) < sample_limit:
                report.samples.append(
                    RefillResult(
                        citation_id=row["id"],
                        etymon_ref=row["etymon_ref"],
                        status="hallucinated",
                        old_short_quote=sq,
                    )
                )
            continue
        report.refilled += 1
        if apply:
            conn.execute(
                "UPDATE etymon_citation SET short_quote = ? WHERE id = ?",
                (new_quote, row["id"]),
            )
        if len(report.samples) < sample_limit:
            report.samples.append(
                RefillResult(
                    citation_id=row["id"],
                    etymon_ref=row["etymon_ref"],
                    status="refilled",
                    old_short_quote=sq,
                    new_short_quote=new_quote,
                    recovered_chars=recovered,
                )
            )
    if apply:
        conn.commit()
    return report
