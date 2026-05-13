"""Truncated-short_quote audit — wyrd-bd68.

Walks ``data/mining/*.jsonl``, looks at every ``citation`` row's
``short_quote`` field, and flags those that look truncated. The
truncation pattern showed up across the scholar-mined corpus in PR
#175 review (bannister:101 'extracted… | Ab' etc.) — LLM extraction
ran into a context-window cap mid-sentence.

We can't perfectly classify truncation without re-running the LLM,
but a high-recall heuristic catches the bulk: a non-trivially-long
short_quote that doesn't end in terminal punctuation almost always
indicates a cut-off mid-sentence.

Output is markdown — operator decides whether to re-mine or
context-snippet-refill, per the wyrd-bd68 ticket.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Below this length a non-terminal-punctuation tail is plausibly
# intentional (one-word entry refs, fragmentary citations). Only
# flag long ones, where the LLM almost certainly ran past a context
# limit.
MIN_TRUNCATION_LENGTH = 200

# Characters that indicate a complete sentence / quote / citation.
# Generous on purpose — many scholar citations end in ``'`` after a
# quoted form, or ``]`` after a bracketed editor note.
_TERMINAL_CHARS: frozenset[str] = frozenset(".!?\"')]}")

# The canonical bannister:101 shape: ``...prose | Ab``. The pipe is
# the LLM-output's "commentary | source context" delimiter; a short
# trailing word means the source-context snippet got guillotined.
# Six chars catches `Abber` / `Thorne` / `Cottes` etc. Past that
# we're more likely seeing a legitimate-but-short citation token.
_PIPE_TAIL_RE = re.compile(r"\|\s*\w{1,6}$")

# Mid-word cutoff: the last whitespace-bounded token has 2-5 chars
# AND looks like the start of something (no internal punctuation).
# Combined with the length floor this catches "...See Do" /
# "...Tornelay) 12" / similar — short fragmentary tails the LLM was
# about to expand.
#
# We skip SINGLE-char tails: many scholar citations end in a known
# abbreviation (``P`` = Pipe Rolls, ``B`` = Bede, ``DB`` = Domesday)
# that would be a false positive at length 1. Two-or-more chars is
# still noisy (``DB``, ``LF``, ``IPM`` are real refs) but the false
# positive rate is much lower at the larger character window.
_SHORT_TAIL_RE = re.compile(r"(?:^|\s)([A-Za-z0-9]{2,5})$")


def looks_truncated(short_quote: str | None) -> bool:
    """High-precision heuristic, intentionally conservative to keep
    the false-positive rate low. Flags a short_quote as
    probably-truncated if ALL of these hold:

    - non-empty and at least :data:`MIN_TRUNCATION_LENGTH` chars
      (an LLM truncation almost always happens at a context-window
      cap, so length is high)
    - terminal character is not in :data:`_TERMINAL_CHARS` (no
      sentence/quote/bracket closure)

    AND one of:

    - ends with the ``| <1-6 char fragment>`` pattern (canonical
      bannister shape — LLM cut off mid-place-name after the pipe
      delimiter), OR
    - the last whitespace-bounded token is 2-5 chars (a short
      fragment that the LLM was about to expand: ``See Do`` is
      truncated where ``See Donegal`` is complete)

    Single-letter trailing tokens are NOT flagged — many scholar
    citations end in a known abbreviation (``P`` = Pipe Rolls, ``B``
    = Bede) that would otherwise be a false positive. Multi-char
    abbreviations (``DB``, ``LF``, ``IPM``) DO still flag; the audit
    is a recall sieve, not a final verdict. Operators eyeball samples
    and decide whether to re-mine.
    """
    if not short_quote:
        return False
    stripped = short_quote.rstrip()
    if len(stripped) < MIN_TRUNCATION_LENGTH:
        return False
    if stripped[-1] in _TERMINAL_CHARS:
        return False
    if _PIPE_TAIL_RE.search(stripped):
        return True
    # A no-whitespace single-token blob can't end in a "short trailing
    # fragment" — the whole quote IS the token. Bail before the regex
    # so the short-tail logic only runs on multi-word quotes where the
    # last whitespace-bounded token might be the truncation point.
    if " " not in stripped:
        return False
    last_tok = stripped.rsplit(maxsplit=1)[-1]
    return bool(_SHORT_TAIL_RE.fullmatch(" " + last_tok))


@dataclass
class SourceAuditResult:
    """Per-source audit summary."""

    source_id: str
    total_citations: int = 0
    truncated_citations: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def truncation_rate(self) -> float:
        if self.total_citations == 0:
            return 0.0
        return 100.0 * self.truncated_citations / self.total_citations


@dataclass
class AuditReport:
    """Aggregate over all sources."""

    sources: list[SourceAuditResult]
    sample_limit_per_source: int

    @property
    def total_citations(self) -> int:
        return sum(s.total_citations for s in self.sources)

    @property
    def total_truncated(self) -> int:
        return sum(s.truncated_citations for s in self.sources)

    @property
    def overall_rate(self) -> float:
        if self.total_citations == 0:
            return 0.0
        return 100.0 * self.total_truncated / self.total_citations


def _source_id_from_path(path: Path) -> str:
    """File ``foo_bar.jsonl`` → source_id ``foo_bar``. Matches the
    naming the per-source dumper produces."""
    return path.stem


def audit_jsonl_file(path: Path, sample_limit: int = 3) -> SourceAuditResult:
    """Scan one ``data/mining/<source_id>.jsonl`` for truncated
    short_quotes. Returns a per-source summary.

    Only ``citation`` rows (with ``_op`` either absent or ``add``)
    are considered. Rows with ``_op == "remove"`` are skipped — the
    operator already retired them; flagging would be double counting.
    """
    result = SourceAuditResult(source_id=_source_id_from_path(path))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Skip un-parseable rows; the schema-validation pass
                # is a separate concern. The audit shouldn't abort on
                # one bad row.
                continue
            if row.get("_type") != "citation":
                continue
            if row.get("_op") == "remove":
                continue
            result.total_citations += 1
            sq = row.get("short_quote")
            if looks_truncated(sq):
                result.truncated_citations += 1
                if len(result.samples) < sample_limit:
                    result.samples.append(sq)
    return result


def audit_jsonl_dir(
    directory: Path, *, sample_limit: int = 3, sources: Iterable[str] | None = None
) -> AuditReport:
    """Audit every ``*.jsonl`` in ``directory``, skipping the
    ``_curation.jsonl`` log (operator event-log, not source-curated
    data). Pass ``sources`` to restrict to a named subset."""
    wanted: set[str] | None = set(sources) if sources is not None else None
    results: list[SourceAuditResult] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem.startswith("_"):
            continue
        if wanted is not None and path.stem not in wanted:
            continue
        results.append(audit_jsonl_file(path, sample_limit=sample_limit))
    return AuditReport(sources=results, sample_limit_per_source=sample_limit)


def _truncate_sample(sample: str, max_len: int = 200) -> str:
    """Shorten a sample for the report. Long quotes themselves are
    the FLAG — there's no value in printing all 1KB."""
    if len(sample) <= max_len:
        return sample
    return sample[: max_len - 3] + "..."


def format_audit_report(report: AuditReport, *, top_n: int = 20) -> str:
    """Markdown rendering. Top-N by absolute truncation count (not
    rate — the operator wants to fix the high-volume offenders first).
    """
    sorted_sources = sorted(
        report.sources,
        key=lambda s: (-s.truncated_citations, -s.total_citations, s.source_id),
    )
    lines: list[str] = ["# Truncated-short_quote audit (wyrd-bd68)", ""]
    lines.append(f"- Sources scanned: {len(report.sources)}")
    lines.append(f"- Total citations: {report.total_citations:,}")
    lines.append(f"- Probably truncated: {report.total_truncated:,} ({report.overall_rate:.1f}%)")
    lines.append("")
    lines.append("## Top sources by truncation count")
    lines.append("")
    lines.append("| Source | Truncated / Total | Rate |")
    lines.append("| --- | ---: | ---: |")
    for s in sorted_sources[:top_n]:
        if s.truncated_citations == 0:
            continue
        lines.append(
            f"| `{s.source_id}` | {s.truncated_citations} / {s.total_citations} "
            f"| {s.truncation_rate:.1f}% |"
        )
    lines.append("")
    affected = [s for s in sorted_sources if s.truncated_citations > 0]
    if affected:
        lines.append(f"## Samples (up to {report.sample_limit_per_source} per source)")
        lines.append("")
        for s in affected[:top_n]:
            lines.append(f"### `{s.source_id}` ({s.truncated_citations} truncated)")
            for sample in s.samples:
                lines.append(f"- `{_truncate_sample(sample)}`")
            lines.append("")
    else:
        lines.append("_No truncations detected. The corpus is clean._")
    return "\n".join(lines).rstrip() + "\n"
