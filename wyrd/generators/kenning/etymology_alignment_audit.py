"""Toponym/etymology cross-misalignment audit — wyrd-8upf.

Walks ``data/mining/*.jsonl`` looking at every ``etymology_element``
row (LLM-extracted scholar etymologies for toponyms) and flags rows
where the ``elements[]`` list doesn't appear supported by the
``historical_form`` reconstruction. The misalignment pattern showed
up in PR #175 review (the ``DownLanp Woop`` example —
old-english:hoh proposed but citation prose says ``William ater
Dune`` pointing at dun + land).

Like the short_quote audit (wyrd-bd68), this is a recall sieve:
high-precision heuristics that operator eyeballs and decides
whether to emit ``remove`` / ``set`` curation events on the
flagged rows. The audit can't perfectly classify alignment without
re-running the LLM — but mechanizable signals catch the bulk.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


def _normalize(text: str) -> str:
    """Lowercase + strip diacritics + collapse non-alphanumeric to
    a single space. Substring matching works on the normalized form,
    so ``Céolwig`` ↔ ``ceolwig`` succeeds, ``tūn`` ↔ ``tun``,
    ``æ`` ↔ ``ae``, and trailing punctuation doesn't break alignment.

    Uses ``unicodedata.normalize("NFKD")`` to decompose combining
    diacritics (covering OE long-vowel macrons that hand-mapping
    misses) plus a small fixup for OE/OF ligatures."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    # NFKD doesn't split ligatures (æ, œ); do that explicitly.
    decomposed = (
        decomposed.replace("æ", "ae").replace("Æ", "AE").replace("œ", "oe").replace("Œ", "OE")
    )
    # Drop combining marks (the decomposed accents).
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


# Morphemes that are too short / common to carry signal. Latinate
# inflections (-es genitive, -ing patronymic) and frequent OE/ON
# endings (-a, -e, -u) would otherwise flag false alignment-misses
# when they're absent from the historical_form (they're often
# elided in attested spellings).
#
# wyrd-j6co extension: added non-OE inflections that elide in scholar
# reconstructions of Irish / Norse / French place-names:
#   - OE  -um  dative/accusative plural
#   - OE  -on  variant of -an (already present), found in older
#               attestations
#   - ON  -ar  nominative/accusative plural
#   - ON  -ir  nominative/accusative plural
# Already covered for non-OE: "a" (Irish gen sing), "e" (OF silent
# endings), "es" (OF/OE -es genitive/plural).
_STOP_MORPHEMES: frozenset[str] = frozenset(
    {"a", "e", "i", "o", "u", "es", "ing", "an", "en", "s", "n", "r", "um", "on", "ar", "ir"}
)


def _etymon_canonical_form(etymon_ref: str) -> str:
    """Strip the ``language:`` prefix from an etymon ref. ``old-english:cot``
    → ``cot``."""
    return etymon_ref.rsplit(":", 1)[-1]


# NOTE: the per-element single-pass walk lives inline in
# :func:`looks_misaligned` (wyrd-j6co Gemini round 3 — was an extracted
# ``_missing_morphemes`` helper, but inlining the loop lets us share
# ``canonical_forms`` between the missing-list and the finding's
# elements field without re-iterating, and lets ``_normalize(historical_form)``
# run once instead of twice).


# wyrd-j6co: categorize MisalignmentFinding by reason so the operator
# can triage one failure mode at a time. The two flag paths in
# :func:`looks_misaligned` produce different evidence kinds and
# warrant different reactions:
#
# * ``empty_form`` — elements exist but the LLM emitted no
#   reconstruction. The morphemes may have been over-confidently
#   posited without prose support. Operator may want to retry with
#   ``mine-llm`` at higher reasoning effort.
# * ``hallucinated`` — elements exist, reconstruction exists, but
#   one or more element canonical_forms don't appear as substrings
#   of the reconstruction. The LLM probably fabricated those
#   morphemes (or the wrong inflection). Operator targets these for
#   manual prune via ``lexicon prune-etymon`` or curation.
MISALIGNMENT_REASON_EMPTY_FORM = "empty_form"
MISALIGNMENT_REASON_HALLUCINATED = "hallucinated"


@dataclass(frozen=True)
class MisalignmentFinding:
    """One flagged etymology_element row.

    ``reason`` is one of ``MISALIGNMENT_REASON_*`` and tells the
    operator which heuristic branch tripped (wyrd-j6co)."""

    toponym_ref: str
    historical_form: str
    elements: list[str]
    missing_morphemes: list[str]
    confidence: str | None
    notes_preview: str
    reason: str = MISALIGNMENT_REASON_HALLUCINATED


def looks_misaligned(row: dict) -> MisalignmentFinding | None:
    """Apply the heuristic to one ``etymology_element`` row. Returns
    a finding when the alignment looks suspect, ``None`` otherwise.

    Rules (high-precision; operator eyeballs the report):

    - At least one element AND historical_form is empty → flag with
      ``reason="empty_form"``. The LLM proposed morphemes but no
      reconstruction; the prose may not actually support those
      morphemes.
    - At least one element's canonical_form doesn't appear as a
      substring of the normalized historical_form (skipping
      stop-morphemes) → flag with ``reason="hallucinated"``. The
      element wasn't pinned to the reconstruction; might be
      hallucinated.
    """
    elements = row.get("elements") or []
    historical_form = (row.get("historical_form") or "").strip()
    if not elements:
        return None
    # Single pass over elements (wyrd-j6co Gemini round 3 perf): collect
    # canonical_forms (for the finding) and missing entries (for the
    # alignment check) in one loop, with _normalize(historical_form)
    # computed once outside.
    #
    # Skip elements with no etymon_ref (wyrd-j6co round 1 robustness:
    # they'd produce empty strings in the report otherwise).
    # Stop-morphemes (single letters, common inflections) are
    # excluded from `missing` since their absence isn't signal — same
    # rule applies whether or not historical_form is populated.
    norm_form = _normalize(historical_form)
    canonical_forms: list[str] = []
    missing: list[str] = []
    for el in elements:
        ref = el.get("etymon_ref")
        if not ref:
            continue
        canonical = _etymon_canonical_form(ref)
        canonical_forms.append(canonical)
        norm_canonical = _normalize(canonical)
        if not norm_canonical or norm_canonical in _STOP_MORPHEMES:
            continue
        if norm_canonical not in norm_form:
            missing.append(canonical)
    if not canonical_forms:
        return None
    if not missing:
        return None
    reason = MISALIGNMENT_REASON_EMPTY_FORM if not norm_form else MISALIGNMENT_REASON_HALLUCINATED
    return MisalignmentFinding(
        toponym_ref=row.get("toponym_ref") or "<missing>",
        historical_form=historical_form,
        elements=canonical_forms,
        missing_morphemes=missing,
        confidence=row.get("confidence"),
        notes_preview=_preview_notes(row.get("notes") or ""),
        reason=reason,
    )


def _preview_notes(notes: str, max_len: int = 200) -> str:
    """Keep the report rows tight — long notes are the SIGNAL, not
    useful in a sample line. The operator can open the file to see
    full context."""
    notes = notes.replace("\n", " ").strip()
    if len(notes) <= max_len:
        return notes
    return notes[: max_len - 3] + "..."


@dataclass
class SourceAuditResult:
    """Per-source audit summary.

    ``reason_counts`` tracks every flagged row's reason (not just the
    capped ``samples`` list), so :func:`format_audit_report`'s header
    breakdown accurately reflects the full scan rather than the
    capped sample. wyrd-j6co Gemini round 2."""

    source_id: str
    total_etymologies: int = 0
    flagged_count: int = 0
    samples: list[MisalignmentFinding] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def flag_rate(self) -> float:
        if self.total_etymologies == 0:
            return 0.0
        return 100.0 * self.flagged_count / self.total_etymologies


@dataclass
class AuditReport:
    """Aggregate over all sources."""

    sources: list[SourceAuditResult]
    sample_limit_per_source: int

    @property
    def total_etymologies(self) -> int:
        return sum(s.total_etymologies for s in self.sources)

    @property
    def total_flagged(self) -> int:
        return sum(s.flagged_count for s in self.sources)

    @property
    def overall_rate(self) -> float:
        if self.total_etymologies == 0:
            return 0.0
        return 100.0 * self.total_flagged / self.total_etymologies


def _source_id_from_path(path: Path) -> str:
    return path.stem


def audit_jsonl_file(path: Path, sample_limit: int = 3) -> SourceAuditResult:
    """Scan one source file for misaligned etymology_element rows."""
    result = SourceAuditResult(source_id=_source_id_from_path(path))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_type") != "etymology_element":
                continue
            if row.get("_op") == "remove":
                continue
            result.total_etymologies += 1
            finding = looks_misaligned(row)
            if finding is not None:
                result.flagged_count += 1
                result.reason_counts[finding.reason] = (
                    result.reason_counts.get(finding.reason, 0) + 1
                )
                if len(result.samples) < sample_limit:
                    result.samples.append(finding)
    return result


def audit_jsonl_dir(
    directory: Path, *, sample_limit: int = 3, sources: Iterable[str] | None = None
) -> AuditReport:
    """Walk every ``*.jsonl`` in ``directory``. Skips ``_curation.jsonl``
    (operator event log, not source data)."""
    wanted: set[str] | None = set(sources) if sources is not None else None
    results: list[SourceAuditResult] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem.startswith("_"):
            continue
        if wanted is not None and path.stem not in wanted:
            continue
        results.append(audit_jsonl_file(path, sample_limit=sample_limit))
    return AuditReport(sources=results, sample_limit_per_source=sample_limit)


def _audit_summary_lines(report: AuditReport) -> list[str]:
    """Header block: totals plus the per-reason flag breakdown."""
    lines = [
        "# Toponym/etymology alignment audit (wyrd-8upf)",
        "",
        f"- Sources scanned: {len(report.sources)}",
        f"- Total etymology rows: {report.total_etymologies:,}",
        f"- Probably misaligned: {report.total_flagged:,} ({report.overall_rate:.1f}%)",
    ]
    # wyrd-j6co: split the flag count by reason so operators can pick
    # one failure mode at a time (empty_form needs a re-mine; hallucinated
    # needs a prune/curate). Counts are global across sources, drawn
    # from per-source reason_counts (every flagged row, not just the
    # capped samples list — wyrd-j6co Gemini round 2).
    reason_counts: dict[str, int] = {}
    for s in report.sources:
        for reason, count in s.reason_counts.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + count
    if reason_counts:
        breakdown = ", ".join(
            f"{count} {reason}" for reason, count in sorted(reason_counts.items())
        )
        lines.append(f"- By reason: {breakdown}")
    return lines


def _audit_top_sources_lines(sorted_sources: list[SourceAuditResult], top_n: int) -> list[str]:
    """Markdown table of the top sources by flag count (skips clean sources), or
    ``[]`` when the section would be empty — no flagged source within ``top_n``,
    or ``top_n <= 0`` — so ``format_audit_report`` omits the header (wyrd-73ma)
    rather than emit an orphaned '## Top sources by flag count' + empty table."""
    rows = [
        f"| `{s.source_id}` | {s.flagged_count} / {s.total_etymologies} | {s.flag_rate:.1f}% |"
        for s in sorted_sources[:top_n]
        if s.flagged_count > 0
    ]
    if not rows:
        return []
    return [
        "## Top sources by flag count",
        "",
        "| Source | Flagged / Total | Rate |",
        "| --- | ---: | ---: |",
        *rows,
    ]


def _audit_samples_lines(
    sorted_sources: list[SourceAuditResult], report: AuditReport, top_n: int
) -> list[str]:
    """Per-source sample misses, a placeholder when the corpus is clean, or ``[]``
    when there ARE misses but ``top_n <= 0`` shows none — so the header isn't
    emitted orphaned over an empty body (wyrd-73ma). The clean-corpus placeholder
    is kept: it's an informative signal, not an empty section."""
    affected = [s for s in sorted_sources if s.flagged_count > 0]
    if not affected:
        return ["_No alignment misses detected._"]
    shown = affected[:top_n]
    if not shown:  # top_n <= 0: misses exist but none rendered → no orphaned header
        return []
    lines = [f"## Samples (up to {report.sample_limit_per_source} per source)", ""]
    for s in shown:
        lines.append(f"### `{s.source_id}` ({s.flagged_count} flagged)")
        for f in s.samples:
            lines.append(
                f"- `{f.toponym_ref}` [{f.reason}] — historical_form=`{f.historical_form or '∅'}`, "
                f"elements=[{', '.join(f.elements)}], "
                f"missing=[{', '.join(f.missing_morphemes)}]"
            )
            if f.notes_preview:
                lines.append(f"  - notes: `{f.notes_preview}`")
        lines.append("")
    return lines


def format_audit_report(report: AuditReport, *, top_n: int = 20) -> str:
    """Markdown rendering. Ranks by absolute flag count (volume
    drives where operator work has highest payoff)."""
    # Clamp so a NEGATIVE top_n suppresses the ranked sections rather than
    # triggering Python's negative slicing (``list[:-1]`` returns all-but-last,
    # not ``[]``) — the section helpers slice ``[:top_n]``, so ``top_n <= 0`` must
    # mean "show none" (wyrd-73ma).
    top_n = max(0, top_n)
    sorted_sources = sorted(
        report.sources,
        key=lambda s: (-s.flagged_count, -s.total_etymologies, s.source_id),
    )
    # Join only non-empty sections with a single blank line between them, so an
    # empty top-sources / samples section leaves no orphaned header (wyrd-73ma).
    # With all sections present this is byte-identical to the prior fixed layout.
    sections = [
        _audit_summary_lines(report),
        _audit_top_sources_lines(sorted_sources, top_n),
        _audit_samples_lines(sorted_sources, report, top_n),
    ]
    body = "\n\n".join("\n".join(section) for section in sections if section)
    return body.rstrip() + "\n"
