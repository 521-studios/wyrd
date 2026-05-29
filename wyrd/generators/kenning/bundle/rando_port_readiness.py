"""Rando-port retirement readiness — wyrd-j2bv (Phase 4b).

The rando-port source is the legacy hand-curated seed lexicon that
contributed ~1599 etymon citations before scholar mining caught up.
It's been the de-facto attribution-of-record for entries the newer
mining pipeline hasn't independently attested. The epic's terminal
goal is to retire it (append ``_op=remove`` events to
``rando-port.jsonl``) once the per-language scholar + modern-corpus
coverage is rich enough.

This module reports whether the retirement gate is open. Three
criteria (from the wyrd-j2bv ticket):

1. **Per-language coverage threshold** — every target language has
   ≥80% of its bundle subjects scholar- or empirical-attested.
2. **Rando-only-zero** — no live bundle subject is attested ONLY by
   rando-port (it's the sole citation source for some entry).
3. **Slice comparison** — for each target language, the rando-only
   subject count is at most the scholar-attested count.

The CLI prints a pass/fail per criterion + an aggregate verdict.
Operator runs the retirement step (append remove events) only when
all three pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.language_quality import _bundle_attestation_breakdown

# Bundle-sibling keys this gate cares about. These are the runtime
# bundle's column names (underscore-separated), not the DB's
# language refs (hyphen-separated). Modern English isn't
# rando-grandfathered. Welsh + Irish are conflated under celtic_mix
# in the current bundle shape (wyrd-c1vq dict-shape migration); the
# wyrd-j2bv text talks about them as separate languages, but the
# rando-port story is per-bundle-sibling.
DEFAULT_TARGET_LANGUAGES: tuple[str, ...] = (
    "old_english",
    "old_french",
    "old_scandinavian",
    "celtic_mix",
    "latin",
)

# Per-language coverage threshold for criterion 1. The 80% number
# is from the wyrd-j2bv ticket and matches the dashboard's "actionable
# coverage" tier. Below this, the language still has substantive
# rando-only gaps and retirement would dump real attestation.
DEFAULT_COVERAGE_THRESHOLD = 0.80


@dataclass(frozen=True)
class LanguageCriterion:
    """Per-language slice of the readiness report."""

    language: str
    total_subjects: int
    scholar_attested: int
    empirical_only: int
    rando_only: int
    uncited: int

    @property
    def attested_count(self) -> int:
        """Scholar + empirical (the two non-rando-grandfather paths)."""
        return self.scholar_attested + self.empirical_only

    @property
    def coverage_pct(self) -> float:
        if self.total_subjects == 0:
            return 0.0
        return 100.0 * self.attested_count / self.total_subjects

    @property
    def scholar_only_pct(self) -> float:
        """wyrd-w1ak: scholar-attested share of bundle subjects, ignoring
        the empirical wedge. Surfaces 'how much of this language's
        coverage rests on one mining pipeline (wiktionary-empirical)
        vs the 40+ named bibliographic sources'.

        C1 verdict math is unchanged — still uses ``coverage_pct``
        (scholar + empirical). This is a visibility metric, not a gate
        change. A language can clear C1 cleanly while scholar_only_pct
        sits low because empirical is doing most of the work.
        """
        if self.total_subjects == 0:
            return 0.0
        return 100.0 * self.scholar_attested / self.total_subjects

    def passes_coverage(self, threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> bool:
        """Criterion 1: ≥threshold attested. A language with zero
        subjects vacuously passes — we're not gating retirement on
        languages we don't yet ship."""
        if self.total_subjects == 0:
            return True
        return self.attested_count / self.total_subjects >= threshold

    def passes_rando_only_zero(self) -> bool:
        """Criterion 2: no rando-only subjects in this language."""
        return self.rando_only == 0

    def passes_slice_comparison(self) -> bool:
        """Criterion 3: rando_only ≤ scholar_attested for the language.
        A vacuous pass when both are zero — there's no rando dependence
        to outweigh."""
        return self.rando_only <= self.scholar_attested


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate readiness across all target languages."""

    coverage_threshold: float
    per_language: tuple[LanguageCriterion, ...]

    @property
    def overall_passes(self) -> bool:
        """All three criteria pass for ALL target languages."""
        return all(
            lc.passes_coverage(self.coverage_threshold)
            and lc.passes_rando_only_zero()
            and lc.passes_slice_comparison()
            for lc in self.per_language
        )


def _sibling_for_language(language: str) -> str:
    """Bundle-subject sibling key for a language. Accept both the
    underscore form (``old_english`` — bundle's native shape) and the
    hyphen form (``old-english`` — DB's ref language). Returns the
    underscore form for bundle lookups."""
    return language.replace("-", "_")


def compute_readiness(
    bundle: Any,
    *,
    target_languages: Iterable[str] = DEFAULT_TARGET_LANGUAGES,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> ReadinessReport:
    """Score the readiness criteria against a loaded bundle."""
    per_lang: list[LanguageCriterion] = []
    for lang in target_languages:
        breakdown = _bundle_attestation_breakdown(bundle, _sibling_for_language(lang))
        per_lang.append(
            LanguageCriterion(
                language=lang,
                total_subjects=breakdown["total"],
                scholar_attested=breakdown["scholar_attested"],
                empirical_only=breakdown["empirical_only"],
                rando_only=breakdown["rando_only"],
                uncited=breakdown["uncited"],
            )
        )
    return ReadinessReport(
        coverage_threshold=coverage_threshold,
        per_language=tuple(per_lang),
    )


def load_bundle(bundle_path: Path) -> Any:
    """Read meanings.json. Returns the raw parsed object; downstream
    helpers handle both list and dict shapes."""
    with open(bundle_path, encoding="utf-8") as fh:
        return json.load(fh)


def format_readiness(report: ReadinessReport) -> str:
    """Markdown rendering, one section per criterion + verdict."""
    lines: list[str] = ["# Rando-port retirement readiness (wyrd-j2bv)", ""]
    verdict = (
        "PASS — retirement gate open" if report.overall_passes else "FAIL — retirement gate closed"
    )
    lines.append(f"**Overall: {verdict}**")
    lines.append("")
    lines.append(f"Coverage threshold: {report.coverage_threshold * 100:.0f}%")
    lines.append("")
    lines.append("## Per-language criteria")
    lines.append("")
    lines.append(
        "| Language | Total | Scholar | Empirical | Rando-only | Uncited "
        "| Scholar only | Coverage | C1 ≥thresh | C2 rando=0 | C3 slice |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |")
    for lc in report.per_language:
        c1 = "✓" if lc.passes_coverage(report.coverage_threshold) else "✗"
        c2 = "✓" if lc.passes_rando_only_zero() else "✗"
        c3 = "✓" if lc.passes_slice_comparison() else "✗"
        lines.append(
            f"| {lc.language} | {lc.total_subjects} | {lc.scholar_attested} "
            f"| {lc.empirical_only} | {lc.rando_only} | {lc.uncited} "
            f"| {lc.scholar_only_pct:.1f}% | {lc.coverage_pct:.1f}% "
            f"| {c1} | {c2} | {c3} |"
        )
    lines.append("")
    lines.append("## Criterion definitions")
    lines.append("")
    lines.append(
        "- **C1 (coverage)** — scholar + empirical attestation covers "
        f"≥{report.coverage_threshold * 100:.0f}% of bundle subjects."
    )
    lines.append("- **C2 (rando-only zero)** — no bundle subject is attested ONLY by rando-port.")
    lines.append(
        "- **C3 (slice comparison)** — rando-only count ≤ scholar-attested "
        "count per language (rando is no longer the dominant slice)."
    )
    lines.append(
        "- **Scholar only** (visibility, not a gate) — scholar-attested "
        "share of bundle subjects (excluding the empirical wedge). "
        "Surfaces how much of a language's coverage rests on the "
        "wiktionary-empirical pipeline vs the named bibliographic sources."
    )
    lines.append("")
    if not report.overall_passes:
        lines.append("## Why the gate is closed")
        lines.append("")
        for lc in report.per_language:
            failures: list[str] = []
            if not lc.passes_coverage(report.coverage_threshold):
                failures.append(
                    f"C1 ({lc.coverage_pct:.1f}% < {report.coverage_threshold * 100:.0f}%)"
                )
            if not lc.passes_rando_only_zero():
                failures.append(f"C2 ({lc.rando_only} rando-only)")
            if not lc.passes_slice_comparison():
                failures.append(f"C3 (rando_only={lc.rando_only} > scholar={lc.scholar_attested})")
            if failures:
                lines.append(f"- `{lc.language}` — {', '.join(failures)}")
    return "\n".join(lines).rstrip() + "\n"
