"""Tests for the rando-port retirement readiness gate — wyrd-j2bv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.rando_port_readiness import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_TARGET_LANGUAGES,
    LanguageCriterion,
    ReadinessReport,
    _sibling_for_language,
    compute_readiness,
    format_readiness,
    load_bundle,
)

# ---------------------------------------------------------------------------
# _sibling_for_language
# ---------------------------------------------------------------------------


def test_sibling_for_language_passthrough_underscore():
    """Bundle's native shape is underscore-separated; pass through unchanged."""
    assert _sibling_for_language("old_english") == "old_english"


def test_sibling_for_language_converts_hyphen_to_underscore():
    """DB ref language uses hyphens (``old-english``); the bundle
    uses underscores. The helper bridges them so a caller can pass
    either form."""
    assert _sibling_for_language("old-english") == "old_english"


# ---------------------------------------------------------------------------
# LanguageCriterion thresholds
# ---------------------------------------------------------------------------


def test_passes_coverage_above_threshold():
    """80%+ scholar/empirical → coverage passes."""
    lc = LanguageCriterion(
        language="old_english",
        total_subjects=100,
        scholar_attested=50,
        empirical_only=30,  # combined: 80% attested
        rando_only=0,
        uncited=20,
    )
    assert lc.coverage_pct == pytest.approx(80.0)
    assert lc.passes_coverage(0.80)


def test_passes_coverage_below_threshold():
    """Just under 80% → fails."""
    lc = LanguageCriterion(
        language="latin",
        total_subjects=100,
        scholar_attested=50,
        empirical_only=29,  # 79% < 80%
        rando_only=0,
        uncited=21,
    )
    assert not lc.passes_coverage(0.80)


def test_passes_coverage_zero_subjects_vacuous_pass():
    """A language with zero bundle subjects can't fail the gate —
    we're not blocking retirement on languages we don't ship."""
    lc = LanguageCriterion(
        language="proto-germanic",
        total_subjects=0,
        scholar_attested=0,
        empirical_only=0,
        rando_only=0,
        uncited=0,
    )
    assert lc.passes_coverage(0.80)


def test_passes_rando_only_zero():
    """Criterion 2: zero rando-only subjects."""
    lc = LanguageCriterion(
        language="old_english",
        total_subjects=10,
        scholar_attested=10,
        empirical_only=0,
        rando_only=0,
        uncited=0,
    )
    assert lc.passes_rando_only_zero()


def test_passes_rando_only_one_fails():
    """A single rando-only subject closes the gate."""
    lc = LanguageCriterion(
        language="old_english",
        total_subjects=10,
        scholar_attested=9,
        empirical_only=0,
        rando_only=1,
        uncited=0,
    )
    assert not lc.passes_rando_only_zero()


def test_passes_slice_comparison_rando_below_scholar():
    """C3: rando_only ≤ scholar. With more scholar than rando-only,
    passes."""
    lc = LanguageCriterion(
        language="old_english",
        total_subjects=10,
        scholar_attested=8,
        empirical_only=0,
        rando_only=2,
        uncited=0,
    )
    assert lc.passes_slice_comparison()


def test_passes_slice_comparison_rando_above_scholar_fails():
    lc = LanguageCriterion(
        language="latin",
        total_subjects=10,
        scholar_attested=2,
        empirical_only=0,
        rando_only=8,
        uncited=0,
    )
    assert not lc.passes_slice_comparison()


def test_passes_slice_comparison_both_zero_vacuous_pass():
    """Both rando and scholar at zero — no rando dependence to
    outweigh."""
    lc = LanguageCriterion(
        language="proto-germanic",
        total_subjects=0,
        scholar_attested=0,
        empirical_only=0,
        rando_only=0,
        uncited=0,
    )
    assert lc.passes_slice_comparison()


# ---------------------------------------------------------------------------
# ReadinessReport.overall_passes — the gate
# ---------------------------------------------------------------------------


def test_overall_passes_all_languages_pass():
    """All three criteria pass for ALL languages → gate open."""
    report = ReadinessReport(
        coverage_threshold=0.80,
        per_language=(
            LanguageCriterion(
                language="old_english",
                total_subjects=100,
                scholar_attested=100,
                empirical_only=0,
                rando_only=0,
                uncited=0,
            ),
            LanguageCriterion(
                language="old_french",
                total_subjects=50,
                scholar_attested=50,
                empirical_only=0,
                rando_only=0,
                uncited=0,
            ),
        ),
    )
    assert report.overall_passes


def test_overall_fails_one_language_coverage_below():
    """A single language failing C1 → gate closed."""
    report = ReadinessReport(
        coverage_threshold=0.80,
        per_language=(
            LanguageCriterion(
                language="latin",
                total_subjects=20,
                scholar_attested=5,
                empirical_only=0,
                rando_only=0,
                uncited=15,
            ),
        ),
    )
    assert not report.overall_passes


def test_overall_fails_one_rando_only_subject():
    """Any single rando-only subject anywhere closes the gate."""
    report = ReadinessReport(
        coverage_threshold=0.80,
        per_language=(
            LanguageCriterion(
                language="old_english",
                total_subjects=100,
                scholar_attested=99,
                empirical_only=0,
                rando_only=1,
                uncited=0,
            ),
        ),
    )
    assert not report.overall_passes


# ---------------------------------------------------------------------------
# compute_readiness with a synthetic bundle
# ---------------------------------------------------------------------------


def _bundle(subjects):
    return {"subjects": subjects}


def _word(siblings_with_citations):
    """Synthetic bundle word. ``siblings_with_citations`` is a dict
    of ``sibling_name -> list of citation source IDs``. Empty list =
    sibling present but uncited."""
    out = {}
    for sib, cites in siblings_with_citations.items():
        out[sib] = "form"  # any truthy value triggers has_lang detection
        out[f"{sib}_citations"] = list(cites)
    return out


def test_compute_readiness_all_scholar_clean(tmp_path: Path):
    """Bundle where every target language has only scholar attestation:
    all three criteria pass."""
    bundle = _bundle(
        [
            {"words": [_word({"old_english": ["mawer_1920_northumberland_durham"]})]},
        ]
    )
    report = compute_readiness(bundle, target_languages=("old_english",))
    lc = report.per_language[0]
    assert lc.scholar_attested == 1
    assert lc.rando_only == 0
    assert report.overall_passes


def test_compute_readiness_rando_only_closes_gate():
    """Bundle where the only citation is rando-port: C2 fails."""
    bundle = _bundle(
        [
            {"words": [_word({"old_english": ["rando-port"]})]},
        ]
    )
    report = compute_readiness(bundle, target_languages=("old_english",))
    lc = report.per_language[0]
    assert lc.rando_only == 1
    assert not lc.passes_rando_only_zero()
    assert not report.overall_passes


def test_compute_readiness_low_coverage_closes_gate():
    """Half the subjects uncited → C1 fails."""
    bundle = _bundle(
        [
            {"words": [_word({"old_english": ["mawer_1920_northumberland_durham"]})]},
            {"words": [_word({"old_english": []})]},  # uncited
        ]
    )
    report = compute_readiness(bundle, target_languages=("old_english",), coverage_threshold=0.80)
    lc = report.per_language[0]
    assert lc.coverage_pct == 50.0
    assert not lc.passes_coverage(0.80)
    assert not report.overall_passes


def test_compute_readiness_empirical_counts_as_attested():
    """C1 thresholds scholar + empirical (both non-rando paths). An
    all-empirical language still passes coverage."""
    bundle = _bundle(
        [
            {"words": [_word({"old_english": ["wiktionary-empirical"]})]},
        ]
    )
    report = compute_readiness(bundle, target_languages=("old_english",))
    lc = report.per_language[0]
    assert lc.empirical_only == 1
    assert lc.scholar_attested == 0
    assert lc.coverage_pct == 100.0
    assert report.overall_passes  # vacuous rando-only zero + slice


def test_compute_readiness_uses_default_target_languages():
    """No explicit target list → uses DEFAULT_TARGET_LANGUAGES."""
    bundle = _bundle([])
    report = compute_readiness(bundle)
    assert {lc.language for lc in report.per_language} == set(DEFAULT_TARGET_LANGUAGES)


# ---------------------------------------------------------------------------
# load_bundle + format_readiness
# ---------------------------------------------------------------------------


def test_load_bundle_reads_json(tmp_path: Path):
    path = tmp_path / "meanings.json"
    path.write_text(json.dumps({"subjects": []}))
    out = load_bundle(path)
    assert out == {"subjects": []}


def test_format_readiness_pass_marker():
    """Passing report includes the PASS verdict."""
    report = ReadinessReport(coverage_threshold=0.80, per_language=())
    out = format_readiness(report)
    assert "PASS — retirement gate open" in out


def test_format_readiness_fail_marker_shows_why():
    """Failing report enumerates the failed criteria."""
    report = ReadinessReport(
        coverage_threshold=0.80,
        per_language=(
            LanguageCriterion(
                language="latin",
                total_subjects=10,
                scholar_attested=2,
                empirical_only=0,
                rando_only=0,
                uncited=8,
            ),
        ),
    )
    out = format_readiness(report)
    assert "FAIL" in out
    assert "Why the gate is closed" in out
    assert "latin" in out
    assert "C1" in out


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_rando_port_readiness_passes_exits_0(tmp_path: Path):
    """When all criteria pass, CLI prints the report + exit 0."""
    path = tmp_path / "meanings.json"
    path.write_text(
        json.dumps(
            {
                "subjects": [
                    {"words": [_word({"old_english": ["mawer_1920_northumberland_durham"]})]},
                ]
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "rando-port-readiness",
            "--bundle",
            str(path),
            "--language",
            "old_english",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_rando_port_readiness_fails_exits_1(tmp_path: Path):
    """When any criterion fails, CLI exits 1 so CI / deploy gates
    can detect a closed retirement gate without parsing markdown."""
    path = tmp_path / "meanings.json"
    path.write_text(
        json.dumps(
            {
                "subjects": [
                    {"words": [_word({"old_english": ["rando-port"]})]},  # closes C2
                ]
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "rando-port-readiness",
            "--bundle",
            str(path),
            "--language",
            "old_english",
        ],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_default_coverage_threshold_value():
    """80% is the wyrd-j2bv-defined gate; pin it so a typo at the
    declaration site fails loudly."""
    assert DEFAULT_COVERAGE_THRESHOLD == 0.80
