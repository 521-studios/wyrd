"""Tests for the rando-port retirement readiness gate — wyrd-j2bv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.bundle.rando_port_readiness import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_TARGET_LANGUAGES,
    LanguageCriterion,
    ReadinessReport,
    _sibling_for_language,
    compute_readiness,
    format_readiness,
    load_rando_attested_refs,
)
from wyrd.generators.kenning.cli import cli as cli_root

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


def test_load_rando_attested_refs_reads_citations_and_dedashes(tmp_path: Path):
    """wyrd-qkn0: the loader keeps only ``citation`` records, splits
    ``etymon_ref`` into language:form, and de-dashes the form (L2 refs may
    still carry affix-position dashes) so it matches the bundle's bare
    ``morpheme_id``. Non-citation records (source/etymon) are ignored."""
    ledger = tmp_path / "rando-port.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_type": "source", "ref": "rando-port"},
                {"_type": "etymon", "ref": "old-english:tun", "canonical_form": "tun"},
                {"_type": "citation", "etymon_ref": "old-english:-ton"},  # dashed
                {"_type": "citation", "etymon_ref": "celtic:aber"},
                {"_type": "citation", "etymon_ref": "no-colon-ref"},  # skipped (no ':')
                {
                    "_type": "citation",
                    "etymon_ref": "old-english:-",
                },  # form de-dashes to empty → dropped
            ]
        )
    )
    refs = load_rando_attested_refs(ledger)
    assert refs == frozenset({"old-english:ton", "celtic:aber"})


def test_load_rando_attested_refs_missing_file_warns_and_is_empty(tmp_path: Path):
    """A missing ledger degrades to an empty set rather than raising — but
    warns, because a silently-empty set re-creates the rando_only=0 bug and
    could falsely PASS a gate that authorizes destructive removes."""
    with pytest.warns(UserWarning, match="rando-port ledger not found"):
        refs = load_rando_attested_refs(tmp_path / "nope.jsonl")
    assert refs == frozenset()


def test_load_rando_attested_refs_mid_file_corruption_raises(tmp_path: Path):
    """A truncated TRAILING line is tolerated (crash mid-append), but a corrupt
    MID-FILE line (e.g. an unresolved merge marker) RAISES — silently dropping
    valid rando-only citations would let this remove-authorizing gate falsely
    PASS."""
    bad_mid = tmp_path / "mid.jsonl"
    bad_mid.write_text(
        '{"_type": "citation", "etymon_ref": "old-english:tun"}\n'
        "<<<<<<< HEAD\n"  # mid-file corruption
        '{"_type": "citation", "etymon_ref": "celtic:aber"}\n'
    )
    with pytest.raises(json.JSONDecodeError):
        load_rando_attested_refs(bad_mid)
    # A truncated trailing record is still tolerated; the valid line loads.
    trailing = tmp_path / "trail.jsonl"
    trailing.write_text(
        '{"_type": "citation", "etymon_ref": "old-english:tun"}\n'
        '{"_type": "citation", "etymon_ref": "celtic:ab'  # truncated, no newline
    )
    assert load_rando_attested_refs(trailing) == frozenset({"old-english:tun"})


def test_compute_readiness_rando_refs_surfaces_live_rando_only():
    """wyrd-qkn0 end-to-end: a live morpheme whose only attestation is
    rando-port reaches the bundle with NO citations (rando-port is scrubbed
    from <lang>_citations). Without rando_refs it misbins as uncited and the
    gate falsely passes C2; with rando_refs it surfaces as rando_only and the
    gate correctly closes."""
    word = {"old_english": "form", "old_english_citations": [], "morpheme_id": "old-english:tun"}
    bundle = _bundle([{"words": [word]}])
    # Pre-fix blind spot: looks clean.
    blind = compute_readiness(bundle, target_languages=("old_english",))
    assert blind.per_language[0].rando_only == 0
    assert blind.per_language[0].uncited == 1
    # With the ledger: the rando-only morpheme is surfaced, gate closes on C2.
    seen = compute_readiness(
        bundle,
        target_languages=("old_english",),
        rando_refs=frozenset({"old-english:tun"}),
    )
    assert seen.per_language[0].rando_only == 1
    assert seen.per_language[0].uncited == 0
    assert not seen.overall_passes


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
# format_readiness
# ---------------------------------------------------------------------------


def test_scholar_only_pct_matches_direct_ratio():
    """wyrd-w1ak: scholar_only_pct is scholar_attested / total, NOT
    (scholar + empirical) / total. Visibility metric — surfaces how
    much of a language's coverage rests on scholarly sources alone
    vs the empirical-mining wedge."""
    lc = LanguageCriterion(
        language="old_french",
        total_subjects=548,
        scholar_attested=16,
        empirical_only=505,
        rando_only=0,
        uncited=27,
    )
    # Fixture shape mirrors the Old French operator-lexicon profile
    # where empirical does most of the work: clears C1 via combined
    # coverage even though scholar-only share is small.
    assert lc.scholar_only_pct == pytest.approx(100.0 * 16 / 548)
    assert lc.coverage_pct == pytest.approx(100.0 * (16 + 505) / 548)


def test_scholar_only_pct_zero_subjects_returns_zero():
    """Defensive division: a language with zero bundle subjects gets
    a 0.0 scholar-only fraction (consistent with coverage_pct's
    zero-subjects return)."""
    lc = LanguageCriterion(
        language="proto-germanic",
        total_subjects=0,
        scholar_attested=0,
        empirical_only=0,
        rando_only=0,
        uncited=0,
    )
    assert lc.scholar_only_pct == 0.0


def test_scholar_only_pct_does_not_change_c1_verdict():
    """wyrd-w1ak invariance: adding the scholar_only column is a
    visibility change, not a gate change. C1 still uses the combined
    (scholar + empirical) ratio. Pin so a future refactor that
    accidentally re-routes passes_coverage through scholar_only_pct
    fails fast."""
    lc = LanguageCriterion(
        language="old_french",
        total_subjects=100,
        scholar_attested=3,  # 3% scholar-only — below any reasonable threshold
        empirical_only=95,  # but combined 98% — clears C1
        rando_only=0,
        uncited=2,
    )
    assert lc.scholar_only_pct == pytest.approx(3.0)
    assert lc.coverage_pct == pytest.approx(98.0)
    assert lc.passes_coverage(0.80), (
        "C1 must remain driven by combined coverage; scholar_only_pct is a "
        "visibility metric, not a gate"
    )


def test_format_readiness_includes_scholar_only_column():
    """wyrd-w1ak: the markdown table renders a 'Scholar only' column
    alongside 'Coverage'. Test asserts the header is present plus a
    rendered percentage row matches the property."""
    report = ReadinessReport(
        coverage_threshold=0.80,
        per_language=(
            LanguageCriterion(
                language="old_french",
                total_subjects=548,
                scholar_attested=16,
                empirical_only=505,
                rando_only=0,
                uncited=27,
            ),
        ),
    )
    out = format_readiness(report)
    assert "| Scholar only |" in out, (
        f"markdown table must include 'Scholar only' column header; got:\n{out}"
    )
    # The rendered value matches the property to one decimal.
    assert "| 2.9% | 95.1% |" in out, (
        f"row must render scholar_only_pct (2.9%) before coverage_pct (95.1%); got:\n{out}"
    )


def test_format_readiness_documents_scholar_only_column():
    """wyrd-w1ak: criterion-definitions section explains the column
    is visibility-only (NOT a gate)."""
    report = ReadinessReport(coverage_threshold=0.80, per_language=())
    out = format_readiness(report)
    assert "Scholar only" in out
    assert "visibility" in out, (
        "the docstring must call out that this is a visibility metric, "
        f"not a gate change; got:\n{out}"
    )


def test_format_readiness_pass_marker():
    """Passing report includes the PASS verdict and omits the failure section."""
    report = ReadinessReport(coverage_threshold=0.80, per_language=())
    out = format_readiness(report)
    assert "PASS — retirement gate open" in out
    # The "why closed" section must be absent on a pass — pins the
    # _readiness_failure_lines early-return (returns [] when overall_passes).
    assert "Why the gate is closed" not in out


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


def test_cli_rando_refs_surfaces_rando_only_on_c2(tmp_path: Path):
    """wyrd-qkn0: the CLI loads the rando-port ledger (--rando-refs) and threads
    it into the gate. Bundle = 4 scholar subjects + 1 uncited morpheme whose
    morpheme_id is in the ledger → coverage 80% (C1 passes), so the gate can
    only close on C2 (rando-only). Pins the full --bundle + --rando-refs wiring:
    a revert would drop rando_only to 0, C2 would pass, and the gate would
    falsely report OPEN (exit 0)."""
    scholar = {"words": [_word({"old_english": ["mawer_1920_northumberland_durham"]})]}
    rando = {
        "words": [
            {"old_english": "tun", "old_english_citations": [], "morpheme_id": "old-english:tun"}
        ]
    }
    bundle = tmp_path / "meanings.json"
    bundle.write_text(json.dumps({"subjects": [scholar, scholar, scholar, scholar, rando]}))
    ledger = tmp_path / "rando-port.jsonl"
    ledger.write_text(json.dumps({"_type": "citation", "etymon_ref": "old-english:tun"}))
    result = CliRunner().invoke(
        cli_root,
        [
            "lexicon",
            "rando-port-readiness",
            "--bundle",
            str(bundle),
            "--rando-refs",
            str(ledger),
            "--language",
            "old_english",
        ],
    )
    assert result.exit_code == 1, result.output  # closed on C2, not C1 (coverage is 80%)
    assert "C2 (1 rando-only)" in result.output


@pytest.mark.no_lexicon_isolation
def test_cli_rando_port_readiness_rehydrates_from_runtime_db():
    """wyrd-52ha: with no --bundle, the command rehydrates the bundle from the
    L4 runtime DB (the on-disk meanings.json was retired, d90t) instead of
    erroring on a missing file. Produces the readiness report against the
    bundled seed; exit code is the gate verdict (0 open / 1 closed), not a usage
    error."""
    runner = CliRunner()
    result = runner.invoke(cli_root, ["lexicon", "rando-port-readiness"])
    # exit 1 must be the gate-closed SystemExit, NOT an unhandled crash (which
    # CliRunner also surfaces as exit 1 — and whose traceback would contain the
    # report header, false-passing the asserts below).
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert result.exit_code in (0, 1), result.output
    assert "Rando-port retirement readiness" in result.output
    # The retired-meanings.json failure mode must be gone.
    assert "does not exist" not in result.output


def test_default_coverage_threshold_value():
    """80% is the wyrd-j2bv-defined gate; pin it so a typo at the
    declaration site fails loudly."""
    assert DEFAULT_COVERAGE_THRESHOLD == 0.80
