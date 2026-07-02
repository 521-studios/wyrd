"""Tests for the toponym/etymology alignment audit — wyrd-8upf."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.etymology_alignment_audit import (
    _STOP_MORPHEMES,
    MISALIGNMENT_REASON_EMPTY_FORM,
    MISALIGNMENT_REASON_HALLUCINATED,
    _normalize,
    audit_jsonl_dir,
    audit_jsonl_file,
    format_audit_report,
    looks_misaligned,
)

# ---------------------------------------------------------------------------
# _normalize — the load-bearing substring matcher
# ---------------------------------------------------------------------------


def test_normalize_strips_acute_diacritic():
    """``Céolwig`` ↔ ``ceolwig`` is the most common scholar-form
    diacritic pair: the LLM emits the bare canonical_form while the
    historical_form preserves the original accent."""
    assert "ceolwig" in _normalize("Céolwigesham")


def test_normalize_strips_macron():
    """Old English long-vowel marks (``tūn``, ``hām``) come through
    Unicode NFKD decomposition, which the hand-mapped diacritic
    table misses. Must round-trip ``tūn`` → ``tun``."""
    assert _normalize("tūn") == "tun"
    assert _normalize("hām") == "ham"


def test_normalize_splits_ae_ligature():
    """``æ`` doesn't decompose under NFKD; explicit fixup turns it
    into ``ae``. OE rendition uses ``æ`` interchangeably with
    ``ae`` in different transcription schemes."""
    assert _normalize("æsc") == "aesc"
    assert _normalize("Æthelwig") == "aethelwig"


def test_normalize_collapses_punctuation_to_space():
    """Asterisks (reconstruction markers) and hyphens shouldn't
    prevent substring matching. ``*Albentio`` should still match
    against an element whose canonical_form is ``albentio``."""
    norm = _normalize("*Albentio-Reconstruction")
    assert "albentio" in norm
    assert "reconstruction" in norm


def test_normalize_empty_string():
    """Defensive — empty input shouldn't crash the matcher."""
    assert _normalize("") == ""
    assert _normalize(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# looks_misaligned
# ---------------------------------------------------------------------------


def _row(
    *,
    toponym_ref="X@Y",
    elements=None,
    historical_form="",
    notes="",
    confidence="medium",
):
    return {
        "_type": "etymology_element",
        "toponym_ref": toponym_ref,
        "elements": elements or [],
        "historical_form": historical_form,
        "notes": notes,
        "confidence": confidence,
    }


def test_looks_misaligned_empty_elements_not_flagged():
    """No elements means nothing to align against. The audit is
    scoped to FALSE-element claims, not missing data."""
    finding = looks_misaligned(_row(historical_form="anything"))
    assert finding is None


def test_looks_misaligned_perfect_alignment_not_flagged():
    """All canonical_forms appear in the historical_form → no flag."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:beorg"},
                {"etymon_ref": "old-english:glind"},
            ],
            historical_form="Beorg-glind",
        )
    )
    assert finding is None


def test_looks_misaligned_missing_canonical_form_flagged():
    """An element whose canonical_form doesn't appear in the
    historical_form → flag. The Gemini-flagged shape: elements
    propose a morpheme the reconstruction doesn't support."""
    finding = looks_misaligned(
        _row(
            elements=[{"etymon_ref": "old-english:hoh"}],
            historical_form="Dune",
        )
    )
    assert finding is not None
    assert finding.missing_morphemes == ["hoh"]


def test_looks_misaligned_empty_historical_form_flagged():
    """Elements without a reconstruction string — the LLM proposed
    morphemes but didn't commit to a historical form. Suspect."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:beorg"},
                {"etymon_ref": "old-english:glind"},
            ],
            historical_form="",
        )
    )
    assert finding is not None
    assert finding.missing_morphemes == ["beorg", "glind"]


def test_looks_misaligned_stop_morpheme_absence_not_flagged():
    """Stop morphemes (single letters, common inflections) being
    absent isn't signal. ``es`` (genitive) is routinely elided in
    attested spellings."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:beorg"},
                {"etymon_ref": "old-english:es"},  # genitive marker
            ],
            historical_form="Beorg",
        )
    )
    assert finding is None  # "es" is a stop morpheme, absence not flagged


def test_looks_misaligned_diacritic_aware_match():
    """The substring match runs through _normalize, so ``céolwig``
    matches ``Céolwigesham``. This is the most common scholar-form
    pairing in the corpus."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:céolwig"},
                {"etymon_ref": "old-english:ham"},
            ],
            historical_form="Céolwigesham",
        )
    )
    assert finding is None


def test_looks_misaligned_macron_aware_match():
    """OE long-vowel marks (tūn, hām) must match against their
    plain-vowel reconstructions and vice versa."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:aecceling"},
                {"etymon_ref": "old-english:tūn"},
            ],
            historical_form="Aecceling-tun",
        )
    )
    assert finding is None


def test_looks_misaligned_preserves_canonical_form_in_finding():
    """The finding's elements list and missing_morphemes use the
    ORIGINAL canonical_form (with diacritics), not the normalized
    one. Operator needs to see the actual etymon ref to fix it."""
    finding = looks_misaligned(
        _row(
            elements=[{"etymon_ref": "old-english:tūn"}],
            historical_form="something_else",
        )
    )
    assert finding is not None
    assert finding.missing_morphemes == ["tūn"]


# ---------------------------------------------------------------------------
# audit_jsonl_file
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def test_audit_jsonl_file_counts_etymology_only(tmp_path: Path):
    """Non-etymology_element rows (toponym, etymon, source, citation)
    are not scanned — the audit is scoped to element-list alignment."""
    path = _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {"_type": "toponym", "ref": "T@-"},
            _row(
                elements=[{"etymon_ref": "oe:cot"}],
                historical_form="cot",
            ),
            _row(
                elements=[{"etymon_ref": "oe:foo"}],
                historical_form="bar",
            ),
        ],
    )
    result = audit_jsonl_file(path)
    assert result.total_etymologies == 2
    assert result.flagged_count == 1


def test_audit_jsonl_file_skips_remove_events(tmp_path: Path):
    """Operator-retired rows shouldn't count — they've already been
    addressed."""
    path = _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {
                **_row(
                    elements=[{"etymon_ref": "oe:foo"}],
                    historical_form="bar",
                ),
                "_op": "remove",
            },
            _row(
                elements=[{"etymon_ref": "oe:foo"}],
                historical_form="bar",
            ),
        ],
    )
    result = audit_jsonl_file(path)
    assert result.total_etymologies == 1
    assert result.flagged_count == 1


def test_audit_jsonl_file_tolerates_corrupt_lines(tmp_path: Path):
    """Schema validation isn't the audit's job — corrupt lines are
    skipped, not raised."""
    path = tmp_path / "src.jsonl"
    path.write_text(
        '{"_type": "etymology_element", "elements": [{"etymon_ref": "oe:cot"}], '
        '"historical_form": "cot"}\n'
        "not json\n"
        '{"_type": "etymology_element", "elements": [{"etymon_ref": "oe:foo"}], '
        '"historical_form": "bar"}\n'
    )
    result = audit_jsonl_file(path)
    assert result.total_etymologies == 2
    assert result.flagged_count == 1


def test_audit_jsonl_file_sample_limit_capped(tmp_path: Path):
    """Per-source sample list is capped to keep the report tight."""
    rows = [
        _row(
            elements=[{"etymon_ref": f"oe:foo{i}"}],
            historical_form="bar",
        )
        for i in range(10)
    ]
    path = _write_jsonl(tmp_path / "src.jsonl", rows)
    result = audit_jsonl_file(path, sample_limit=3)
    assert result.flagged_count == 10
    assert len(result.samples) == 3


# ---------------------------------------------------------------------------
# audit_jsonl_dir + format_audit_report
# ---------------------------------------------------------------------------


def test_audit_jsonl_dir_skips_underscore_files(tmp_path: Path):
    _write_jsonl(
        tmp_path / "good.jsonl",
        [_row(elements=[{"etymon_ref": "oe:cot"}], historical_form="cot")],
    )
    _write_jsonl(
        tmp_path / "_curation.jsonl",
        [_row(elements=[{"etymon_ref": "oe:foo"}], historical_form="bar")],
    )
    report = audit_jsonl_dir(tmp_path)
    assert [s.source_id for s in report.sources] == ["good"]


def test_format_audit_report_ranks_by_flag_count(tmp_path: Path):
    """Top-N sorted by absolute flag count, not rate."""
    _write_jsonl(
        tmp_path / "high_volume.jsonl",
        [_row(elements=[{"etymon_ref": "oe:foo"}], historical_form="bar")] * 10,
    )
    _write_jsonl(
        tmp_path / "high_rate.jsonl",
        [_row(elements=[{"etymon_ref": "oe:foo"}], historical_form="bar")],
    )
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert out.index("high_volume") < out.index("high_rate")
    # wyrd-73ma: suppression is narrow — with flags present, both section headers
    # still render (the normal top_n=20 path is unchanged). The leading "\n\n"
    # pins the single blank-line inter-section separator, so a regression from the
    # "\n\n"-join back to a "\n"-join (losing the byte-identical layout) fails here.
    assert "\n\n## Top sources by flag count" in out
    assert "\n\n## Samples" in out


def test_format_audit_report_clean_corpus(tmp_path: Path):
    _write_jsonl(
        tmp_path / "clean.jsonl",
        [_row(elements=[{"etymon_ref": "oe:cot"}], historical_form="cot")],
    )
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert "_No alignment misses detected" in out
    # wyrd-73ma: a clean corpus has no flagged sources, so the top-sources table
    # is suppressed rather than rendered as an orphaned header + empty table.
    assert "## Top sources by flag count" not in out


def test_format_audit_report_suppresses_empty_sections_when_top_n_non_positive(tmp_path: Path):
    """wyrd-73ma: with misses present but ``top_n <= 0`` rendering none, neither
    the '## Top sources' table nor the '## Samples' section emits an orphaned
    header — only the summary. Covers 0 AND negative (a negative top_n is clamped
    to 0, not left to Python's negative slicing). Degenerate input; the CLI
    defaults to top_n=20."""
    _write_jsonl(
        tmp_path / "flagged.jsonl",
        [_row(elements=[{"etymon_ref": "oe:foo"}], historical_form="bar")] * 3,
    )
    report = audit_jsonl_dir(tmp_path)
    for val in (0, -1):
        out = format_audit_report(report, top_n=val)
        assert "## Top sources by flag count" not in out, val
        assert "## Samples" not in out, val
        # There ARE misses (so not the clean-corpus placeholder), but none shown.
        assert "_No alignment misses detected" not in out, val
        assert "Probably misaligned:" in out, val  # the summary section still renders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_audit_etymology_alignment(tmp_path: Path):
    _write_jsonl(
        tmp_path / "src.jsonl",
        [
            _row(elements=[{"etymon_ref": "oe:cot"}], historical_form="cot"),
            _row(elements=[{"etymon_ref": "oe:foo"}], historical_form="bar"),
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "audit-etymology-alignment", "--dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "alignment audit" in result.output
    assert "src" in result.output


# ---------------------------------------------------------------------------
# wyrd-j6co: finding.reason categorization + extended _STOP_MORPHEMES
# ---------------------------------------------------------------------------


def test_misalignment_finding_reason_empty_form():
    """When historical_form is empty, the finding's reason is
    'empty_form' — operator should re-mine, not prune."""
    finding = looks_misaligned(
        _row(
            elements=[{"etymon_ref": "old-english:beorg"}],
            historical_form="",
        )
    )
    assert finding is not None
    assert finding.reason == MISALIGNMENT_REASON_EMPTY_FORM


def test_misalignment_finding_reason_hallucinated():
    """When elements don't substring-match the historical_form,
    reason is 'hallucinated' — operator should prune/curate."""
    finding = looks_misaligned(
        _row(
            elements=[{"etymon_ref": "old-english:beorg"}],
            historical_form="Tuneham",  # 'beorg' not in form
        )
    )
    assert finding is not None
    assert finding.reason == MISALIGNMENT_REASON_HALLUCINATED


def test_stop_morphemes_includes_non_oe_inflections():
    """wyrd-j6co: _STOP_MORPHEMES extended for OE/ON/Irish/OF
    inflections that elide in attested forms. Verifies the new
    additions are present (regression-pin against accidental removal)."""
    for stop in ("um", "on", "ar", "ir"):
        assert stop in _STOP_MORPHEMES, f"{stop!r} should be a stop-morpheme"
    # Confirms pre-existing entries are still there (no accidental clobber).
    for legacy in ("a", "e", "es", "ing"):
        assert legacy in _STOP_MORPHEMES


def test_on_plural_ar_not_flagged_when_missing():
    """ON nom/acc plural -ar is now a stop morpheme. A reconstruction
    that elides it shouldn't flag as misaligned. Concrete case: a
    toponym whose elements include an ON 'ar' suffix marker that
    doesn't appear in the singular-form historical_form."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-norse:steinn"},
                {"etymon_ref": "old-norse:ar"},  # plural marker
            ],
            historical_form="Steinn",
        )
    )
    assert finding is None, "ON -ar should be in _STOP_MORPHEMES; absence shouldn't flag"


def test_looks_misaligned_filters_empty_etymon_refs():
    """wyrd-j6co Gemini round-1 robustness: an element row with a
    missing/empty etymon_ref shouldn't produce an empty string in
    the report's elements list. Such elements are dropped before
    the alignment check runs."""
    finding = looks_misaligned(
        _row(
            elements=[
                {"etymon_ref": "old-english:beorg"},
                {"etymon_ref": ""},  # malformed
                {"etymon_ref": None},  # malformed
            ],
            historical_form="Dune",  # beorg not in form → flag
        )
    )
    assert finding is not None
    assert finding.elements == ["beorg"], (
        "empty refs should be filtered before being placed in elements"
    )
    assert "" not in finding.missing_morphemes


def test_looks_misaligned_returns_none_when_all_etymon_refs_empty():
    """If every element has a missing etymon_ref, the audit can't
    say anything meaningful — return None rather than flag noise."""
    finding = looks_misaligned(
        _row(
            elements=[{"etymon_ref": ""}, {"etymon_ref": None}],
            historical_form="Anything",
        )
    )
    assert finding is None


def test_format_audit_report_breakdown_by_reason():
    """The report header includes a per-reason breakdown so operators
    can pick a failure mode to triage first."""
    from wyrd.generators.kenning.etymology_alignment_audit import (
        AuditReport,
        MisalignmentFinding,
        SourceAuditResult,
    )

    src = SourceAuditResult(source_id="test", total_etymologies=10, flagged_count=3)
    src.samples = [
        MisalignmentFinding(
            toponym_ref="A@-",
            historical_form="",
            elements=["x"],
            missing_morphemes=["x"],
            confidence="medium",
            notes_preview="",
            reason=MISALIGNMENT_REASON_EMPTY_FORM,
        ),
        MisalignmentFinding(
            toponym_ref="B@-",
            historical_form="Other",
            elements=["y"],
            missing_morphemes=["y"],
            confidence="high",
            notes_preview="",
            reason=MISALIGNMENT_REASON_HALLUCINATED,
        ),
        MisalignmentFinding(
            toponym_ref="C@-",
            historical_form="Different",
            elements=["z"],
            missing_morphemes=["z"],
            confidence="low",
            notes_preview="",
            reason=MISALIGNMENT_REASON_HALLUCINATED,
        ),
    ]
    src.reason_counts = {
        MISALIGNMENT_REASON_EMPTY_FORM: 1,
        MISALIGNMENT_REASON_HALLUCINATED: 2,
    }
    report = AuditReport(sources=[src], sample_limit_per_source=3)
    output = format_audit_report(report)
    # Header summary line includes a by-reason breakdown.
    assert "By reason" in output
    assert "1 empty_form" in output
    assert "2 hallucinated" in output
    # Per-sample lines tag the reason inline so operators see it
    # while scanning the report.
    assert "[empty_form]" in output
    assert "[hallucinated]" in output


def test_format_audit_report_breakdown_reflects_full_scan_not_samples(tmp_path: Path):
    """wyrd-j6co Gemini round 2: with sample_limit smaller than the
    flagged_count, the report's reason breakdown MUST reflect every
    flagged row's reason — not just the truncated samples list.
    Otherwise the operator sees a misleading distribution on large
    scans where samples are heavily capped."""
    rows = []
    # 5 empty_form findings (no historical_form) + 1 hallucinated
    for i in range(5):
        rows.append(
            _row(
                toponym_ref=f"E{i}@-",
                elements=[{"etymon_ref": f"old-english:beorg{i}"}],
                historical_form="",
            )
        )
    rows.append(
        _row(
            toponym_ref="H1@-",
            elements=[{"etymon_ref": "old-english:hoh"}],
            historical_form="Dune",
        )
    )
    path = tmp_path / "src.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # sample_limit=2 — only 2 of the 5 empty_form rows make it into
    # samples; the breakdown must still report 5 empty_form + 1 hallucinated.
    result = audit_jsonl_file(path, sample_limit=2)
    assert result.flagged_count == 6
    assert result.reason_counts == {
        MISALIGNMENT_REASON_EMPTY_FORM: 5,
        MISALIGNMENT_REASON_HALLUCINATED: 1,
    }
    assert len(result.samples) == 2  # capped

    from wyrd.generators.kenning.etymology_alignment_audit import AuditReport

    report = AuditReport(sources=[result], sample_limit_per_source=2)
    output = format_audit_report(report)
    # Breakdown uses reason_counts (full), not samples (capped).
    assert "5 empty_form" in output
    assert "1 hallucinated" in output
