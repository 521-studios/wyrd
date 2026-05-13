"""Tests for the toponym/etymology alignment audit — wyrd-8upf."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.etymology_alignment_audit import (
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


def test_format_audit_report_clean_corpus(tmp_path: Path):
    _write_jsonl(
        tmp_path / "clean.jsonl",
        [_row(elements=[{"etymon_ref": "oe:cot"}], historical_form="cot")],
    )
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert "_No alignment misses detected" in out


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
