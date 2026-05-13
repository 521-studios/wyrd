"""Tests for the truncated-short_quote audit — wyrd-bd68."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.short_quote_audit import (
    MIN_TRUNCATION_LENGTH,
    _truncate_sample,
    audit_jsonl_dir,
    audit_jsonl_file,
    format_audit_report,
    looks_truncated,
)

# ---------------------------------------------------------------------------
# looks_truncated — the load-bearing classifier
# ---------------------------------------------------------------------------


def _pad(s: str) -> str:
    """Pad a tail to clear MIN_TRUNCATION_LENGTH so the heuristic
    actually runs the trailing-fragment test (the threshold otherwise
    filters short citations out)."""
    padding = "x " * MIN_TRUNCATION_LENGTH
    return padding + s


def test_looks_truncated_empty_or_none():
    """No-quote rows aren't truncations — they're missing data."""
    assert not looks_truncated(None)
    assert not looks_truncated("")


def test_looks_truncated_short_quote_not_flagged():
    """Below the length floor, even mid-clause endings are allowed.
    Short citations legitimately end mid-fragment ('cf. 1086 DB')."""
    assert not looks_truncated("cf. 1086 DB")


def test_looks_truncated_terminal_punctuation_passes():
    """Standard sentence endings: period, exclamation, question,
    quotation, bracket. All clear-completion signals."""
    for terminal in (".", "!", "?", '"', "'", ")", "]", "}"):
        assert not looks_truncated(_pad("text" + terminal)), terminal


def test_looks_truncated_pipe_short_tail_flagged():
    """The canonical bannister shape: LLM commentary, then pipe,
    then a short word-fragment. The page-101 example."""
    assert looks_truncated(_pad("Scholar prose | Ab"))
    assert looks_truncated(_pad("Scholar prose | Abber"))
    assert looks_truncated(_pad("Scholar prose | 1268"))


def test_looks_truncated_pipe_long_tail_not_flagged():
    """A long token after the pipe is most likely a full
    sentence-or-citation — the LLM didn't cut off there."""
    assert not looks_truncated(_pad("Scholar prose | Aberystwyth_full_word."))


def test_looks_truncated_short_trailing_token_flagged():
    """Without a pipe but ending in a short fragment after whitespace:
    likely truncated. ``See Do`` is incomplete where
    ``See Donegal.`` would be done."""
    assert looks_truncated(_pad("the great northern road. See Do"))
    assert looks_truncated(_pad("references: Aclitone DB Tornelay 1268"))


def test_looks_truncated_single_letter_tail_not_flagged():
    """Scholarly abbreviations like ``P`` (Pipe Rolls), ``B`` (Bede)
    are 1-char terminal tokens that look like truncation but aren't.
    Skipping single-letter tails kills the dominant false-positive
    class without losing the multi-char truncation signal."""
    assert not looks_truncated(_pad("cf. Aclitone P"))
    assert not looks_truncated(_pad("cf. Eboracum B"))


def test_looks_truncated_terminal_punctuation_beats_short_tail():
    """A short token followed by terminal punctuation is complete,
    not truncated. ``cf. DB.`` is a fine closing."""
    assert not looks_truncated(_pad("cf. DB."))
    assert not looks_truncated(_pad("the road. See Do."))


# ---------------------------------------------------------------------------
# audit_jsonl_file
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def test_audit_jsonl_file_counts_citations_only(tmp_path: Path):
    """Non-citation rows (etymon, toponym, source) are not scanned
    — the audit is scoped to citation.short_quote per the wyrd-bd68
    ticket."""
    path = _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {"_type": "etymon", "ref": "oe:cot", "language": "old-english"},
            {"_type": "citation", "etymon_ref": "oe:cot", "short_quote": _pad("complete.")},
            {"_type": "citation", "etymon_ref": "oe:cot", "short_quote": _pad("prose | Ab")},
        ],
    )
    result = audit_jsonl_file(path)
    assert result.total_citations == 2
    assert result.truncated_citations == 1


def test_audit_jsonl_file_skips_remove_events(tmp_path: Path):
    """_op=remove events are operator retirements; counting them as
    truncated would double-count work the operator already did."""
    path = _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {
                "_op": "remove",
                "_type": "citation",
                "etymon_ref": "oe:cot",
                "short_quote": _pad("prose | Ab"),
            },
            {"_type": "citation", "etymon_ref": "oe:cot", "short_quote": _pad("prose | Ab")},
        ],
    )
    result = audit_jsonl_file(path)
    assert result.total_citations == 1
    assert result.truncated_citations == 1


def test_audit_jsonl_file_handles_unparseable_lines(tmp_path: Path):
    """Schema validation isn't the audit's job — a corrupt row
    shouldn't abort the scan."""
    path = tmp_path / "src.jsonl"
    path.write_text(
        '{"_type": "citation", "short_quote": "complete."}\n'
        "this is not json at all\n"
        '{"_type": "citation", "short_quote": "' + ("y " * 200) + 'prose | Ab"}\n'
    )
    result = audit_jsonl_file(path)
    assert result.total_citations == 2
    assert result.truncated_citations == 1


def test_audit_jsonl_file_captures_samples_under_limit(tmp_path: Path):
    """Per-source samples are capped at `sample_limit` — keeps the
    report from drowning in identical-shape rows."""
    rows = [{"_type": "citation", "short_quote": _pad(f"prose {i} | Ab")} for i in range(10)]
    path = _write_jsonl(tmp_path / "src.jsonl", rows)
    result = audit_jsonl_file(path, sample_limit=3)
    assert result.truncated_citations == 10
    assert len(result.samples) == 3


def test_audit_jsonl_file_source_id_from_stem(tmp_path: Path):
    """source_id = filename without ``.jsonl`` — matches the
    per-source dumper's naming so the audit report keys align with
    L2 file paths."""
    path = _write_jsonl(tmp_path / "mawer_1920_northumberland_durham.jsonl", [])
    result = audit_jsonl_file(path)
    assert result.source_id == "mawer_1920_northumberland_durham"


# ---------------------------------------------------------------------------
# audit_jsonl_dir
# ---------------------------------------------------------------------------


def test_audit_jsonl_dir_skips_underscore_files(tmp_path: Path):
    """``_curation.jsonl`` is the operator event-log, not source data.
    Audit should skip anything starting with underscore."""
    _write_jsonl(tmp_path / "good.jsonl", [{"_type": "citation", "short_quote": "ok."}])
    _write_jsonl(
        tmp_path / "_curation.jsonl",
        [{"_type": "citation", "short_quote": _pad("prose | Ab")}],
    )
    report = audit_jsonl_dir(tmp_path)
    assert [s.source_id for s in report.sources] == ["good"]


def test_audit_jsonl_dir_source_filter(tmp_path: Path):
    """Passing ``sources=[...]`` restricts the scan."""
    _write_jsonl(tmp_path / "a.jsonl", [])
    _write_jsonl(tmp_path / "b.jsonl", [])
    _write_jsonl(tmp_path / "c.jsonl", [])
    report = audit_jsonl_dir(tmp_path, sources=("a", "c"))
    assert [s.source_id for s in report.sources] == ["a", "c"]


# ---------------------------------------------------------------------------
# format_audit_report
# ---------------------------------------------------------------------------


def test_format_audit_report_empty_corpus(tmp_path: Path):
    """No sources → 0 / 0% clean-corpus message, no crash on
    division by zero."""
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert "Sources scanned: 0" in out
    assert "Total citations: 0" in out
    assert "0.0%" in out


def test_format_audit_report_ranks_by_truncation_count(tmp_path: Path):
    """Top-N table sorts by absolute truncation count (operator wants
    to fix the high-volume offenders first), not by rate."""
    _write_jsonl(
        tmp_path / "high_volume.jsonl",
        [{"_type": "citation", "short_quote": _pad("p | Ab")}] * 10,
    )
    _write_jsonl(
        tmp_path / "high_rate.jsonl",
        [{"_type": "citation", "short_quote": _pad("p | Ab")}],
    )
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert out.index("high_volume") < out.index("high_rate")


def test_format_audit_report_clean_corpus_message(tmp_path: Path):
    """If no truncations are detected, the report says so explicitly
    rather than producing an empty samples section."""
    _write_jsonl(tmp_path / "clean.jsonl", [{"_type": "citation", "short_quote": "ok."}])
    report = audit_jsonl_dir(tmp_path)
    out = format_audit_report(report)
    assert "_No truncations detected" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_truncate_sample_ellipses_over_max_len():
    """Long samples get an ellipsis suffix exactly at max_len. The
    full quote is the flag — printing >200 chars in a report row
    wastes screen space without adding information."""
    long_sample = "x" * 500
    out = _truncate_sample(long_sample, max_len=200)
    assert len(out) == 200
    assert out.endswith("...")


def test_truncate_sample_short_passthrough():
    """Samples under max_len pass through unchanged — no spurious
    ellipses on already-short rows."""
    assert _truncate_sample("short", max_len=200) == "short"


def test_cli_audit_short_quotes(tmp_path: Path):
    _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {"_type": "citation", "short_quote": "ok."},
            {"_type": "citation", "short_quote": _pad("prose | Ab")},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(cli_root, ["lexicon", "audit-short-quotes", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Truncated-short_quote audit" in result.output
    assert "src" in result.output
