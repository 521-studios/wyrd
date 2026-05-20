"""Tests for ``wyrd kenning lexicon recover-inprogress-chunks`` (wyrd-w7i3 /
wyrd-81i3 — refactored helpers covered directly)."""

from __future__ import annotations

import json
from pathlib import Path

from wyrd.generators.kenning.cli.lexicon._dedup import dedup_key
from wyrd.generators.kenning.cli.lexicon.recover_inprogress_chunks import (
    _handle_empty_log,
    _merge_candidate,
    _promote_candidate,
    _read_jsonl_rows,
)

# ---------------------------------------------------------------------
# dedup_key — shared seam between recover + staged-resume
# ---------------------------------------------------------------------


def test_dedup_key_collapses_identical_rows() -> None:
    """Two rows with the same (form, date_year, region_hint, context)
    tuple hash to identical keys. The staged-resume + recover paths
    both consume this so a recover after a partial resume can't
    double-insert."""
    a = {"form": "Stratford", "date_year": 1086, "region_hint": "BK", "context": "context A"}
    b = {"form": "Stratford", "date_year": 1086, "region_hint": "BK", "context": "context A"}
    assert dedup_key(a) == dedup_key(b)


def test_dedup_key_distinguishes_year_change() -> None:
    """A row with a different date_year hashes to a different key —
    same toponym attested at different years stays as separate rows."""
    a = {"form": "Stratford", "date_year": 1086, "region_hint": None, "context": "ctx"}
    b = {"form": "Stratford", "date_year": 1224, "region_hint": None, "context": "ctx"}
    assert dedup_key(a) != dedup_key(b)


def test_dedup_key_tolerates_missing_fields() -> None:
    """A row missing some fields gets defaulted (form/context → '',
    date_year/region_hint → None) so a malformed row doesn't blow up
    the merge loop."""
    assert dedup_key({}) == ("", None, None, "")


# ---------------------------------------------------------------------
# _handle_empty_log
# ---------------------------------------------------------------------


def test_handle_empty_log_removes_file_when_not_dry_run(tmp_path: Path) -> None:
    ip = tmp_path / "src.chunks.jsonl"
    ip.write_text("")
    _handle_empty_log(ip, "src", dry_run=False)
    assert not ip.exists()


def test_handle_empty_log_preserves_file_when_dry_run(tmp_path: Path) -> None:
    ip = tmp_path / "src.chunks.jsonl"
    ip.write_text("")
    _handle_empty_log(ip, "src", dry_run=True)
    assert ip.exists()


# ---------------------------------------------------------------------
# _promote_candidate
# ---------------------------------------------------------------------


def test_promote_candidate_atomic_renames_when_not_dry_run(tmp_path: Path) -> None:
    """No canonical file exists → in-progress log is renamed to the
    canonical path. Atomic-rename preserves content bit-for-bit."""
    ip = tmp_path / "src.chunks.jsonl"
    out = tmp_path / "src.jsonl"
    row = {"form": "X", "date_year": 1086, "region_hint": "BK", "context": "ctx"}
    ip.write_text(json.dumps(row) + "\n")
    _promote_candidate(ip, out, "src", [(json.dumps(row), row)], dry_run=False)
    assert not ip.exists()
    assert out.exists()
    assert json.loads(out.read_text().splitlines()[0]) == row


def test_promote_candidate_dry_run_skips_rename(tmp_path: Path) -> None:
    ip = tmp_path / "src.chunks.jsonl"
    out = tmp_path / "src.jsonl"
    ip.write_text('{"form": "X"}\n')
    _promote_candidate(ip, out, "src", [('{"form": "X"}', {"form": "X"})], dry_run=True)
    assert ip.exists()
    assert not out.exists()


# ---------------------------------------------------------------------
# _merge_candidate
# ---------------------------------------------------------------------


def test_merge_candidate_dedups_new_against_existing(tmp_path: Path) -> None:
    """Rows in the in-progress log whose dedup_key matches an existing
    canonical row are dropped (counted in dup_count). New rows are
    appended verbatim."""
    out = tmp_path / "src.jsonl"
    existing = {"form": "A", "date_year": 1086, "region_hint": None, "context": "x"}
    out.write_text(json.dumps(existing) + "\n")
    ip = tmp_path / "src.chunks.jsonl"
    duplicate = dict(existing)
    new = {"form": "B", "date_year": 1086, "region_hint": None, "context": "y"}
    ip.write_text(json.dumps(duplicate) + "\n" + json.dumps(new) + "\n")
    new_rows, _ = _read_jsonl_rows(ip)
    _merge_candidate(ip, out, "src", new_rows, dry_run=False)
    final_lines = out.read_text().splitlines()
    forms = [json.loads(line)["form"] for line in final_lines]
    assert forms == ["A", "B"]  # existing preserved verbatim, new appended


def test_merge_candidate_preserves_existing_verbatim(tmp_path: Path) -> None:
    """Stage-specific fields on existing rows (e.g. extractor) must
    survive the merge — recover must not re-serialize existing rows."""
    out = tmp_path / "src.jsonl"
    existing_line = (
        '{"form": "A", "date_year": 1086, "region_hint": null, '
        '"context": "x", "extractor": "tier-1-gemini"}'
    )
    out.write_text(existing_line + "\n")
    ip = tmp_path / "src.chunks.jsonl"
    new = {"form": "B", "date_year": 1086, "region_hint": None, "context": "y"}
    ip.write_text(json.dumps(new) + "\n")
    new_rows, _ = _read_jsonl_rows(ip)
    _merge_candidate(ip, out, "src", new_rows, dry_run=False)
    surviving_first_line = out.read_text().splitlines()[0]
    # Byte-for-byte preservation — no re-serialization (which would
    # change key order, formatting, etc.).
    assert surviving_first_line == existing_line


def test_merge_candidate_dry_run_leaves_files_alone(tmp_path: Path) -> None:
    out = tmp_path / "src.jsonl"
    out.write_text('{"form": "A"}\n')
    ip = tmp_path / "src.chunks.jsonl"
    new = {"form": "B"}
    ip.write_text(json.dumps(new) + "\n")
    before = out.read_text()
    new_rows, _ = _read_jsonl_rows(ip)
    _merge_candidate(ip, out, "src", new_rows, dry_run=True)
    assert ip.exists()  # not unlinked
    assert out.read_text() == before  # unchanged
