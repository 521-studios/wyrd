"""CLI-level tests for the element-gloss commands (wyrd-4yfc / wyrd-u9k6).

The backfill module logic is covered in test_element_gloss_backfill.py; this
pins the CLI glue the test-coverage-reviewer flagged on PR #410:
- ``mine-element-glosses`` — the confident/weak/no-match summary branching.
- ``adjudicate-element-glosses`` — the resume-dedup (``done`` set) + the
  per-record error capture (an LLM failure is recorded, not fatal).

The LLM transport (Ollama ``requests.post``) and the DB-touching backfill
helpers are stubbed — no Ollama, no real lexicon DB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
from click.testing import CliRunner

import wyrd.generators.kenning.lexicon.element_gloss_backfill as eg
from wyrd.generators.kenning.cli.lexicon.adjudicate_element_glosses import (
    lexicon_adjudicate_element_glosses,
)
from wyrd.generators.kenning.cli.lexicon.mine_element_glosses import lexicon_mine_element_glosses


def _touch(p: Path) -> Path:
    p.write_text("x", encoding="utf-8")  # a non-empty file so click.Path(exists=True) is satisfied
    return p


def test_mine_element_glosses_summary_counts(tmp_path: Path, monkeypatch) -> None:
    """The summary tallies is_element → confident, candidates-but-not-element →
    weak, no-candidates → no-match. Stub mine_to_jsonl with one of each."""
    rows = [
        {"surface": "a", "is_element": True, "candidates": [{"x": 1}]},  # confident
        {"surface": "b", "is_element": False, "candidates": [{"x": 1}]},  # weak
        {"surface": "c", "is_element": False, "candidates": []},  # no-match
    ]
    monkeypatch.setattr(
        eg,
        "mine_to_jsonl",
        lambda *a, **k: eg.ElementGlossMineResult(
            rows=rows, written=rows, existing_kept=0, new_added=len(rows), overwritten=False
        ),
    )
    census = _touch(tmp_path / "census.db")
    db = _touch(tmp_path / "lex.db")
    out = tmp_path / "_element_glosses.jsonl"
    result = CliRunner().invoke(
        lexicon_mine_element_glosses,
        ["--census", str(census), "--db", str(db), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "[3/3]  confident=1 weak=1 no-match=1" in result.output
    assert "merged into" in result.output and "added 3 new" in result.output


def test_mine_element_glosses_overwrite_flag_forwarded(tmp_path: Path, monkeypatch) -> None:
    """The --overwrite flag is forwarded to mine_to_jsonl and the summary reports
    the destructive 'overwrote' branch (wyrd-5f5w)."""
    captured: dict = {}

    def _stub(*a, **k):
        captured["overwrite"] = k.get("overwrite")
        return eg.ElementGlossMineResult(
            rows=[], written=[], existing_kept=0, new_added=0, overwritten=k.get("overwrite", False)
        )

    monkeypatch.setattr(eg, "mine_to_jsonl", _stub)
    census = _touch(tmp_path / "census.db")
    db = _touch(tmp_path / "lex.db")
    out = tmp_path / "_element_glosses.jsonl"
    result = CliRunner().invoke(
        lexicon_mine_element_glosses,
        ["--census", str(census), "--db", str(db), "--output", str(out), "--overwrite"],
    )
    assert result.exit_code == 0, result.output
    assert captured["overwrite"] is True  # flag forwarded
    assert "overwrote" in result.output  # destructive-branch verb


@pytest.fixture
def _stub_adjudicate(monkeypatch):
    """Stub the DB-touching backfill helpers + the Ollama transport. Returns a
    mutable call counter for the (default-OK) ``requests.post`` stub."""
    monkeypatch.setattr(eg, "load_toponym_lowers", lambda db: [])
    monkeypatch.setattr(eg, "toponym_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(eg, "build_adjudication_user", lambda *a, **k: "user")
    monkeypatch.setattr(
        eg,
        "accept_adjudication",
        lambda row, response: row["candidates"][0] if response.get("choice") else None,
    )
    calls = {"n": 0}

    class _Resp:
        def json(self):
            return {
                "message": {"content": json.dumps({"choice": 1, "confidence": "high", "why": "w"})}
            }

    def _post(url, **kw):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(requests, "post", _post)
    return calls


def _weak_row(surface: str, weight: int) -> str:
    return json.dumps(
        {
            "surface": surface,
            "position": "post",
            "is_element": False,
            "candidates": [{"c": 1}],
            "weight": weight,
        }
    )


def test_adjudicate_resume_skips_already_done(tmp_path: Path, _stub_adjudicate) -> None:
    """A surface already in the output ledger is in the ``done`` set, so the
    second run only adjudicates the remaining surface (one LLM call)."""
    consensus = tmp_path / "consensus.jsonl"
    consensus.write_text(_weak_row("A", 2) + "\n" + _weak_row("B", 1) + "\n")
    db = _touch(tmp_path / "lex.db")
    out = tmp_path / "adj.jsonl"
    out.write_text(json.dumps({"surface": "A", "accepted": True}) + "\n")  # A already adjudicated
    result = CliRunner().invoke(
        lexicon_adjudicate_element_glosses,
        ["--consensus", str(consensus), "--db", str(db), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert _stub_adjudicate["n"] == 1  # only B asked — A skipped via the done set
    surfaces = [
        json.loads(line)["surface"] for line in out.read_text().splitlines() if line.strip()
    ]
    assert surfaces == ["A", "B"]  # A pre-existing, B appended


def test_adjudicate_captures_per_record_error(
    tmp_path: Path, monkeypatch, _stub_adjudicate
) -> None:
    """An LLM transport failure is recorded on the row (``error``) and the batch
    continues + exits cleanly, rather than aborting."""

    def _boom(url, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(requests, "post", _boom)
    consensus = tmp_path / "consensus.jsonl"
    consensus.write_text(_weak_row("X", 1) + "\n")
    db = _touch(tmp_path / "lex.db")
    out = tmp_path / "adj.jsonl"
    result = CliRunner().invoke(
        lexicon_adjudicate_element_glosses,
        ["--consensus", str(consensus), "--db", str(db), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output  # error captured, not raised
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["surface"] == "X"
    assert "ollama down" in rec["error"]
    assert "accepted" not in rec  # never reached the accept path
