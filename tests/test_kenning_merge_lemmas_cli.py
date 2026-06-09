"""CLI-level tests for ``wyrd kenning lexicon merge-lemmas`` (v3ow P3).

The meaning_merge_detect unit pieces (detect_meaning_merge_candidates /
build_merge_judge_prompt / parse_merge_verdict / merge_verdict_to_row) are
covered in test_meaning_merge_detect.py. These pin the wyrd-8uvi-extracted
command orchestration (``_load_judged`` / ``_select_todo`` /
``_judge_and_record`` / ``_MergeLemmaRun``) end-to-end via CliRunner, with the
candidate detection and the LLM transport stubbed — no DB scan, no Ollama call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import wyrd.generators.kenning.extractors.llm as llm
import wyrd.generators.kenning.lexicon.meaning_merge_detect as mm
from wyrd.generators.kenning.cli.lexicon.merge_lemmas import lexicon_merge_lemmas
from wyrd.generators.kenning.lexicon import init_schema

_CANDS = [
    mm.MeaningMergeCandidate("oe:foo", "oe:foa", "foo", "foa", ("g1",), ("g2",), 0.9, 1, 5),
    mm.MeaningMergeCandidate("oe:bar", "oe:baz", "bar", "baz", ("g3",), ("g4",), 0.8, 1, 4),
]


@pytest.fixture
def _stub_detect_and_llm(monkeypatch):
    """Stub candidate detection (canned minor↔major pairs) + the Ollama
    transport (1st call → confident same-morpheme merge; 2nd → low reject)."""
    monkeypatch.setattr(mm, "detect_meaning_merge_candidates", lambda conn, **kw: list(_CANDS))
    calls = {"n": 0}

    def _fake_chat(self, system, user, schema):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return {"same_morpheme": True, "confidence": "high", "reason": "same"}
        return {"same_morpheme": False, "confidence": "low", "reason": "distinct"}

    monkeypatch.setattr(llm.OllamaClient, "chat_json", _fake_chat)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "lex.db"
    init_schema(db)  # so db.exists() + _readonly_lexicon open succeed (detect is stubbed)
    return db


def test_merge_lemmas_dry_run_writes_nothing(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_merge_lemmas, ["--db", str(db), "--collapse-file", str(ledger), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not ledger.exists()
    assert "recorded: 1 merges + 1 rejections (0 skipped)" in result.output


def test_merge_lemmas_apply_records_merge_and_reject(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_merge_lemmas, ["--db", str(db), "--collapse-file", str(ledger)]
    )
    assert result.exit_code == 0
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(rows) == 2
    merge, reject = rows
    # Confident same-morpheme → a real `into` fold; low not-same → no-op.
    assert merge["ref"] == "oe:foo" and merge["into"] == "oe:foa"
    assert reject["ref"] == "oe:bar" and reject["into"] == ""
    assert "recorded: 1 merges + 1 rejections (0 skipped)" in result.output


def test_merge_lemmas_rerun_skips_already_judged(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    runner = CliRunner()
    runner.invoke(lexicon_merge_lemmas, ["--db", str(db), "--collapse-file", str(ledger)])
    # The merged minor is in merged_refs; the rejected pair is in `judged` →
    # the second run judges nothing.
    result = runner.invoke(lexicon_merge_lemmas, ["--db", str(db), "--collapse-file", str(ledger)])
    assert result.exit_code == 0
    assert "already-judged=2 to-judge=0" in result.output
    assert "Nothing to judge." in result.output
