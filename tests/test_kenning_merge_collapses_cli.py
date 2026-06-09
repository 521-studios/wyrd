"""CLI-level tests for ``wyrd kenning lexicon merge-collapses`` (wyrd-6xzs).

The collapse_merge unit pieces (detect_merge_candidates / build_judge_prompt /
parse_verdict / verdict_to_row) are covered in test_kenning_collapse_merge.py.
These pin the wyrd-8uvi-extracted command orchestration (``_load_judged`` /
``_select_todo`` / ``_judge_and_record`` / ``_MergeRun``) end-to-end via
CliRunner, with the candidate detection and the LLM transport stubbed — no DB
scan, no Ollama call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import wyrd.generators.kenning.extractors.llm as llm
import wyrd.generators.kenning.lexicon.collapse_merge as cm
from wyrd.generators.kenning.cli.lexicon.merge_collapses import lexicon_merge_collapses
from wyrd.generators.kenning.lexicon import init_schema

_CANDS = [
    cm.MergeCandidate("oe:foo", "oe:foal", "foo", "foal", ("a young horse",), ("a foal",), "ocr"),
    cm.MergeCandidate("oe:bar", "oe:bear", "bar", "bear", ("barley",), ("a bear",), "ocr"),
]


@pytest.fixture
def _stub_detect_and_llm(monkeypatch):
    """Stub candidate detection (canned pair list) + the Ollama transport
    (1st call → confident same-meaning fold; 2nd → low not-same reject)."""
    monkeypatch.setattr(cm, "detect_merge_candidates", lambda conn: list(_CANDS))
    calls = {"n": 0}

    def _fake_chat(self, system, user, schema):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return {"same_meaning": True, "confidence": "high", "reason": "same item"}
        return {"same_meaning": False, "confidence": "low", "reason": "different"}

    monkeypatch.setattr(llm.OllamaClient, "chat_json", _fake_chat)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "lex.db"
    init_schema(db)  # so db.exists() + _readonly_lexicon open succeed (detect is stubbed)
    return db


def test_merge_collapses_dry_run_writes_nothing(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_merge_collapses, ["--db", str(db), "--collapse-file", str(ledger), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not ledger.exists()  # dry-run never opens the ledger for writing
    assert "would record: 1 folds + 1 rejections (0 skipped)" in result.output


def test_merge_collapses_apply_records_fold_and_reject(
    tmp_path: Path, _stub_detect_and_llm
) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_merge_collapses, ["--db", str(db), "--collapse-file", str(ledger)]
    )
    assert result.exit_code == 0
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(rows) == 2
    fold, reject = rows
    # Confident same-meaning → a real collapse (into set); low not-same → no-op.
    assert fold["ref"] == "oe:foo" and fold["into"] == "oe:foal"
    assert reject["ref"] == "oe:bar" and reject["into"] == ""
    assert "recorded: 1 folds + 1 rejections (0 skipped)" in result.output


def test_merge_collapses_rerun_skips_already_judged(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    runner = CliRunner()
    runner.invoke(lexicon_merge_collapses, ["--db", str(db), "--collapse-file", str(ledger)])
    # Both refs are now in the ledger → the second run judges nothing.
    result = runner.invoke(
        lexicon_merge_collapses, ["--db", str(db), "--collapse-file", str(ledger)]
    )
    assert result.exit_code == 0
    assert "already-judged=2 to-judge=0" in result.output
    assert "Nothing to judge." in result.output
