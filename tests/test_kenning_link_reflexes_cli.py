"""CLI-level tests for ``wyrd kenning lexicon link-reflexes`` (wyrd-rogd.15).

The reflex_link_detect unit pieces (detect_reflex_link_candidates /
build_reflex_judge_prompt / parse_reflex_verdict / reflex_verdict_to_row) are
covered in test_rogd15_reflex_detect.py. These pin the wyrd-8uvi-extracted
command orchestration (``_load_judged_links`` / ``_select_todo`` /
``_judge_and_record`` / ``_LinkRun``) end-to-end via CliRunner, with the
candidate detection and the LLM transport stubbed — no DB scan, no Ollama call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import wyrd.generators.kenning.extractors.llm as llm
import wyrd.generators.kenning.lexicon.reflex_link_detect as rl
from wyrd.generators.kenning.cli.lexicon.link_reflexes import lexicon_link_reflexes
from wyrd.generators.kenning.lexicon import init_schema

_CANDS = [
    rl.ReflexLinkCandidate("me:foo", "oe:foa", "foo", "foa", ("a gloss",), 0.9),
    rl.ReflexLinkCandidate("me:bar", "oe:baer", "bar", "baer", ("b gloss",), 0.8),
]


@pytest.fixture
def _stub_detect_and_llm(monkeypatch):
    """Stub candidate detection (canned cross-era pairs) + the Ollama transport
    (1st call → confident is_reflex link; 2nd → low not-reflex reject)."""
    monkeypatch.setattr(rl, "detect_reflex_link_candidates", lambda conn, **kw: list(_CANDS))
    calls = {"n": 0}

    def _fake_chat(self, system, user, schema):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return {"is_reflex": True, "confidence": "high", "reason": "is reflex"}
        return {"is_reflex": False, "confidence": "low", "reason": "coincidence"}

    monkeypatch.setattr(llm.OllamaClient, "chat_json", _fake_chat)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "lex.db"
    init_schema(db)  # so db.exists() + _readonly_lexicon open succeed (detect is stubbed)
    return db


def test_link_reflexes_dry_run_writes_nothing(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_link_reflexes, ["--db", str(db), "--collapse-file", str(ledger), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not ledger.exists()
    assert "would record: 1 links + 1 rejections (0 skipped)" in result.output


def test_link_reflexes_apply_records_link_and_reject(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    result = CliRunner().invoke(
        lexicon_link_reflexes, ["--db", str(db), "--collapse-file", str(ledger)]
    )
    assert result.exit_code == 0
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(rows) == 2
    link, reject = rows
    # Confident is_reflex → an inheritance LINK; low not-reflex → no-op.
    assert link["ref"] == "me:foo" and link["inherits"] == "oe:foa"
    assert reject["ref"] == "me:bar" and reject["inherits"] == ""
    assert "recorded: 1 links + 1 rejections (0 skipped)" in result.output


def test_link_reflexes_rerun_skips_already_judged(tmp_path: Path, _stub_detect_and_llm) -> None:
    db = _db(tmp_path)
    ledger = tmp_path / "_collapses.jsonl"
    runner = CliRunner()
    runner.invoke(lexicon_link_reflexes, ["--db", str(db), "--collapse-file", str(ledger)])
    # Both child refs are now in the ledger (with `inherits`) → second run judges nothing.
    result = runner.invoke(lexicon_link_reflexes, ["--db", str(db), "--collapse-file", str(ledger)])
    assert result.exit_code == 0
    assert "already-judged=2 to-judge=0" in result.output
    assert "Nothing to judge." in result.output
