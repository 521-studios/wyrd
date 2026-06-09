"""Tests for ``wyrd kenning lexicon audit-reflexes`` (wyrd-862b).

The command LLM-judges suspicious reflex_etymon links (logging verdicts to
``_reflex_audit.jsonl``) and, with ``--apply``, prunes the PRUNE-verdict links +
re-dumps ``_reflexes.jsonl``. These pin the wyrd-8uvi-extracted helpers
(``_select_pending`` / ``_judge_one`` / ``_run_apply_prune``) end-to-end with a
STUB client (no real model call), via CliRunner against a seeded SQLite DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.audit_reflexes import lexicon_audit_reflexes
from wyrd.generators.kenning.lexicon import init_schema


class _StubClient:
    """Deterministic stand-in for GeminiClient — every judgment is a PRUNE
    verdict so the run-path + apply-path are exercised without a model call."""

    def chat_json(self, system, user, schema):  # noqa: ANN001 — matches the real signature
        return {"valid": False, "reason": "stub: not a real reflex"}


@pytest.fixture
def _stub_gemini(monkeypatch):
    # The command does a function-local `from ...extractors.gemini import
    # GeminiClient`, so patch the attribute on the source module.
    import wyrd.generators.kenning.extractors.gemini as gem

    monkeypatch.setattr(gem, "GeminiClient", _StubClient)


def _seed(db_path: Path) -> tuple[int, int]:
    """hām (OE, gloss 'dwelling') with two reflex links: '-burnham' (suspicious
    — not a known form) and '-ham' (the canonical's own form — pre-screen-kept).
    Returns the two reflex ids."""
    init_schema(db_path)
    c = sqlite3.connect(db_path)
    c.execute("INSERT INTO etymon(id, canonical_form, language) VALUES (1, 'hām', 'old-english')")
    c.execute("INSERT INTO etymon_gloss(etymon_id, gloss) VALUES (1, 'dwelling')")
    rid_burn = c.execute(
        "INSERT INTO reflex(surface_form, position, productivity) VALUES ('-burnham', 'post', 0)"
    ).lastrowid
    rid_ham = c.execute(
        "INSERT INTO reflex(surface_form, position, productivity) VALUES ('-ham', 'post', 0)"
    ).lastrowid
    c.execute("INSERT INTO reflex_etymon(reflex_id, etymon_id) VALUES (?, 1)", (rid_burn,))
    c.execute("INSERT INTO reflex_etymon(reflex_id, etymon_id) VALUES (?, 1)", (rid_ham,))
    c.commit()
    c.close()
    return rid_burn, rid_ham


def test_audit_reflexes_run_judges_only_suspicious(tmp_path: Path, _stub_gemini) -> None:
    """The run path judges only the suspicious link ('-burnham'); the known form
    ('-ham') is pre-screened out and never reaches the ledger."""
    db_path = tmp_path / "lex.db"
    _seed(db_path)
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    result = CliRunner().invoke(
        lexicon_audit_reflexes,
        ["--db", str(db_path), "--jsonl-dir", str(jsonl_dir), "--concurrency", "1"],
    )
    assert result.exit_code == 0
    rows = [
        json.loads(line)
        for line in (jsonl_dir / "_reflex_audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["surface_strip"] == "burnham"
    assert row["canon"] == "ham"
    assert row["gloss"] == "dwelling"
    assert row["valid"] is False
    assert row["reason"] == "stub: not a real reflex"


def test_audit_reflexes_apply_prunes_pruned_links(tmp_path: Path, _stub_gemini) -> None:
    """After a run logs a PRUNE verdict for '-burnham', --apply deletes that
    reflex_etymon link while keeping the pre-screen-kept '-ham' link."""
    db_path = tmp_path / "lex.db"
    _rid_burn, rid_ham = _seed(db_path)
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    runner = CliRunner()
    # Run (judge → ledger), then apply (prune).
    runner.invoke(
        lexicon_audit_reflexes,
        ["--db", str(db_path), "--jsonl-dir", str(jsonl_dir), "--concurrency", "1"],
    )
    result = runner.invoke(
        lexicon_audit_reflexes, ["--db", str(db_path), "--jsonl-dir", str(jsonl_dir), "--apply"]
    )
    assert result.exit_code == 0
    c = sqlite3.connect(db_path)
    links = sorted(c.execute("SELECT reflex_id, etymon_id FROM reflex_etymon").fetchall())
    c.close()
    assert links == [(rid_ham, 1)]  # only the kept '-ham' link survives


def test_audit_reflexes_no_pending_is_noop(tmp_path: Path, _stub_gemini) -> None:
    """A DB with no suspicious links writes no ledger and exits cleanly."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)  # empty — no reflex links at all
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    result = CliRunner().invoke(
        lexicon_audit_reflexes, ["--db", str(db_path), "--jsonl-dir", str(jsonl_dir)]
    )
    assert result.exit_code == 0
    assert not (jsonl_dir / "_reflex_audit.jsonl").exists()
