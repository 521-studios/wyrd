"""End-to-end CLI test for mine-cognate-descents-llm (wyrd-zrce.2) with a FAKE Ollama
client — no live LLM. Proves the two-pass judge → audit-log → emit-descends-from
wiring + resumability + the dry-run-emits-from-log path.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from wyrd.generators.kenning.canonicalization import load_assertions
from wyrd.generators.kenning.cli.lexicon import mine_cognate_descents_llm as cli_mod
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


class _FakeClient:
    """Stands in for OllamaClient: confirms the proposed link. Distinguishes the two
    passes by the refute prompt's wording."""

    def __init__(self, **_kw):
        self.calls = 0

    def chat_json(self, system, user, _schema):
        self.calls += 1
        if "efute" in user:  # PASS 2 refute prompt
            return {"confirmed": True, "confidence": "medium", "reason": "genuine cognate"}
        return {"choice": 1, "confidence": "medium", "reason": "matches candidate 1"}


def _seed_db(path):
    init_schema(path)
    db = LexiconDB(path)
    db.conn.execute("PRAGMA foreign_keys = OFF")
    root = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stanaz', 'proto-germanic')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    bridge = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stane', 'old-norse')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, bridge))
    db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stone', NULL)", (bridge,))
    x = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stant', 'old-english')"
    ).lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (x, x),
    )
    db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stony place', NULL)", (x,))
    db.commit()
    db.close()
    return root, x


def test_cli_judges_logs_and_emits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "OllamaClient", _FakeClient)
    db_path = tmp_path / "lexicon.db"
    mining = tmp_path / "mining"
    mining.mkdir()
    root, x = _seed_db(db_path)

    result = CliRunner().invoke(
        cli_mod.lexicon_mine_cognate_descents_llm,
        ["--db", str(db_path), "--mining-dir", str(mining), "--apply"],
    )
    assert result.exit_code == 0, result.output

    # Audit log recorded the confirmed judgment.
    log_lines = (mining / "_cognate_descent_audit.jsonl").read_text().strip().splitlines()
    rows = [
        json.loads(ln) for ln in log_lines if json.loads(ln).get("_type") == "cognate_descent_audit"
    ]
    assert len(rows) == 1 and rows[0]["confirmed"] is True and rows[0]["cluster_root"] == root

    # A descends-from assertion was authored (x -> root).
    assertions = [a for a in load_assertions(mining) if a.predicate == "descends-from"]
    assert len(assertions) == 1
    assert assertions[0].subject.ref == str(x) and assertions[0].object.ref == str(root)
    assert assertions[0].confidence == "medium"


def test_cli_resumes_and_dry_run_emits_from_log(tmp_path, monkeypatch):
    # First --apply run judges + logs. A second run finds nothing fresh (resumed), and
    # a dry-run still re-derives edges from the log without any LLM call.
    monkeypatch.setattr(cli_mod, "OllamaClient", _FakeClient)
    db_path = tmp_path / "lexicon.db"
    mining = tmp_path / "mining"
    mining.mkdir()
    _seed_db(db_path)
    runner = CliRunner()
    base = ["--db", str(db_path), "--mining-dir", str(mining)]

    first = runner.invoke(cli_mod.lexicon_mine_cognate_descents_llm, [*base, "--apply"])
    assert first.exit_code == 0
    log_after_first = (mining / "_cognate_descent_audit.jsonl").read_text()

    # Dry-run: no LLM, no new judgments, but reports edges derivable from the log.
    dry = runner.invoke(cli_mod.lexicon_mine_cognate_descents_llm, base)
    assert dry.exit_code == 0
    assert "dry-run" in dry.output
    assert "1 fresh" not in dry.output  # already judged → resumed
    assert (mining / "_cognate_descent_audit.jsonl").read_text() == log_after_first  # unchanged
